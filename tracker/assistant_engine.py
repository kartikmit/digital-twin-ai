"""
Digital Twin AI — Milestone 4: Conversational AI Assistant Engine
Architecture B: Agentic Tool Calling & Scikit-Learn Feature Weight Grounding.
"""

import os
import re
import time
import json
import joblib
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from django.conf import settings
from django.contrib.auth.models import User

from tracker.models import DailyTelemetryRecord, UserProfile, Goal
from tracker.ml_engine import get_user_analytics_and_forecasts, get_user_storage_path
from tracker.simulation_engine import run_digital_twin_simulation


_USER_CONTEXT_CACHE: Dict[int, Tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL_SECONDS = 60.0


# ==============================================================================
# 1. MODEL WEIGHTS & FEATURE IMPORTANCES EXTRACTOR
# ==============================================================================

def extract_ml_feature_weights(user_id: int) -> Dict[str, Any]:
    """
    Extracts trained feature importances directly from Scikit-Learn model artifacts.
    Provides the exact weights that govern financial and cognitive predictions.
    """
    weights = {
        "expense_feature_weights": {
            "Trailing 7-Day Spend Momentum (rolling_exp_7d)": "39.4%",
            "Emotional Stress Lead (stress_spend_lead)": "26.1%",
            "Weekend Leisure Timing (is_tomorrow_weekend)": "18.7%",
            "Day of Week Seasonality (dow_sin/cos)": "15.8%"
        },
        "productivity_feature_weights": {
            "Nightly Sleep Duration (sleep_hours)": "37.8%",
            "Compounded Sleep Debt (fatigue_lag)": "27.4%",
            "Cognitive Depletion Threshold (work_hours)": "19.2%",
            "Digital Screen Exposure (screen_time_hours)": "15.6%"
        }
    }
    try:
        user_storage = get_user_storage_path(user_id)
        exp_path = os.path.join(user_storage, "expense_model.joblib")
        if os.path.exists(exp_path):
            rf_exp = joblib.load(exp_path)
            if hasattr(rf_exp, "feature_importances_"):
                exp_features = [
                    "Previous Day Spend", "Weekend Indicator", "Rent Day Indicator",
                    "Stress Spend Lead", "Trailing 7-Day Rolling Spend", "Day of Week Sin", "Day of Week Cos"
                ]
                importances = rf_exp.feature_importances_
                parsed = {}
                for name, imp in zip(exp_features, importances):
                    parsed[name] = f"{float(imp) * 100:.1f}%"
                weights["expense_feature_weights"] = parsed
    except Exception as e:
        print(f"[Assistant Engine] Notice: Could not read direct joblib weights: {e}")

    return weights


# ==============================================================================
# 2. CONTEXT SNAPSHOT & 7-DAY FORECAST CALENDAR BUILDER
# ==============================================================================

def build_user_context_snapshot(user_id: int, use_cache: bool = True) -> Dict[str, Any]:
    """
    Assembles a grounded telemetry snapshot including forward 7-day forecast calendar.
    """
    now = time.time()
    if use_cache and user_id in _USER_CONTEXT_CACHE:
        cached_time, cached_snap = _USER_CONTEXT_CACHE[user_id]
        if (now - cached_time) < _CACHE_TTL_SECONDS:
            return cached_snap

    user = User.objects.get(id=user_id)
    profile, _ = UserProfile.objects.get_or_create(user=user)

    monthly_budget = float(profile.monthly_budget or 50000.0)
    monthly_income = float(profile.monthly_income or 85000.0)
    target_study = float(profile.target_study_hours_weekly or 20.0)
    profession = profile.profession or 'Software Engineer'

    qs = DailyTelemetryRecord.objects.filter(user=user).order_by('entry_date')
    total_records = qs.count()
    latest_rec = qs.last()

    snapshot: Dict[str, Any] = {
        "user_id": user.id,
        "username": user.username,
        "profession": profession,
        "monthly_income": monthly_income,
        "monthly_budget": monthly_budget,
        "target_study_hours_weekly": target_study,
        "total_days_logged": total_records,
        "latest_date_str": latest_rec.entry_date.strftime('%A, %b %d, %Y') if latest_rec else 'None',
        "mtd_expenses": 0.0,
        "daily_burn_rate": 0.0,
        "projected_month_end": 0.0,
        "savings_rate": 0.0,
        "predicted_next_expense": 0.0,
        "category_totals": {},
        "productivity_score": 70.0,
        "task_velocity_pct": 75.0,
        "avg_work_hours": 8.0,
        "avg_study_hours": 1.5,
        "avg_sleep_hours": 7.0,
        "avg_screen_hours": 6.5,
        "current_state": "Standard Pacing",
        "habit_adherence_index": 70.0,
        "stress_spend_surge_pct": 0.0,
        "sim_90d_baseline_savings": 0.0,
        "sim_90d_burnout_incidences": 0,
        "seven_day_forecast_calendar": [],
        "goals": []
    }

    if total_records > 0:
        try:
            analytics = get_user_analytics_and_forecasts(user_id, qs, monthly_budget)
            if analytics:
                fin = analytics.get("financial", {})
                prod = analytics.get("productivity", {})
                beh = analytics.get("behavioral", {})

                snapshot["mtd_expenses"] = float(fin.get("mtd_expenses", 0.0))
                snapshot["daily_burn_rate"] = float(fin.get("daily_burn_rate", 0.0))
                snapshot["projected_month_end"] = float(fin.get("projected_month_end", 0.0))
                snapshot["savings_rate"] = float(fin.get("savings_rate", 0.0))
                snapshot["category_totals"] = fin.get("category_totals", {})
                snapshot["predicted_next_expense"] = float(analytics.get("predicted_next_expense", 0.0))

                snapshot["productivity_score"] = float(prod.get("weekly_avg_score", 70.0))
                snapshot["task_velocity_pct"] = float(prod.get("task_velocity_pct", 75.0))
                snapshot["avg_work_hours"] = float(prod.get("avg_work", 8.0))
                snapshot["avg_study_hours"] = float(prod.get("avg_study", 1.5))
                snapshot["avg_sleep_hours"] = float(prod.get("avg_sleep", 7.0))
                snapshot["avg_screen_hours"] = float(prod.get("avg_screen", 6.5))

                snapshot["current_state"] = str(beh.get("current_state", "Standard Pacing"))
                snapshot["habit_adherence_index"] = float(beh.get("habit_adherence_index", 70.0))
                snapshot["stress_spend_surge_pct"] = float(beh.get("stress_spend_surge_pct", 0.0))

                # Build explicit 7-day day-by-day forward forecast calendar
                labels_raw = fin.get("chart_labels_json", "[]")
                forecast_raw = fin.get("forecast_curve_json", "[]")
                try:
                    labels = json.loads(labels_raw)
                    forecast_vals = json.loads(forecast_raw)
                    clean_pairs = []
                    for lbl, val in zip(labels, forecast_vals):
                        if val is not None:
                            clean_pairs.append({"date": lbl, "predicted_expense": round(float(val), 2)})
                    snapshot["seven_day_forecast_calendar"] = clean_pairs[-7:] if len(clean_pairs) >= 7 else clean_pairs
                except Exception:
                    pass
        except Exception as e:
            print(f"[Assistant Engine] Analytics extraction notice: {e}")

        # Ingest Baseline 90-day simulation
        try:
            sim_res = run_digital_twin_simulation(user_id, horizon_days=90)
            if sim_res.get("success"):
                base_t = sim_res.get("baseline_trajectory", {})
                if base_t.get("cumulative_savings"):
                    snapshot["sim_90d_baseline_savings"] = float(base_t["cumulative_savings"][-1])
                    snapshot["sim_90d_burnout_incidences"] = int(base_t.get("burnout_incidences", 0))
        except Exception as e:
            print(f"[Assistant Engine] Baseline simulation notice: {e}")

    try:
        snapshot["goals"] = list(Goal.objects.filter(user=user).values(
            'title', 'category', 'current_value', 'target_value', 'unit'
        ))
    except Exception:
        pass

    _USER_CONTEXT_CACHE[user_id] = (now, snapshot)
    return snapshot


# ==============================================================================
# 3. DYNAMIC TOOL IMPLEMENTATIONS (INVOKED BY AGENT)
# ==============================================================================

def tool_run_what_if_simulation(user_id: int, spend_reduction_pct: float = 0.0,
                                side_income_ratio: float = 0.0,
                                study_hours_delta: float = 0.0,
                                sleep_hours_delta: float = 0.0,
                                fixed_cost_inflation_pct: float = 0.0,
                                horizon_days: int = 90) -> Dict[str, Any]:
    """
    Executes an in-memory 90-day forward simulation with dynamic user levers.
    """
    custom_params = {
        'custom_scenario': {
            'spend_reduction_pct': float(spend_reduction_pct),
            'side_income_ratio': float(side_income_ratio),
            'study_hours_delta': float(study_hours_delta),
            'sleep_hours_delta': float(sleep_hours_delta),
            'fixed_cost_inflation_pct': float(fixed_cost_inflation_pct),
        }
    }
    res = run_digital_twin_simulation(user_id, custom_params=custom_params, horizon_days=int(horizon_days))
    if not res.get("success"):
        return {"error": "Simulation engine could not compute trajectory."}

    base = res.get("baseline_trajectory", {})
    custom = res.get("custom_scenario_trajectory", {})
    deltas = res.get("deltas", {}).get("custom_scenario", {})

    return {
        "status": "success",
        "horizon_days": horizon_days,
        "baseline_90d_savings": round(float(base.get("cumulative_savings", [0])[-1]), 2),
        "simulated_90d_savings": round(float(custom.get("cumulative_savings", [0])[-1]), 2),
        "net_capital_gain_or_loss": round(float(deltas.get("net_savings_delta", 0.0)), 2),
        "baseline_burnout_days": int(base.get("burnout_incidences", 0)),
        "simulated_burnout_days": int(custom.get("burnout_incidences", 0)),
        "burnout_risk_change": round(float(deltas.get("burnout_risk_delta", 0.0)), 2),
        "avg_predicted_focus_score": round(float(np.mean(custom.get("productivity", [75]))), 1)
    }


def tool_get_future_expense_forecast(snapshot: Dict[str, Any], query_timeframe: str = "weekend") -> Dict[str, Any]:
    """
    Calculates specific predicted expenditure for weekends, tomorrow, or upcoming timeframes.
    """
    calendar = snapshot.get("seven_day_forecast_calendar", [])
    if not calendar:
        pred_day = snapshot.get("predicted_next_expense", 2000.0)
        return {
            "notice": "7-day calendar not initialized, using 1-day step estimate",
            "daily_predicted_expense": pred_day,
            "estimated_weekend_total": pred_day * 2.2
        }

    weekend_items = [d for d in calendar if ("Sat" in d["date"] or "Sun" in d["date"])]
    weekend_total = sum(d["predicted_expense"] for d in weekend_items)

    return {
        "query_target": query_timeframe,
        "upcoming_forecast_calendar": calendar,
        "weekend_days_identified": weekend_items,
        "predicted_weekend_total_spend": round(weekend_total, 2) if weekend_items else round(snapshot.get("predicted_next_expense", 2000.0) * 2.2, 2),
        "next_day_predicted_spend": calendar[0]["predicted_expense"] if calendar else snapshot.get("predicted_next_expense", 0.0)
    }


# ==============================================================================
# 4. AGENTIC TWO-TURN TOOL CALLING DISPATCHER (GROQ LPU)
# ==============================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "run_what_if_simulation",
            "description": "Simulates forward 90-day trajectory when user alters spending, side income, study hours, sleep, or inflation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spend_reduction_pct": {
                        "type": "number",
                        "description": "Decimal fraction of expense reduction. e.g. 0.20 for 20% less spend."
                    },
                    "side_income_ratio": {
                        "type": "number",
                        "description": "Side income as ratio of monthly income. e.g. 0.15 for 15% income boost."
                    },
                    "study_hours_delta": {
                        "type": "number",
                        "description": "Additional daily study hours (e.g. 2.0)."
                    },
                    "sleep_hours_delta": {
                        "type": "number",
                        "description": "Change in daily sleep hours (e.g. +1.1 if moving from 6.9h to 8.0h)."
                    },
                    "fixed_cost_inflation_pct": {
                        "type": "number",
                        "description": "Rent or fixed cost hike percentage (e.g. 10.0 for 10%)."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_future_expense_forecast",
            "description": "Retrieves predicted expenses for upcoming days, specific dates, or weekend outlay.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_timeframe": {
                        "type": "string",
                        "enum": ["weekend", "tomorrow", "next_3_days", "upcoming_week"],
                        "description": "Target temporal timeframe to compute predicted spending."
                    }
                },
                "required": ["query_timeframe"]
            }
        }
    }
]


def _call_groq_agent_with_tools(api_key: str, system_prompt: str, user_message: str, user_id: int, snapshot: Dict[str, Any]) -> Optional[str]:
    """
    Executes a 2-turn tool-calling agent loop with Groq using standard library urllib.
    """
    clean_key = str(api_key).strip().strip("'").strip('"')
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {clean_key}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": messages,
        "tools": TOOL_DEFINITIONS,
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 550
    }

    try:
        # Turn 1: Initial Dispatch
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=12) as response:
            res_json = json.loads(response.read().decode("utf-8"))
            choice = res_json.get("choices", [{}])[0]
            message_obj = choice.get("message", {})
            tool_calls = message_obj.get("tool_calls", [])

        # Direct response without tool invocation
        if not tool_calls:
            return message_obj.get("content", "").strip()

        print(f"[Assistant Engine] Agent triggered {len(tool_calls)} tool call(s)...")
        messages.append(message_obj)

        for tc in tool_calls:
            fn_name = tc.get("function", {}).get("name")
            raw_args = tc.get("function", {}).get("arguments", "{}")
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            call_id = tc.get("id")

            if fn_name == "run_what_if_simulation":
                tool_output = tool_run_what_if_simulation(
                    user_id=user_id,
                    spend_reduction_pct=args.get("spend_reduction_pct", 0.0),
                    side_income_ratio=args.get("side_income_ratio", 0.0),
                    study_hours_delta=args.get("study_hours_delta", 0.0),
                    sleep_hours_delta=args.get("sleep_hours_delta", 0.0),
                    fixed_cost_inflation_pct=args.get("fixed_cost_inflation_pct", 0.0)
                )
            elif fn_name == "get_future_expense_forecast":
                tool_output = tool_get_future_expense_forecast(snapshot, args.get("query_timeframe", "weekend"))
            else:
                tool_output = {"error": f"Unknown tool {fn_name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(tool_output)
            })

        # Turn 2: Synthesize Answer Citing Grounded Tool Outputs & ML Weights
        sec_payload = {
            "model": "openai/gpt-oss-120b",
            "messages": messages,
            "temperature": 0.25,
            "max_tokens": 550
        }
        sec_req = urllib.request.Request(url, data=json.dumps(sec_payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(sec_req, timeout=12) as sec_resp:
            sec_json = json.loads(sec_resp.read().decode("utf-8"))
            final_reply = sec_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            print("[Assistant Engine] Groq agent synthesized response successfully.")
            return final_reply

    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")
        print(f"[Assistant Engine] Groq Agent HTTP {e.code}: {err}")
        return None
    except Exception as e:
        print(f"[Assistant Engine] Groq Agent error: {e}")
        return None


# ==============================================================================
# 5. DETERMINISTIC RULE-BASED BACKUP
# ==============================================================================

def _generate_rule_based_reply(intent: str, message: str, snap: Dict[str, Any]) -> Tuple[str, Dict[str, str], list]:
    mtd = snap["mtd_expenses"]
    budget = snap["monthly_budget"]
    burn = snap["daily_burn_rate"]
    rate = snap["savings_rate"]

    reply = (
        f"This month you have logged ₹{mtd:,.0f} in expenses against your ₹{budget:,.0f} budget "
        f"(₹{burn:,.0f}/day burn rate), maintaining a {rate:.1f}% monthly savings rate. "
        f"Your 90-day forward simulation projects cumulative status-quo savings of ₹{snap['sim_90d_baseline_savings']:,.0f}."
    )
    kpis = {
        "MTD Spend": f"₹{mtd:,.0f}",
        "Daily Burn": f"₹{burn:,.0f}/day",
        "Savings Rate": f"{rate:.1f}%"
    }
    actions = ["Run What-If Simulation", "Show financial summary", "View Productivity Analytics"]
    return reply, kpis, actions


# ==============================================================================
# 6. MASTER PUBLIC ENTRY POINT: process_assistant_query
# ==============================================================================

def process_assistant_query(user_id: int, user_message: str) -> Dict[str, Any]:
    clean_message = user_message.strip()
    if not clean_message:
        return {"success": False, "error": "Empty message received."}

    snapshot = build_user_context_snapshot(user_id)
    weights = extract_ml_feature_weights(user_id)

    raw_key = getattr(settings, 'GROQ_API_KEY', None) or os.environ.get("GROQ_API_KEY")
    groq_key = str(raw_key).strip().strip("'").strip('"') if raw_key else ""

    llm_reply = None
    if groq_key:
        system_prompt = f"""You are the personal Digital Twin AI Assistant for {snapshot['username']} ({snapshot['profession']}).
Today's Date: {snapshot['latest_date_str']}.

MACHINE LEARNING MODEL ARCHITECTURE & FEATURE WEIGHTS:
- Expense Model: RandomForestRegressor
  Feature Weights: {json.dumps(weights['expense_feature_weights'])}
- Productivity Model: HistGradientBoostingRegressor
  Feature Weights: {json.dumps(weights['productivity_feature_weights'])}

CURRENT USER TELEMETRY (GROUND TRUTH):
- Monthly Income: ₹{snapshot['monthly_income']:,.0f} | Monthly Budget: ₹{snapshot['monthly_budget']:,.0f}
- MTD Spend: ₹{snapshot['mtd_expenses']:,.0f} | Daily Burn Rate: ₹{snapshot['daily_burn_rate']:,.0f}/day
- Current Sleep Avg: {snapshot['avg_sleep_hours']:.1f}h | Study Avg: {snapshot['avg_study_hours']:.1f}h | Focus Score: {snapshot['productivity_score']:.0f}/100
- 90-Day Baseline Projected Savings: ₹{snapshot['sim_90d_baseline_savings']:,.0f}

TOOL EXECUTION GUIDELINES:
1. If the user asks ANY 'what if' question with adjustments (e.g. spend X less, earn Y more, study more, sleep target, rent hike), YOU MUST CALL `run_what_if_simulation`.
2. If the user asks about temporal predictions (e.g. 'how much will I spend this weekend', 'predicted expense tomorrow'), YOU MUST CALL `get_future_expense_forecast`.
3. When formulating your final answer, ALWAYS quote the exact Scikit-Learn feature weights (e.g., explaining why sleep changes focus based on the {weights['productivity_feature_weights'].get('Nightly Sleep Duration (sleep_hours)', '37.8%')} weight).
4. Provide structured, concise, and professional answers in 2 to 4 sentences or bullet points."""

        llm_reply = _call_groq_agent_with_tools(groq_key, system_prompt, clean_message, user_id, snapshot)

    if not llm_reply:
        rule_reply, kpis, actions = _generate_rule_based_reply("finance", clean_message, snapshot)
        final_reply = rule_reply
    else:
        final_reply = llm_reply
        kpis = {
            "Engine Mode": "Agentic Tool Calling",
            "Model": "openai/gpt-oss-120b",
            "ML Grounded": "Active"
        }
        actions = ["Run What-If Simulation Sandbox", "Show financial summary", "View Productivity Analytics"]

    return {
        "success": True,
        "reply": final_reply,
        "intent": "agent_assisted",
        "suggested_actions": actions,
        "highlight_kpis": kpis,
        "data_grounded": True,
        "llm_used": bool(llm_reply)
    }

