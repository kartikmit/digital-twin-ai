"""
Digital Twin AI — Milestone 3: What-If Simulation Engine
100% In-Memory Forward Trajectory Projection Controller.

Non-Negotiable Invariants:
1. Zero Database Writes: DailyTelemetryRecord is strictly read-only during simulation.
2. Conservation of Cash Flow: Net savings balance strictly to the rupee:
   Simulated Savings = Projected Income - Simulated Expenses
3. Dynamic Forward Causal State Updates: dow_sin, dow_cos, is_tomorrow_weekend,
   is_tomorrow_rent, rolling_exp_7d, fatigue_lag, and stress_spend_lead update dynamically each day.
4. Decimal-Float Safety: Explicit float casting across all Django ORM and NumPy computations.
"""
import os
import sys
import copy
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

# Ensure project root is on sys.path for direct script execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import django
from django.apps import apps
if not apps.ready:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "digitaltwin_project.settings")
    django.setup()

import joblib
import numpy as np
import pandas as pd

from tracker.ml_engine import get_user_storage_path
from tracker.models import DailyTelemetryRecord, UserProfile

# In-memory model and result caches to eliminate server latency and socket timeouts
_LOADED_MODELS_CACHE = {}
_SIMULATION_RESULT_CACHE = {}


def _calculate_burnout_risk_index(work_hours: float, stress_level: float, sleep_hours: float) -> float:
    """
    Computes a calibrated, continuous burnout risk index (0.0 to 100.0%).
    Weights:
      - Work hours intensity (35%): Work beyond 8 hours adds significant strain.
      - Daily stress level (40%): Subjective and physiological stress rating.
      - Sleep debt / deprivation (25%): Deficit below optimal 8.0 hours.
    """
    work_component = (float(work_hours) / 10.0) * 35.0
    stress_component = (float(stress_level) / 10.0) * 40.0
    sleep_deficit = max(0.0, (8.0 - float(sleep_hours)) / 8.0) * 25.0
    raw_risk = work_component + stress_component + sleep_deficit
    return round(float(np.clip(raw_risk, 0.0, 100.0)), 1)


def _calculate_fallback_cognitive_score(
    work_hours: float,
    study_hours: float,
    screen_time_hours: float,
    sleep_hours: float,
    stress_level: float,
    tasks_assigned: int = 8,
    tasks_completed: int = 7
) -> float:
    """
    Fallback deterministic cognitive readiness formula matching manual_daily_entry
    when Scikit-Learn productivity model artifact is unavailable (< 7 days cold start).
    """
    c_task = min(1.0, float(tasks_completed) / max(1.0, float(tasks_assigned)))
    tot_act = float(work_hours) + float(study_hours)
    denom = max(float(screen_time_hours), tot_act) + 0.5
    e_focus = float(np.clip(tot_act / max(0.1, denom), 0.0, 1.0))
    r_cog = float(np.clip(float(sleep_hours) / 8.0, 0.0, 1.0) * ((11.0 - float(stress_level)) / 10.0))
    score = 100.0 * (0.40 * c_task + 0.35 * e_focus + 0.25 * r_cog)
    return round(float(np.clip(score, 0.0, 100.0)), 2)


def _extract_user_telemetry_baselines(user_id: int):
    """
    Extracts personalized baseline habits, spending splits, and starting points
    from the user's historical telemetry records without mutating the database.
    """
    records_qs = DailyTelemetryRecord.objects.filter(user_id=user_id).order_by('entry_date')
    count = records_qs.count()

    if count == 0:
        return None, None, None

    latest_record = records_qs.last()

    # Pre-extract numerical values into arrays for vector statistics
    recent_records_qs = records_qs.order_by('-entry_date')[:60]
    records_data = list(reversed(list(recent_records_qs.values(
    'entry_date', 'total_expenses', 'food_expenses', 'shopping_expenses',
    'work_hours', 'study_hours', 'screen_time_hours', 'sleep_hours',
    'stress_level', 'steps_count', 'productivity_score'
))))

    wkday_works, wknd_works = [], []
    wkday_sleeps, wknd_sleeps = [], []
    wkday_stresses, wknd_stresses = [], []
    wkday_screens, wknd_screens = [], []
    wkday_studies, wknd_studies = [], []
    wkday_steps, wknd_steps = [], []
    wkday_disc_ratios, wknd_disc_ratios = [], []

    for r in records_data:
        dow = r['entry_date'].weekday()
        is_wknd = dow in [5, 6]
        tot_exp = float(r['total_expenses'] or 0.0)
        disc = float(r['food_expenses'] or 0.0) + float(r['shopping_expenses'] or 0.0)
        ratio = (disc / tot_exp) if tot_exp > 0 else 0.70

        if is_wknd:
            wknd_works.append(float(r['work_hours'] or 0.0))
            wknd_sleeps.append(float(r['sleep_hours'] or 7.0))
            wknd_stresses.append(float(r['stress_level'] or 3.0))
            wknd_screens.append(float(r['screen_time_hours'] or 7.0))
            wknd_studies.append(float(r['study_hours'] or 1.5))
            wknd_steps.append(float(r['steps_count'] or 7000))
            wknd_disc_ratios.append(ratio)
        else:
            wkday_works.append(float(r['work_hours'] or 8.0))
            wkday_sleeps.append(float(r['sleep_hours'] or 7.0))
            wkday_stresses.append(float(r['stress_level'] or 4.5))
            wkday_screens.append(float(r['screen_time_hours'] or 10.0))
            wkday_studies.append(float(r['study_hours'] or 1.5))
            wkday_steps.append(float(r['steps_count'] or 6000))
            wkday_disc_ratios.append(ratio)

    baselines = {
        'wkday': {
            'work_hours': float(np.mean(wkday_works)) if wkday_works else 8.0,
            'sleep_hours': float(np.mean(wkday_sleeps)) if wkday_sleeps else 6.8,
            'stress_level': float(np.mean(wkday_stresses)) if wkday_stresses else 4.2,
            'screen_time_hours': float(np.mean(wkday_screens)) if wkday_screens else 10.5,
            'study_hours': float(np.mean(wkday_studies)) if wkday_studies else 1.5,
            'steps_count': int(np.mean(wkday_steps)) if wkday_steps else 6000,
            'disc_ratio': float(np.mean(wkday_disc_ratios)) if wkday_disc_ratios else 0.65,
        },
        'wknd': {
            'work_hours': float(np.mean(wknd_works)) if wknd_works else 0.8,
            'sleep_hours': float(np.mean(wknd_sleeps)) if wknd_sleeps else 7.2,
            'stress_level': float(np.mean(wknd_stresses)) if wknd_stresses else 2.5,
            'screen_time_hours': float(np.mean(wknd_screens)) if wknd_screens else 7.0,
            'study_hours': float(np.mean(wknd_studies)) if wknd_studies else 1.8,
            'steps_count': int(np.mean(wknd_steps)) if wknd_steps else 8000,
            'disc_ratio': float(np.mean(wknd_disc_ratios)) if wknd_disc_ratios else 0.90,
        },
        'rolling_7d_history': [float(x['total_expenses'] or 0.0) for x in records_data[-7:]]
    }

    # Ensure buffer has at least 7 elements
    while len(baselines['rolling_7d_history']) < 7:
        pad_val = baselines['rolling_7d_history'][-1] if baselines['rolling_7d_history'] else 750.0
        baselines['rolling_7d_history'].insert(0, pad_val)

    return latest_record, baselines, count


def _build_scenario_configs(custom_params=None, monthly_income=85000.0, monthly_budget=50000.0):
    """
    Normalizes scenario parameters for all 4 trajectories:
      1. Baseline (Status Quo): Zero intervention across all levers.
      2. Custom Scenario: Dynamically driven by live user slider positions and multi-levers.
      3. Scenario A (Controlled Benchmark): Evidence-based optimization (+10% savings, +1h study, +0.5h sleep, -0.5h screen, 4% yield).
      4. Scenario B (Stress Shock Benchmark): Adverse shock (+25% spend surge, emergency shock of 35% budget, -1h study, -1.5h sleep, +4h screen).
    """
    custom_params = custom_params or {}
    monthly_income = float(monthly_income or 85000.0)
    monthly_budget = float(monthly_budget or 50000.0)

    # 1. Baseline: zero intervention
    scenario_base = {
        'name': 'Baseline (Status Quo)',
        'description': 'Current habits and financial spending patterns forward projection.',
        'spend_reduction_pct': 0.0,
        'study_hours_delta': 0.0,
        'sleep_hours_delta': 0.0,
        'screen_time_delta': 0.0,
        'work_hours_delta': 0.0,
        'stress_level_delta': 0.0,
        'steps_delta': 0,
        'fixed_cost_inflation_pct': 0.0,
        'side_income_ratio': 0.0,
        'daily_side_income': 0.0,
        'shock_ratio': 0.0,
        'resolved_shock_amount': 0.0,
        'shock_day': 30,
        'savings_yield_annual_pct': 0.04,
        'sprint_mode_active': False,
    }

    # 2. Custom Scenario (Live User Slider & Multi-Lever Driven)
    custom_cfg = custom_params.get('custom_scenario', {})
    custom_spend_pct = float(custom_cfg.get('spend_reduction_pct', 0.10))
    custom_side_ratio = float(custom_cfg.get('side_income_ratio', 0.0))
    custom_daily_side = round((monthly_income * custom_side_ratio) / 30.0, 2)
    custom_shock_ratio = float(custom_cfg.get('shock_ratio', 0.0))
    custom_shock_amount = float(custom_cfg.get(
        'resolved_shock_amount',
        custom_cfg.get('shock_amount', round(monthly_budget * custom_shock_ratio, 2))
    ))

    scenario_custom = {
        'name': custom_cfg.get('name', 'Custom What-If Scenario'),
        'description': custom_cfg.get('description', 'Live user-adjusted parameters and dynamic lifestyle tuning.'),
        'spend_reduction_pct': custom_spend_pct,
        'study_hours_delta': float(custom_cfg.get('study_hours_delta', 1.0)),
        'sleep_hours_delta': float(custom_cfg.get('sleep_hours_delta', 0.5)),
        'screen_time_delta': float(custom_cfg.get('screen_time_delta', -0.5)),
        'work_hours_delta': float(custom_cfg.get('work_hours_delta', 0.0)),
        'stress_level_delta': float(custom_cfg.get('stress_level_delta', -1.0 if custom_spend_pct >= 0 else 1.5)),
        'steps_delta': int(custom_cfg.get('steps_delta', 2000 if custom_spend_pct >= 0 else -1000)),
        'fixed_cost_inflation_pct': float(custom_cfg.get('fixed_cost_inflation_pct', 0.0)),
        'side_income_ratio': custom_side_ratio,
        'daily_side_income': custom_daily_side,
        'shock_ratio': custom_shock_ratio,
        'resolved_shock_amount': custom_shock_amount,
        'shock_day': int(custom_cfg.get('shock_day', 30)),
        'savings_yield_annual_pct': float(custom_cfg.get('savings_yield_annual_pct', 0.04)),
        'sprint_mode_active': bool(custom_cfg.get('sprint_mode_active', False)),
    }

    # 3. Scenario A (Controlled Benchmark)
    custom_a = custom_params.get('scenario_a', {})
    scenario_a = {
        'name': custom_a.get('name', 'Scenario A (Controlled Benchmark)'),
        'description': custom_a.get('description', 'Conservative lifestyle tuning: -10% discretionary spend, +1h study, +0.5h sleep, -0.5h screen time, 4% compound savings yield.'),
        'spend_reduction_pct': float(custom_a.get('spend_reduction_pct', 0.10)),
        'study_hours_delta': float(custom_a.get('study_hours_delta', 1.0)),
        'sleep_hours_delta': float(custom_a.get('sleep_hours_delta', 0.5)),
        'screen_time_delta': float(custom_a.get('screen_time_delta', -0.5)),
        'work_hours_delta': float(custom_a.get('work_hours_delta', 0.0)),
        'stress_level_delta': float(custom_a.get('stress_level_delta', -1.0)),
        'steps_delta': int(custom_a.get('steps_delta', 2000)),
        'fixed_cost_inflation_pct': float(custom_a.get('fixed_cost_inflation_pct', 0.0)),
        'side_income_ratio': float(custom_a.get('side_income_ratio', 0.0)),
        'daily_side_income': 0.0,
        'shock_ratio': 0.0,
        'resolved_shock_amount': 0.0,
        'shock_day': int(custom_a.get('shock_day', 30)),
        'savings_yield_annual_pct': float(custom_a.get('savings_yield_annual_pct', 0.04)),
        'sprint_mode_active': bool(custom_a.get('sprint_mode_active', False)),
    }

    # 4. Scenario B (Stress Shock Benchmark)
    custom_b = custom_params.get('scenario_b', {})
    mode_b = custom_b.get('mode', custom_params.get('mode', 'stress_shock'))

    if mode_b == 'aggressive':
        scenario_b = {
            'name': custom_b.get('name', 'Scenario B (Aggressive Optimization)'),
            'description': custom_b.get('description', 'Maximized optimization: -20% discretionary spend, +2h study, +1h sleep, -1h screen time, 4% compound yield.'),
            'mode': 'aggressive',
            'spend_reduction_pct': float(custom_b.get('spend_reduction_pct', 0.20)),
            'study_hours_delta': float(custom_b.get('study_hours_delta', 2.0)),
            'sleep_hours_delta': float(custom_b.get('sleep_hours_delta', 1.0)),
            'screen_time_delta': float(custom_b.get('screen_time_delta', -1.0)),
            'work_hours_delta': float(custom_b.get('work_hours_delta', 0.0)),
            'stress_level_delta': float(custom_b.get('stress_level_delta', -2.0)),
            'steps_delta': int(custom_b.get('steps_delta', 4000)),
            'fixed_cost_inflation_pct': float(custom_b.get('fixed_cost_inflation_pct', 0.0)),
            'side_income_ratio': float(custom_b.get('side_income_ratio', 0.0)),
            'daily_side_income': 0.0,
            'shock_ratio': 0.0,
            'resolved_shock_amount': 0.0,
            'shock_day': int(custom_b.get('shock_day', 30)),
            'savings_yield_annual_pct': float(custom_b.get('savings_yield_annual_pct', 0.04)),
            'sprint_mode_active': bool(custom_b.get('sprint_mode_active', False)),
        }
    else:
        b_shock_ratio = float(custom_b.get('shock_ratio', 0.35))
        b_shock_day = int(custom_b.get('shock_day', 30))
        b_resolved_shock = float(custom_b.get(
            'resolved_shock_amount',
            custom_b.get('shock_amount', round(monthly_budget * b_shock_ratio, 2))
        ))
        scenario_b = {
            'name': custom_b.get('name', 'Scenario B (Stress Shock Benchmark)'),
            'description': custom_b.get(
                'description',
                f"Adverse shock: +25% stress spend surge, emergency capital shock (35% budget = ₹{b_resolved_shock:,.0f} on Day {b_shock_day}), -1.0h study, -1.5h sleep, +2.5 stress level, +4.0h screen time."
            ),
            'mode': 'stress_shock',
            'spend_reduction_pct': float(custom_b.get('spend_reduction_pct', -0.25)),  # Negative reduction = surge
            'study_hours_delta': float(custom_b.get('study_hours_delta', -1.0)),
            'sleep_hours_delta': float(custom_b.get('sleep_hours_delta', -1.5)),
            'screen_time_delta': float(custom_b.get('screen_time_delta', 4.0)),
            'work_hours_delta': float(custom_b.get('work_hours_delta', 0.0)),
            'stress_level_delta': float(custom_b.get('stress_level_delta', 2.5)),
            'steps_delta': int(custom_b.get('steps_delta', -2000)),
            'fixed_cost_inflation_pct': float(custom_b.get('fixed_cost_inflation_pct', 0.0)),
            'side_income_ratio': float(custom_b.get('side_income_ratio', 0.0)),
            'daily_side_income': 0.0,
            'shock_ratio': b_shock_ratio,
            'resolved_shock_amount': b_resolved_shock,
            'shock_day': b_shock_day,
            'savings_yield_annual_pct': float(custom_b.get('savings_yield_annual_pct', 0.04)),
            'sprint_mode_active': bool(custom_b.get('sprint_mode_active', False)),
        }

    return scenario_base, scenario_custom, scenario_a, scenario_b


def run_digital_twin_simulation(user_id: int, custom_params: dict = None, horizon_days: int = 90):
    """
    Executes a multi-scenario forward digital twin simulation over horizon_days (default: 90).
    Guaranteed 100% in-memory with zero database side-effects.

    Parameters:
      user_id (int): ID of the authenticated user.
      custom_params (dict, optional): Overrides for scenarios A and B.
      horizon_days (int): Forward forecasting horizon (typically 30, 60, or 90).

    Returns:
      dict: Standardized simulation payload with baseline, scenario_a, scenario_b trajectories,
            exact deltas, checkpoints, and execution metadata.
    """
    user_id = int(user_id)
    horizon_days = max(7, min(365, int(horizon_days)))

    # Step 1: Read-only telemetry extraction
    latest_record, baselines, total_records = _extract_user_telemetry_baselines(user_id)
    if not latest_record:
        return {
            'success': False,
            'error': f'No telemetry records found for user_id={user_id}. Log daily entries or upload CSV.'
        }

    # Step 2: User profile & income parameters
    profile = UserProfile.objects.filter(user_id=user_id).first()
    monthly_income = float(profile.monthly_income) if (profile and profile.monthly_income) else 85000.0
    monthly_budget = float(profile.monthly_budget) if (profile and profile.monthly_budget) else 50000.0
    daily_income = round(float(monthly_income) / 30.0, 2)

    # Fast in-memory result cache check (TTL: 60s, keyed on user, latest record, horizon & params)
    cache_key = (user_id, latest_record.id, horizon_days, json.dumps(custom_params, sort_keys=True) if custom_params else '')
    now_ts = time.time()
    if cache_key in _SIMULATION_RESULT_CACHE:
        cached_ts, cached_res = _SIMULATION_RESULT_CACHE[cache_key]
        if now_ts - cached_ts < 60.0:
            return copy.deepcopy(cached_res)

    # Step 3: Load trained Scikit-Learn model artifacts (in-memory cached)
    user_storage = get_user_storage_path(user_id)
    exp_path = os.path.join(user_storage, "expense_model.joblib")
    prod_path = os.path.join(user_storage, "productivity_model.joblib")

    if exp_path not in _LOADED_MODELS_CACHE and os.path.exists(exp_path):
        _LOADED_MODELS_CACHE[exp_path] = joblib.load(exp_path)
    if prod_path not in _LOADED_MODELS_CACHE and os.path.exists(prod_path):
        _LOADED_MODELS_CACHE[prod_path] = joblib.load(prod_path)

    rf_exp = _LOADED_MODELS_CACHE.get(exp_path)
    hgb_prod = _LOADED_MODELS_CACHE.get(prod_path)

    # Step 4: Scenario setups
    scenario_base, scenario_custom, scenario_a, scenario_b = _build_scenario_configs(
        custom_params, monthly_income=monthly_income, monthly_budget=monthly_budget
    )
    scenarios = [
        ('baseline', scenario_base),
        ('custom_scenario', scenario_custom),
        ('scenario_a', scenario_a),
        ('scenario_b', scenario_b)
    ]

    exp_features = [
        'total_expenses', 'is_tomorrow_weekend', 'is_tomorrow_rent',
        'stress_spend_lead', 'rolling_exp_7d', 'dow_sin', 'dow_cos'
    ]
    prod_features = [
        'productivity_score', 'sleep_hours', 'fatigue_lag', 'work_hours',
        'is_tomorrow_weekend', 'stress_level', 'screen_time_hours'
    ]

    start_date = latest_record.entry_date
    simulation_trajectories = {}

    # High-performance tree references and pre-allocated prediction buffers
    rf_trees = [getattr(e, 'tree_', e) for e in rf_exp.estimators_] if (rf_exp and hasattr(rf_exp, 'estimators_')) else None
    exp_arr = np.zeros((1, 7), dtype=np.float32)
    prod_arr = np.zeros((1, 7), dtype=np.float64)

    # Step 5: Day-by-day forward autoregressive iteration per scenario
    for sc_key, sc_cfg in scenarios:
        dates_list = []
        expenses_list = []
        cumulative_savings_list = []
        productivity_list = []
        burnout_risk_list = []
        yields_list = []

        # In-memory working buffers
        rolling_exp_buf = copy.deepcopy(baselines['rolling_7d_history'])
        prev_exp = float(latest_record.total_expenses or 750.0)
        prev_prod = float(latest_record.productivity_score or 80.0)
        cum_savings = 0.0
        burnout_incidences_count = 0

        # Dynamic economic and cognitive levers for this scenario
        daily_inflow = round(daily_income + float(sc_cfg.get('daily_side_income', 0.0)), 2)
        yield_rate = float(sc_cfg.get('savings_yield_annual_pct', 0.04)) / 365.0
        shock_day = int(sc_cfg.get('shock_day', 30))
        resolved_shock = float(sc_cfg.get('resolved_shock_amount', 0.0))
        fixed_inflation = float(sc_cfg.get('fixed_cost_inflation_pct', 0.0))
        spend_cut_pct = float(sc_cfg.get('spend_reduction_pct', 0.0))
        sprint_active = bool(sc_cfg.get('sprint_mode_active', False))

        for day_step in range(1, horizon_days + 1):
            curr_target_date = start_date + timedelta(days=day_step)
            next_calendar_date = curr_target_date + timedelta(days=1)
            f_dow = curr_target_date.weekday()
            is_wknd = f_dow in [5, 6]

            # Cyclic trigonometric coordinates
            dow_sin = float(np.sin(2.0 * np.pi * f_dow / 7.0))
            dow_cos = float(np.cos(2.0 * np.pi * f_dow / 7.0))
            is_tomorrow_weekend = 1.0 if next_calendar_date.weekday() in [5, 6] else 0.0
            is_tomorrow_rent = 1.0 if next_calendar_date.day == 1 else 0.0

            # Dynamic Behavioral State for this scenario
            day_type = 'wknd' if is_wknd else 'wkday'
            base_habits = baselines[day_type]

            work_hrs = max(0.0, base_habits['work_hours'])
            sleep_hrs = min(11.0, max(3.5, base_habits['sleep_hours'] + sc_cfg['sleep_hours_delta']))
            stress_lvl = min(10.0, max(1.0, base_habits['stress_level'] + sc_cfg['stress_level_delta']))
            screen_hrs = max(1.0, base_habits['screen_time_hours'] + sc_cfg['screen_time_delta'])
            study_hrs = max(0.0, base_habits['study_hours'] + sc_cfg['study_hours_delta'])
            # Sprint mode temporary crunch (Days 20 to 34 inclusive = 14 days)
            in_sprint = sprint_active and (20 <= day_step <= 34)

            work_hrs = max(0.0, base_habits['work_hours'] + sc_cfg.get('work_hours_delta', 0.0) + (2.0 if in_sprint else 0.0))
            sleep_hrs = min(11.0, max(3.5, base_habits['sleep_hours'] + sc_cfg.get('sleep_hours_delta', 0.0) - (1.0 if in_sprint else 0.0)))
            stress_lvl = min(10.0, max(1.0, base_habits['stress_level'] + sc_cfg.get('stress_level_delta', 0.0) + (1.5 if in_sprint else 0.0)))
            screen_hrs = max(1.0, base_habits['screen_time_hours'] + sc_cfg.get('screen_time_delta', 0.0))
            study_hrs = max(0.0, base_habits['study_hours'] + sc_cfg.get('study_hours_delta', 0.0))

            # Dynamic Causal Lags
            fatigue_lag = float((work_hrs / 8.0) - (sleep_hrs / 8.0))
            rolling_7d = float(np.mean(rolling_exp_buf))

            if (fatigue_lag > 0.35) or (stress_lvl >= 7.0):
                burnout_incidences_count += 1

            # Base expense prediction
            disc_ratio = base_habits['disc_ratio']
            base_est_disc = prev_exp * disc_ratio
            stress_spend_lead = base_est_disc if stress_lvl >= 7.0 else 0.0

            if rf_trees:
                exp_arr[0, 0] = prev_exp
                exp_arr[0, 1] = is_tomorrow_weekend
                exp_arr[0, 2] = is_tomorrow_rent
                exp_arr[0, 3] = stress_spend_lead
                exp_arr[0, 4] = rolling_7d
                exp_arr[0, 5] = dow_sin
                exp_arr[0, 6] = dow_cos
                pred_raw_exp = float(np.mean([t.predict(exp_arr)[0] for t in rf_trees]))
            elif rf_exp:
                exp_input_df = pd.DataFrame([[
                    prev_exp, is_tomorrow_weekend, is_tomorrow_rent,
                    stress_spend_lead, rolling_7d, dow_sin, dow_cos
                ]], columns=exp_features).astype(float)
                pred_raw_exp = float(rf_exp.predict(exp_input_df)[0])
            else:
                # Fallback causal burn rate
                daily_burn = float(monthly_budget) / 30.0
                pred_raw_exp = daily_burn * (1.75 if is_wknd else 0.95)
                if is_tomorrow_rent:
                    pred_raw_exp += 22000.0 # Rent proxy

            pred_raw_exp = max(50.0, pred_raw_exp)

            # Apply Scenario Discretionary Spend Intervention
            # Economic Scaling & Discretionary Adjustment
            disc_component = pred_raw_exp * disc_ratio
            fixed_component = pred_raw_exp - disc_component

            # spend_reduction_pct: 0.10 means 10% reduction. -0.25 means 25% surge.
            savings_cut_rupees = disc_component * sc_cfg['spend_reduction_pct']
            simulated_expense = round(float(fixed_component + (disc_component - savings_cut_rupees)), 2)
            # Non-discretionary inflation
            fixed_comp = fixed_component * (1.0 + fixed_inflation)

            # Emergency shock debit on shock_day
            is_shock_day = (day_step == shock_day)
            shock_debit = resolved_shock if is_shock_day else 0.0

            # Discretionary spend reduction / surge
            savings_cut_rupees = disc_component * spend_cut_pct
            operational_expense = fixed_comp + (disc_component - savings_cut_rupees)
            simulated_expense = round(float(operational_expense + shock_debit), 2)

            # Daily compounding interest yield on positive cumulative savings balance
            daily_yield = round(max(0.0, cum_savings) * yield_rate, 2)

            # Strict conservation of cash flow:
            daily_net_savings = round(daily_income - simulated_expense, 2)
            daily_net_savings = round(daily_inflow - simulated_expense + daily_yield, 2)
            cum_savings = round(cum_savings + daily_net_savings, 2)

            # Autoregressive updates for next day step
            prev_exp = simulated_expense
            rolling_exp_buf.append(simulated_expense)
            prev_exp = round(float(operational_expense), 2)
            rolling_exp_buf.append(prev_exp)
            if len(rolling_exp_buf) > 7:
                rolling_exp_buf.pop(0)

            # Productivity Prediction
            if hgb_prod and hasattr(hgb_prod, '_raw_predict'):
                prod_arr[0, 0] = prev_prod
                prod_arr[0, 1] = sleep_hrs
                prod_arr[0, 2] = fatigue_lag
                prod_arr[0, 3] = work_hrs
                prod_arr[0, 4] = is_tomorrow_weekend
                prod_arr[0, 5] = stress_lvl
                prod_arr[0, 6] = screen_hrs
                pred_prod = float(hgb_prod._raw_predict(prod_arr)[0, 0])
            elif hgb_prod:
                prod_input_df = pd.DataFrame([[
                    prev_prod, sleep_hrs, fatigue_lag, work_hrs,
                    is_tomorrow_weekend, stress_lvl, screen_hrs
                ]], columns=prod_features).astype(float)
                pred_prod = float(hgb_prod.predict(prod_input_df)[0])
            else:
                pred_prod = _calculate_fallback_cognitive_score(
                    work_hrs, study_hrs, screen_hrs, sleep_hrs, stress_lvl
                )

            # Holistic cognitive lifestyle adjustments from scenario interventions
            # (sleep restoration, stress reduction, study focus, screen time balance)
            sleep_effect = sc_cfg['sleep_hours_delta'] * 2.0
            stress_effect = - (sc_cfg['stress_level_delta'] * 1.5)
            study_effect = sc_cfg['study_hours_delta'] * 1.0
            screen_effect = - (sc_cfg['screen_time_delta'] * 0.75)
            cognitive_adjustment = sleep_effect + stress_effect + study_effect + screen_effect
            sprint_cognitive_drag = -4.0 if in_sprint else 0.0
            sleep_effect = (sc_cfg.get('sleep_hours_delta', 0.0) - (1.0 if in_sprint else 0.0)) * 2.0
            stress_effect = - ((sc_cfg.get('stress_level_delta', 0.0) + (1.5 if in_sprint else 0.0)) * 1.5)
            study_effect = sc_cfg.get('study_hours_delta', 0.0) * 1.0
            screen_effect = - (sc_cfg.get('screen_time_delta', 0.0) * 0.75)
            cognitive_adjustment = sleep_effect + stress_effect + study_effect + screen_effect + sprint_cognitive_drag

            pred_prod = round(float(np.clip(pred_prod + cognitive_adjustment, 0.0, 100.0)), 1)
            prev_prod = pred_prod

            # Burnout Risk Index
            burnout_risk = _calculate_burnout_risk_index(work_hrs, stress_lvl, sleep_hrs)

            # Append to time-series
            dates_list.append(curr_target_date.strftime('%Y-%m-%d'))
            expenses_list.append(simulated_expense)
            cumulative_savings_list.append(cum_savings)
            productivity_list.append(pred_prod)
            burnout_risk_list.append(burnout_risk)
            yields_list.append(daily_yield)

        simulation_trajectories[sc_key] = {
            'name': sc_cfg['name'],
            'description': sc_cfg['description'],
            'dates': dates_list,
            'expenses': expenses_list,
            'cumulative_savings': cumulative_savings_list,
            'productivity': productivity_list,
            'burnout_risk': burnout_risk_list,
            'burnout_incidences': burnout_incidences_count,
            'daily_inflow': daily_inflow,
            'total_inflow': round(daily_inflow * horizon_days, 2),
            'total_yield_earned': round(float(sum(yields_list)), 2),
            'yields': yields_list,
            'resolved_shock_amount': resolved_shock,
            'shock_day': shock_day,
        }

    # Step 6: Exact Mathematical Deltas between Baseline and Scenarios
    base_traj = simulation_trajectories['baseline']
    traj_custom = simulation_trajectories['custom_scenario']
    traj_a = simulation_trajectories['scenario_a']
    traj_b = simulation_trajectories['scenario_b']

    total_base_exp = round(float(sum(base_traj['expenses'])), 2)
    avg_base_prod = round(float(np.mean(base_traj['productivity'])), 1)
    avg_base_burnout = round(float(np.mean(base_traj['burnout_risk'])), 1)
    base_burnout_incidences = base_traj.get('burnout_incidences', 0)
    final_base_savings = base_traj['cumulative_savings'][-1]
    base_total_inflow = base_traj.get('total_inflow', round(daily_income * horizon_days, 2))
    base_total_yield = base_traj.get('total_yield_earned', 0.0)

    def compute_scenario_delta(target_traj):
        tot_exp = round(float(sum(target_traj['expenses'])), 2)
        avg_prod = round(float(np.mean(target_traj['productivity'])), 1)
        avg_burnout = round(float(np.mean(target_traj['burnout_risk'])), 1)
        target_burnout_incidences = target_traj.get('burnout_incidences', 0)
        final_savings = target_traj['cumulative_savings'][-1]
        target_total_inflow = target_traj.get('total_inflow', round(target_traj.get('daily_inflow', daily_income) * horizon_days, 2))
        target_total_yield = target_traj.get('total_yield_earned', 0.0)

        net_savings_delta = round(float(final_savings - final_base_savings), 2)
        expenses_saved = round(float(total_base_exp - tot_exp), 2)
        inflow_delta = round(float(target_total_inflow - base_total_inflow), 2)
        yield_delta = round(float(target_total_yield - base_total_yield), 2)
        focus_pct_gained = round(float(avg_prod - avg_base_prod), 1)
        burnout_risk_reduction = round(float(avg_base_burnout - avg_burnout), 1)
        burnout_incidences_delta = target_burnout_incidences - base_burnout_incidences

        cum_savings_delta_series = [
            round(float(s - b), 2)
            for s, b in zip(target_traj['cumulative_savings'], base_traj['cumulative_savings'])
        ]
        prod_delta_series = [
            round(float(p - b), 1)
            for p, b in zip(target_traj['productivity'], base_traj['productivity'])
        ]

        return {
            'net_savings_delta': net_savings_delta,
            'total_expenses_saved': expenses_saved,
            'inflow_delta': inflow_delta,
            'yield_delta': yield_delta,
            'total_inflow': target_total_inflow,
            'total_yield_earned': target_total_yield,
            'shock_amount': target_traj.get('resolved_shock_amount', 0.0),
            'focus_pct_gained': focus_pct_gained,
            'burnout_risk_reduction': burnout_risk_reduction,
            'burnout_incidences': target_burnout_incidences,
            'burnout_incidences_delta': burnout_incidences_delta,
            'total_expenses': tot_exp,
            'final_cumulative_savings': final_savings,
            'avg_productivity': avg_prod,
            'avg_burnout_risk': avg_burnout,
            'cumulative_savings_delta_series': cum_savings_delta_series,
            'productivity_delta_series': prod_delta_series,
        }

    deltas = {
        'custom_scenario': compute_scenario_delta(traj_custom),
        'scenario_a': compute_scenario_delta(traj_a),
        'scenario_b': compute_scenario_delta(traj_b),
    }

    # Step 7: Checkpoint Aggregations (Monthly at 30, 60, 90 & Weekly)
    checkpoints = {'monthly': [], 'weekly': []}
    for day_target in [30, 60, 90]:
        if day_target <= horizon_days:
            idx = day_target - 1
            checkpoints['monthly'].append({
                'day': day_target,
                'date': base_traj['dates'][idx],
                'baseline_savings': base_traj['cumulative_savings'][idx],
                'custom_savings': traj_custom['cumulative_savings'][idx],
                'scenario_a_savings': traj_a['cumulative_savings'][idx],
                'scenario_b_savings': traj_b['cumulative_savings'][idx],
                'custom_net_gain': deltas['custom_scenario']['cumulative_savings_delta_series'][idx],
                'scenario_a_net_gain': deltas['scenario_a']['cumulative_savings_delta_series'][idx],
                'scenario_b_net_gain': deltas['scenario_b']['cumulative_savings_delta_series'][idx],
                'custom_prod_gain': deltas['custom_scenario']['productivity_delta_series'][idx],
                'scenario_a_prod_gain': deltas['scenario_a']['productivity_delta_series'][idx],
                'scenario_b_prod_gain': deltas['scenario_b']['productivity_delta_series'][idx],
            })

    for wk in range(1, (horizon_days // 7) + 1):
        idx = (wk * 7) - 1
        if idx < horizon_days:
            checkpoints['weekly'].append({
                'week': wk,
                'day': wk * 7,
                'date': base_traj['dates'][idx],
                'baseline_savings': base_traj['cumulative_savings'][idx],
                'custom_savings': traj_custom['cumulative_savings'][idx],
                'scenario_a_savings': traj_a['cumulative_savings'][idx],
                'scenario_b_savings': traj_b['cumulative_savings'][idx],
            })

    result_payload = {
        'success': True,
        'user_id': user_id,
        'horizon_days': horizon_days,
        'start_date': (start_date + timedelta(days=1)).strftime('%Y-%m-%d'),
        'end_date': (start_date + timedelta(days=horizon_days)).strftime('%Y-%m-%d'),
        'baseline_trajectory': base_traj,
        'custom_trajectory': traj_custom,
        'scenario_a_trajectory': traj_a,
        'scenario_b_trajectory': traj_b,
        'trajectories': {
            'baseline': base_traj,
            'custom_scenario': traj_custom,
            'scenario_a': traj_a,
            'scenario_b': traj_b,
        },
        'deltas': deltas,
        'checkpoints': checkpoints,
        'metadata': {
            'monthly_income': monthly_income,
            'monthly_budget': monthly_budget,
            'daily_income': daily_income,
            'total_historical_records': total_records,
            'models_active': {
                'expense_model': bool(rf_exp),
                'productivity_model': bool(hgb_prod),
            },
            'scenarios_info': {
                'baseline': scenario_base,
                'custom_scenario': scenario_custom,
                'scenario_a': scenario_a,
                'scenario_b': scenario_b,
            }
        }
    }

    _SIMULATION_RESULT_CACHE[cache_key] = (now_ts, copy.deepcopy(result_payload))
    return result_payload


if __name__ == "__main__":
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "digitaltwin_project.settings")
    django.setup()

    print("\n" + "=" * 70)
    print("DIGITAL TWIN SIMULATION ENGINE — STANDALONE VERIFICATION SUITE")
    print("=" * 70)

    # 1. Test with mature intelligence user (User 4)
    TEST_USER_ID = 4
    horizon = 90
    print(f"\n[Test 1] Running 90-Day Digital Twin Simulation for user_id={TEST_USER_ID}...")

    initial_record_count = DailyTelemetryRecord.objects.count()
    result = run_digital_twin_simulation(user_id=TEST_USER_ID, horizon_days=horizon)

    assert result['success'] is True, f"Simulation failed: {result.get('error')}"
    post_record_count = DailyTelemetryRecord.objects.count()

    # Invariant 1: Zero DB mutations
    assert initial_record_count == post_record_count, (
        f"INVARIANT VIOLATION: Database was modified! Initial: {initial_record_count}, Post: {post_record_count}"
    )
    print("  [PASS] Invariant 1: 100% In-Memory (Zero DB writes or mutations).")

    # Invariant 2: Correct lengths and shapes across all 4 trajectories
    all_trajectories = ['baseline_trajectory', 'custom_trajectory', 'scenario_a_trajectory', 'scenario_b_trajectory']
    for sc in all_trajectories:
        traj = result[sc]
        assert len(traj['dates']) == horizon, f"{sc} dates length mismatch"
        assert len(traj['expenses']) == horizon, f"{sc} expenses length mismatch"
        assert len(traj['cumulative_savings']) == horizon, f"{sc} cumulative_savings length mismatch"
        assert len(traj['productivity']) == horizon, f"{sc} productivity length mismatch"
        assert len(traj['burnout_risk']) == horizon, f"{sc} burnout_risk length mismatch"
    print(f"  [PASS] Invariant 2: Exact trajectory lengths ({horizon} days each across all 4 trajectories).")

    # Invariant 3: Zero NaNs, Infs, or Decimal leaks
    for sc in all_trajectories:
        traj = result[sc]
        for key in ['expenses', 'cumulative_savings', 'productivity', 'burnout_risk']:
            series = traj[key]
            for idx, val in enumerate(series):
                assert not np.isnan(val), f"NaN detected in {sc}['{key}'] at index {idx}"
                assert not np.isinf(val), f"Inf detected in {sc}['{key}'] at index {idx}"
                assert isinstance(val, (float, int)), f"Non-float type {type(val)} in {sc}['{key}']"
    print("  [PASS] Invariant 3: Clean numeric float arrays (Zero NaNs, Infs, or Decimal type leaks).")

    # Invariant 4: Conservation of Cash Flow
    # Net savings delta MUST equal (expenses saved + inflow delta + yield delta) within 0.05 rounding tolerance
    for sc_key in ['custom_scenario', 'scenario_a', 'scenario_b']:
        sc_delta = result['deltas'][sc_key]
        expected_savings_delta = round(sc_delta['total_expenses_saved'] + sc_delta['inflow_delta'] + sc_delta['yield_delta'], 2)
        diff = abs(sc_delta['net_savings_delta'] - expected_savings_delta)
        assert diff < 0.05, (
            f"CASH CONSERVATION LEAK in {sc_key}: Net savings delta ({sc_delta['net_savings_delta']}) "
            f"!= Expected ({expected_savings_delta}) [Expenses saved: {sc_delta['total_expenses_saved']}, "
            f"Inflow delta: {sc_delta['inflow_delta']}, Yield delta: {sc_delta['yield_delta']}]"
        )
    print("  [PASS] Invariant 4: Conservation of Cash Flow verified across custom_scenario, scenario_a, and scenario_b.")

    # Invariant 5: Test Live Custom Slider Dynamics (Thrift vs Surge)
    print("\n[Test 2] Testing Custom Scenario Live Slider Mechanics (Thrift vs Surge)...")
    res_thrift = run_digital_twin_simulation(
        user_id=TEST_USER_ID,
        custom_params={'custom_scenario': {'spend_reduction_pct': 0.30}},
        horizon_days=90
    )
    res_surge = run_digital_twin_simulation(
        user_id=TEST_USER_ID,
        custom_params={'custom_scenario': {'spend_reduction_pct': -0.30}},
        horizon_days=90
    )
    thrift_savings = res_thrift['deltas']['custom_scenario']['net_savings_delta']
    surge_savings = res_surge['deltas']['custom_scenario']['net_savings_delta']
    assert thrift_savings > 0, f"Thrift (-30% spend) must yield positive net savings gain, got: {thrift_savings}"
    assert surge_savings < 0, f"Surge (+30% spend) must yield negative net savings delta, got: {surge_savings}"
    assert thrift_savings > surge_savings, "Thrift must produce strictly greater savings than surge"
    print(f"  [PASS] Custom Slider Dynamics: -30% spend yields INR +{thrift_savings:,.2f}, +30% spend yields INR {surge_savings:,.2f}.")

    # Invariant 6: Test Stress Shock scenario
    print("\n[Test 3] Testing Stress Shock parameterization in Scenario B...")
    shock_params = {
        'scenario_b': {
            'mode': 'stress_shock',
            'name': 'Extreme Crunch Period',
        }
    }
    shock_result = run_digital_twin_simulation(user_id=TEST_USER_ID, custom_params=shock_params, horizon_days=90)
    assert shock_result['success'] is True
    shock_delta_b = shock_result['deltas']['scenario_b']
    assert shock_delta_b['net_savings_delta'] < 0, "Stress shock should decrease cumulative savings"
    assert shock_delta_b['focus_pct_gained'] < 0, "Stress shock should decrease productivity"
    assert shock_delta_b['burnout_risk_reduction'] < 0, "Stress shock should increase burnout risk (negative reduction)"
    print(f"  [PASS] Stress Shock verified: Net savings impact: INR {shock_delta_b['net_savings_delta']:,.2f}, Focus impact: {shock_delta_b['focus_pct_gained']}%, Burnout surge: +{abs(shock_delta_b['burnout_risk_reduction'])}%.")
    assert shock_delta_b['shock_amount'] > 0, "Emergency shock amount must be greater than 0"
    print(f"  [PASS] Stress Shock verified: Shock debit: INR {shock_delta_b['shock_amount']:,.2f}, Net savings impact: INR {shock_delta_b['net_savings_delta']:,.2f}, Focus impact: {shock_delta_b['focus_pct_gained']}%, Burnout surge: +{abs(shock_delta_b['burnout_risk_reduction'])}%.")

    # Invariant 7: Test Early Learning user (User 5, 21 records) and Cold Start (User 6, 5 records)
    print("\n[Test 4] Testing graceful execution across cold-start and early-learning phases...")
    # Invariant 7: Test Multi-Lever Economic Scaling & Sprint Mode
    print("\n[Test 4] Testing Economic Scaling Levers & Sprint Crunch Mode...")
    # Side income test (+20% of monthly income)
    res_side = run_digital_twin_simulation(
        user_id=TEST_USER_ID,
        custom_params={'custom_scenario': {'side_income_ratio': 0.20}},
        horizon_days=90
    )
    side_delta = res_side['deltas']['custom_scenario']
    assert side_delta['inflow_delta'] > 0, "Side income ratio must yield positive inflow delta"
    print(f"  [PASS] Side Income Lever: +20% income yields INR +{side_delta['inflow_delta']:,.2f} total inflows over 90 days.")

    # Sprint Mode test (Days 20-34 crunch)
    res_sprint = run_digital_twin_simulation(
        user_id=TEST_USER_ID,
        custom_params={'custom_scenario': {'sprint_mode_active': True}},
        horizon_days=90
    )
    sprint_delta = res_sprint['deltas']['custom_scenario']
    assert sprint_delta['burnout_incidences'] > 0, "Sprint mode must register burnout risk days during 14-day crunch"
    print(f"  [PASS] Sprint Mode Lever: 14-day sprint triggered {sprint_delta['burnout_incidences']} critical fatigue/stress days.")

    # Fixed cost inflation test (+10% inflation)
    res_inf = run_digital_twin_simulation(
        user_id=TEST_USER_ID,
        custom_params={'custom_scenario': {'fixed_cost_inflation_pct': 0.10, 'spend_reduction_pct': 0.0}},
        horizon_days=90
    )
    inf_delta = res_inf['deltas']['custom_scenario']
    assert inf_delta['total_expenses_saved'] < 0, "Fixed cost inflation must increase expenses (negative saved)"
    print(f"  [PASS] Fixed Cost Inflation Lever: +10% inflation increased expenses by INR {abs(inf_delta['total_expenses_saved']):,.2f}.")

    # Invariant 8: Test Early Learning user (User 5, 21 records) and Cold Start (User 6, 5 records)
    print("\n[Test 5] Testing graceful execution across cold-start and early-learning phases...")
    for uid in [5, 6]:
        r = run_digital_twin_simulation(user_id=uid, horizon_days=90)
        assert r['success'] is True
        print(f"  [PASS] user_id={uid} ({r['metadata']['total_historical_records']} records) simulation completed cleanly.")

    print("\n" + "=" * 70)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY (8/8).")
    print("=" * 70)
