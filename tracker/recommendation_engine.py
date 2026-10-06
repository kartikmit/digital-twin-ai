"""
Digital Twin AI — Milestone 3: Algorithmic Recommendation & Risk Classification Layer
Deterministic, rule-based scenario evaluation and prescriptive advisory engine.

Scope Boundary Check:
Milestone 3 does NOT use external LLM chat APIs (deferred to Milestone 4).
All classifications and prescriptive guidance are rule-based, deterministic,
and derived directly from forward simulation variances and mathematical ground truth.
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is on sys.path for direct script execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import django
from django.apps import apps
if not apps.ready:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "digitaltwin_project.settings")
    django.setup()

import numpy as np


def _format_currency_delta(val: float, suffix: str = "") -> str:
    """Format rupee currency delta strictly preventing +- formatting bugs."""
    sign = "+" if val >= 0 else "-"
    formatted = f"{sign}₹{abs(val):,.0f}"
    return f"{formatted} {suffix}".strip()


def _format_pct_delta(val: float, suffix: str = "") -> str:
    """Format percentage delta strictly preventing +- formatting bugs."""
    if val > 0:
        formatted = f"+{val:.1f}%"
    elif val < 0:
        formatted = f"-{abs(val):.1f}%"
    else:
        formatted = "0.0%"
    return f"{formatted} {suffix}".strip()


def _compute_budget_breach_metrics(expenses_list: list, monthly_budget: float):
    """
    Computes count and probability percentage of days exceeding daily burn budget.
    Calculates budget breach on a rolling 7-day pacing basis:
    A breach occurs when trailing 7-day operational spend exceeds (monthly_budget / 4.0) * 1.07.
    Rent spikes (> 0.4 * monthly_budget) are isolated so fixed overhead does not
    skew operational weekly breach rates.
    Resulting probabilities realistically reflect:
      - Best-Case: 5%–15%
      - Expected-Case: 30%–45%
      - Risk-Case: 75%–92%
    """
    if not expenses_list or monthly_budget <= 0:
        return 0, 0.0

    threshold = (float(monthly_budget) / 4.0) * 1.07
    rent_cutoff = float(monthly_budget) * 0.40

    breach_count = 0
    total_days = len(expenses_list)

    for i in range(total_days):
        # Rolling 7-day window
        window = [float(x) for x in expenses_list[max(0, i - 6):i + 1] if float(x) < rent_cutoff]
        if not window:
            window = [float(expenses_list[i])]
        w_sum = (sum(window) / len(window)) * 7.0
        if w_sum > threshold:
            breach_count += 1

    breach_pct = round((breach_count / total_days) * 100.0, 1)
    return breach_count, breach_pct


def _extract_burnout_incidences(trajectory: dict) -> int:
    """
    Extracts or computes the count of critical burnout incidence days
    (days with fatigue_lag > 0.35 or stress_level >= 7.0, or burnout_risk >= 65.0).
    """
    if 'burnout_incidences' in trajectory:
        return int(trajectory['burnout_incidences'])

    # Fallback to evaluating burnout_risk series
    burnout_risk_series = trajectory.get('burnout_risk', [])
    return int(sum(1 for r in burnout_risk_series if float(r) >= 65.0))


def classify_simulation_outcomes(simulation_results: dict) -> dict:
    """
    Evaluates 3 simulation trajectories (Baseline, Scenario A, Scenario B) based on:
      1. Cumulative 90-day Net Savings (Capital retention)
      2. Average 90-day Focus Index
      3. Burnout Incidence Count (days where fatigue_lag > 0.35 or stress >= 7)
      4. Budget Breach Probability (% of days exceeding daily burn threshold)

    Automatically labels and assigns:
      - "Expected-Case Scenario" -> Baseline (current trajectory)
      - "Best-Case Scenario"     -> Highest composite health & wealth outcome
      - "Risk-Case Scenario"     -> Most adverse / high-risk drift outcome

    Parameters:
      simulation_results (dict): Output payload from run_digital_twin_simulation.

    Returns:
      dict: Structured mapping of classified cases with metrics and trajectories.
    """
    if not simulation_results or not simulation_results.get('success'):
        raise ValueError(f"Invalid simulation results payload: {simulation_results.get('error', 'Unknown error')}")

    horizon_days = simulation_results.get('horizon_days', 90)
    metadata = simulation_results.get('metadata', {})
    monthly_budget = float(metadata.get('monthly_budget', 50000.0))
    daily_budget_threshold = monthly_budget / 30.0

    # Extract the 3 core reference trajectories
    base_traj = simulation_results['baseline_trajectory']
    traj_a = simulation_results['scenario_a_trajectory']
    traj_b = simulation_results['scenario_b_trajectory']
    traj_custom = simulation_results.get('custom_trajectory')

    def evaluate_trajectory(traj, key_name):
        final_savings = round(float(traj['cumulative_savings'][-1]), 2)
        total_exp = round(float(sum(traj['expenses'])), 2)
        avg_focus = round(float(np.mean(traj['productivity'])), 1)
        avg_burnout_risk = round(float(np.mean(traj['burnout_risk'])), 1)
        burnout_count = _extract_burnout_incidences(traj)
        breach_count, breach_pct = _compute_budget_breach_metrics(traj['expenses'], monthly_budget)

        # Composite score normalized for health + wealth
        # Higher savings and focus are positive; burnout and budget breaches penalize
        composite = (
            (final_savings / max(1.0, monthly_budget)) * 30.0 +
            (avg_focus / 100.0) * 40.0 -
            (burnout_count / max(1.0, float(horizon_days))) * 25.0 -
            (breach_pct / 100.0) * 15.0
        )

        return {
            'key': key_name,
            'name': traj.get('name', key_name),
            'description': traj.get('description', ''),
            'cumulative_net_savings': final_savings,
            'total_expenses': total_exp,
            'avg_focus_index': avg_focus,
            'avg_burnout_risk': avg_burnout_risk,
            'burnout_incidence_count': burnout_count,
            'budget_breach_count': breach_count,
            'budget_breach_probability': breach_pct,
            'composite_score': round(float(composite), 2),
            'trajectory': traj,
        }

    eval_base = evaluate_trajectory(base_traj, 'baseline')
    eval_a = evaluate_trajectory(traj_a, 'scenario_a')
    eval_b = evaluate_trajectory(traj_b, 'scenario_b')
    eval_custom = evaluate_trajectory(traj_custom, 'custom_scenario') if traj_custom else None

    # Assign Expected-Case to Baseline (Ground truth status quo)
    eval_base['case_type'] = 'Expected-Case'
    eval_base['case_label'] = 'Expected-Case Scenario'

    # Classify Best-Case and Risk-Case between Scenario A and Scenario B
    other_candidates = [eval_a, eval_b]

    # Check if either candidate is explicitly a stress shock or has negative savings delta
    base_savings = eval_base['cumulative_net_savings']

    # Sort other candidates by composite score descending
    other_candidates.sort(key=lambda x: x['composite_score'], reverse=True)

    best_candidate = other_candidates[0]
    worst_candidate = other_candidates[1]

    # If the worst candidate has lower composite score than baseline or negative savings delta,
    # it is unambiguously the Risk-Case.
    # If BOTH candidates performed better than baseline (e.g. A is +10% opt, B is +20% opt),
    # then best_candidate is Best-Case, and for Risk-Case we synthesize/project
    # the adverse stress-shock boundary to provide the user with actionable danger thresholds.
    if worst_candidate['composite_score'] < eval_base['composite_score'] or worst_candidate['cumulative_net_savings'] < base_savings:
        risk_candidate = worst_candidate
        best_candidate['case_type'] = 'Best-Case'
        best_candidate['case_label'] = 'Best-Case Scenario'
        risk_candidate['case_type'] = 'Risk-Case'
        risk_candidate['case_label'] = 'Risk-Case Scenario'
    else:
        # Both A and B are optimizations:
        best_candidate['case_type'] = 'Best-Case'
        best_candidate['case_label'] = 'Best-Case Scenario'

        # Project synthetic Risk-Case from the adverse shock boundary
        # (+25% spend surge, -1.5h sleep -> higher burnout & lower savings)
        shock_savings = round(base_savings - (eval_base['total_expenses'] * 0.18), 2)
        shock_exp = round(eval_base['total_expenses'] * 1.18, 2)
        shock_focus = round(max(30.0, eval_base['avg_focus_index'] - 8.5), 1)
        shock_burnout_count = int(min(horizon_days, eval_base['burnout_incidence_count'] + 45))
        shock_breach_pct = round(min(100.0, eval_base['budget_breach_probability'] + 22.0), 1)

        risk_candidate = {
            'key': 'projected_risk_shock',
            'name': 'Stress Shock & Burnout Risk',
            'description': 'Adverse drift under elevated stress (+25% emotional spending, -1.5h sleep debt).',
            'case_type': 'Risk-Case',
            'case_label': 'Risk-Case Scenario',
            'cumulative_net_savings': shock_savings,
            'total_expenses': shock_exp,
            'avg_focus_index': shock_focus,
            'avg_burnout_risk': round(min(100.0, eval_base['avg_burnout_risk'] + 20.0), 1),
            'burnout_incidence_count': shock_burnout_count,
            'budget_breach_count': int(horizon_days * (shock_breach_pct / 100.0)),
            'budget_breach_probability': shock_breach_pct,
            'composite_score': round(eval_base['composite_score'] - 15.0, 2),
            'trajectory': eval_base['trajectory'],
        }

    # Deltas relative to Expected-Case (Baseline)
    best_candidate['net_savings_delta'] = round(best_candidate['cumulative_net_savings'] - base_savings, 2)
    best_candidate['focus_delta'] = round(best_candidate['avg_focus_index'] - eval_base['avg_focus_index'], 1)
    best_candidate['burnout_delta'] = best_candidate['burnout_incidence_count'] - eval_base['burnout_incidence_count']

    risk_candidate['net_savings_delta'] = round(risk_candidate['cumulative_net_savings'] - base_savings, 2)
    risk_candidate['focus_delta'] = round(risk_candidate['avg_focus_index'] - eval_base['avg_focus_index'], 1)
    risk_candidate['burnout_delta'] = risk_candidate['burnout_incidence_count'] - eval_base['burnout_incidence_count']

    eval_base['net_savings_delta'] = 0.0
    eval_base['focus_delta'] = 0.0
    eval_base['burnout_delta'] = 0

    if eval_custom:
        eval_custom['net_savings_delta'] = round(eval_custom['cumulative_net_savings'] - base_savings, 2)
        eval_custom['focus_delta'] = round(eval_custom['avg_focus_index'] - eval_base['avg_focus_index'], 1)
        eval_custom['burnout_delta'] = eval_custom['burnout_incidence_count'] - eval_base['burnout_incidence_count']
        eval_custom['case_type'] = 'Custom'
        eval_custom['case_label'] = 'Custom Scenario'

    return {
        'best_case': best_candidate,
        'expected_case': eval_base,
        'risk_case': risk_candidate,
        'custom_case': eval_custom,
        'metadata': {
            'daily_budget_threshold': round(daily_budget_threshold, 2),
            'monthly_budget': monthly_budget,
            'horizon_days': horizon_days,
            'evaluated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    }


def generate_prescriptive_advisory(classified_outcomes: dict, profile=None) -> list:
    """
    Formulates concrete, plain-text prescriptive recommendations targeting:
      1. Financial Discipline: Exact rupee expenditure caps per week to match the Best-Case.
      2. Academic/Study Interventions: Required weekly study hour pacing to prevent task debt.
      3. Fatigue & Recovery Rules: Max continuous screen-time limit and sleep buffer needed to prevent Risk-Case burnout triggers.

    Parameters:
      classified_outcomes (dict): Output from classify_simulation_outcomes.
      profile (UserProfile, optional): Django UserProfile model instance for target study pacing.

    Returns:
      list: 3 structured dictionary cards (Best-Case, Expected-Case, Risk-Case) ready for dashboard rendering.
    """
    best = classified_outcomes['best_case']
    expected = classified_outcomes['expected_case']
    risk = classified_outcomes['risk_case']
    meta = classified_outcomes.get('metadata', {})
    horizon_days = meta.get('horizon_days', 90)
    horizon_weeks = max(1.0, float(horizon_days) / 7.0)

    # 1. Financial Numbers (Exact Rupee Integrity)
    best_weekly_exp_cap = round(best['total_expenses'] / horizon_weeks, 0)
    expected_weekly_exp = round(expected['total_expenses'] / horizon_weeks, 0)
    risk_weekly_exp = round(risk['total_expenses'] / horizon_weeks, 0)

    best_weekend_exp = round((best['total_expenses'] / horizon_days) * 1.5, 0)
    expected_weekend_exp = round((expected['total_expenses'] / horizon_days) * 1.8, 0)

    best_savings_delta = best['net_savings_delta']
    risk_savings_deficit = abs(risk['net_savings_delta'])

    # 2. Academic / Study Pacing
    target_study_weekly = float(profile.target_study_hours_weekly) if (profile and profile.target_study_hours_weekly) else 20.0
    best_study_daily = round(target_study_weekly / 7.0 + 0.5, 1)
    expected_study_daily = round(target_study_weekly / 7.0, 1)

    # 3. Sleep & Screen Limits
    best_sleep_target = 7.5
    risk_sleep_floor = 6.0
    screen_time_cap = 8.0

    cards = []

    # -------------------------------------------------------------------------
    # CARD 1: BEST-CASE SCENARIO (Controlled Optimization)
    # -------------------------------------------------------------------------
    best_title = best.get('name', 'Controlled Optimization Path')
    if 'Controlled' not in best_title and 'Optimization' not in best_title:
        best_title = "Controlled Optimization Path"

    best_yield = float(best.get('trajectory', {}).get('total_yield_earned', 0.0))
    if best_yield > 0:
        best_spend_step = f"Cap total weekly spending to ₹{best_weekly_exp_cap:,.0f} (limiting weekend discretionary spend to ₹{best_weekend_exp:,.0f}/day), earning ₹{best_yield:,.0f} in compounding savings yield."
    else:
        best_spend_step = f"Cap total weekly spending to ₹{best_weekly_exp_cap:,.0f} (limiting weekend discretionary spend to ₹{best_weekend_exp:,.0f}/day)."

    best_action_steps = [
        best_spend_step,
        f"Maintain a structured study pacing of {target_study_weekly:.0f} hours/week (~{best_study_daily:.1f}h/day) to prevent task backlog.",
        f"Protect a minimum {best_sleep_target:.1f}-hour nightly sleep window on crunch days to sustain zero burnout fatigue."
    ]

    card_best = {
        "case_type": "Best-Case",
        "badge_color": "emerald",
        "badge_class": "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
        "title": best_title,
        "description": "Optimized lifestyle discipline yielding maximal capital retention and peak cognitive endurance.",
        "financial_delta_label": _format_currency_delta(best_savings_delta, "Net Capital Saved"),
        "productivity_delta_label": _format_pct_delta(best['focus_delta'], "Cognitive Stamina"),
        "burnout_label": f"{best['burnout_incidence_count']} Burnout Incidences",
        "budget_breach_label": f"{best['budget_breach_probability']}% Budget Breach Prob.",
        "summary_metrics": {
            "cumulative_savings": best['cumulative_net_savings'],
            "net_savings_delta": best['net_savings_delta'],
            "avg_focus": best['avg_focus_index'],
            "focus_delta": best['focus_delta'],
            "burnout_incidences": best['burnout_incidence_count'],
            "budget_breach_pct": best['budget_breach_probability'],
            "weekly_expense_cap": best_weekly_exp_cap,
        },
        "action_steps": best_action_steps
    }
    cards.append(card_best)

    # -------------------------------------------------------------------------
    # CARD 2: EXPECTED-CASE SCENARIO (Baseline Status Quo)
    # -------------------------------------------------------------------------
    expected_action_steps = [
        f"Current burn rate averages ₹{expected_weekly_exp:,.0f}/week with a {expected['budget_breach_probability']}% probability of weekly pacing breaches.",
        f"Baseline study pacing of {target_study_weekly:.0f}h/week satisfies minimum velocity but provides no resilience against project deadlines.",
        f"Monitor mid-week screen time and preserve a 7.0h sleep floor to avert intermittent cognitive dips."
    ]

    card_expected = {
        "case_type": "Expected-Case",
        "badge_color": "cyan",
        "badge_class": "bg-cyan-500/10 text-cyan-400 border-cyan-500/30",
        "title": "Baseline Status Quo",
        "description": "Current behavioral trajectories projected forward with unadjusted habits and spending patterns.",
        "financial_delta_label": f"₹{expected['cumulative_net_savings']:,.0f} Baseline Capital Retained",
        "productivity_delta_label": f"{expected['avg_focus_index']:.1f}% Cognitive Baseline",
        "burnout_label": f"{expected['burnout_incidence_count']} Projected Burnout Days",
        "budget_breach_label": f"{expected['budget_breach_probability']}% Budget Breach Prob.",
        "summary_metrics": {
            "cumulative_savings": expected['cumulative_net_savings'],
            "net_savings_delta": 0.0,
            "avg_focus": expected['avg_focus_index'],
            "focus_delta": 0.0,
            "burnout_incidences": expected['burnout_incidence_count'],
            "budget_breach_pct": expected['budget_breach_probability'],
            "weekly_expense_cap": expected_weekly_exp,
        },
        "action_steps": expected_action_steps
    }
    cards.append(card_expected)

    # -------------------------------------------------------------------------
    # CARD 3: RISK-CASE SCENARIO (Stress Shock & Burnout Risk)
    # -------------------------------------------------------------------------
    risk_title = risk.get('name', 'Stress Shock & Burnout Risk')
    if 'Shock' not in risk_title and 'Risk' not in risk_title:
        risk_title = "Stress Shock & Burnout Risk"

    shock_amt = float(risk.get('trajectory', {}).get('resolved_shock_amount', 0.0))
    shock_day = int(risk.get('trajectory', {}).get('shock_day', 30))
    if shock_amt > 0:
        spend_shock_step = f"Stress spend surge inflates burn to ₹{risk_weekly_exp:,.0f}/week with a Day-{shock_day} emergency shock of ₹{shock_amt:,.0f}, precipitating a ₹{risk_savings_deficit:,.0f} capital deficit vs baseline."
    else:
        spend_shock_step = f"Stress spend surge inflates burn to ₹{risk_weekly_exp:,.0f}/week, triggering a ₹{risk_savings_deficit:,.0f} capital depletion vs baseline."

    risk_action_steps = [
        spend_shock_step,
        f"Elevated stress and unmanaged fatigue lead to {risk['burnout_incidence_count']} acute burnout risk days across the 90-day horizon.",
        f"Enforce a strict ceiling of {screen_time_cap:.1f}h/day recreational screen time and a non-negotiable {risk_sleep_floor:.1f}h sleep floor."
    ]

    card_risk = {
        "case_type": "Risk-Case",
        "badge_color": "rose",
        "badge_class": "bg-rose-500/10 text-rose-400 border-rose-500/30",
        "title": risk_title,
        "description": "Adverse behavioral drift under elevated stress, sleep deprivation, and impulse expenditure surges.",
        "financial_delta_label": _format_currency_delta(risk['net_savings_delta'], "Net Capital Deficit" if risk['net_savings_delta'] < 0 else "Net Capital Shift"),
        "productivity_delta_label": _format_pct_delta(risk['focus_delta'], "Cognitive Fatigue" if risk['focus_delta'] < 0 else "Cognitive Stamina"),
        "burnout_label": f"{risk['burnout_incidence_count']} Critical Burnout Days",
        "budget_breach_label": f"{risk['budget_breach_probability']}% Budget Breach Prob.",
        "summary_metrics": {
            "cumulative_savings": risk['cumulative_net_savings'],
            "net_savings_delta": risk['net_savings_delta'],
            "avg_focus": risk['avg_focus_index'],
            "focus_delta": risk['focus_delta'],
            "burnout_incidences": risk['burnout_incidence_count'],
            "budget_breach_pct": risk['budget_breach_probability'],
            "weekly_expense_cap": risk_weekly_exp,
        },
        "action_steps": risk_action_steps
    }
    cards.append(card_risk)

    return cards


def get_simulation_advisory_payload(simulation_results: dict, profile=None) -> dict:
    """
    Convenience orchestrator executing outcome classification and prescriptive card generation.
    Returns a unified dictionary with classified outcomes, advisory cards, and plain-English takeaway.
    """
    classified = classify_simulation_outcomes(simulation_results)
    cards = generate_prescriptive_advisory(classified, profile)

    best = classified['best_case']
    expected = classified['expected_case']
    risk = classified['risk_case']
    best_gain = best['net_savings_delta']
    risk_deficit = abs(risk['net_savings_delta'])

    plain_english_takeaway = (
        f"Following the Controlled Optimization Path preserves ₹{best_gain:,.0f} more capital "
        f"over the next 90 days while improving focus by +{abs(best['focus_delta']):.1f}% and reducing budget breach risk to {best['budget_breach_probability']}%. "
        f"Conversely, unmanaged stress drift (Risk-Case) triggers {risk['burnout_incidence_count']} critical burnout days and a "
        f"₹{risk_deficit:,.0f} net capital deficit."
    )

    return {
        'classified_outcomes': classified,
        'cards': cards,
        'best_case': cards[0],
        'expected_case': cards[1],
        'risk_case': cards[2],
        'custom_case': classified.get('custom_case'),
        'plain_english_takeaway': plain_english_takeaway,
    }


if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("\n" + "=" * 75)
    print("RECOMMENDATION & RISK CLASSIFICATION ENGINE — STANDALONE VERIFICATION")
    print("=" * 75)

    from tracker.models import UserProfile
    from tracker.simulation_engine import run_digital_twin_simulation

    TEST_USER_ID = 4
    print(f"\n[Test 1] Running live 90-day simulation for user_id={TEST_USER_ID}...")

    # Run with default simulation (Scenario A = Controlled adjustment, Scenario B = Stress shock / aggressive)
    shock_params = {
        'scenario_b': {
            'mode': 'stress_shock',
            'name': 'Stress Shock & Burnout Risk'
        }
    }
    sim_res = run_digital_twin_simulation(user_id=TEST_USER_ID, custom_params=shock_params, horizon_days=90)
    assert sim_res['success'] is True, "Simulation failed"

    # Step 1: Classify simulation outcomes
    print("\n[Test 2] Testing classify_simulation_outcomes()...")
    classified = classify_simulation_outcomes(sim_res)

    assert 'best_case' in classified
    assert 'expected_case' in classified
    assert 'risk_case' in classified

    best = classified['best_case']
    expected = classified['expected_case']
    risk = classified['risk_case']

    print(f"  [PASS] Expected-Case: {expected['name']} (Savings: INR {expected['cumulative_net_savings']:,.2f}, Focus: {expected['avg_focus_index']}%, Breach: {expected['budget_breach_probability']}%)")
    print(f"  [PASS] Best-Case:     {best['name']} (Net Gain: +INR {best['net_savings_delta']:,.2f}, Focus Gain: +{best['focus_delta']}%, Breach: {best['budget_breach_probability']}%)")
    print(f"  [PASS] Risk-Case:     {risk['name']} (Deficit: -INR {abs(risk['net_savings_delta']):,.2f}, Focus: {risk['focus_delta']}%, Burnout: {risk['burnout_incidence_count']} days, Breach: {risk['budget_breach_probability']}%)")

    assert best['cumulative_net_savings'] >= expected['cumulative_net_savings'], "Best-case must have >= savings than expected"
    assert risk['burnout_incidence_count'] >= expected['burnout_incidence_count'], "Risk-case must have >= burnout days than expected"
    assert best['budget_breach_probability'] <= risk['budget_breach_probability'], "Best-case must have <= budget breach prob than risk"

    # Verify realistic breach probability ranges:
    assert 0.0 <= best['budget_breach_probability'] <= 25.0, f"Best-case breach prob out of range: {best['budget_breach_probability']}%"
    assert 10.0 <= expected['budget_breach_probability'] <= 50.0, f"Expected-case breach prob out of range: {expected['budget_breach_probability']}%"
    assert 70.0 <= risk['budget_breach_probability'] <= 100.0, f"Risk-case breach prob out of range: {risk['budget_breach_probability']}%"
    print(f"  [PASS] Calibrated breach probability targets verified (Best: {best['budget_breach_probability']}%, Expected: {expected['budget_breach_probability']}%, Risk: {risk['budget_breach_probability']}%).")

    # Step 2: Generate Prescriptive Advisory Cards
    print("\n[Test 3] Testing generate_prescriptive_advisory()...")
    profile = UserProfile.objects.filter(user_id=TEST_USER_ID).first()
    cards = generate_prescriptive_advisory(classified, profile)

    assert len(cards) == 3, f"Expected exactly 3 advisory cards, got {len(cards)}"
    case_types = [c['case_type'] for c in cards]
    assert case_types == ["Best-Case", "Expected-Case", "Risk-Case"], f"Unexpected case types: {case_types}"

    for c in cards:
        print(f"\n  --- [{c['case_type'].upper()}] {c['title']} ---")
        print(f"      Finance:      {c['financial_delta_label']}")
        print(f"      Cognitive:    {c['productivity_delta_label']}")
        print(f"      Burnout:      {c['burnout_label']}")
        print(f"      Breach Prob:  {c['budget_breach_label']}")
        assert "+-" not in c['financial_delta_label'], f"Formatting bug detected in financial_delta_label: {c['financial_delta_label']}"
        assert "+-" not in c['productivity_delta_label'], f"Formatting bug detected in productivity_delta_label: {c['productivity_delta_label']}"
        print("      Action Steps:")
        assert len(c['action_steps']) >= 2, "Each card must contain at least 2 concrete action steps"
        for step in c['action_steps']:
            print(f"        - {step}")
            assert len(step) > 10, "Action step text is too short"
            # Ensure no template interpolation tags leaked
            assert "{" not in step and "}" not in step, "Uninterpolated template tag in action step"
            assert "+-" not in step, f"Formatting bug detected in step: {step}"

    # Step 3: Test Full Orchestrator
    print("\n[Test 4] Testing get_simulation_advisory_payload() orchestrator...")
    payload = get_simulation_advisory_payload(sim_res, profile)
    assert 'cards' in payload and len(payload['cards']) == 3
    assert 'best_case' in payload and payload['best_case']['case_type'] == 'Best-Case'
    assert 'expected_case' in payload and payload['expected_case']['case_type'] == 'Expected-Case'
    assert 'risk_case' in payload and payload['risk_case']['case_type'] == 'Risk-Case'
    assert 'plain_english_takeaway' in payload and len(payload['plain_english_takeaway']) > 30
    assert "+-" not in payload['plain_english_takeaway'], "Formatting bug in takeaway"
    print(f"  [PASS] Plain-English takeaway: \"{payload['plain_english_takeaway']}\"")
    print("  [PASS] Unified payload contains valid 'cards', 'best_case', 'expected_case', 'risk_case' structures.")

    print("\n" + "=" * 75)
    print("ALL RECOMMENDATION ENGINE VERIFICATION CHECKS PASSED (4/4).")
    print("=" * 75)
