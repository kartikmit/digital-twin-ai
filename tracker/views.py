import io
import pandas as pd
import numpy as np
from datetime import datetime
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import UserRegistrationForm, UserProfileForm, DailyTelemetryManualForm
from .models import UserProfile, Goal, DailyTelemetryRecord, ActivityLog
from .ml_engine import (
    train_user_models,
    get_user_analytics_and_forecasts,
)
from .simulation_engine import run_digital_twin_simulation
from .recommendation_engine import (
    classify_simulation_outcomes,
    generate_prescriptive_advisory,
    get_simulation_advisory_payload,
)

def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            Goal.objects.create(user=user, title='Emergency Fund Vault', current_value=25000, target_value=100000, unit='₹', category='Finance')
            Goal.objects.create(user=user, title='Distributed Systems Mastery', current_value=45, target_value=100, unit='%', category='Study')
            Goal.objects.create(user=user, title='Cardio Conditioning', current_value=15, target_value=30, unit='Days', category='Habit')
            ActivityLog.objects.create(user=user, category='Auth', action='Account created.')
            login(request, user)
            messages.success(request, f"Welcome to Digital Twin AI, {user.first_name}!")
            return redirect('dashboard')
    else:
        form = UserRegistrationForm()
    return render(request, 'tracker/register.html', {'form': form})

def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            ActivityLog.objects.create(user=user, category='Auth', action='User authenticated successfully.')
            return redirect('dashboard')
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
    return render(request, 'tracker/login.html', {'form': form})

@login_required
def logout_view(request):
    ActivityLog.objects.create(user=request.user, category='Auth', action='User session terminated.')
    logout(request)
    return redirect('login')

@login_required
def dashboard_view(request):
    user = request.user
    profile, _ = UserProfile.objects.get_or_create(user=user)
    telemetry_qs = DailyTelemetryRecord.objects.filter(user=user).order_by('entry_date')

    if telemetry_qs.count() == 0:
        return render(request, 'tracker/dashboard.html', {
            'has_data': False,
            'profile': profile,
            'goals': user.goals.all().order_by('-id'),
            'total_telemetry_count': 0,
            'activity_logs': user.activity_logs.all()[:6],
            'latest_record': None,
        })

    # 1. Trailing Analytics & Forecasts (Milestone 2 Engine)
    monthly_budget = float(profile.monthly_budget or 50000.0)
    monthly_income = float(profile.monthly_income or 85000.0)
    target_study_hours_weekly = float(profile.target_study_hours_weekly or 20.0)

    analytics = get_user_analytics_and_forecasts(user.id, telemetry_qs, monthly_budget)

    mtd_expenses = float(analytics['financial']['mtd_expenses']) if analytics else 0.0
    daily_burn_rate = float(analytics['financial']['daily_burn_rate']) if analytics else 0.0
    projected_month_end = float(analytics['financial']['projected_month_end']) if analytics else 0.0
    savings_rate = float(analytics['financial']['savings_rate']) if analytics else 0.0
    habit_score = float(analytics['behavioral']['habit_adherence_index']) if analytics else 75.0

    monthly_savings = round(monthly_income - mtd_expenses, 2)

    # Study Hours Pacing (Trailing 30 days)
    recent_30 = list(telemetry_qs.order_by('-entry_date')[:30])
    total_study_hours = round(float(sum(r.study_hours for r in recent_30)), 1)
    target_monthly_study = round(target_study_hours_weekly * 4.0, 1)
    study_pacing_pct = round((total_study_hours / target_monthly_study) * 100.0, 1) if target_monthly_study > 0 else 100.0

    # Expense pacing indicator
    expense_pacing_pct = round(((projected_month_end - monthly_budget) / monthly_budget) * 100.0, 1) if monthly_budget > 0 else 0.0
    expense_pacing_label = f"+{expense_pacing_pct}% over budget" if expense_pacing_pct > 0 else f"{abs(expense_pacing_pct)}% under budget"

    # 2. Multi-Scenario Savings Projection (Baseline vs. +₹5,000 Savings)
    sim_res = run_digital_twin_simulation(user.id, horizon_days=90)
    sim_res_plus5k = run_digital_twin_simulation(
        user.id, 
        custom_params={'custom_scenario': {'spend_reduction_pct': 0.15}}, 
        horizon_days=90
    )

    base_dates = sim_res['baseline_trajectory']['dates'] if sim_res.get('success') else []
    sim_dates_formatted = [
        datetime.strptime(d, '%Y-%m-%d').strftime('%b %d, %Y') if '-' in d else d 
        for d in base_dates
    ]
    savings_current = sim_res['baseline_trajectory']['cumulative_savings'] if sim_res.get('success') else []
    savings_plus5k = sim_res_plus5k['custom_trajectory']['cumulative_savings'] if sim_res_plus5k.get('success') else []

    sim_dates_json = json.dumps(sim_dates_formatted)
    savings_current_plan_json = json.dumps(savings_current)
    savings_plus5k_plan_json = json.dumps(savings_plus5k)

    # 3. Categorical Expense Allocation (Donut Chart)
    category_totals = analytics['financial'].get('category_totals', {}) if analytics else {}
    if not category_totals and recent_30:
        category_totals = {
            'Food': round(sum(float(r.food_expenses) for r in recent_30), 2),
            'Travel': round(sum(float(r.travel_expenses) for r in recent_30), 2),
            'Shopping': round(sum(float(r.shopping_expenses) for r in recent_30), 2),
            'Utilities': round(sum(float(r.utilities_expenses) for r in recent_30), 2),
            'Health': round(sum(float(r.health_expenses) for r in recent_30), 2),
            'Other': round(sum(float(r.other_expenses) for r in recent_30), 2),
        }
    expense_category_labels_json = json.dumps(list(category_totals.keys()))
    expense_category_values_json = json.dumps(list(category_totals.values()))

    # 4. Toggleable Study & Productivity Series (Trailing 14 days, Chronological)
    recent_14 = list(telemetry_qs.order_by('-entry_date')[:14])
    recent_14.reverse()

    prod_dates_14 = [r.entry_date.strftime('%b %d') for r in recent_14]
    study_hours_14 = [float(r.study_hours) for r in recent_14]
    work_hours_14 = [float(r.work_hours) for r in recent_14]
    prod_scores_14 = [round(float(r.productivity_score), 1) for r in recent_14]

    prod_dates_14_json = json.dumps(prod_dates_14)
    study_hours_14_json = json.dumps(study_hours_14)
    work_hours_14_json = json.dumps(work_hours_14)
    productivity_score_14_json = json.dumps(prod_scores_14)

    # 5. Toggleable Fitness & Health Series (Trailing 14 days)
    health_dates_14 = [r.entry_date.strftime('%b %d') for r in recent_14]
    steps_14 = [int(r.steps_count) for r in recent_14]
    sleep_14 = [float(r.sleep_hours) for r in recent_14]
    exercise_14 = [int(r.exercise_duration_min) for r in recent_14]

    health_dates_14_json = json.dumps(health_dates_14)
    steps_14_json = json.dumps(steps_14)
    sleep_14_json = json.dumps(sleep_14)
    exercise_14_json = json.dumps(exercise_14)

    # 6. Goals Progress & Recommendations
    goals = user.goals.all().order_by('-id')
    advisory_payload = get_simulation_advisory_payload(sim_res, profile) if sim_res.get('success') else {}
    advisory_cards = advisory_payload.get('cards', [])
    best_case_card = advisory_payload.get('best_case', {})
    risk_case_card = advisory_payload.get('risk_case', {})
    expected_case_card = advisory_payload.get('expected_case', {})
    plain_english_takeaway = advisory_payload.get('plain_english_takeaway', '')

    # Extract flat, guaranteed cumulative savings metrics for the prescription cards
    rec_val = (
        expected_case_card.get('summary_metrics', {}).get('cumulative_savings') or
        (sim_res['baseline_trajectory']['cumulative_savings'][-1] if sim_res.get('success') else 0)
    )
    best_val = (
        best_case_card.get('summary_metrics', {}).get('cumulative_savings') or
        (sim_res.get('scenario_a_trajectory', {}).get('cumulative_savings', [0])[-1] if sim_res.get('success') else 0)
    )
    risk_val = (
        risk_case_card.get('summary_metrics', {}).get('cumulative_savings') or
        (sim_res.get('scenario_b_trajectory', {}).get('cumulative_savings', [0])[-1] if sim_res.get('success') else 0)
    )

    rec_savings_val = float(rec_val)
    best_savings_val = float(best_val)
    risk_savings_val = float(risk_val)
    risk_savings_val_abs = abs(float(risk_val))

    if expected_case_card:
        expected_case_card['projected_savings'] = rec_savings_val
        expected_case_card['projected_savings_abs'] = abs(rec_savings_val)
    if best_case_card:
        best_case_card['projected_savings'] = best_savings_val
        best_case_card['projected_savings_abs'] = abs(best_savings_val)
    if risk_case_card:
        risk_case_card['projected_savings'] = risk_savings_val
        risk_case_card['projected_savings_abs'] = risk_savings_val_abs

    assistant_preset_chips = [
        "Show my financial summary",
        "How much can I save in 3 years with ₹5,000 more?",
        "Suggest a study plan",
        "How can I improve my fitness?"
    ]

    context = {
        'has_data': True,
        'profile': profile,
        'total_telemetry_count': telemetry_qs.count(),
        'latest_record': telemetry_qs.last(),
        'activity_logs': user.activity_logs.all()[:6],

        # Top 5 Executive Metric Cards
        'monthly_income': monthly_income,
        'monthly_expenses': mtd_expenses,
        'monthly_savings': monthly_savings,
        'total_study_hours': total_study_hours,
        'monthly_study_hours': total_study_hours,
        'target_study_hours': target_monthly_study,
        'habit_score': habit_score,
        'expense_pacing_label': expense_pacing_label,
        'savings_rate': savings_rate,
        'study_pacing_pct': study_pacing_pct,
        'daily_burn_rate': daily_burn_rate,
        'projected_month_end': projected_month_end,
        'kpis': {
            'income': monthly_income,
            'expenses': mtd_expenses,
            'savings': monthly_savings,
            'study_hours': total_study_hours,
            'target_study': target_monthly_study,
            'habit_score': habit_score,
            'expense_pacing_label': expense_pacing_label,
            'savings_rate': savings_rate,
            'study_pacing_pct': study_pacing_pct,
            'daily_burn_rate': daily_burn_rate,
            'projected_month_end': projected_month_end,
        },

        # Multi-Scenario Savings Projection
        'sim_dates_json': sim_dates_json,
        'savings_current_plan_json': savings_current_plan_json,
        'savings_plus5k_plan_json': savings_plus5k_plan_json,

        # Categorical Expense Allocation
        'category_totals': category_totals,
        'expense_category_labels_json': expense_category_labels_json,
        'expense_category_values_json': expense_category_values_json,

        # Toggleable Study & Productivity Series
        'prod_dates_14_json': prod_dates_14_json,
        'study_hours_14_json': study_hours_14_json,
        'work_hours_14_json': work_hours_14_json,
        'productivity_score_14_json': productivity_score_14_json,

        # Toggleable Fitness & Health Series
        'health_dates_14_json': health_dates_14_json,
        'steps_14_json': steps_14_json,
        'sleep_14_json': sleep_14_json,
        'exercise_14_json': exercise_14_json,

        # Goals & Recommendations
        'goals': goals,
        'advisory_cards': advisory_cards,
        'best_case_card': best_case_card,
        'risk_case_card': risk_case_card,
        'expected_case_card': expected_case_card,
        'rec_val': rec_savings_val,
        'best_val': best_savings_val,
        'risk_val': risk_savings_val,
        'rec_savings_val': rec_savings_val,
        'best_savings_val': best_savings_val,
        'risk_savings_val': risk_savings_val,
        'risk_savings_val_abs': risk_savings_val_abs,
        'plain_english_takeaway': plain_english_takeaway,
        'assistant_preset_chips': assistant_preset_chips,
    }

    return render(request, 'tracker/dashboard.html', context)

@login_required
def analytics_finance_view(request):
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    telemetry_qs = DailyTelemetryRecord.objects.filter(user=request.user)
    analytics_data = get_user_analytics_and_forecasts(request.user.id, telemetry_qs, profile.monthly_budget)
    return render(request, 'tracker/analytics_finance.html', {
        'profile': profile,
        'analytics': analytics_data,
        'has_data': analytics_data is not None,
    })

@login_required
def analytics_productivity_view(request):
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    telemetry_qs = DailyTelemetryRecord.objects.filter(user=request.user)
    analytics_data = get_user_analytics_and_forecasts(request.user.id, telemetry_qs, profile.monthly_budget)
    return render(request, 'tracker/analytics_productivity.html', {
        'profile': profile,
        'analytics': analytics_data,
        'has_data': analytics_data is not None,
    })

@login_required
def analytics_behavior_view(request):
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    telemetry_qs = DailyTelemetryRecord.objects.filter(user=request.user)
    analytics_data = get_user_analytics_and_forecasts(request.user.id, telemetry_qs, profile.monthly_budget)
    return render(request, 'tracker/analytics_behavior.html', {
        'profile': profile,
        'analytics': analytics_data,
        'has_data': analytics_data is not None,
    })

@login_required
def manual_daily_entry(request):
    user = request.user
    if request.method == 'POST':
        form = DailyTelemetryManualForm(request.POST)
        if form.is_valid():
            rec = form.save(commit=False)
            rec.user = user
            rec.total_expenses = (
                rec.food_expenses + rec.travel_expenses + rec.shopping_expenses +
                rec.utilities_expenses + rec.health_expenses + rec.other_expenses
            )
            rec.savings = rec.income - rec.total_expenses
            C_task = min(1.0, rec.tasks_completed / max(1, rec.tasks_assigned))
            tot_act = float(rec.work_hours + rec.study_hours)
            denom = max(float(rec.screen_time_hours), tot_act) + (rec.break_minutes / 60.0)
            E_focus = float(np.clip(tot_act / max(0.1, denom), 0.0, 1.0))
            R_cog = float(np.clip(float(rec.sleep_hours) / 8.0, 0.0, 1.0) * ((11 - rec.stress_level) / 10.0))
            rec.productivity_score = round(float(np.clip(100.0 * (0.40 * C_task + 0.35 * E_focus + 0.25 * R_cog), 0.0, 100.0)), 2)
            
            if (rec.work_hours + rec.study_hours) >= 8.0 and E_focus >= 0.65 and rec.stress_level <= 6:
                rec.behavioral_state = 'Peak Focus'
            elif rec.work_hours >= 9.5 or (rec.stress_level >= 8 and rec.sleep_hours <= 6.0):
                rec.behavioral_state = 'Burnout Risk'
            elif rec.screen_time_hours >= 8.5 and E_focus < 0.45:
                rec.behavioral_state = 'Distracted'
            else:
                rec.behavioral_state = 'Recovery'
            rec.save()

            ActivityLog.objects.create(user=user, category='Telemetry', action=f"Daily log saved for {rec.entry_date} (₹{rec.total_expenses})")
            train_user_models(user.id, DailyTelemetryRecord.objects.filter(user=user))
            messages.success(request, f"Daily telemetry for {rec.entry_date} synchronized. Models updated.")
            return redirect('dashboard')
    else:
        form = DailyTelemetryManualForm(initial={'entry_date': datetime.now().strftime("%Y-%m-%d"), 'day_of_week': datetime.now().strftime("%A")})
    return render(request, 'tracker/daily_entry.html', {'form': form})

@login_required
def upload_csv_view(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request, "Uploaded file must be a .csv format.")
            return redirect('dashboard')
        try:
            data_set = csv_file.read().decode('UTF-8')
            df = pd.read_csv(io.StringIO(data_set))

            def clean_dec(val, default=0.0):
                if pd.isna(val) or val is None or str(val).strip().lower() in ['nan', 'null', '']:
                    return default
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default

            def clean_nullable_dec(val):
                if pd.isna(val) or val is None or str(val).strip().lower() in ['nan', 'null', '']:
                    return None
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None

            records_to_create = []
            for _, row in df.iterrows():
                next_exp = clean_nullable_dec(row.get('next_day_expenses'))
                if next_exp is None:
                    next_exp = clean_dec(row.get('total_expenses'))
                next_prod = clean_nullable_dec(row.get('next_day_productivity'))
                if next_prod is None:
                    next_prod = clean_dec(row.get('productivity_score'))

                records_to_create.append(
                    DailyTelemetryRecord(
                        user=request.user,
                        entry_date=pd.to_datetime(row['entry_date']).date(),
                        day_of_week=str(row['day_of_week']),
                        income=clean_dec(row.get('income')),
                        food_expenses=clean_dec(row.get('food_expenses')),
                        travel_expenses=clean_dec(row.get('travel_expenses')),
                        shopping_expenses=clean_dec(row.get('shopping_expenses')),
                        utilities_expenses=clean_dec(row.get('utilities_expenses')),
                        health_expenses=clean_dec(row.get('health_expenses')),
                        other_expenses=clean_dec(row.get('other_expenses')),
                        total_expenses=clean_dec(row.get('total_expenses')),
                        savings=clean_dec(row.get('savings')),
                        work_hours=clean_dec(row.get('work_hours')),
                        study_hours=clean_dec(row.get('study_hours')),
                        break_minutes=int(clean_dec(row.get('break_minutes'))),
                        screen_time_hours=clean_dec(row.get('screen_time_hours')),
                        tasks_assigned=max(1, int(clean_dec(row.get('tasks_assigned'), default=1))),
                        tasks_completed=int(clean_dec(row.get('tasks_completed'))),
                        productivity_score=clean_dec(row.get('productivity_score')),
                        steps_count=int(clean_dec(row.get('steps_count'), default=5000)),
                        exercise_duration_min=int(clean_dec(row.get('exercise_duration_min'))),
                        sleep_hours=clean_dec(row.get('sleep_hours'), default=7.0),
                        reading_meditation_min=int(clean_dec(row.get('reading_meditation_min'))),
                        habit_completion_pct=clean_dec(row.get('habit_completion_pct')),
                        stress_level=int(clean_dec(row.get('stress_level'), default=5)),
                        mood_score=int(clean_dec(row.get('mood_score'), default=7)),
                        behavioral_state=str(row.get('behavioral_state', 'Recovery')),
                        next_day_expenses=next_exp,
                        next_day_productivity=next_prod
                    )
                )

            DailyTelemetryRecord.objects.filter(user=request.user).delete()
            DailyTelemetryRecord.objects.bulk_create(records_to_create)

            ActivityLog.objects.create(
                user=request.user,
                category='Telemetry',
                action=f"Imported {len(records_to_create)} telemetry records via CSV."
            )
            train_user_models(request.user.id, DailyTelemetryRecord.objects.filter(user=request.user))
            messages.success(request, f"Successfully synchronized {len(records_to_create)} telemetry records. Models updated.")
            return redirect('analytics_finance')
        except Exception as e:
            messages.error(request, f"CSV parsing error: {str(e)}")
            return redirect('dashboard')
    return redirect('dashboard')

@login_required
def add_goal(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        category = request.POST.get('category', 'Finance')
        current_val = request.POST.get('current_value', 0)
        target_val = request.POST.get('target_value', 100)
        unit = request.POST.get('unit', '₹')
        if title and target_val:
            Goal.objects.create(user=request.user, title=title, category=category, current_value=current_val, target_value=target_val, unit=unit)
            ActivityLog.objects.create(user=request.user, category='Goals', action=f"Added target objective: {title} ({unit}{target_val})")
            messages.success(request, f"Objective '{title}' created successfully.")
    return redirect('dashboard')

@login_required
def update_goal_progress(request, goal_id):
    goal = get_object_or_404(Goal, id=goal_id, user=request.user)
    if request.method == 'POST':
        new_val = request.POST.get('current_value')
        if new_val is not None:
            goal.current_value = new_val
            goal.save()
            ActivityLog.objects.create(user=request.user, category='Goals', action=f"Updated progress for '{goal.title}'")
            messages.success(request, f"Progress updated for '{goal.title}'.")
    return redirect('dashboard')

@login_required
def delete_goal(request, goal_id):
    goal = get_object_or_404(Goal, id=goal_id, user=request.user)
    title = goal.title
    goal.delete()
    ActivityLog.objects.create(user=request.user, category='Goals', action=f"Removed objective: {title}")
    messages.info(request, f"Objective '{title}' removed.")
    return redirect('dashboard')

@login_required
def profile_view(request):
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    if request.method == 'POST':
        form = UserProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            ActivityLog.objects.create(user=request.user, category='Profile', action="Updated Twin profile.")
            messages.success(request, "Digital Twin configuration updated.")
            return redirect('profile')
    else:
        form = UserProfileForm(instance=profile)
    return render(request, 'tracker/profile.html', {'form': form, 'profile': profile})


@login_required
def simulation_sandbox_view(request):
    """
    Milestone 3 Dedicated Simulation Sandbox View.
    Ingests user telemetry, runs 90-day forward multi-scenario simulations in memory,
    evaluates rule-based advisory cards, and passes full JSON series for Chart.js.
    """
    user = request.user
    profile, _ = UserProfile.objects.get_or_create(user=user)
    telemetry_qs = DailyTelemetryRecord.objects.filter(user=user)

    if telemetry_qs.count() == 0:
        return render(request, 'tracker/analytics_simulation.html', {
            'has_data': False,
            'profile': profile,
        })

    monthly_budget = float(profile.monthly_budget or 50000.0)
    monthly_income = float(profile.monthly_income or 85000.0)

    # Run default 90-day forward simulation with initial custom scenario presets
    default_custom_params = {
        'custom_scenario': {
            'name': 'Custom What-If Scenario',
            'spend_reduction_pct': 0.10, # -10% spend default
            'study_hours_delta': 1.0,
            'sleep_hours_delta': 0.5,
            'screen_time_delta': -0.5,
            'work_hours_delta': 0.0,
            'stress_level_delta': -1.0,
            'steps_delta': 2000,
            'fixed_cost_inflation_pct': 0.0,
            'side_income_ratio': 0.0,
            'shock_ratio': 0.0,
            'shock_day': 30,
            'savings_yield_annual_pct': 0.04,
            'sprint_mode_active': False,
        },
        'scenario_a': {
            'name': 'Scenario A (Controlled Benchmark)',
            'spend_reduction_pct': 0.10,
            'study_hours_delta': 1.0,
            'sleep_hours_delta': 0.5,
            'screen_time_delta': -0.5,
            'work_hours_delta': 0.0,
            'stress_level_delta': -1.0,
            'steps_delta': 2000,
            'savings_yield_annual_pct': 0.04,
        },
        'scenario_b': {
            'name': 'Scenario B (Stress Shock Benchmark)',
            'mode': 'stress_shock',
            'spend_reduction_pct': -0.25,
            'study_hours_delta': -1.0,
            'sleep_hours_delta': -1.5,
            'screen_time_delta': 2.0,
            'screen_time_delta': 4.0,
            'work_hours_delta': 0.0,
            'stress_level_delta': 2.5,
            'steps_delta': -2000,
            'shock_ratio': 0.35,
            'shock_day': 30,
            'savings_yield_annual_pct': 0.04,
        }
    }
    sim_res = run_digital_twin_simulation(user.id, custom_params=default_custom_params, horizon_days=90)
    if not sim_res.get('success'):
        return render(request, 'tracker/analytics_simulation.html', {
            'has_data': False,
            'error': sim_res.get('error'),
            'profile': profile,
        })

    # Generate prescriptive advisory payload
    advisory_payload = get_simulation_advisory_payload(sim_res, profile)

    base_traj = sim_res['baseline_trajectory']
    traj_custom = sim_res['custom_trajectory']
    traj_a = sim_res['scenario_a_trajectory']
    traj_b = sim_res['scenario_b_trajectory']

    # Initial Live Custom Impact
    custom_delta = sim_res.get('deltas', {}).get('custom_scenario', {})
    c_savings = custom_delta.get('net_savings_delta', 0.0)
    c_focus = custom_delta.get('focus_pct_gained', 0.0)
    c_burnout_avg = custom_delta.get('avg_burnout_risk', 30.0)
    c_burnout_level = 'Low' if c_burnout_avg < 35.0 else ('Moderate' if c_burnout_avg < 60.0 else 'High')
    c_avg_focus = round(float(np.mean(traj_custom['productivity'])), 2) if len(traj_custom['productivity']) > 0 else 0.0

    custom_impact = {
        'net_savings_delta': c_savings,
        'net_savings_formatted': f"+₹{abs(c_savings):,.0f}" if c_savings >= 0 else f"-₹{abs(c_savings):,.0f}",
        'focus_pct_gained': c_focus,
        'focus_formatted': f"+{c_focus:.1f}%" if c_focus >= 0 else f"-{abs(c_focus):.1f}%",
        'avg_focus_index': c_avg_focus,
        'burnout_risk_level': c_burnout_level,
        'avg_burnout_risk': c_burnout_avg,
    }

    context = {
        'has_data': True,
        'profile': profile,
        'monthly_budget': monthly_budget,
        'monthly_income': monthly_income,
        'user_budget_json': json.dumps(monthly_budget),
        'user_income_json': json.dumps(monthly_income),
        'sim_results': sim_res,
        'horizon_days': sim_res.get('horizon_days', 90),

        # Serialized JSON series for Chart.js (All 4 Trajectories)
        'sim_dates_json': json.dumps(base_traj['dates']),
        'baseline_spend_json': json.dumps(base_traj['expenses']),
        'custom_spend_json': json.dumps(traj_custom['expenses']),
        'case_a_spend_json': json.dumps(traj_a['expenses']),
        'case_b_spend_json': json.dumps(traj_b['expenses']),

        'baseline_savings_json': json.dumps(base_traj['cumulative_savings']),
        'custom_savings_json': json.dumps(traj_custom['cumulative_savings']),
        'case_a_savings_json': json.dumps(traj_a['cumulative_savings']),
        'case_b_savings_json': json.dumps(traj_b['cumulative_savings']),

        'baseline_focus_json': json.dumps(base_traj['productivity']),
        'custom_focus_json': json.dumps(traj_custom['productivity']),
        'case_a_focus_json': json.dumps(traj_a['productivity']),
        'case_b_focus_json': json.dumps(traj_b['productivity']),

        'baseline_burnout_json': json.dumps(base_traj['burnout_risk']),
        'custom_burnout_json': json.dumps(traj_custom['burnout_risk']),
        'case_a_burnout_json': json.dumps(traj_a['burnout_risk']),
        'case_b_burnout_json': json.dumps(traj_b['burnout_risk']),

        # Structured Advisory Cards (Best-Case, Expected-Case, Risk-Case)
        'advisory_cards': advisory_payload['cards'],
        'best_case': advisory_payload['best_case'],
        'expected_case': advisory_payload['expected_case'],
        'risk_case': advisory_payload['risk_case'],
        'plain_english_takeaway': advisory_payload.get('plain_english_takeaway', ''),

        # Live Custom Impact Card & Deltas
        'custom_impact': custom_impact,
        'deltas': sim_res.get('deltas', {}),
        'checkpoints': sim_res.get('checkpoints', {}),
        'metadata': sim_res.get('metadata', {}),
    }

    return render(request, 'tracker/analytics_simulation.html', context)


@csrf_exempt
def api_run_simulation(request):
    """
    Asynchronous AJAX simulation endpoint for What-If sandbox slider interactions.
    Executes in-memory forward projections across all 4 parallel trajectories and returns
    updated curves, custom impact KPIs, and recalculated advisory steps in clean JSON format.
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST method required.'}, status=405)

    if request.user.is_authenticated:
        user = request.user
    else:
        # Fallback for headless API / cURL testing without session cookie
        from django.contrib.auth.models import User
        user = User.objects.filter(telemetry_records__isnull=False).distinct().first()
        if not user:
            user = User.objects.first()

    if not user:
        return JsonResponse({'success': False, 'error': 'No active user found for simulation.'}, status=400)

    profile, _ = UserProfile.objects.get_or_create(user=user)
    monthly_budget = float(profile.monthly_budget or 50000.0)
    monthly_income = float(profile.monthly_income or 85000.0)

    try:
        if request.body:
            body_data = json.loads(request.body.decode('utf-8'))
        else:
            body_data = {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        body_data = {}

    # 1. Parse slider parameters
    # Spend slider: negative value (-20) means cut spending by 20% (spend_reduction_pct = +0.20)
    # Spend slider: positive value (+20) means surge spending by 20% (spend_reduction_pct = -0.20)
    try:
        spend_val = float(body_data.get('expense_delta_pct', -10))
        spend_reduction_pct = -(spend_val / 100.0)
    except (ValueError, TypeError):
        spend_reduction_pct = 0.10

    study_delta = float(body_data.get('study_hours_delta', 1.0))

    # Sleep slider: delta from 7.0h baseline
    if 'sleep_hours_delta' in body_data:
        sleep_delta = float(body_data['sleep_hours_delta'])
    elif 'sleep_target_hours' in body_data:
        sleep_delta = float(body_data['sleep_target_hours']) - 7.0
    elif 'sleep_hours' in body_data:
        sleep_delta = float(body_data['sleep_hours']) - 7.0
    else:
        sleep_delta = 0.5

    # Screen slider: delta from 7.0h baseline
    if 'screen_time_delta' in body_data:
        screen_delta = float(body_data['screen_time_delta'])
    elif 'screen_limit_hours' in body_data:
        screen_delta = float(body_data['screen_limit_hours']) - 7.0
    elif 'screen_time_cap' in body_data:
        screen_delta = float(body_data['screen_time_cap']) - 7.0
    else:
        screen_delta = -0.5

    work_delta = float(body_data.get('work_hours_delta', 0.0))

    # Fixed bills inflation: 0.0 to 0.20 (accept percentage e.g. 5 or ratio 0.05)
    if 'fixed_inflation_pct' in body_data:
        raw_inf = float(body_data['fixed_inflation_pct'])
        fixed_inflation_pct = raw_inf / 100.0 if raw_inf > 1.0 else raw_inf
    elif 'fixed_cost_inflation_pct' in body_data:
        raw_inf = float(body_data['fixed_cost_inflation_pct'])
        fixed_inflation_pct = raw_inf / 100.0 if raw_inf > 1.0 else raw_inf
    else:
        fixed_inflation_pct = 0.0

    # Side income: 0% to 50%
    if 'side_income_pct' in body_data:
        side_income_ratio = float(body_data['side_income_pct']) / 100.0
    elif 'side_income_ratio' in body_data:
        side_income_ratio = float(body_data['side_income_ratio'])
    else:
        side_income_ratio = 0.0

    # Emergency shock: 0% to 80%
    if 'shock_pct' in body_data:
        shock_ratio = float(body_data['shock_pct']) / 100.0
    elif 'shock_ratio' in body_data:
        shock_ratio = float(body_data['shock_ratio'])
    else:
        shock_ratio = 0.0

    shock_day = int(body_data.get('shock_day', 30))
    yield_annual_pct = float(body_data.get('savings_yield_annual_pct', 0.04))
    sprint_active = bool(body_data.get('sprint_mode', body_data.get('sprint_mode_active', False)))

    stress_delta = float(body_data.get('stress_level_delta', -1.0 if spend_reduction_pct >= 0 else 1.5))
    steps_delta = int(body_data.get('steps_delta', 2000 if spend_reduction_pct >= 0 else -1000))
    horizon_days = int(body_data.get('horizon_days', 90))

    # Multi-lever economic & cognitive parameters
    # fixed_inflation_pct = float(body_data.get('fixed_cost_inflation_pct', 0.0))
    # side_income_ratio = float(body_data.get('side_income_ratio', 0.0))
    # shock_ratio = float(body_data.get('shock_ratio', 0.0))
    # shock_day = int(body_data.get('shock_day', 30))
    # yield_annual_pct = float(body_data.get('savings_yield_annual_pct', 0.04))
    # sprint_active = bool(body_data.get('sprint_mode_active', False))
    # work_delta = float(body_data.get('work_hours_delta', 0.0))

    custom_params = {
        'custom_scenario': {
            'name': 'Custom What-If Scenario',
            'spend_reduction_pct': spend_reduction_pct,
            'study_hours_delta': study_delta,
            'sleep_hours_delta': sleep_delta,
            'screen_time_delta': screen_delta,
            'work_hours_delta': work_delta,
            'stress_level_delta': stress_delta,
            'steps_delta': steps_delta,
            'fixed_cost_inflation_pct': fixed_inflation_pct,
            'side_income_ratio': side_income_ratio,
            'shock_ratio': shock_ratio,
            'shock_day': shock_day,
            'savings_yield_annual_pct': yield_annual_pct,
            'sprint_mode_active': sprint_active,
        },
        'scenario_a': {
            'name': 'Scenario A (Controlled Benchmark)',
            'spend_reduction_pct': 0.10,
            'study_hours_delta': 1.0,
            'sleep_hours_delta': 0.5,
            'screen_time_delta': -0.5,
            'work_hours_delta': 0.0,
            'stress_level_delta': -1.0,
            'steps_delta': 2000,
            'savings_yield_annual_pct': 0.04,
        },
        'scenario_b': {
            'name': 'Scenario B (Stress Shock Benchmark)',
            'mode': 'stress_shock',
            'spend_reduction_pct': -0.25,
            'study_hours_delta': -1.0,
            'sleep_hours_delta': -1.5,
            'screen_time_delta': 4.0,
            'work_hours_delta': 0.0,
            'stress_level_delta': 2.5,
            'steps_delta': -2000,
            'shock_ratio': 0.35,
            'shock_day': 30,
            'savings_yield_annual_pct': 0.04,
        }
    }

    # 2. In-memory simulation execution across all 4 scenarios
    sim_res = run_digital_twin_simulation(user.id, custom_params=custom_params, horizon_days=horizon_days)
    if not sim_res.get('success'):
        return JsonResponse({'success': False, 'error': sim_res.get('error')}, status=400)

    # 3. Dynamic advisory recalculation
    profile, _ = UserProfile.objects.get_or_create(user=user)
    advisory_payload = get_simulation_advisory_payload(sim_res, profile)

    base_traj = sim_res['baseline_trajectory']
    traj_custom = sim_res['custom_trajectory']
    traj_a = sim_res['scenario_a_trajectory']
    traj_b = sim_res['scenario_b_trajectory']

    # 4. Compute Live Custom Impact Card Metrics
    custom_delta = sim_res.get('deltas', {}).get('custom_scenario', {})
    c_savings = custom_delta.get('net_savings_delta', 0.0)
    c_focus = custom_delta.get('focus_pct_gained', 0.0)
    c_burnout_avg = custom_delta.get('avg_burnout_risk', 30.0)
    c_burnout_level = 'Low' if c_burnout_avg < 35.0 else ('Moderate' if c_burnout_avg < 60.0 else 'High')
    c_avg_focus = round(float(np.mean(traj_custom['productivity'])), 2) if len(traj_custom['productivity']) > 0 else 0.0

    custom_impact = {
        'net_savings_delta': c_savings,
        'net_savings_formatted': f"+₹{abs(c_savings):,.0f}" if c_savings >= 0 else f"-₹{abs(c_savings):,.0f}",
        'focus_pct_gained': c_focus,
        'focus_formatted': f"+{c_focus:.1f}%" if c_focus >= 0 else f"-{abs(c_focus):.1f}%",
        'avg_focus_index': c_avg_focus,
        'burnout_risk_level': c_burnout_level,
        'avg_burnout_risk': c_burnout_avg,
    }

    # 5. Standardized JSON payload response with all 4 trajectories
    return JsonResponse({
        'success': True,
        'user_id': user.id,
        'horizon_days': sim_res['horizon_days'],
        'sim_dates': base_traj['dates'],
        'trajectories': {
            'baseline': {
                'expenses': base_traj['expenses'],
                'cumulative_savings': base_traj['cumulative_savings'],
                'productivity': base_traj['productivity'],
                'burnout_risk': base_traj['burnout_risk'],
            },
            'custom_scenario': {
                'expenses': traj_custom['expenses'],
                'cumulative_savings': traj_custom['cumulative_savings'],
                'productivity': traj_custom['productivity'],
                'burnout_risk': traj_custom['burnout_risk'],
            },
            'scenario_a': {
                'expenses': traj_a['expenses'],
                'cumulative_savings': traj_a['cumulative_savings'],
                'productivity': traj_a['productivity'],
                'burnout_risk': traj_a['burnout_risk'],
            },
            'scenario_b': {
                'expenses': traj_b['expenses'],
                'cumulative_savings': traj_b['cumulative_savings'],
                'productivity': traj_b['productivity'],
                'burnout_risk': traj_b['burnout_risk'],
            },
        },
        'deltas': sim_res.get('deltas', {}),
        'custom_impact': custom_impact,
        'advisory_cards': advisory_payload['cards'],
        'plain_english_takeaway': advisory_payload.get('plain_english_takeaway', ''),
        'checkpoints': sim_res.get('checkpoints', {}),
        'metadata': sim_res.get('metadata', {}),
    }, status=200)

@csrf_exempt
def api_assistant_chat(request):
    """
    Milestone 4 - Phase 1: Conversational AI Assistant API Endpoint.
    Ingests natural language queries, grounds them in database telemetry,
    and returns rich structured answers with KPIs and recommended actions.
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST method required.'}, status=405)

    if request.user.is_authenticated:
        user = request.user
    else:
        # Fallback for headless API testing / cURL
        from tracker.models import DailyTelemetryRecord
        first_rec = DailyTelemetryRecord.objects.first()
        if first_rec:
            user = first_rec.user
        else:
            from django.contrib.auth.models import User
            user = User.objects.first()

    if not user:
        return JsonResponse({'success': False, 'error': 'No active user found.'}, status=400)

    try:
        if request.body:
            body_data = json.loads(request.body.decode('utf-8'))
        else:
            body_data = {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        body_data = {}

    user_message = body_data.get('message', '').strip()
    if not user_message:
        return JsonResponse({'success': False, 'error': 'Message content cannot be empty.'}, status=400)

    from tracker.assistant_engine import process_assistant_query
    response_payload = process_assistant_query(user.id, user_message)
    return JsonResponse(response_payload, status=200)
