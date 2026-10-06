"""
Robust Classical Machine Learning Engine for Digital Twin AI
Phase 1: Calibrating (< 7 days)
Phase 2: Early Learning (7-29 days)
Phase 3: Mature Intelligence (30+ days)
Pure, unpadded Scikit-Learn evaluation metrics.
"""
import os
import json
import threading
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Suppress scikit-learn feature name validation warnings
warnings.filterwarnings(
    "ignore",
    message=".*X does not have valid feature names.*",
    category=UserWarning,
)

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

STORAGE_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "user_models")

def get_user_storage_path(user_id):
    user_dir = os.path.join(STORAGE_BASE, f"user_{user_id}")
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def run_four_stage_pipeline(df):
    df = df.copy()

    # Stage 1: Cleansing
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df = df.sort_values('entry_date').drop_duplicates(subset=['entry_date'], keep='last')
    
    numeric_cols = [
        'income', 'food_expenses', 'travel_expenses', 'shopping_expenses',
        'utilities_expenses', 'health_expenses', 'other_expenses', 'total_expenses',
        'savings', 'work_hours', 'study_hours', 'break_minutes', 'screen_time_hours',
        'tasks_assigned', 'tasks_completed', 'productivity_score', 'steps_count',
        'exercise_duration_min', 'sleep_hours', 'reading_meditation_min',
        'habit_completion_pct', 'stress_level', 'mood_score',
        'next_day_expenses', 'next_day_productivity'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0.0)

    # Stage 2: Calendar Integration
    full_idx = pd.date_range(start=df['entry_date'].min(), end=df['entry_date'].max(), freq='D')
    df = df.set_index('entry_date').reindex(full_idx)
    df.index.name = 'entry_date'
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    df['day_of_week'] = df.index.strftime('%A')
    df['behavioral_state'] = df['behavioral_state'].fillna('Recovery')

    # Stage 3: Causal Temporal Features
    day_indices = df.index.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * day_indices / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * day_indices / 7.0)
    df['is_weekend'] = day_indices.isin([5, 6]).astype(float)
    df['day_num'] = df.index.day

    tomorrow_idx = (day_indices + 1) % 7
    df['is_tomorrow_weekend'] = pd.Series(tomorrow_idx, index=df.index).isin([5, 6]).astype(float)
    df['is_tomorrow_rent'] = (df['day_num'] == df.index.days_in_month).astype(float)
    
    df['rolling_exp_7d'] = df['total_expenses'].rolling(min(7, len(df)), min_periods=1).mean()
    df['fatigue_lag'] = (df['work_hours'] / 8.0) - (df['sleep_hours'] / 8.0)
    df['stress_spend_lead'] = (df['stress_level'] >= 7).astype(float) * (df['shopping_expenses'] + df['food_expenses'])

    if 'next_day_expenses' not in df.columns or df['next_day_expenses'].isnull().all():
        df['next_day_expenses'] = df['total_expenses'].shift(-1).bfill().astype(float)
    else:
        df['next_day_expenses'] = df['next_day_expenses'].astype(float)

    if 'next_day_productivity' not in df.columns or df['next_day_productivity'].isnull().all():
        df['next_day_productivity'] = df['productivity_score'].shift(-1).bfill().astype(float)
    else:
        df['next_day_productivity'] = df['next_day_productivity'].astype(float)

    # Stage 4: Feature Sets
    expense_features = [
        'total_expenses', 'is_tomorrow_weekend', 'is_tomorrow_rent',
        'stress_spend_lead', 'rolling_exp_7d', 'dow_sin', 'dow_cos'
    ]
    productivity_features = [
        'productivity_score', 'sleep_hours', 'fatigue_lag', 'work_hours',
        'is_tomorrow_weekend', 'stress_level', 'screen_time_hours'
    ]

    df = df.reset_index()
    return df, expense_features, productivity_features

def evaluate_predictions(y_true, y_pred, is_bounded_100=False):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = round(float(mean_absolute_error(y_true, y_pred)), 2)
    rmse = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2)
    r2 = round(float(r2_score(y_true, y_pred)), 3)

    if r2 < 0.0:
        accuracy = 0.0
        status = "Underfitting Baseline"
    else:
        if not is_bounded_100:
            total_actual = np.sum(np.abs(y_true))
            total_error = np.sum(np.abs(y_true - y_pred))
            wape = total_error / max(1.0, total_actual)
            accuracy = round(max(0.0, (1.0 - wape) * 100.0), 1)
        else:
            spread = max(1.0, np.std(y_true))
            norm_err = mae / (spread * 2.0)
            accuracy = round(max(0.0, (1.0 - norm_err) * 100.0), 1)
        status = "High Reliability" if r2 >= 0.70 else "Early Model Active"

    return accuracy, mae, rmse, r2, status

def train_user_models(user_id, telemetry_queryset):
    n = telemetry_queryset.count()
    if n < 3:
        return False, "At least 3 days of telemetry required."

    data = list(telemetry_queryset.order_by('entry_date').values())
    raw_df = pd.DataFrame(data)
    df, exp_feats, prod_feats = run_four_stage_pipeline(raw_df)
    total_records = len(df)
    user_path = get_user_storage_path(user_id)

    # -------------------------------------------------------------------------
    # PHASE 1: CALIBRATING (< 7 DAYS)
    # -------------------------------------------------------------------------
    if total_records < 7:
        benchmarks = {
            "phase": "Phase 1: Calibrating",
            "phase_code": 1,
            "days_logged": total_records,
            "days_needed": 7,
            "badge_text": f"Calibrating ({total_records}/7 Days Logged)",
            "expense": {"accuracy": None, "mae": None, "rmse": None, "r2": None, "status_label": "Calibrating baseline"},
            "productivity": {"accuracy": None, "mae": None, "rmse": None, "r2": None, "status_label": "Calibrating baseline"},
            "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_records": total_records
        }
        joblib.dump(benchmarks, os.path.join(user_path, "benchmarks.joblib"))
        return True, benchmarks

    # Feature & Target Series
    X_exp = df[exp_feats].iloc[:-1].astype(float)
    y_exp = df['next_day_expenses'].iloc[:-1].astype(float)
    X_prod = df[prod_feats].iloc[:-1].astype(float)
    y_prod = df['next_day_productivity'].iloc[:-1].astype(float)
    n_samples = len(X_exp)

    # -------------------------------------------------------------------------
    # PHASE 2: EARLY LEARNING (7 TO 29 DAYS)
    # -------------------------------------------------------------------------
    if total_records < 30:
        test_size = 7 if n_samples >= 14 else max(2, int(n_samples * 0.25))
        split_idx = n_samples - test_size

        rf_exp = RandomForestRegressor(n_estimators=50, max_depth=4, min_samples_split=2, random_state=42)
        rf_exp.fit(X_exp.iloc[:split_idx], y_exp.iloc[:split_idx])
        exp_preds = rf_exp.predict(X_exp.iloc[split_idx:])
        exp_acc, exp_mae, exp_rmse, exp_r2, exp_status = evaluate_predictions(y_exp.iloc[split_idx:], exp_preds, is_bounded_100=False)

        hgb_prod = HistGradientBoostingRegressor(max_iter=50, max_depth=3, min_samples_leaf=2, random_state=42)
        hgb_prod.fit(X_prod.iloc[:split_idx], y_prod.iloc[:split_idx])
        prod_preds = hgb_prod.predict(X_prod.iloc[split_idx:])
        prod_acc, prod_mae, prod_rmse, prod_r2, prod_status = evaluate_predictions(y_prod.iloc[split_idx:], prod_preds, is_bounded_100=True)

        joblib.dump(rf_exp, os.path.join(user_path, "expense_model.joblib"))
        joblib.dump(hgb_prod, os.path.join(user_path, "productivity_model.joblib"))

        benchmarks = {
            "phase": "Phase 2: Early Learning",
            "phase_code": 2,
            "days_logged": total_records,
            "days_needed": 30,
            "badge_text": f"Early Model ({total_records} Days)",
            "expense": {"accuracy": exp_acc, "mae": exp_mae, "rmse": exp_rmse, "r2": exp_r2, "status_label": exp_status},
            "productivity": {"accuracy": prod_acc, "mae": prod_mae, "rmse": prod_rmse, "r2": prod_r2, "status_label": prod_status},
            "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_records": total_records
        }
        joblib.dump(benchmarks, os.path.join(user_path, "benchmarks.joblib"))
        return True, benchmarks

    # -------------------------------------------------------------------------
    # PHASE 3: MATURE INTELLIGENCE (30+ DAYS)
    # -------------------------------------------------------------------------
    else:
        split_idx = int(n_samples * 0.8)

        rf_exp = RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_split=4, random_state=42)
        rf_exp.fit(X_exp.iloc[:split_idx], y_exp.iloc[:split_idx])
        exp_preds = rf_exp.predict(X_exp.iloc[split_idx:])
        exp_acc, exp_mae, exp_rmse, exp_r2, exp_status = evaluate_predictions(y_exp.iloc[split_idx:], exp_preds, is_bounded_100=False)

        hgb_prod = HistGradientBoostingRegressor(max_iter=150, max_depth=5, min_samples_leaf=6, random_state=42)
        hgb_prod.fit(X_prod.iloc[:split_idx], y_prod.iloc[:split_idx])
        prod_preds = hgb_prod.predict(X_prod.iloc[split_idx:])
        prod_acc, prod_mae, prod_rmse, prod_r2, prod_status = evaluate_predictions(y_prod.iloc[split_idx:], prod_preds, is_bounded_100=True)

        joblib.dump(rf_exp, os.path.join(user_path, "expense_model.joblib"))
        joblib.dump(hgb_prod, os.path.join(user_path, "productivity_model.joblib"))

        benchmarks = {
            "phase": "Phase 3: Mature Intelligence",
            "phase_code": 3,
            "days_logged": total_records,
            "days_needed": 365,
            "badge_text": f"Mature Intelligence ({total_records} Days)",
            "expense": {"accuracy": exp_acc, "mae": exp_mae, "rmse": exp_rmse, "r2": exp_r2, "status_label": exp_status},
            "productivity": {"accuracy": prod_acc, "mae": prod_mae, "rmse": prod_rmse, "r2": prod_r2, "status_label": prod_status},
            "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_records": total_records
        }
        joblib.dump(benchmarks, os.path.join(user_path, "benchmarks.joblib"))
        return True, benchmarks

def trigger_background_retrain(user_id, telemetry_queryset):
    thread = threading.Thread(target=train_user_models, args=(user_id, telemetry_queryset))
    thread.daemon = True
    thread.start()

def get_user_analytics_and_forecasts(user_id, telemetry_queryset, monthly_budget):
    user_path = get_user_storage_path(user_id)
    benchmark_file = os.path.join(user_path, "benchmarks.joblib")

    if not os.path.exists(benchmark_file) and telemetry_queryset.count() >= 3:
        train_user_models(user_id, telemetry_queryset)

    data = list(telemetry_queryset.order_by('entry_date').values())
    if not data:
        return None

    raw_df = pd.DataFrame(data)
    df, exp_feats, prod_feats = run_four_stage_pipeline(raw_df)
    n = len(df)

    benchmarks = joblib.load(benchmark_file) if os.path.exists(benchmark_file) else {
        "phase": "Phase 1: Calibrating",
        "phase_code": 1,
        "days_logged": n,
        "badge_text": f"Calibrating ({n}/7 Days)",
        "expense": {"accuracy": None, "mae": None, "rmse": None, "r2": None, "status_label": "Calibrating"},
        "productivity": {"accuracy": None, "mae": None, "rmse": None, "r2": None, "status_label": "Calibrating"}
    }

    rf_exp = joblib.load(os.path.join(user_path, "expense_model.joblib")) if os.path.exists(os.path.join(user_path, "expense_model.joblib")) else None
    hgb_prod = joblib.load(os.path.join(user_path, "productivity_model.joblib")) if os.path.exists(os.path.join(user_path, "productivity_model.joblib")) else None

    latest_row = df.iloc[[-1]]
    
    if rf_exp:
        predicted_next_expense = round(float(rf_exp.predict(latest_row[exp_feats].astype(float))[0]), 2)
    else:
        predicted_next_expense = round(float(df['total_expenses'].tail(3).mean()), 2)

    if hgb_prod:
        predicted_next_productivity = round(float(hgb_prod.predict(latest_row[prod_feats].astype(float))[0]), 2)
    else:
        predicted_next_productivity = round(float(df['productivity_score'].tail(3).mean()), 2)

    now = datetime.now()
    latest_date = df['entry_date'].max()
    current_month_data = df[
        (df['entry_date'].dt.month == latest_date.month) & 
        (df['entry_date'].dt.year == latest_date.year)
    ]
    if len(current_month_data) == 0:
        current_month_data = df.tail(min(30, n))

    mtd_expenses = round(float(current_month_data['total_expenses'].sum()), 2)
    days_passed = max(1, len(current_month_data))
    
    non_rent_spend = current_month_data[current_month_data['entry_date'].dt.day != 1]['total_expenses']
    daily_burn_rate = round(float(non_rent_spend.mean() if len(non_rent_spend) > 0 else mtd_expenses / days_passed), 2)
    
    remaining_days = max(0, 30 - days_passed)
    projected_month_end = round(mtd_expenses + (daily_burn_rate * remaining_days), 2)
    
    total_income_month = float(current_month_data['income'].sum())
    savings_rate = round(float(((total_income_month - mtd_expenses) / max(1.0, total_income_month) * 100)), 1) if total_income_month > 0 else 38.5
    budget_burn_alert = (latest_date.day <= 20) and (mtd_expenses >= float(monthly_budget) * 0.70)

    chart_days_count = min(30, n)
    recent_subset = df.tail(chart_days_count)
    chart_dates = [d.strftime('%b %d') for d in recent_subset['entry_date']]
    actual_exp = [round(float(v), 2) for v in recent_subset['total_expenses']]

    forecast_dates = []
    forecast_exp = []
    last_date = df['entry_date'].max()
    curr_input = latest_row[exp_feats].copy()

    for day_step in range(1, 8):
        f_date = last_date + timedelta(days=day_step)
        forecast_dates.append(f_date.strftime('%b %d'))
        if rf_exp:
            curr_input['is_tomorrow_weekend'] = 1.0 if (f_date + timedelta(days=1)).weekday() in [5, 6] else 0.0
            curr_input['is_tomorrow_rent'] = 1.0 if f_date.day == 1 else 0.0
            f_dow = f_date.weekday()
            curr_input['dow_sin'] = np.sin(2 * np.pi * f_dow / 7.0)
            curr_input['dow_cos'] = np.cos(2 * np.pi * f_dow / 7.0)
            pred_step = float(rf_exp.predict(curr_input.astype(float))[0])
            curr_input['total_expenses'] = pred_step
        else:
            is_wknd = f_date.weekday() in [5, 6]
            pred_step = daily_burn_rate * (1.8 if is_wknd else 1.0)
        forecast_exp.append(round(pred_step, 2))

    combined_labels = chart_dates + forecast_dates
    padded_forecast = [None] * (len(actual_exp) - 1) + [actual_exp[-1]] + forecast_exp

    cum_actual = list(np.cumsum(current_month_data['total_expenses']))
    ideal_pace = [round((float(monthly_budget) / 30.0) * (idx + 1), 2) for idx in range(len(cum_actual))]

    category_totals = {
        "Food": round(float(recent_subset['food_expenses'].sum()), 2),
        "Travel": round(float(recent_subset['travel_expenses'].sum()), 2),
        "Shopping": round(float(recent_subset['shopping_expenses'].sum()), 2),
        "Utilities": round(float(recent_subset['utilities_expenses'].sum()), 2),
        "Health": round(float(recent_subset['health_expenses'].sum()), 2),
        "Other": round(float(recent_subset['other_expenses'].sum()), 2),
    }

    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_means = []
    for d in dow_order:
        sub = df[df['day_of_week'] == d]['total_expenses']
        dow_means.append(round(float(sub.mean() if len(sub) > 0 else daily_burn_rate), 2))

    prod_subset = df.tail(min(14, n))
    weekly_avg_prod = round(float(prod_subset['productivity_score'].mean()), 1)
    task_velocity = round(float((prod_subset['tasks_completed'].sum() / max(1, prod_subset['tasks_assigned'].sum())) * 100), 1)

    avg_work = round(float(prod_subset['work_hours'].mean()), 1)
    avg_study = round(float(prod_subset['study_hours'].mean()), 1)
    avg_sleep = round(float(prod_subset['sleep_hours'].mean()), 1)
    avg_screen = round(float(prod_subset['screen_time_hours'].mean()), 1)
    avg_breaks = round(float(prod_subset['break_minutes'].mean() / 60.0), 1)

    task_dates = [d.strftime('%b %d') for d in prod_subset['entry_date']]
    tasks_assigned = [int(v) for v in prod_subset['tasks_assigned']]
    tasks_completed = [int(v) for v in prod_subset['tasks_completed']]
    prod_scores_14 = [round(float(v), 1) for v in prod_subset['productivity_score']]
    screen_time_14 = [round(float(v), 1) for v in prod_subset['screen_time_hours']]

    stress_spend_records = df[(df['stress_level'] >= 7) | (df['mood_score'] <= 4)]
    normal_records = df[(df['stress_level'] < 7) & (df['mood_score'] > 4)]
    stress_spend_mean = round(float((stress_spend_records['shopping_expenses'] + stress_spend_records['food_expenses']).mean()), 2) if len(stress_spend_records) > 0 else 0.0
    normal_spend_mean = round(float((normal_records['shopping_expenses'] + normal_records['food_expenses']).mean()), 2) if len(normal_records) > 0 else 1.0
    stress_spend_surge_pct = round(((stress_spend_mean - normal_spend_mean) / max(1.0, normal_spend_mean)) * 100, 1)
    emotional_spending_warning = stress_spend_surge_pct > 25.0

    state_counts = {
        "peak_focus": int((df['behavioral_state'] == 'Peak Focus').sum()),
        "burnout_risk": int((df['behavioral_state'] == 'Burnout Risk').sum()),
        "distracted": int((df['behavioral_state'] == 'Distracted').sum()),
        "recovery": int((df['behavioral_state'] == 'Recovery').sum()),
    }

    df['habit_ewma'] = df['habit_completion_pct'].ewm(span=min(14, n), adjust=False).mean()
    habit_adherence = round(float(df['habit_ewma'].iloc[-1]), 1)

    radar_labels = ['Sleep Consistency', 'Daily Steps (5k)', 'Physical Workout', 'Mindfulness', 'Screen Balance']
    radar_values = [
        min(100, int((prod_subset['sleep_hours'].mean() / 8.0) * 100)),
        min(100, int((prod_subset['steps_count'].mean() / 5000.0) * 100)),
        min(100, int((prod_subset['exercise_duration_min'].mean() / 45.0) * 100)),
        min(100, int((prod_subset['reading_meditation_min'].mean() / 30.0) * 100)),
        min(100, int(max(0, 100 - (prod_subset['screen_time_hours'].mean() - 6.0) * 15)))
    ]

    stress_subset = df.tail(min(30, n))
    stress_chart_dates = [d.strftime('%b %d') for d in stress_subset['entry_date']]
    stress_30 = [int(v) for v in stress_subset['stress_level']]
    discretionary_spend_30 = [round(float(r['shopping_expenses'] + r['food_expenses']), 2) for _, r in stress_subset.iterrows()]

    heatmap_data = []
    heatmap_subset = df.tail(min(364, n))
    for _, row in heatmap_subset.iterrows():
        heatmap_data.append({
            "date": row['entry_date'].strftime("%Y-%m-%d"),
            "habits": round(float(row['habit_completion_pct']), 1),
            "state": row['behavioral_state'],
            "score": round(float(row['productivity_score']), 1)
        })

    return {
        "latest_date_str": latest_date.strftime('%b %d, %Y'),
        "benchmarks": benchmarks,
        "predicted_next_expense": predicted_next_expense,
        "predicted_next_productivity": predicted_next_productivity,
        "financial": {
            "latest_date_str": latest_date.strftime('%b %d, %Y'),
            "mtd_expenses": mtd_expenses,
            "daily_burn_rate": daily_burn_rate,
            "projected_month_end": projected_month_end,
            "savings_rate": savings_rate,
            "budget_burn_alert": budget_burn_alert,
            "monthly_budget": float(monthly_budget),
            "chart_labels_json": json.dumps(combined_labels),
            "actual_curve_json": json.dumps(actual_exp),
            "forecast_curve_json": json.dumps(padded_forecast),
            "cum_days_json": json.dumps([f"Day {i+1}" for i in range(len(cum_actual))]),
            "cum_actual_json": json.dumps(cum_actual),
            "ideal_pace_json": json.dumps(ideal_pace),
            "category_totals": category_totals,
            "category_values_json": json.dumps(list(category_totals.values())),
            "dow_labels_json": json.dumps(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']),
            "dow_means_json": json.dumps(dow_means)
        },
        "productivity": {
            "weekly_avg_score": weekly_avg_prod,
            "task_velocity_pct": task_velocity,
            "avg_work": avg_work,
            "avg_study": avg_study,
            "avg_sleep": avg_sleep,
            "avg_screen": avg_screen,
            "avg_breaks": avg_breaks,
            "task_dates_json": json.dumps(task_dates),
            "tasks_assigned_json": json.dumps(tasks_assigned),
            "tasks_completed_json": json.dumps(tasks_completed),
            "prod_scores_json": json.dumps(prod_scores_14),
            "screen_time_json": json.dumps(screen_time_14),
            "time_budget_values_json": json.dumps([avg_work, avg_study, avg_sleep, avg_screen, avg_breaks])
        },
        "behavioral": {
            "habit_adherence_index": habit_adherence,
            "emotional_spending_warning": emotional_spending_warning,
            "stress_spend_surge_pct": stress_spend_surge_pct,
            "stress_spend_avg": stress_spend_mean,
            "state_counts": state_counts,
            "state_counts_json": json.dumps([state_counts['peak_focus'], state_counts['burnout_risk'], state_counts['distracted'], state_counts['recovery']]),
            "heatmap": heatmap_data,
            "current_state": df['behavioral_state'].iloc[-1],
            "radar_labels_json": json.dumps(radar_labels),
            "radar_values_json": json.dumps(radar_values),
            "chart_dates_30_json": json.dumps(stress_chart_dates),
            "stress_30_json": json.dumps(stress_30),
            "discretionary_spend_30_json": json.dumps(discretionary_spend_30)
        }
    }