# Digital Twin AI — System Context, Invariants & Developer Handover
> **Target Audience:** Core Developers, ML Engineers, and AI Coding Assistants (Antigravity).  
> **Mandate:** Read this file completely before creating or modifying any code in Milestones 3 & 4.

---

## 1. High-Level Architecture & Current State
Digital Twin AI is a Django 6 application backed by PostgreSQL, running classical Scikit-Learn regression models to forecast personal financial expenditures, cognitive productivity scores, and behavioral state drift.

* **Framework:** Django 6.1 (Python 3.13)
* **Database:** PostgreSQL (`digitaltwin_db` on `localhost:5432`)
* **Telemetry Source:** `tracker.models.DailyTelemetryRecord`
* **ML Artifacts Directory:** `storage/user_models/user_<id>/`
* **Status:** Milestones 1 & 2 are 100% complete and verified. The system currently ingests daily telemetry, trains per-user classical ML regressors, and renders 3 dedicated analytics dashboards.

---

## 2. Telemetry Schema & Ground Truth Models

### Primary Model: `DailyTelemetryRecord` (`tracker/models.py`)
Unique together on `('user', 'entry_date')`. All calculations must map strictly to these field names:
* **Temporal:** `entry_date` (DateField, indexed), `day_of_week` (CharField)
* **Financial (₹):** `income`, `food_expenses`, `travel_expenses`, `shopping_expenses`, `utilities_expenses`, `health_expenses`, `other_expenses`, `total_expenses`, `savings` (all `DecimalField(max_digits=10/12, decimal_places=2)`)
* **Productivity:** `work_hours`, `study_hours`, `break_minutes`, `screen_time_hours`, `tasks_assigned`, `tasks_completed`, `productivity_score`
* **Wellness & Biometrics:** `steps_count`, `exercise_duration_min`, `sleep_hours`, `reading_meditation_min`, `habit_completion_pct`, `stress_level` (1-10), `mood_score` (1-10)
* **Behavioral State:** `behavioral_state` (`'Peak Focus'`, `'Burnout Risk'`, `'Distracted'`, `'Recovery'`)
* **Supervised Forward Targets:** `next_day_expenses`, `next_day_productivity` (next calendar day's real values)

### Supporting Models
* `UserProfile`: Contains `user`, `monthly_income` (default: 85000.00), `monthly_budget` (default: 50000.00), `target_study_hours_weekly`, `goal_score`, `days_active`.
* `Goal`: Quantitative objectives (`Finance`, `Study`, `Habit`) with dynamic `progress_percentage`.
* `FinancialRecord`, `StudyRecord`, `HabitRecord`: Preserved legacy models. **DO NOT DELETE OR MIGRATE OUT.**

---

## 3. Exact Machine Learning Feature Contracts (`tracker/ml_engine.py`)

Trained models expect **exact feature names and array order**. Any deviation in feature naming or shape causes a Scikit-Learn `ValueError: feature_names mismatch`.

### A. Financial Regressor: `RandomForestRegressor`
* **Artifact:** `storage/user_models/user_<id>/expense_model.joblib`
* **Target:** `next_day_expenses` (continuous ₹ spend)
* **Exact Feature List (Ordered):**
  1. `total_expenses` (float)
  2. `is_tomorrow_weekend` (1.0 if next day is Sat/Sun, else 0.0)
  3. `is_tomorrow_rent` (1.0 if current day is end of month, else 0.0)
  4. `stress_spend_lead` (`(stress_level >= 7) * (shopping + food)`)
  5. `rolling_exp_7d` (7-day rolling expenditure mean)
  6. `dow_sin` (`sin(2 * pi * day_of_week / 7.0)`)
  7. `dow_cos` (`cos(2 * pi * day_of_week / 7.0)`)

### B. Productivity Regressor: `HistGradientBoostingRegressor`
* **Artifact:** `storage/user_models/user_<id>/productivity_model.joblib`
* **Target:** `next_day_productivity` (continuous 0-100 cognitive score)
* **Exact Feature List (Ordered):**
  1. `productivity_score` (float)
  2. `sleep_hours` (float)
  3. `fatigue_lag` (`(work_hours / 8.0) - (sleep_hours / 8.0)`)
  4. `work_hours` (float)
  5. `is_tomorrow_weekend` (float)
  6. `stress_level` (float)
  7. `screen_time_hours` (float)

---

## 4. Cold-Start Phases & Benchmark Persistence
* **Phase 1 (<7 days):** Calibrating. No model exists. 3-day rolling heuristics with 1.8x weekend uplift are used.
* **Phase 2 (7 to 29 days):** Early Learning. Constrained trees evaluated on a full 7-day holdout cycle (`test_size = 7`).
* **Phase 3 (30+ days):** Mature Intelligence. Full-depth ensembles evaluated on an 80:20 chronological split.
* **Metadata Artifact:** `storage/user_models/user_<id>/benchmarks.joblib` holds phase codes, raw R², MAE, RMSE, WAPE-based accuracy, and status strings.

---

## 5. Non-Negotiable Engineering Invariants (The Landmines)

1. **Strict Type Safety (`float` vs `Decimal`):**  
   Django fields return Python `decimal.Decimal`. Scikit-Learn outputs NumPy `float64`. Subtracting or multiplying them directly will raise `TypeError: unsupported operand type(s) for -: 'decimal.Decimal' and 'float'`.  
   *Rule:* Always explicitly cast data using `np.asarray(..., dtype=float)` or `.astype(float)` before any ML inference or arithmetic.

2. **Simulation Must Be 100% In-Memory:**  
   Milestone 3's "What-If" engine is a sandbox. **NEVER write, update, or delete records in `DailyTelemetryRecord` during simulation.** All forward projections are generated in memory and returned as JSON/context.

3. **Strict Mathematical Integrity (Conservation of Cash Flow):**  
   If an intervention reduces expenses by 10%, that reduction MUST produce an exact rupee-for-rupee increase in savings:  
   $$\text{Simulated Savings} = \text{Projected Income} - \text{Simulated Expenses}$$  
   Never invent arbitrary savings deltas; math must balance to the rupee.

4. **Dynamic Forward Causal State Updates:**  
   When projecting 90 days forward day-by-day, you cannot use static feature values. Each step must dynamically update:  
   * `is_tomorrow_weekend = 1.0 if next_date.weekday() in [5, 6] else 0.0`
   * `is_tomorrow_rent = 1.0 if next_date.day == 1 else 0.0`
   * `dow_sin = sin(2 * pi * current_date.weekday() / 7.0)`
   * `dow_cos = cos(2 * pi * current_date.weekday() / 7.0)`
   * `fatigue_lag = (work_hours / 8.0) - (sleep_hours / 8.0)`

5. **Zero Artificial Padding or Metric Floors:**  
   Never inject cosmetic floors like `max(0.72, r2)` or `clip(accuracy, 70, 95)`. All metrics must represent true Scikit-Learn variance.

---

## 6. Milestone 3 Roadmap & Scope Boundaries

### What Milestone 3 Builds:
1. `tracker/simulation_engine.py`: Multi-scenario 90-day forward inference loop (Baseline vs. Case 1 vs. Case 2) without database writes.
2. `tracker/recommendation_engine.py`: Algorithmic rule-based classifier labeling Best-Case, Expected-Case, and Risk-Case with actionable next steps.
3. `tracker/views.py` & `tracker/urls.py`: `simulation_sandbox_view` (`/simulation/`) and an asynchronous AJAX endpoint `api_run_simulation` (`/api/simulate/`).
4. `tracker/templates/tracker/analytics_simulation.html`: Sandbox UI with interactive sliders, 3 advisory cards, and multi-line Chart.js projections.

### What is Deferred to Milestone 4:
* Conversational LLM chat interfaces ("Ask Your Twin").
* External market/financial API connectors.
* PDF report generation.