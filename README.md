# Digital Twin AI: Personal Finance, Habit, and Productivity System

An integrated Python/Django application that creates a synchronized computational replica of an individual's financial, academic, and physiological trajectory. It utilizes classical Scikit-Learn regression models for time-series forecasting and an agentic OpenAI-compatible chatbot (powered by Groq LPU) to run multi-scenario 90-day forward simulations.

## Key Engineering Milestones
* **Milestone 1:** Atomic relational telemetry schema in PostgreSQL/SQLite capturing 28 daily features.
* **Milestone 2:** Supervised ML forecasting (`RandomForestRegressor` and `HistGradientBoostingRegressor`) predicting daily spend and focus.
* **Milestone 3:** 100% In-Memory 90-Day Simulation Engine enforcing strict cash-flow conservation.
* **Milestone 4:** Agentic conversational AI assistant executing tool calls and citing empirical model feature weights.

## Local Setup & Installation

**1. Clone the Repository**
```bash
git clone [https://github.com/](https://github.com/)[Your-Username]/digital-twin-ai.git
cd digital-twin-ai
