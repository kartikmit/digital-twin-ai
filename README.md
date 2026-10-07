# Digital Twin AI: Personal Finance, Habit, and Productivity System

An integrated Python/Django application that creates a synchronized computational replica of an individual's financial, academic, and physiological trajectory. It utilizes classical Scikit-Learn regression models for time-series forecasting and an agentic OpenAI-compatible chatbot (powered by Groq LPU) to run multi-scenario 90-day forward simulations.

## Key Engineering Milestones
* **Milestone 1:** Atomic relational telemetry schema in PostgreSQL/SQLite capturing 28 daily features.
* **Milestone 2:** Supervised ML forecasting (`RandomForestRegressor` and `HistGradientBoostingRegressor`) predicting daily spend and focus.
* **Milestone 3:** 100% In-Memory 90-Day Simulation Engine enforcing strict cash-flow conservation.
* **Milestone 4:** Agentic conversational AI assistant executing tool calls and citing empirical model feature weights.

## Local Setup & Installation

> ⚠️ **Important Note on API Keys:** 
> The conversational AI features of this project require a Groq API key to function. For security reasons, API keys are not committed to this repository. You can obtain a free API key by creating an account on the [Groq Cloud Console](https://console.groq.com/). You will need to create a `.env` file to store this key as shown in Step 2.

**1. Clone the Repository & Install Dependencies**
Open your terminal, clone the project, and install the required Python packages:
```bash
git clone [https://github.com/kartikmit/digital-twin-ai.git](https://github.com/kartikmit/digital-twin-ai.git)
cd digital-twin-ai
pip install -r requirements.txt
```

**2. Configure the API Key**
Create a new file named exactly `.env` in the root directory of the project (the exact same folder where `manage.py` is located) and add your Groq API key using the exact variable name shown below:
```text
GROQ_API_KEY=your_actual_api_key_here
```

**3. Initialize the Database**
Run the migration command to construct the required database tables locally:
```bash
python manage.py migrate
```

**4. Start the Server**
```bash
python manage.py runserver
```

