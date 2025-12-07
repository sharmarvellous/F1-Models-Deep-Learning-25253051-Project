F1 Models – Deep Learning Race Strategy Project

A hybrid Machine Learning + Deep Learning system designed to predict lap times, identify tyre degradation phases, and recommend race strategies for Autodromo Nazionale Monza. The project combines a Gradient Boosting regressor for lap-time forecasting with a CNN-LSTM sequence classifier for tyre-health prediction, all packaged as a production-ready API powering a real-time race-engineer dashboard.

🚀 Project Features
Lap Time Prediction (Machine Learning)

A Gradient Boosting Regressor trained on engineered telemetry-derived features to forecast the next-lap pace under varying stint conditions.

Tyre Degradation Classification (Deep Learning)

A CNN-LSTM network trained on sliding sequences of real lap times to classify tyre condition into:

Fresh

Optimal

Worn

Critical

Race Strategy Engine

Combines ML + DL outputs to recommend:

Pit stop timing

Compound switching

Pace management

Risk scenarios (puncture-like slow laps, pit out-laps, SC/VSC distortions)

REST API Server

Backend endpoints served using Flask:

/api/analyze – full model inference

/api/batch-analyze – multiple laps

/api/drivers – available drivers

/api/health – server status

Interactive Frontend

A race-engineer UI displaying:

Lap comparison

Tyre condition visualization

Strategy recommendations

Driver selection

Circuit statistics

Stint history

📂 Repository Structure
|-- models/
|     |-- gradient_boosting_model.pkl
|     |-- tire_degradation_model_final.h5
|     |-- feature_scaler.pkl
|     |-- target_scaler.pkl
|
|-- api_server.py              # Flask API backend
|-- monza_strategy_backend.py  # Core hybrid strategy engine
|-- strategy_decisions.csv
|-- requirements.txt
|-- README.md

🔧 1. Installation
Clone the repository
git clone https://github.com/sharmarvellous/F1-Models-Deep-Learning-25253051-Project.git
cd F1-Models-Deep-Learning-25253051-Project

Install all dependencies
pip install -r requirements.txt
pip install tensorflow       # CPU version recommended

🧠 2. Running the Race Strategy Backend

Start the model API server:

python api_server.py


You will see:

🔥 MONZA STRATEGY API SERVER
Available endpoints:
/api/health
/api/analyze
/api/drivers
/api/batch-analyze


Server URL:

http://localhost:5000

📡 3. API Usage
POST /api/analyze – Predict for a single lap
Request
{
  "driver": "16",
  "compound": "MEDIUM",
  "lap_number": 32,
  "last_10_laps": [83.2, 83.6, 84.0, 84.7, 85.5, 86.3, 87.1, 88.2, 89.4, 90.1]
}

Response
{
  "predicted_lap_time": 84.327,
  "tire_condition": "Worn",
  "confidence": 0.91,
  "recommended_strategy": "Pit within 2 laps"
}

🖥️ 4. Connecting the Frontend

Your frontend can call the backend like this:

fetch("http://localhost:5000/api/analyze", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({...})
})
.then(res => res.json())
.then(data => console.log(data));


The frontend outputs:

Predicted lap time

Tyre condition & confidence

Strategy (maintain pace, pit now, extend, etc.)

Driver comparison panel

Circuit metrics (Monza map, lap length, laps remaining)

🎥 5. Demo Video (Hosted on Google Drive)

GitHub cannot store the full-quality demo video due to file-size limits.

🔗 Full Demo Video:
👉 https://drive.google.com/file/d/1nGqoFcQRsTgP9b3mn2Wf7BL0UkcWGaQl/view?usp=sharing

The video includes:

Full frontend UI demo

Real-time API interaction

Head-to-head driver comparison

Tyre degradation predictions

Lap-time forecasting workflow

📊 6. Model Training Resources

Included in this repository:

training_history.png – learning curves

model_performance.txt – evaluation summary

tire_degradation_model_final.h5 – trained CNN-LSTM

gradient_boosting_model.pkl – ML model
