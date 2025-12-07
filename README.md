# 🏎️ **F1 Models – Deep Learning Race Strategy Project**

A hybrid **Machine Learning + Deep Learning** system designed to predict lap times, classify tyre degradation, and recommend race strategies specifically for **Autodromo Nazionale Monza**.

This project integrates:
- **Gradient Boosting Regressor** → Lap-time forecasting  
- **CNN-LSTM Network** → Tyre degradation classification  
- **Flask API Server** → Real-time race predictions  
- **Frontend Dashboard** → Live race-engineer interface  

---

# 🚀 **Project Features**

## 🔹 **Lap Time Prediction (Machine Learning)**  
A **Gradient Boosting Regressor** trained on high-resolution telemetry-derived features to forecast next-lap pace under live race conditions.

---

## 🔹 **Tyre Degradation Classification (Deep Learning)**  
A hybrid **CNN-LSTM** model that learns degradation patterns from sequences of ten consecutive lap times to classify tyres into:

- 🟢 **Fresh**
- 🔵 **Optimal**
- 🟠 **Worn**
- 🔴 **Critical**

---

## 🔹 **Race Strategy Engine (ML + DL)**  
Automatically recommends:

- 🛑 Pit instructions  
- 🔄 Compound switching (Soft ↔ Medium ↔ Hard)  
- 🏎️ Pace management suggestions  
- ⚠️ Risk detection (punctures, SC/VSC distortions, pit out-laps)

---


---

## 🔹 **Frontend Dashboard (Race Engineer UI)**  
Displays:

- Lap comparison  
- Tyre condition gauges  
- Confidence metrics  
- Strategy panel  
- Track map & sector layout  
- Stint history and compound usage  

---

# 🧩 **System Architecture**

```text
      ┌─────────────────────┐
      │  Frontend Dashboard │
      └──────────┬──────────┘
                 │ HTTP (JSON)
                 ▼
        ┌────────────────────┐
        │    Flask API       │
        └──────┬──────┬─────┘
               │      │
               │      │
  ┌────────────▼┐  ┌──▼─────────────────┐
  │ Lap Time ML │  │ Tyre DL (CNN-LSTM) │
  └──────┬──────┘  └──────────┬─────────┘
         │                     │
         └──────────┬─────────┘
                    ▼
          ┌──────────────────┐
          │ Strategy Engine  │
          └──────────────────┘
Endpoints:

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

```

🔧 1. Installation
Clone the repository
git clone https://github.com/sharmarvellous/F1-Models-Deep-Learning-25253051-Project.git
cd F1-Models-Deep-Learning-25253051-Project

Install dependencies
pip install -r requirements.txt
pip install tensorflow

🧠 2. Running the API Server

Run:

python api_server.py


Expected startup:

🔥 MONZA STRATEGY API SERVER
Available endpoints:
/api/health
/api/analyze
/api/drivers
/api/batch-analyze


Server runs at:

http://localhost:5000


📡 3. API Usage
POST /api/analyze
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

Example fetch call:

fetch("http://localhost:5000/api/analyze", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({...})
})
.then(res => res.json())
.then(data => console.log(data));

🎥 5. Demo Video (Google Drive)

GitHub cannot host large high-quality videos, so the full demonstration is here:

👉 Full Demo Video:
https://drive.google.com/file/d/1nGqoFcQRsTgP9b3mn2Wf7BL0UkcWGaQl/view?usp=sharing

Includes:

Frontend walkthrough

Side-by-side driver comparison

Tyre health predictions

Real-time strategy engine demo

📊 6. Model Training Resources Included

training_history.png → Loss & accuracy curves

model_performance.txt → Evaluation summary

tire_degradation_model_final.h5 → CNN-LSTM

gradient_boosting_model.pkl → ML model
