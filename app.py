import os
import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

# -----------------------------
# 1️⃣ Initialize Flask
# -----------------------------
app = Flask(__name__)
CORS(app)  # Allow frontend requests from any domain

# -----------------------------
# 2️⃣ File paths (relative)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "D:/forecasting/lstm_weather_model.h5")
SCALER_X_FILE = os.path.join(BASE_DIR, "D:/forecasting/scaler_X.pkl")
SCALER_Y_FILE = os.path.join(BASE_DIR, "D:/forecasting/scaler_y.pkl")
X_SCALED_FILE = os.path.join(BASE_DIR, "D:/forecasting/X_scaled.pkl")

# -----------------------------
# 3️⃣ Load model & scalers
# -----------------------------
for f in [MODEL_FILE, SCALER_X_FILE, SCALER_Y_FILE, X_SCALED_FILE]:
    if not os.path.isfile(f):
        raise FileNotFoundError(f"Required file not found: {f}")

model = load_model(MODEL_FILE, compile=False)
scaler_X = joblib.load(SCALER_X_FILE)
scaler_y = joblib.load(SCALER_Y_FILE)
X_scaled = joblib.load(X_SCALED_FILE)

SEQ_LEN = 7  # LSTM sequence length

# -----------------------------
# 4️⃣ Flask routes
# -----------------------------
@app.route("/")
def home():
    return "Flask LSTM API is running!"

@app.route("/predict", methods=["GET"])
def predict_energy():
    try:
        city = request.args.get("city", "Patiala,IN")
        API_KEY = os.environ.get("OPENWEATHER_API_KEY")
        if not API_KEY:
            return jsonify({"error": "OpenWeatherMap API key not set in environment variables."})

        # -----------------------------
        # Fetch weather forecast for tomorrow
        # -----------------------------
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url).json()
        if "list" not in response:
            return jsonify({"error": "Weather API returned invalid response. Check city/API key."})

        tomorrow = datetime.now() + timedelta(days=1)
        tmr_date = tomorrow.date()

        temps, hums, winds, precs = [], [], [], []
        for entry in response["list"]:
            dt = pd.to_datetime(entry["dt"], unit="s")
            if dt.date() == tmr_date:
                temps.append(entry["main"]["temp"])
                hums.append(entry["main"]["humidity"])
                winds.append(entry["wind"]["speed"])
                precs.append(entry.get("rain", {}).get("3h", 0))

        if not temps:
            return jsonify({"error": "No forecast available for tomorrow."})

        # Aggregate features
        tomorrow_features = [
            float(np.mean(temps)),      # temperature
            float(np.mean(hums)),       # humidity
            float(np.mean(winds)),      # wind speed
            float(np.sum(precs))        # precipitation
        ]

        # -----------------------------
        # Handle missing features
        # -----------------------------
        expected_features = X_scaled.shape[1]
        if len(tomorrow_features) < expected_features:
            tomorrow_features += [0] * (expected_features - len(tomorrow_features))
        elif len(tomorrow_features) > expected_features:
            tomorrow_features = tomorrow_features[:expected_features]

        tomorrow_features = np.array([tomorrow_features])

        # -----------------------------
        # Scale & prepare LSTM input
        # -----------------------------
        X_tomorrow_scaled = scaler_X.transform(tomorrow_features)
        recent_days = X_scaled[-(SEQ_LEN-1):]
        seq_input = np.vstack([recent_days, X_tomorrow_scaled])
        seq_input = seq_input.reshape(1, SEQ_LEN, X_scaled.shape[1])

        # -----------------------------
        # Predict energy
        # -----------------------------
        pred_scaled = model.predict(seq_input)
        pred_kWh = scaler_y.inverse_transform(pred_scaled)[0, 0]

        return jsonify({
            "city": city,
            "predicted_energy_kWh": round(float(pred_kWh), 2)
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"})

# -----------------------------
# 5️⃣ Run Flask (production ready)
# -----------------------------
if __name__ == "__main__":
    # debug=False for production
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7000)), debug=False)
