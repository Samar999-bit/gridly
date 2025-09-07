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
CORS(app)

# -----------------------------
# 2️⃣ File paths (relative)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "lstm_weather_model.h5")
SCALER_X_FILE = os.path.join(BASE_DIR, "scaler_X.pkl")
SCALER_Y_FILE = os.path.join(BASE_DIR, "scaler_y.pkl")
X_SCALED_FILE = os.path.join(BASE_DIR, "X_scaled.pkl")

SEQ_LEN = 7

# -----------------------------
# 3️⃣ Lazy-load artifacts
# -----------------------------
model = None
scaler_X = None
scaler_y = None
X_scaled = None

def load_artifacts():
    """Load model & scalers only once (lazy loading)."""
    global model, scaler_X, scaler_y, X_scaled
    if model is None:
        if not os.path.isfile(MODEL_FILE):
            raise FileNotFoundError(f"Missing model file: {MODEL_FILE}")
        if not os.path.isfile(SCALER_X_FILE):
            raise FileNotFoundError(f"Missing scaler_X file: {SCALER_X_FILE}")
        if not os.path.isfile(SCALER_Y_FILE):
            raise FileNotFoundError(f"Missing scaler_y file: {SCALER_Y_FILE}")
        if not os.path.isfile(X_SCALED_FILE):
            raise FileNotFoundError(f"Missing X_scaled file: {X_SCALED_FILE}")

        print("Loading model and scalers...")
        model = load_model(MODEL_FILE, compile=False)
        scaler_X = joblib.load(SCALER_X_FILE)
        scaler_y = joblib.load(SCALER_Y_FILE)
        X_scaled = joblib.load(X_SCALED_FILE)
        print("✅ Model & scalers loaded successfully")

# -----------------------------
# 4️⃣ Flask routes
# -----------------------------
@app.route("/")
def home():
    return "✅ Flask LSTM API is running!"

@app.route("/predict", methods=["GET"])
def predict_energy():
    try:
        # ensure model is loaded
        load_artifacts()

        city = request.args.get("city", "Patiala,IN")
        API_KEY = os.environ.get("OPENWEATHER_API_KEY")
        if not API_KEY:
            return jsonify({"error": "OpenWeatherMap API key not set in environment variables."})

        # --- Fetch tomorrow’s weather ---
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

        # --- Aggregate features ---
        tomorrow_features = [
            float(np.mean(temps)),
            float(np.mean(hums)),
            float(np.mean(winds)),
            float(np.sum(precs))
        ]

        expected_features = X_scaled.shape[1]
        if len(tomorrow_features) < expected_features:
            tomorrow_features += [0] * (expected_features - len(tomorrow_features))
        elif len(tomorrow_features) > expected_features:
            tomorrow_features = tomorrow_features[:expected_features]

        tomorrow_features = np.array([tomorrow_features])
        X_tomorrow_scaled = scaler_X.transform(tomorrow_features)

        # --- Build LSTM input ---
        recent_days = X_scaled[-(SEQ_LEN - 1):]
        seq_input = np.vstack([recent_days, X_tomorrow_scaled])
        seq_input = seq_input.reshape(1, SEQ_LEN, X_scaled.shape[1])

        # --- Predict ---
        pred_scaled = model.predict(seq_input)
        pred_kWh = scaler_y.inverse_transform(pred_scaled)[0, 0]

        return jsonify({
            "city": city,
            "predicted_energy_kWh": round(float(pred_kWh), 2)
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"})

# -----------------------------
# 5️⃣ Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7000)), debug=False)

