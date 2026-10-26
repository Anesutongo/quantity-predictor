from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import pandas as pd
import joblib
import os

# ====================================================
# APP SETUP
# ====================================================

app = Flask(__name__)

# Allow requests from Netlify
CORS(app)

# ====================================================
# LOAD MODEL
# ====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "xgboost_quantity_model.pkl"

model = joblib.load(MODEL_PATH)

# ====================================================
# MODEL FEATURES
# ====================================================

FEATURES = [
    "Branch",
    "Product line",
    "Unit price",
    "Sales",
    "Month",
    "Day",
    "DayOfWeek",
    "Quarter",
    "Week",
    "Weekend",
    "Lag_1",
    "Lag_3",
    "Lag_7",
    "Rolling_Mean_3",
    "Rolling_Mean_7"
]

NUMERIC_FEATURES = [
    "Unit price",
    "Sales",
    "Month",
    "Day",
    "DayOfWeek",
    "Quarter",
    "Week",
    "Weekend",
    "Lag_1",
    "Lag_3",
    "Lag_7",
    "Rolling_Mean_3",
    "Rolling_Mean_7"
]

# ====================================================
# HOME
# ====================================================

@app.route("/")
def home():

    return jsonify({
        "status": "online",
        "message": "Retail Demand Forecasting API",
        "model": "XGBoost"
    })

# ====================================================
# HEALTH CHECK
# ====================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "healthy"
    })

# ====================================================
# SHOW EXPECTED FEATURES
# ====================================================

@app.route("/features")
def features():

    return jsonify({
        "expected_features": FEATURES
    })

# ====================================================
# PREDICT
# ====================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True)

        if not isinstance(data, dict):

            return jsonify({
                "error": "Please send valid JSON."
            }), 400

        missing_features = [
            feature
            for feature in FEATURES
            if feature not in data
        ]

        if missing_features:

            return jsonify({
                "error": "Missing required features.",
                "missing_features": missing_features
            }), 400

        row = {}

        for feature in FEATURES:

            value = data[feature]

            if feature in NUMERIC_FEATURES:
                value = float(value)

            row[feature] = value

        input_df = pd.DataFrame(
            [row],
            columns=FEATURES
        )

        print("\n========== INPUT DATA ==========")
        print(input_df)

        prediction = model.predict(input_df)

        predicted_quantity = float(
            prediction[0]
        )

        stockout_risk = (
            "YES"
            if predicted_quantity > 20
            else "NO"
        )

        return jsonify({
            "status": "success",
            "predicted_quantity": round(
                predicted_quantity,
                2
            ),
            "stockout_risk": stockout_risk
        })

    except Exception as e:

        print("\n========== ERROR ==========")
        print(str(e))

        return jsonify({
            "error": str(e)
        }), 500

# ====================================================
# RUN APP
# ====================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                10000
            )
        )
    )