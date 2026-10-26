from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Allow requests from Netlify
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgboost_quantity_model.pkl"

model = joblib.load(MODEL_PATH)

FEATURES = [
    "Branch",
    "City",
    "Customer type",
    "Gender",
    "Product line",
    "Unit price",
    "Payment",
    "Month",
    "Day",
    "DayOfWeek",
    "Quarter",
    "Week"
]

NUMERIC_FEATURES = [
    "Unit price",
    "Month",
    "Day",
    "DayOfWeek",
    "Quarter",
    "Week"
]


@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "message": "XGBoost Quantity Prediction API"
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy"
    })


@app.route("/features")
def features():
    return jsonify({
        "expected_features": FEATURES
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Please send valid JSON."
            }), 400

        missing = [
            feature
            for feature in FEATURES
            if feature not in data
        ]

        if missing:
            return jsonify({
                "error": "Missing required features.",
                "missing_features": missing
            }), 400

        row = {
            feature: data[feature]
            for feature in FEATURES
        }

        # Convert numeric fields
        for feature in NUMERIC_FEATURES:
            row[feature] = float(row[feature])

        input_df = pd.DataFrame(
            [row],
            columns=FEATURES
        )

        print("\n========== REQUEST RECEIVED ==========")
        print(input_df)
        print("\nCOLUMN TYPES")
        print(input_df.dtypes)

        prediction = model.predict(input_df)

        predicted_quantity = float(prediction[0])

        return jsonify({
            "predicted_quantity": round(
                predicted_quantity,
                2
            ),
            "status": "success"
        })

    except Exception as e:

        print("\n========== ERROR ==========")
        print(str(e))

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )