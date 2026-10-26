
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import pandas as pd
import joblib
import os

app = Flask(__name__)

# For initial testing. Restrict this to your Netlify domain
# after the website has been deployed.
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgboost_quantity_model.pkl"

model = joblib.load(MODEL_PATH)

FEATURES = [
    "Branch", "City", "Customer type", "Gender",
    "Product line", "Unit price", "Payment",
    "Month", "Day", "DayOfWeek", "Quarter", "Week"
]

NUMERIC_FEATURES = [
    "Unit price", "Month", "Day",
    "DayOfWeek", "Quarter", "Week"
]


@app.get("/")
def home():
    return jsonify({
        "status": "online",
        "message": "XGBoost Quantity Prediction API"
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Please send valid JSON."}), 400

    missing = [col for col in FEATURES if col not in data]
    if missing:
        return jsonify({
            "error": "Missing required features.",
            "missing_features": missing
        }), 400

    try:
        row = {col: data[col] for col in FEATURES}

        for col in NUMERIC_FEATURES:
            row[col] = float(row[col])

        if not all(
            pd.notna(row[col]) for col in NUMERIC_FEATURES
        ):
            return jsonify({
                "error": "Numeric features must be valid numbers."
            }), 400

        input_df = pd.DataFrame([row], columns=FEATURES)
        prediction = float(model.predict(input_df)[0])

        return jsonify({
            "predicted_quantity": round(prediction, 2)
        })

    except (TypeError, ValueError, OverflowError):
        return jsonify({
            "error": "Check that all numeric inputs are valid."
        }), 400

    except Exception:
        app.logger.exception("Prediction error")
        return jsonify({
            "error": "The model could not generate a prediction."
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
