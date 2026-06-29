from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
DATA_PATH = PROJECT_ROOT / "Data" / "archive" / "customer_churn_dataset-testing-master.csv"

FEATURE_COLUMNS = [
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
]

NUMERIC_COLUMNS = [
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
]


def load_model():
    return joblib.load(MODEL_PATH)


def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    return pd.read_json(METRICS_PATH, typ="series").to_dict()


def load_reference_data():
    return pd.read_csv(DATA_PATH)


def prepare_features(data):
    features = pd.DataFrame(data).copy()
    features = features[FEATURE_COLUMNS]

    for column in NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    features["Subscription Type"] = features["Subscription Type"].astype(str).str.strip()
    features["Contract Length"] = features["Contract Length"].astype(str).str.strip()
    return features


def predict_churn(model, data):
    features = prepare_features(data)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return predictions, probabilities


def risk_band(probability):
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Watch"
    return "Low"
