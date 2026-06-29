from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Data" / "archive" / "customer_churn_dataset-testing-master.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

TARGET_COLUMN = "Churn"
ID_COLUMN = "CustomerID"

FEATURE_COLUMNS = [
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
]

NUMERIC_FEATURES = [
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
]

CATEGORICAL_FEATURES = [
    "Subscription Type",
    "Contract Length",
]
