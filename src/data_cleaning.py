import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN


def clean_data(df):
    df = df.copy()
    keep_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df[keep_columns]

    for column in ["Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Subscription Type"] = df["Subscription Type"].astype(str).str.strip()
    df["Contract Length"] = df["Contract Length"].astype(str).str.strip()
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    df = df.dropna()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df
