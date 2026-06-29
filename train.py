from sklearn.model_selection import train_test_split

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.data_cleaning import clean_data
from src.data_loader import load_data
from src.evaluation import evaluate_model
from src.model import build_model
from src.save_model import save_metrics, save_model


def main():
    raw_data = load_data()
    data = clean_data(raw_data)

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    save_model(model)
    save_metrics(metrics)

    print("Model trained and saved successfully.")
    print(metrics)


if __name__ == "__main__":
    main()
