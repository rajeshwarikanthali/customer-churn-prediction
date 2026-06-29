from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


def build_model():
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=160,
                    max_depth=10,
                    min_samples_leaf=4,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
