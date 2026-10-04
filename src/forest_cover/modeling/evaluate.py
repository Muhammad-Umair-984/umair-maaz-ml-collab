"""Evaluate the trained Forest CoverType pipeline on held-out data."""

import json
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from forest_cover.config import MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT
from forest_cover.eda import TARGET_COLUMN

TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv.gz"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = PROJ_ROOT / "metrics.json"


def calculate_metrics(target: pd.Series, predictions: pd.Series) -> dict[str, float]:
    """Calculate finite multiclass metrics suitable for DVC tracking."""
    metrics = {
        "accuracy": float(accuracy_score(target, predictions)),
        "f1_macro": float(f1_score(target, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(target, predictions, average="weighted", zero_division=0)),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Evaluation produced a non-finite metric.")
    return metrics


def evaluate_model(test_data: pd.DataFrame, model: object) -> dict[str, float]:
    """Evaluate a fitted model using only the held-out test artifact."""
    if TARGET_COLUMN not in test_data.columns:
        raise ValueError(f"Test data must contain target column {TARGET_COLUMN!r}.")

    features = test_data.drop(columns=TARGET_COLUMN)
    target = test_data[TARGET_COLUMN]
    predictions = model.predict(features)
    return calculate_metrics(target, predictions)


def write_metrics(metrics: dict[str, float], path: Path = METRICS_PATH) -> None:
    """Write deterministic DVC-readable JSON metrics."""
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    """Run evaluation against the prepared test artifact."""
    test_data = pd.read_csv(TEST_DATA_PATH, compression="gzip")
    model = joblib.load(MODEL_PATH)
    metrics = evaluate_model(test_data, model)
    write_metrics(metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
