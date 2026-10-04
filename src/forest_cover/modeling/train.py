"""Fit and persist the deterministic Forest CoverType baseline pipeline."""

from collections.abc import Mapping
import json
from pathlib import Path
import subprocess
from typing import Any

from dvc.api import params_show
import joblib
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from forest_cover.config import MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT
from forest_cover.eda import TARGET_COLUMN

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv.gz"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "training_metadata.json"


def build_model(*, seed: int, model_name: str, parameters: Mapping[str, Any]) -> Pipeline:
    """Build a train-only preprocessing and classification pipeline."""
    if model_name != "sgd_classifier":
        raise ValueError(f"Unsupported model {model_name!r}; expected 'sgd_classifier'.")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SGDClassifier(random_state=seed, **dict(parameters)),
            ),
        ]
    )


def fit_model(
    train_data: pd.DataFrame,
    *,
    seed: int,
    model_name: str,
    parameters: Mapping[str, Any],
) -> Pipeline:
    """Fit the complete sklearn pipeline using only the prepared training artifact."""
    if TARGET_COLUMN not in train_data.columns:
        raise ValueError(f"Training data must contain target column {TARGET_COLUMN!r}.")

    features = train_data.drop(columns=TARGET_COLUMN)
    target = train_data[TARGET_COLUMN]
    model = build_model(seed=seed, model_name=model_name, parameters=parameters)
    model.fit(features, target)
    return model


def get_git_commit() -> str:
    """Return the current Git commit, with a stable fallback outside a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJ_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def git_worktree_is_dirty() -> bool | None:
    """Report whether tracked or untracked workspace changes existed during training."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJ_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def write_training_metadata(metadata: dict[str, Any], path: Path = METADATA_PATH) -> None:
    """Write stable machine-readable training metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    """Run the training stage from the prepared training artifact."""
    params = params_show()
    seed = int(params["seed"])
    train_params = params["train"]
    model_name = str(train_params["model"])
    model_parameters = dict(train_params["parameters"])

    train_data = pd.read_csv(TRAIN_DATA_PATH, compression="gzip")
    model = fit_model(
        train_data,
        seed=seed,
        model_name=model_name,
        parameters=model_parameters,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metadata = {
        "feature_count": train_data.shape[1] - 1,
        "git_commit": get_git_commit(),
        "git_worktree_dirty": git_worktree_is_dirty(),
        "model": model_name,
        "parameters": model_parameters,
        "seed": seed,
        "target": TARGET_COLUMN,
        "training_rows": len(train_data),
    }
    write_training_metadata(metadata)
    print(f"Trained {model_name} on {len(train_data):,} rows and saved {MODEL_PATH}.")


if __name__ == "__main__":
    main()
