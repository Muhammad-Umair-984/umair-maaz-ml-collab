"""Create deterministic raw train and test artifacts for the DVC pipeline."""

from pathlib import Path

from dvc.api import params_show
import pandas as pd
from sklearn.model_selection import train_test_split

from forest_cover.config import PROCESSED_DATA_DIR
from forest_cover.eda import COLUMN_NAMES, TARGET_COLUMN, load_forest_cover

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv.gz"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv.gz"
EXPECTED_FEATURE_COUNT = len(COLUMN_NAMES) - 1


def split_dataset(
    data: pd.DataFrame, *, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split raw features and target deterministically while preserving class proportions."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain target column {TARGET_COLUMN!r}.")

    features = data.drop(columns=TARGET_COLUMN)
    target = data[TARGET_COLUMN]
    if features.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} predictors, found {features.shape[1]}."
        )
    if TARGET_COLUMN in features.columns:
        raise ValueError("Target leakage detected in the feature columns.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=seed,
        stratify=target,
    )
    train_data = x_train.assign(**{TARGET_COLUMN: y_train}).reset_index(drop=True)
    test_data = x_test.assign(**{TARGET_COLUMN: y_test}).reset_index(drop=True)

    if train_data.empty or test_data.empty:
        raise ValueError("The train/test split produced an empty artifact.")
    return train_data, test_data


def write_prepared_data(data: pd.DataFrame, path: Path) -> None:
    """Write a deterministic gzip-compressed CSV artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )


def main() -> None:
    """Run the prepare stage using DVC-tracked parameters."""
    params = params_show()
    seed = int(params["seed"])
    test_size = float(params["prepare"]["test_size"])

    raw_data = load_forest_cover()
    train_data, test_data = split_dataset(raw_data, test_size=test_size, seed=seed)
    write_prepared_data(train_data, TRAIN_DATA_PATH)
    write_prepared_data(test_data, TEST_DATA_PATH)

    print(f"Prepared {len(train_data):,} training rows and {len(test_data):,} test rows.")


if __name__ == "__main__":
    main()
