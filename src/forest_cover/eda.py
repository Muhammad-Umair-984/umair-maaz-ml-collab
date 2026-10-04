"""Reusable loading and summary helpers for Forest CoverType EDA."""

from pathlib import Path

import pandas as pd

from forest_cover.config import RAW_DATA_DIR

QUANTITATIVE_COLUMNS = (
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
)
WILDERNESS_AREA_COLUMNS = tuple(f"Wilderness_Area{index}" for index in range(1, 5))
SOIL_TYPE_COLUMNS = tuple(f"Soil_Type{index}" for index in range(1, 41))
TARGET_COLUMN = "Cover_Type"
COLUMN_NAMES = (
    *QUANTITATIVE_COLUMNS,
    *WILDERNESS_AREA_COLUMNS,
    *SOIL_TYPE_COLUMNS,
    TARGET_COLUMN,
)
EXPECTED_COLUMN_COUNT = 55
EXPECTED_ROW_COUNT = 581_012

COVER_TYPE_LABELS = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}


def load_forest_cover(path: Path = RAW_DATA_DIR / "covtype.data.gz") -> pd.DataFrame:
    """Load the headerless Forest CoverType data and validate its 55-column schema."""
    data = pd.read_csv(Path(path), compression="infer", header=None)
    actual_column_count = data.shape[1]

    if actual_column_count != EXPECTED_COLUMN_COUNT:
        raise ValueError(
            "Invalid Forest CoverType schema: "
            f"expected {EXPECTED_COLUMN_COUNT} columns, found {actual_column_count}."
        )

    data.columns = COLUMN_NAMES
    return data


def dataset_summary(data: pd.DataFrame) -> dict[str, int]:
    """Return compact dataset size, completeness, duplication, and target information."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain the target column {TARGET_COLUMN!r}.")

    return {
        "row_count": len(data),
        "column_count": data.shape[1],
        "total_missing_values": int(data.isna().sum().sum()),
        "duplicate_row_count": int(data.duplicated().sum()),
        "target_class_count": int(data[TARGET_COLUMN].nunique(dropna=True)),
    }
