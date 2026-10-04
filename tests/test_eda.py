from pathlib import Path

import pandas as pd
import pytest

from forest_cover.eda import (
    COLUMN_NAMES,
    EXPECTED_COLUMN_COUNT,
    TARGET_COLUMN,
    dataset_summary,
    load_forest_cover,
)


def write_gzip_csv(path: Path, column_count: int = EXPECTED_COLUMN_COUNT) -> None:
    rows = [list(range(column_count)), list(range(column_count, column_count * 2))]
    pd.DataFrame(rows).to_csv(path, compression="gzip", header=False, index=False)


def test_expected_schema_has_55_columns_and_target() -> None:
    assert len(COLUMN_NAMES) == EXPECTED_COLUMN_COUNT
    assert COLUMN_NAMES[-1] == TARGET_COLUMN


def test_loader_assigns_documented_column_names(tmp_path: Path) -> None:
    data_path = tmp_path / "forest-cover.csv.gz"
    write_gzip_csv(data_path)

    loaded = load_forest_cover(data_path)

    assert tuple(loaded.columns) == COLUMN_NAMES
    assert loaded.shape == (2, EXPECTED_COLUMN_COUNT)


def test_dataset_summary_reports_quality_counts() -> None:
    first_row = {column: 0 for column in COLUMN_NAMES}
    first_row[TARGET_COLUMN] = 1
    second_row = first_row | {"Elevation": pd.NA, TARGET_COLUMN: 2}
    data = pd.DataFrame([first_row, second_row, second_row])

    summary = dataset_summary(data)

    assert summary == {
        "row_count": 3,
        "column_count": EXPECTED_COLUMN_COUNT,
        "total_missing_values": 2,
        "duplicate_row_count": 1,
        "target_class_count": 2,
    }


def test_loader_rejects_invalid_column_count(tmp_path: Path) -> None:
    data_path = tmp_path / "invalid-forest-cover.csv.gz"
    write_gzip_csv(data_path, column_count=EXPECTED_COLUMN_COUNT - 1)

    with pytest.raises(ValueError, match="expected 55 columns, found 54"):
        load_forest_cover(data_path)
