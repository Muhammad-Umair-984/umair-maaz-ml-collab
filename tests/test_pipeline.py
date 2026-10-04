import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from forest_cover.eda import COLUMN_NAMES, TARGET_COLUMN
from forest_cover.modeling.evaluate import calculate_metrics, evaluate_model
from forest_cover.modeling.prepare import split_dataset
from forest_cover.modeling.train import fit_model


def make_sample_data(rows_per_class: int = 8) -> pd.DataFrame:
    feature_columns = COLUMN_NAMES[:-1]
    rows = []
    for target_class in range(1, 8):
        for repetition in range(rows_per_class):
            row = {
                column: target_class * 10 + repetition + feature_index
                for feature_index, column in enumerate(feature_columns)
            }
            row[TARGET_COLUMN] = target_class
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMN_NAMES)


def test_split_is_deterministic_stratified_and_target_safe() -> None:
    data = make_sample_data()

    first_train, first_test = split_dataset(data, test_size=0.25, seed=42)
    second_train, second_test = split_dataset(data, test_size=0.25, seed=42)

    assert_frame_equal(first_train, second_train)
    assert_frame_equal(first_test, second_test)
    assert TARGET_COLUMN not in first_train.drop(columns=TARGET_COLUMN).columns
    assert set(first_train[TARGET_COLUMN]) == set(range(1, 8))
    assert set(first_test[TARGET_COLUMN]) == set(range(1, 8))


def test_scaler_is_fitted_only_from_training_rows() -> None:
    train_data = make_sample_data()
    model = fit_model(
        train_data,
        seed=42,
        model_name="sgd_classifier",
        parameters={"loss": "log_loss", "max_iter": 100, "tol": 0.001},
    )

    expected_means = train_data.drop(columns=TARGET_COLUMN).mean().to_numpy()
    np.testing.assert_allclose(model.named_steps["scaler"].mean_, expected_means)


def test_evaluation_returns_finite_expected_metrics() -> None:
    train_data = make_sample_data()
    model = fit_model(
        train_data,
        seed=42,
        model_name="sgd_classifier",
        parameters={"loss": "log_loss", "max_iter": 100, "tol": 0.001},
    )

    metrics = evaluate_model(train_data, model)

    assert set(metrics) == {"accuracy", "f1_macro", "f1_weighted"}
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_metric_calculation_is_exact_for_perfect_predictions() -> None:
    target = pd.Series([1, 2, 3, 4, 5, 6, 7])

    metrics = calculate_metrics(target, target.copy())

    assert metrics == {"accuracy": 1.0, "f1_macro": 1.0, "f1_weighted": 1.0}
