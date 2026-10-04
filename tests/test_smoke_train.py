"""Smoke test for the model training workflow."""

import pandas as pd

from forest_cover.modeling.train import fit_model


def test_smoke_training() -> None:
    """Verify that the training pipeline can fit a small dataset."""
    train_data = pd.DataFrame(
        {
            "Elevation": [2500, 2600, 2700, 2800, 2900, 3000],
            "Aspect": [10, 20, 30, 40, 50, 60],
            "Cover_Type": [1, 1, 2, 2, 1, 2],
        }
    )

    model = fit_model(
        train_data,
        seed=42,
        model_name="sgd_classifier",
        parameters={
            "loss": "log_loss",
            "max_iter": 100,
            "tol": 0.001,
        },
    )

    predictions = model.predict(train_data.drop(columns="Cover_Type"))

    assert len(predictions) == len(train_data)
