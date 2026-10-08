"""Chronological splits, training-only preprocessing, penalty selection and metrics."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from hf_factor_zoo import forecasting
from tests.conftest import session_timestamps


@pytest.fixture(scope="module")
def history_index() -> pd.DatetimeIndex:
    """Regular-session timestamps from 2000 through 2004."""
    days = pd.bdate_range("2000-01-03", "2004-12-31")
    return session_timestamps(days, include_overnight=False)


def test_year_split_is_chronological_and_matches_reported_2004_cutoff(history_index):
    splits, metadata = forecasting.make_year_split(history_index, test_year=2004)

    assert splits["train"].max() < splits["valid"].min() < splits["test"].min()
    assert set(splits["test"].year) == {2004}
    assert set(splits["valid"].year) == {2003}
    # 80% of 1,461 calendar days (2000-2003): training ends 2003-03-13 (README).
    assert splits["train"].max().date() == date(2003, 3, 13)
    assert splits["valid"].min().date() == date(2003, 3, 14)
    # Every history row belongs to exactly one of train and valid.
    history = history_index[history_index.year < 2004]
    assert len(splits["train"]) + len(splits["valid"]) == len(history)
    assert metadata["test"]["row_count"] == len(splits["test"])


def test_scalers_use_training_rows_only():
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.normal(loc=2.0, size=(200, 3)))
    y_train = pd.Series(rng.normal(loc=0.5, size=200))

    x_scaler, y_scaler = forecasting.fit_preprocessing(X_train, y_train)

    np.testing.assert_allclose(x_scaler.mean_, X_train.mean().to_numpy())
    np.testing.assert_allclose(y_scaler.mean_, [y_train.mean()])
    # Population standard deviation (ddof=0), as documented.
    np.testing.assert_allclose(y_scaler.scale_, [y_train.std(ddof=0)])


def test_ridge_selection_returns_the_minimum_validation_loss_model():
    rng = np.random.default_rng(0)
    beta = rng.normal(size=5)
    X = rng.normal(size=(400, 5))
    y = X @ beta + rng.normal(scale=3.0, size=400)
    X_train, X_valid = X[:300], X[300:]
    y_train, y_valid = pd.Series(y[:300]), pd.Series(y[300:])

    x_scaler, y_scaler = forecasting.fit_preprocessing(pd.DataFrame(X_train), y_train)
    X_train_scaled = x_scaler.transform(pd.DataFrame(X_train))
    X_valid_scaled = x_scaler.transform(pd.DataFrame(X_valid))
    y_train_scaled = y_scaler.transform(y_train.to_frame()).ravel()

    model, table = forecasting.select_ridge_penalty(
        X_train_scaled, y_train_scaled, X_valid_scaled, y_valid, y_scaler
    )

    predictions = y_scaler.inverse_transform(model.predict(X_valid_scaled).reshape(-1, 1)).ravel()
    selected_mse = np.mean((y_valid.to_numpy() - predictions) ** 2)
    np.testing.assert_allclose(selected_mse, table["validation_mse"].min(), rtol=1e-12)
    assert len(table) == 100


def test_compute_metrics_uses_uncentered_r2_against_zero():
    predictions = pd.DataFrame(
        {
            "actual_return": [0.02, -0.01, 0.03],
            "ridge_prediction": [0.01, -0.01, 0.01],
            "zero_prediction": 0.0,
            "training_mean_prediction": 0.001,
        }
    )

    metrics = forecasting.compute_metrics(predictions, model_name="ridge").set_index("model")

    sse = (0.01**2) + 0 + (0.02**2)
    sum_y2 = 0.02**2 + 0.01**2 + 0.03**2
    np.testing.assert_allclose(metrics.at["ridge_prediction", "r2_oos"], 1 - sse / sum_y2)
    assert metrics.at["zero_prediction", "r2_oos"] == 0
    np.testing.assert_allclose(metrics.at["ridge_prediction", "mse"], sse / 3)
