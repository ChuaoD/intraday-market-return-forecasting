"""Annual expanding-window forecasts of 15-minute market returns.

Fit Ridge, Lasso or Elastic Net with training-only scaling and chronological
validation. Predictions and losses use decimal simple-return units.
"""

import argparse
import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
    enet_path,
    lasso_path,
)
from sklearn.preprocessing import StandardScaler

from hf_factor_zoo.data_processing import find_project_root

START_YEAR = 2000
TEST_YEAR = 2004
TRAIN_FRACTION = 0.8
TARGET_COLUMN = "target__ff__mkt"
FEATURE_PREFIXES = ("lag15__", "avg60__", "prevday__")


def load_forecast_panel(
    path: Path,
    start_year: int = START_YEAR,
    test_year: int = TEST_YEAR,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load predictors X, target y and timing metadata for start_year to test_year.

    All three share the target_timestamp index. Metadata holds
    forecast_origin_timestamp, trading_date and slot_id.
    """

    start = pd.Timestamp(year=start_year, month=1, day=1)
    end = pd.Timestamp(year=test_year + 1, month=1, day=1)
    # Read only the rows needed for these years.
    panel = pd.read_parquet(
        path,
        filters=[
            ("target_timestamp", ">=", start),
            ("target_timestamp", "<", end),
        ],
    )
    # Keep the 816 predictors in a fixed, sorted column order.
    feature_columns = sorted(
        column for column in panel.columns if column.startswith(FEATURE_PREFIXES)
    )
    X = panel.loc[:, feature_columns]
    assert X.index.is_monotonic_increasing
    assert X.index.is_unique
    assert X.shape[1] == 816
    assert np.isfinite(X.to_numpy()).all()
    # Targets stay in return units until training-only scaling.
    y = panel[TARGET_COLUMN]
    assert X.index.equals(y.index)
    assert np.isfinite(y.to_numpy()).all()
    metadata = panel.loc[:, ["forecast_origin_timestamp", "trading_date", "slot_id"]]
    assert X.index.equals(metadata.index)
    # Every forecast must be made strictly before its target interval ends.
    origins = metadata["forecast_origin_timestamp"]
    assert origins.notna().all()
    assert (origins < metadata.index).all()

    return X, y, metadata


def make_year_split(
    index: pd.DatetimeIndex,
    test_year: int = TEST_YEAR,
    start_year: int = START_YEAR,
    train_fraction: float = TRAIN_FRACTION,
) -> tuple[dict[str, pd.DatetimeIndex], dict[str, object]]:
    """Split history by calendar days, followed by a separate test year.

    The training cutoff uses floor(0.8 * inclusive historical calendar days),
    not 80% of observations. Metadata record both configured and observed bounds.
    """
    history_start = pd.Timestamp(year=start_year, month=1, day=1)
    history_end = pd.Timestamp(year=test_year, month=1, day=1) - pd.Timedelta(seconds=1)
    calendar_days = (history_end - history_start).days + 1
    train_end = history_start + pd.Timedelta(days=math.floor(calendar_days * train_fraction))
    valid_start = train_end + pd.Timedelta(seconds=1)
    test_start = pd.Timestamp(year=test_year, month=1, day=1)
    test_stop = pd.Timestamp(year=test_year + 1, month=1, day=1)
    test_index = index[(index >= test_start) & (index < test_stop)]
    split_indexes = {
        "train": index[(index >= history_start) & (index <= train_end)],
        "valid": index[(index >= valid_start) & (index <= history_end)],
        "test": test_index,
    }

    train = split_indexes["train"]
    valid = split_indexes["valid"]
    test = split_indexes["test"]

    # All training observations precede validation, which precedes testing.
    assert train.max() < valid.min()
    assert valid.max() < test.min()

    configured_cutoffs = {
        "train": (history_start, train_end),
        "valid": (valid_start, history_end),
        "test": (test_start, test_stop),
    }

    split_metadata = {}

    for name, timestamps in split_indexes.items():
        start, end = configured_cutoffs[name]

        split_metadata[name] = {
            "configured_start": start.isoformat(),
            "configured_end": end.isoformat(),
            "end_inclusive": name != "test",
            "actual_first": timestamps.min().isoformat(),
            "actual_last": timestamps.max().isoformat(),
            "row_count": len(timestamps),
        }

    return split_indexes, split_metadata


def fit_preprocessing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[StandardScaler, StandardScaler]:
    """Fit feature and target scalers using training observations only."""
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(X_train)
    y_scaler.fit(y_train.to_frame())

    y_train_scaled = y_scaler.transform(y_train.to_frame())
    y_train_recovered = y_scaler.inverse_transform(y_train_scaled).ravel()

    assert np.allclose(y_train_recovered, y_train.to_numpy())

    return x_scaler, y_scaler


def select_ridge_penalty(
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    X_valid_scaled: np.ndarray,
    y_valid: pd.Series,
    y_scaler: StandardScaler,
) -> tuple[Ridge, pd.DataFrame]:
    """Return the validation-selected Ridge fit and the candidate loss table.

    The normalized penalty is lambda = alpha / n_train. Validation predictions
    are inverse-scaled before computing MSE; exact ties favor smaller lambda.
    The selected model remains fitted on training rows only.
    """
    y_valid = y_valid.to_numpy()
    lambdas = np.exp(np.linspace(-1, 8, 100))
    records = []
    best_model = None
    best_mse = np.inf
    for lambda_value in lambdas:
        alpha = len(y_train_scaled) * lambda_value
        # fit_intercept=False disables fitting an additional constant term.
        model = Ridge(
            alpha=alpha,
            fit_intercept=False,
            solver="cholesky",
        )
        model.fit(X_train_scaled, y_train_scaled)
        valid_predictions_scaled = model.predict(X_valid_scaled)
        valid_predictions = y_scaler.inverse_transform(
            valid_predictions_scaled.reshape(-1, 1)
        ).ravel()
        validation_mse = np.mean((y_valid - valid_predictions) ** 2)
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_model = model

        records.append(
            {
                "lambda": lambda_value,
                "alpha": alpha,
                "validation_mse": validation_mse,
            }
        )
    validation_results = pd.DataFrame(records)

    return best_model, validation_results


def select_lasso_penalty(
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    X_valid_scaled: np.ndarray,
    y_valid: pd.Series,
    y_scaler: StandardScaler,
) -> tuple[Lasso, pd.DataFrame]:
    """Select alpha using validation MSE; return a training-only Lasso fit."""

    # Column-major storage helps the coordinate-descent solver.
    X_train = np.asfortranarray(X_train_scaled, dtype=float)
    y_train = np.asarray(y_train_scaled, dtype=float).ravel()

    alpha_grid = np.exp(np.linspace(-6, -1, 100))

    # Fit coefficients for every candidate using training observations only.
    # Stop if the solver reports nonconvergence.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)

        alphas, coefficients, dual_gaps, iteration_counts = lasso_path(
            X_train,
            y_train,
            alphas=alpha_grid,
            max_iter=5000,
            tol=1e-4,
            return_n_iter=True,
        )
    # Each path column is the training coefficient vector for one penalty.
    records = []
    y_valid = y_valid.to_numpy()

    for j, alpha in enumerate(alphas):
        beta = coefficients[:, j]

        predictions_scaled = X_valid_scaled @ beta
        predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()

        validation_mse = np.mean((y_valid - predictions) ** 2)

        records.append(
            {
                "alpha": float(alpha),
                "validation_mse": float(validation_mse),
                "n_nonzero": int(np.count_nonzero(beta)),
                "dual_gap": float(dual_gaps[j]),
                "n_iter": int(iteration_counts[j]),
            }
        )

    validation_results = pd.DataFrame(records)

    if not np.isfinite(validation_results["validation_mse"]).all():
        raise ValueError("Validation losses contain invalid values.")

    # Select the alpha with minimum validation MSE.
    # Smaller alpha breaks an exact MSE tie.
    best_row = validation_results.sort_values(["validation_mse", "alpha"]).iloc[0]

    selected_alpha = float(best_row["alpha"])

    model = Lasso(
        alpha=selected_alpha,
        fit_intercept=False,
        max_iter=10000,
        tol=1e-4,
        warm_start=True,
    )

    # Start the final Lasso fit from the minimum-norm least-squares solution.
    n_train = len(y_train)
    model.coef_ = np.linalg.pinv(X_train.T @ X_train / n_train) @ (X_train.T @ y_train / n_train)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X_train, y_train)

    return model, validation_results


def select_elastic_net_penalty(
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    X_valid_scaled: np.ndarray,
    y_valid: pd.Series,
    y_scaler: StandardScaler,
) -> tuple[ElasticNet, pd.DataFrame]:
    """Select alpha and l1_ratio (rho) using validation MSE."""

    X_train = np.asfortranarray(X_train_scaled, dtype=float)
    y_train = np.asarray(y_train_scaled, dtype=float).ravel()
    y_valid = y_valid.to_numpy()

    alpha_grid = np.exp(np.linspace(-6, 1, 50))
    ratio_grid = np.linspace(0.05, 0.95, 50)

    records = []
    for l1_ratio in ratio_grid:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)

            alphas, coefficients, dual_gaps, iteration_counts = enet_path(
                X_train,
                y_train,
                l1_ratio=l1_ratio,
                alphas=alpha_grid,
                max_iter=20_000,
                tol=1e-4,
                return_n_iter=True,
            )
            for j, alpha in enumerate(alphas):
                beta = coefficients[:, j]

                predictions_scaled = X_valid_scaled @ beta
                predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()

                validation_mse = np.mean((y_valid - predictions) ** 2)

                records.append(
                    {
                        "l1_ratio": float(l1_ratio),
                        "alpha": float(alpha),
                        "validation_mse": float(validation_mse),
                        "n_nonzero": int(np.count_nonzero(beta)),
                        "dual_gap": float(dual_gaps[j]),
                        "n_iter": int(iteration_counts[j]),
                    }
                )
    validation_results = pd.DataFrame(records)

    if not np.isfinite(validation_results["validation_mse"]).all():
        raise ValueError("Validation losses contain invalid values.")

    # Select the alpha with minimum validation MSE.
    # Smaller alpha breaks an exact MSE tie.
    best_row = validation_results.sort_values(["validation_mse", "alpha", "l1_ratio"]).iloc[0]

    selected_alpha = float(best_row["alpha"])
    selected_ratio = float(best_row["l1_ratio"])

    model = ElasticNet(
        alpha=selected_alpha,
        l1_ratio=selected_ratio,
        fit_intercept=False,
        tol=1e-4,
        max_iter=20_000,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X_train, y_train)

    return model, validation_results


def forecast_test_year(
    model: Ridge | Lasso | ElasticNet | LinearRegression,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    X_test: pd.DataFrame,
    training_mean: float,
    model_name: str,
) -> pd.DataFrame:
    """Predict the test inputs in return units, with zero and training-mean baselines.

    Test outcomes are intentionally absent from the arguments. The caller joins
    realizations and timing metadata after predictions have been computed.
    """
    X_test_scaled = x_scaler.transform(X_test)
    test_predictions_scaled = model.predict(X_test_scaled)
    test_predictions = y_scaler.inverse_transform(test_predictions_scaled.reshape(-1, 1)).ravel()

    predictions = pd.DataFrame(
        {
            f"{model_name}_prediction": test_predictions,
            "zero_prediction": 0.0,
            "training_mean_prediction": training_mean,
        },
        index=X_test.index,
    )

    # One row per test observation, with identical timestamps and order.
    assert len(predictions) == len(X_test)
    assert predictions.index.is_unique
    assert predictions.index.equals(X_test.index)

    # All prediction values are valid numbers.
    assert np.isfinite(predictions.to_numpy()).all()

    return predictions


def compute_metrics(
    predictions: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Return MSE and uncentered out-of-sample R2 for a model and two baselines.

    The denominator is sum(actual_return**2), the loss of forecasting zero.
    A zero denominator makes R2 undefined; MSE remains available.
    """
    y = predictions["actual_return"].to_numpy()
    denominator = np.sum(y**2)

    prediction_columns = [
        f"{model_name}_prediction",
        "zero_prediction",
        "training_mean_prediction",
    ]

    records = []
    for column in prediction_columns:
        squared_errors = (y - predictions[column].to_numpy()) ** 2

        r2 = np.nan if denominator == 0 else 1 - squared_errors.sum() / denominator

        records.append(
            {
                "model": column,
                "n_obs": len(y),
                "mse": squared_errors.mean(),
                "r2_oos": r2,
                "r2_oos_percent": 100 * r2,
            }
        )

    return pd.DataFrame(records)


def evaluate_and_save(
    predictions: pd.DataFrame,
    validation_results: pd.DataFrame,
    run_metadata: dict[str, object],
    output_dir: Path,
    model: Ridge | Lasso | ElasticNet | LinearRegression,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    model_name: str,
) -> pd.DataFrame:
    """Save predictions, validation losses, metrics, run metadata and fitted objects.

    The model bundle includes both training scalers and the exact feature order.
    CSV return values remain in decimal units without display rounding.
    """

    metrics = compute_metrics(predictions, model_name=model_name)

    # Check that metadata contains the predictor order used during fitting.
    feature_order = list(run_metadata["feature_order"])

    if len(feature_order) != model.n_features_in_:
        raise ValueError("Feature order does not match the fitted model.")

    # Create the destination folder.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save observation-level results and summary tables.
    predictions.to_csv(
        output_dir / "predictions.csv",
        index_label="target_timestamp",
    )
    validation_results.to_csv(
        output_dir / "validation_results.csv",
        index=False,
    )
    metrics.to_csv(
        output_dir / "metrics.csv",
        index=False,
    )

    # run_metadata holds only JSON-compatible values (strings, numbers, lists).
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(run_metadata, file, indent=2, allow_nan=False)

    # Save the fitted objects needed to reproduce predictions.
    joblib.dump(
        {
            "model": model,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "feature_order": feature_order,
        },
        output_dir / "model.joblib",
    )

    return metrics


def run_year(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    forecast_panel_path: Path,
    output_root: Path,
    test_year: int,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit, select and evaluate one model for one calendar test year.

    Each year fits fresh training-only scalers. The validation period selects
    regularization; test outcomes enter only evaluation after fitting.
    """
    split_indexes, split_metadata = make_year_split(
        X.index,
        test_year=test_year,
        start_year=START_YEAR,
        train_fraction=TRAIN_FRACTION,
    )

    X_train = X.loc[split_indexes["train"]]
    y_train = y.loc[split_indexes["train"]]

    X_valid = X.loc[split_indexes["valid"]]
    y_valid = y.loc[split_indexes["valid"]]

    X_test = X.loc[split_indexes["test"]]
    y_test = y.loc[split_indexes["test"]]

    x_scaler, y_scaler = fit_preprocessing(X_train, y_train)
    X_train_scaled = x_scaler.transform(X_train)
    X_valid_scaled = x_scaler.transform(X_valid)
    # Ridge expects a one-dimensional target.
    y_train_scaled = y_scaler.transform(y_train.to_frame()).ravel()

    if model_name == "ridge":
        selector = select_ridge_penalty
    elif model_name == "lasso":
        selector = select_lasso_penalty
    elif model_name == "elastic_net":
        selector = select_elastic_net_penalty
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model, validation_results = selector(
        X_train_scaled,
        y_train_scaled,
        X_valid_scaled,
        y_valid,
        y_scaler,
    )
    training_mean = y_train.mean()
    predictions = forecast_test_year(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        X_test=X_test,
        training_mean=training_mean,
        model_name=model_name,
    )
    predictions["actual_return"] = y_test

    predictions["forecast_origin_timestamp"] = metadata.loc[
        predictions.index, "forecast_origin_timestamp"
    ]

    run_metadata = {
        "test_year": test_year,
        "start_year": START_YEAR,
        "train_fraction": TRAIN_FRACTION,
        "source_path": str(forecast_panel_path),
        "split": split_metadata,
        "feature_order": X_train.columns.tolist(),
        "model_name": model_name,
        "selected_alpha": float(model.alpha),
        "training_target_mean": float(training_mean),
        "row_counts": {
            "train": len(X_train),
            "valid": len(X_valid),
            "test": len(X_test),
        },
    }

    if model_name == "elastic_net":
        run_metadata["selected_l1_ratio"] = float(model.l1_ratio)
        run_metadata["search_method"] = "full_grid"
        run_metadata["candidate_count"] = len(validation_results)

    output_dir = output_root / f"{model_name}_{test_year}"

    metrics = evaluate_and_save(
        predictions=predictions,
        validation_results=validation_results,
        run_metadata=run_metadata,
        output_dir=output_dir,
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        model_name=model_name,
    )

    print(f"\nCompleted {test_year}", flush=True)
    print(metrics.to_string(index=False), flush=True)

    return predictions, metrics


def main() -> None:
    """Run the requested model over an annual evaluation range and save pooled metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("ridge", "lasso", "elastic_net"), default="ridge")
    parser.add_argument("--first-year", type=int, default=2004)
    parser.add_argument("--last-year", type=int, default=2020)
    args = parser.parse_args()
    if not 2004 <= args.first_year <= args.last_year <= 2020:
        parser.error("Require 2004 <= first-year <= last-year <= 2020.")
    project_root = find_project_root(Path.cwd())
    forecast_panel_path = project_root / "data" / "processed" / "forecast_panel.parquet"
    output_root = project_root / "results" / "forecasting"

    model_name = args.model
    first_test_year = args.first_year
    last_test_year = args.last_year

    # Load once, including all years needed by the annual runs.
    X, y, metadata = load_forecast_panel(
        forecast_panel_path,
        start_year=START_YEAR,
        test_year=last_test_year,
    )

    prediction_frames = []
    metric_frames = []

    for test_year in range(first_test_year, last_test_year + 1):
        print(f"\nStarting {test_year}...", flush=True)

        predictions, metrics = run_year(
            X=X,
            y=y,
            metadata=metadata,
            forecast_panel_path=forecast_panel_path,
            output_root=output_root,
            test_year=test_year,
            model_name=model_name,
        )

        prediction_frames.append(predictions.assign(test_year=test_year))
        metric_frames.append(metrics.assign(test_year=test_year))

    pooled_predictions = pd.concat(prediction_frames).sort_index()
    annual_metrics = pd.concat(metric_frames, ignore_index=True)

    # Every eligible test observation should appear exactly once.
    expected_index = X.index[(X.index.year >= first_test_year) & (X.index.year <= last_test_year)]
    assert pooled_predictions.index.is_unique
    assert pooled_predictions.index.equals(expected_index)

    pooled_metrics = compute_metrics(
        pooled_predictions,
        model_name=model_name,
    )

    summary_dir = output_root / f"{model_name}_{first_test_year}_{last_test_year}"
    summary_dir.mkdir(parents=True, exist_ok=True)

    pooled_predictions.to_csv(
        summary_dir / "predictions.csv",
        index_label="target_timestamp",
    )
    annual_metrics.to_csv(
        summary_dir / "annual_metrics.csv",
        index=False,
    )
    pooled_metrics.to_csv(
        summary_dir / "pooled_metrics.csv",
        index=False,
    )

    print("\nPooled out-of-sample results:", flush=True)
    print(pooled_metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
