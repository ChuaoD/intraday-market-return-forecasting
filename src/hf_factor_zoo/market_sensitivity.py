"""Weaker-penalty market-only sensitivity and annual error decomposition.

Run after feature_ablation has completed for the requested years:
    uv run python -m hf_factor_zoo.market_sensitivity --last-year 2020

The original ablation files are read-only inputs. Refit only the three-feature
market model, on every year, using a fixed expanded grid. Existing full and
no-characteristics forecasts are retained. Rerunning overwrites this module's
own output directory, market_sensitivity_<first>_<last>.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression, Ridge
from threadpoolctl import threadpool_limits

from hf_factor_zoo import feature_ablation as ablation
from hf_factor_zoo import forecasting
from hf_factor_zoo.data_processing import find_project_root

EXPERIMENTS = (*ablation.EXPERIMENTS, "market_original")
PAIRS = (
    ("full", "market_only"),
    ("full", "market_original"),
    ("full", "no_characteristics"),
    ("market_only", "market_original"),
)


def expanded_penalty_grid() -> np.ndarray:
    """161 ascending candidates: zero, 60 weaker positives, 100 original values.

    Zero uses least squares with no additional intercept. Positive lambdas use
    Ridge with alpha = n_train * lambda, exactly as in the baseline runner.
    The fixed extension spans 1e-6 up to, but excluding, exp(-1).
    """
    original = np.exp(np.linspace(-1, 8, 100))
    weaker = np.geomspace(1e-6, original[0], 60, endpoint=False)
    return np.concatenate(([0.0], weaker, original))


def select_penalty(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: pd.Series,
    y_scaler,
    lambdas: np.ndarray,
) -> tuple[Ridge | LinearRegression, pd.DataFrame]:
    """Select by raw-return validation MSE; exact ties choose smaller lambda.

    All fits use standardized training data only. Test data are not arguments.
    Retain the winning training fit rather than pooling train and validation.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    if (
        lambdas.ndim != 1
        or len(lambdas) == 0
        or not np.isfinite(lambdas).all()
        or (lambdas < 0).any()
        or (np.diff(lambdas) <= 0).any()
    ):
        raise ValueError("Require finite, nonnegative, strictly ascending penalties.")
    records = []
    best_model, best_loss = None, np.inf
    for penalty in lambdas:
        alpha = len(y_train) * float(penalty)
        model = (
            LinearRegression(fit_intercept=False)
            if penalty == 0
            else Ridge(alpha=alpha, fit_intercept=False, solver="cholesky")
        )
        model.fit(X_train, y_train)
        predictions = y_scaler.inverse_transform(model.predict(X_valid).reshape(-1, 1)).ravel()
        loss = float(np.mean((y_valid.to_numpy() - predictions) ** 2))
        if not np.isfinite(loss):
            raise ValueError("Nonfinite validation error.")
        records.append({"lambda": float(penalty), "alpha": alpha, "validation_mse": loss})
        if loss < best_loss:
            best_model, best_loss = model, loss
    if best_model is None:
        raise ValueError("No fitted model selected.")
    return best_model, pd.DataFrame(records)


def fit_year(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    year: int,
    lambdas: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Fit one market-only year; return predictions, search, metadata and fit bundle."""
    splits, split_metadata = forecasting.make_year_split(
        X.index, year, forecasting.START_YEAR, forecasting.TRAIN_FRACTION
    )
    train, valid, test = (splits[name] for name in ("train", "valid", "test"))
    x_scaler, y_scaler = forecasting.fit_preprocessing(X.loc[train], y.loc[train])
    model, validation = select_penalty(
        x_scaler.transform(X.loc[train]),
        y_scaler.transform(y.loc[train].to_frame()).ravel(),
        x_scaler.transform(X.loc[valid]),
        y.loc[valid],
        y_scaler,
        lambdas,
    )
    winner = validation.iloc[int(validation.validation_mse.argmin())]
    predictions = forecasting.forecast_test_year(
        model, x_scaler, y_scaler, X.loc[test], float(y.loc[train].mean()), "market_only"
    )
    predictions["actual_return"] = y.loc[test]
    predictions["forecast_origin_timestamp"] = metadata.loc[test, "forecast_origin_timestamp"]
    predictions["test_year"] = year
    run_metadata = {
        "test_year": year,
        "start_year": forecasting.START_YEAR,
        "train_fraction": forecasting.TRAIN_FRACTION,
        "split": split_metadata,
        "row_counts": {name: len(index) for name, index in splits.items()},
        "feature_order": X.columns.tolist(),
        "model_name": "market_only",
        "selected_lambda": float(winner["lambda"]),
        "selected_alpha": float(winner["alpha"]),
        "selected_validation_mse": float(winner["validation_mse"]),
        "training_target_mean": float(y.loc[train].mean()),
        "zero_penalty_solver": "LinearRegression(fit_intercept=False)",
        "positive_penalty_solver": "Ridge(fit_intercept=False, solver='cholesky')",
    }
    bundle = {"model": model, "x_scaler": x_scaler, "y_scaler": y_scaler}
    return predictions, validation, run_metadata, bundle


def annual_contributions(
    comparison: pd.DataFrame,
    pairs: tuple[tuple[str, str], ...] = PAIRS,
) -> pd.DataFrame:
    """Split each pooled R2 difference into annual contributions that add up exactly.

    For left minus right, contribution_pp = 100 * (SSE_right - SSE_left) / S,
    where S is the sum of squared returns over the whole period. Because S is
    shared, the contributions sum to the pooled R2 difference in percentage
    points. annual_r2_difference_pp instead divides by that year's own sum.
    """
    denominator = float((comparison.actual_return**2).sum())
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("Require a positive finite full-period denominator.")
    records = []
    for year, frame in comparison.groupby("test_year", sort=True):
        annual_denominator = float((frame.actual_return**2).sum())
        if annual_denominator <= 0:
            raise ValueError(f"Year {year} has no squared target variation.")
        for left, right in pairs:
            left_sse = float(((frame.actual_return - frame[f"{left}_prediction"]) ** 2).sum())
            right_sse = float(((frame.actual_return - frame[f"{right}_prediction"]) ** 2).sum())
            improvement = right_sse - left_sse
            records.append(
                {
                    "test_year": int(year),
                    "comparison": f"{left}_minus_{right}",
                    "n_obs": len(frame),
                    "target_sum_squares": annual_denominator,
                    "target_energy_weight": annual_denominator / denominator,
                    "left_sse": left_sse,
                    "right_sse": right_sse,
                    "improvement_sse": improvement,
                    "annual_r2_difference_pp": 100 * improvement / annual_denominator,
                    "contribution_pp": 100 * improvement / denominator,
                }
            )
    result = pd.DataFrame(records)
    for left, right in pairs:
        key = f"{left}_minus_{right}"
        expected = (
            100
            * (
                ((comparison.actual_return - comparison[f"{right}_prediction"]) ** 2).sum()
                - ((comparison.actual_return - comparison[f"{left}_prediction"]) ** 2).sum()
            )
            / denominator
        )
        np.testing.assert_allclose(
            result.loc[result.comparison == key, "contribution_pp"].sum(),
            expected,
            rtol=1e-9,
            atol=1e-12,
        )
    return result


def plot_contributions(contributions: pd.DataFrame, output_dir: Path) -> None:
    """Plot additive contributions; bars above zero favor the full model."""
    figure = Figure(figsize=(11, 7), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1, sharex=True)
    pivot = contributions.pivot(index="test_year", columns="comparison", values="contribution_pp")
    years = pivot.index.to_numpy()
    axes[0].bar(
        years - 0.2,
        pivot.full_minus_market_original,
        width=0.4,
        label="Full − Market (original grid)",
        color="#5477a4",
    )
    axes[0].bar(
        years + 0.2,
        pivot.full_minus_market_only,
        width=0.4,
        label="Full − Market (expanded grid)",
        color="#d58d36",
    )
    axes[1].bar(
        years,
        pivot.full_minus_no_characteristics,
        color="#439581",
        width=0.65,
        label="Full − Without characteristics",
    )
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Contribution to pooled R²\ndifference (percentage points)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    axes[0].set_title("Which years contribute to the full model's accuracy difference?")
    axes[1].set_xticks(years)
    axes[1].set_xlabel("Test year; annual contributions sum to the pooled difference")
    for extension in ("png", "svg"):
        figure.savefig(output_dir / f"annual_contributions.{extension}", dpi=200)
    figure.clear()


def run_study(root: Path, first_year: int, last_year: int) -> Path:
    """Refit only market-only, validate against saved searches, and save the study."""
    if not 2004 <= first_year <= last_year <= 2020:
        raise ValueError("Require 2004 <= first_year <= last_year <= 2020.")
    results_root = root / "results" / "forecasting"
    source = results_root / f"feature_ablation_{first_year}_{last_year}"
    output = results_root / f"market_sensitivity_{first_year}_{last_year}"
    original = pd.read_csv(
        source / "predictions.csv",
        index_col="target_timestamp",
        parse_dates=["target_timestamp", "forecast_origin_timestamp"],
        float_precision="round_trip",
    ).sort_index()
    panel_path = root / "data" / "processed" / "forecast_panel.parquet"
    X, y, metadata = forecasting.load_forecast_panel(panel_path, test_year=last_year)
    # Recheck full-model source provenance and targets using the existing loader.
    full = ablation.load_full_predictions(results_root, X, y, metadata, first_year, last_year)
    original.index = original.index.as_unit(X.index.unit)
    original["forecast_origin_timestamp"] = original.forecast_origin_timestamp.dt.as_unit(
        metadata.forecast_origin_timestamp.dt.unit
    )
    frames = {
        name: original[[*ablation.SHARED_COLUMNS, f"{name}_prediction"]].copy()
        for name in ablation.EXPERIMENTS
    }
    ablation.build_comparison(frames)
    if not full.index.equals(original.index):
        raise ValueError("Original ablation and full reference use different test rows.")
    for column in ("actual_return", "full_prediction", "training_mean_prediction"):
        ablation._require_close(original[column], full[column], f"Original {column}")
    if not original.forecast_origin_timestamp.equals(full.forecast_origin_timestamp):
        raise ValueError("Original forecast origins differ from full reference.")
    X = X.loc[:, ablation.select_feature_columns(X.columns, "market_only")].copy()
    lambdas = expanded_penalty_grid()
    predictions, search_records = [], []
    for year in range(first_year, last_year + 1):
        print(f"Expanded market-only search: {year} ({len(lambdas)} penalties)", flush=True)
        frame, validation, info, bundle = fit_year(X, y, metadata, year, lambdas)
        old_dir = source / "market_only" / f"ridge_{year}"
        old_info = json.loads((old_dir / "metadata.json").read_text(encoding="utf-8"))
        old_validation = pd.read_csv(
            old_dir / "validation_results.csv", float_precision="round_trip"
        )
        for key in ("feature_order", "split", "row_counts", "start_year", "train_fraction"):
            if old_info[key] != info[key]:
                raise ValueError(f"Original market-only {year}: mismatch in {key}.")
        # Original candidates must reproduce their original validation losses.
        retained = validation.loc[validation["lambda"].isin(old_validation["lambda"])]
        if len(retained) != 100:
            raise ValueError("The original 100 penalties were not retained exactly.")
        np.testing.assert_allclose(
            retained[["lambda", "alpha", "validation_mse"]],
            old_validation[["lambda", "alpha", "validation_mse"]],
            rtol=1e-9,
            atol=1e-18,
        )
        old_winner = old_validation.iloc[int(old_validation.validation_mse.argmin())]
        if info["selected_validation_mse"] > old_winner.validation_mse + 1e-18:
            raise ValueError("Expanded search unexpectedly worsened minimum validation MSE.")
        search_records.append(
            {
                "test_year": year,
                "original_lambda": float(old_winner["lambda"]),
                "expanded_lambda": info["selected_lambda"],
                "original_validation_mse": float(old_winner.validation_mse),
                "expanded_validation_mse": info["selected_validation_mse"],
                "original_at_lower_boundary": bool(
                    old_winner["lambda"] == old_validation["lambda"].min()
                ),
                "expanded_selects_weaker": bool(
                    info["selected_lambda"] < old_validation["lambda"].min()
                ),
                "expanded_selects_ols": info["selected_lambda"] == 0,
            }
        )
        info["source_path"] = str(panel_path)
        info["lambda_grid"] = lambdas.tolist()
        forecasting.evaluate_and_save(
            predictions=frame,
            validation_results=validation,
            run_metadata=info,
            output_dir=output / "market_only" / f"ridge_{year}",
            model_name="market_only",
            **bundle,
        )
        predictions.append(frame)
    frames["market_only"] = pd.concat(predictions).sort_index()
    comparison = ablation.build_comparison(frames)
    comparison["market_original_prediction"] = original.market_only_prediction
    pooled, annual = ablation.evaluate_comparison(comparison)
    old_metrics = (
        forecasting.compute_metrics(comparison, "market_original")
        .iloc[[0]]
        .assign(experiment="market_original", n_features=3)
    )
    pooled = pd.concat([pooled, old_metrics], ignore_index=True)
    old_annual = [
        forecasting.compute_metrics(f, "market_original")
        .iloc[[0]]
        .assign(experiment="market_original", n_features=3, test_year=int(year))
        for year, f in comparison.groupby("test_year")
    ]
    annual = pd.concat([annual, *old_annual], ignore_index=True)
    daily = ablation.aggregate_daily_losses(comparison, EXPERIMENTS)
    points = ablation.compute_statistics(daily, EXPERIMENTS, PAIRS)
    draws = ablation.run_bootstrap(daily, experiments=EXPERIMENTS, pairs=PAIRS)
    intervals = ablation.summarize_intervals(points, draws)
    settings = {
        "first_test_year": first_year,
        "last_test_year": last_year,
        "source_ablation": str(source),
        "panel_path": str(panel_path),
        "lambda_grid": lambdas.tolist(),
        "original_candidates": 100,
        "added_positive_candidates": 60,
        "added_zero_penalty_candidate": True,
        "alpha": "n_train * lambda",
        "tie_break": "smaller lambda for exact validation-MSE ties",
        "selection": "Validation MSE only; training-only scaling and coefficient fitting",
        "scope": "Follow-up sensitivity after inspecting the original historical results",
        "market_only_label": "Expanded-grid forecasts; market_original is the unchanged baseline",
        "n_replicates": ablation.N_REPLICATES,
        "block_lengths": ablation.BLOCK_LENGTHS,
        "seed": ablation.RANDOM_SEED,
        "confidence_level": ablation.CONFIDENCE_LEVEL,
        "bootstrap": (
            "Paired circular days; pointwise percentile intervals conditional on forecasts"
        ),
        "contribution_pp": "100 * (SSE_right_year - SSE_left_year) / sum_all_years(y^2)",
    }
    ablation.save_results(
        comparison, pooled, annual, daily, points, draws, intervals, settings, output
    )
    pd.DataFrame(search_records).to_csv(output / "penalty_selection.csv", index=False)
    contributions = annual_contributions(comparison)
    contributions.to_csv(output / "annual_contributions.csv", index=False)
    # Labels in this output directory refer to the expanded market-only model.
    ablation.plot_results(annual, intervals, output)
    plot_contributions(contributions, output)
    print(pooled.to_string(index=False))
    print(f"Saved sensitivity study to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-year", type=int, default=2004)
    parser.add_argument("--last-year", type=int, default=2020)
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        run_study(find_project_root(Path.cwd()), args.first_year, args.last_year)


if __name__ == "__main__":
    main()
