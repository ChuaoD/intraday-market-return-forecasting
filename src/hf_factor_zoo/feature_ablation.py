"""Compare Ridge forecasts using full, market-only and reduced factor sets.

Reuse saved full-Ridge forecasts; independently tune Ridge for each subset.
All experiments retain identical rows, training-only scaling and annual splits.
Bootstrap intervals are pointwise and conditional on the saved forecasts.

Run a single-year check:
    uv run python -m hf_factor_zoo.feature_ablation
Run the complete historical study:
    uv run python -m hf_factor_zoo.feature_ablation --last-year 2020
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from threadpoolctl import threadpool_limits

from hf_factor_zoo import forecasting
from hf_factor_zoo.bootstrap_forecast import sample_block_indices
from hf_factor_zoo.data_processing import find_project_root

EXPERIMENTS = ("full", "market_only", "no_characteristics")
FIT_EXPERIMENTS = ("market_only", "no_characteristics")
EXPECTED_FEATURE_COUNTS = {"full": 816, "market_only": 3, "no_characteristics": 162}
COMPARISON_PAIRS = (("full", "market_only"), ("full", "no_characteristics"))
SHARED_COLUMNS = (
    "actual_return",
    "forecast_origin_timestamp",
    "test_year",
    "zero_prediction",
    "training_mean_prediction",
)
LABELS = {
    "full": "Full (816 predictors)",
    "market_only": "Market only (3 predictors)",
    "no_characteristics": "Without characteristics (162 predictors)",
}
FIRST_TEST_YEAR = 2004
LAST_TEST_YEAR = 2004
N_REPLICATES = 5_000
BLOCK_LENGTHS = (5, 10, 20)
PRIMARY_BLOCK_DAYS = 5
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42
# Return-unit tolerances allow CSV round trips, not economically different data.
RETURN_RTOL = 1e-10
RETURN_ATOL = 1e-14


def _require_close(actual, expected, description: str) -> None:
    """Compare finite numeric arrays without broadcasting."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape or not np.allclose(
        actual, expected, rtol=RETURN_RTOL, atol=RETURN_ATOL, equal_nan=False
    ):
        raise ValueError(f"{description} differs from the expected values.")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"{description} contains nonfinite values.")


def _validate_frame(frame: pd.DataFrame, experiment: str) -> None:
    """Validate one experiment's timestamp index, shared fields and predictions."""
    required = [*SHARED_COLUMNS, f"{experiment}_prediction"]
    missing = set(required) - set(frame.columns)
    if missing or not frame.columns.is_unique:
        raise ValueError(f"{experiment}: missing {missing} or duplicate columns.")
    index = frame.index
    if (
        frame.empty
        or not isinstance(index, pd.DatetimeIndex)
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise ValueError(f"{experiment}: require a nonempty, sorted, unique datetime index.")
    origins = frame["forecast_origin_timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(origins.dtype):
        raise ValueError(f"{experiment}: forecast origins must be parsed datetimes.")
    if origins.isna().any() or not (origins < index).all():
        raise ValueError(f"{experiment}: forecast origins must precede targets.")
    if not np.array_equal(frame["test_year"].to_numpy(), index.year):
        raise ValueError(f"{experiment}: year labels disagree with timestamps.")
    numeric = frame[
        ["actual_return", "zero_prediction", "training_mean_prediction", f"{experiment}_prediction"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{experiment}: nonfinite return or prediction.")
    if not (frame["zero_prediction"] == 0).all():
        raise ValueError(f"{experiment}: zero-return baseline must be zero.")


def select_feature_columns(columns: pd.Index, experiment: str) -> list[str]:
    """Select complete three-horizon factor sets, retaining original column order."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    if not columns.is_unique:
        raise ValueError("Feature names must be unique.")
    selected = []
    horizons_by_factor: dict[str, set[str]] = {}
    valid_horizons = {"lag15", "avg60", "prevday"}
    for column in columns:
        if not isinstance(column, str) or "__" not in column:
            raise ValueError(f"Malformed feature: {column!r}")
        horizon, factor = column.split("__", 1)
        if horizon not in valid_horizons or not factor.startswith(
            ("ff__", "ind48__", "cz__", "jkp__")
        ):
            raise ValueError(f"Unknown feature horizon or family: {column}")
        keep = (
            experiment == "full"
            or (experiment == "market_only" and factor == "ff__mkt")
            or (experiment == "no_characteristics" and factor.startswith(("ind48__", "ff__")))
        )
        if keep:
            selected.append(column)
            horizons_by_factor.setdefault(factor, set()).add(horizon)
    if len(selected) != EXPECTED_FEATURE_COUNTS[experiment]:
        raise ValueError(f"{experiment}: unexpected number of features: {len(selected)}")
    if any(horizons != valid_horizons for horizons in horizons_by_factor.values()):
        raise ValueError("Each selected factor must contain all three horizons.")
    return selected


def load_full_predictions(
    results_root: Path,
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    first_year: int,
    last_year: int,
) -> pd.DataFrame:
    """Load the saved annual full-Ridge forecasts and check them against the panel.

    Checks the saved splits, feature order, targets, baselines, penalty grid and
    selected alpha, so the reduced models are compared on identical rows. The
    saved runs do not record code versions, so these checks confirm consistency,
    not exact provenance.
    """
    if first_year > last_year:
        raise ValueError("first_year must not exceed last_year.")
    if not X.index.equals(y.index) or not X.index.equals(metadata.index):
        raise ValueError("Panel inputs must share an identical index.")
    feature_order = select_feature_columns(X.columns, "full")
    frames = []
    for year in range(first_year, last_year + 1):
        directory = results_root / f"ridge_{year}"
        saved = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        splits, split_metadata = forecasting.make_year_split(
            X.index, year, forecasting.START_YEAR, forecasting.TRAIN_FRACTION
        )
        expected_metadata = {
            "model_name": "ridge",
            "test_year": year,
            "start_year": forecasting.START_YEAR,
            "train_fraction": forecasting.TRAIN_FRACTION,
            "feature_order": feature_order,
            "split": split_metadata,
            "row_counts": {name: len(index) for name, index in splits.items()},
        }
        for key, expected in expected_metadata.items():
            if saved.get(key) != expected:
                raise ValueError(f"Full Ridge {year}: incompatible saved {key}.")
        validation = pd.read_csv(directory / "validation_results.csv", float_precision="round_trip")
        grid = np.exp(np.linspace(-1, 8, 100))
        _require_close(validation["lambda"], grid, f"Full Ridge {year} lambda grid")
        _require_close(
            validation["alpha"], len(splits["train"]) * grid, f"Full Ridge {year} alpha grid"
        )
        losses = validation["validation_mse"].to_numpy(dtype=float)
        if not np.isfinite(losses).all() or (losses < 0).any():
            raise ValueError(f"Full Ridge {year}: invalid validation losses.")
        winner = validation.iloc[int(np.argmin(losses))]
        _require_close(saved["selected_alpha"], winner["alpha"], f"Full Ridge {year} selection")

        frame = pd.read_csv(
            directory / "predictions.csv",
            parse_dates=["target_timestamp", "forecast_origin_timestamp"],
            index_col="target_timestamp",
            float_precision="round_trip",
        ).sort_index()
        # CSV parsing and Parquet can use different datetime resolutions.
        # Match units before equality checks; do not change timestamp values.
        frame.index = frame.index.as_unit(X.index.unit)
        frame["forecast_origin_timestamp"] = frame["forecast_origin_timestamp"].dt.as_unit(
            metadata["forecast_origin_timestamp"].dt.unit
        )
        if "test_year" in frame and not (frame["test_year"] == year).all():
            raise ValueError(f"Full Ridge {year}: incorrect saved year labels.")
        frame["test_year"] = year
        frame = frame.rename(columns={"ridge_prediction": "full_prediction"})
        _validate_frame(frame, "full")
        if not frame.index.equals(splits["test"]):
            raise ValueError(f"Full Ridge {year}: saved test rows differ from the panel.")
        _require_close(frame["actual_return"], y.loc[frame.index], f"Full Ridge {year} targets")
        expected_origins = metadata.loc[frame.index, "forecast_origin_timestamp"]
        if not frame["forecast_origin_timestamp"].equals(expected_origins):
            raise ValueError(f"Full Ridge {year}: forecast origins differ from the panel.")
        train_mean = float(y.loc[splits["train"]].mean())
        _require_close(saved["training_target_mean"], train_mean, f"Full Ridge {year} mean")
        _require_close(
            frame["training_mean_prediction"],
            np.full(len(frame), train_mean),
            f"Full Ridge {year} mean baseline",
        )
        frames.append(frame.loc[:, [*SHARED_COLUMNS, "full_prediction"]])
    result = pd.concat(frames).sort_index()
    _validate_frame(result, "full")
    return result


def run_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    forecast_panel_path: Path,
    output_root: Path,
    experiment: str,
    first_year: int,
    last_year: int,
) -> pd.DataFrame:
    """Independently tune and fit annual Ridge models for one feature subset."""
    if experiment not in FIT_EXPERIMENTS or first_year > last_year:
        raise ValueError("Require a reduced experiment and an ordered year range.")
    if not X.index.equals(y.index) or not X.index.equals(metadata.index):
        raise ValueError("Panel inputs must share an identical index.")
    subset = X.loc[:, select_feature_columns(X.columns, experiment)]
    experiment_root = output_root / experiment
    frames = []
    for year in range(first_year, last_year + 1):
        print(f"Starting {experiment}, {year}...", flush=True)
        predictions, _ = forecasting.run_year(
            X=subset,
            y=y,
            metadata=metadata,
            forecast_panel_path=forecast_panel_path,
            output_root=experiment_root,
            test_year=year,
            model_name="ridge",
        )
        frames.append(predictions.assign(test_year=year))
    result = (
        pd.concat(frames)
        .sort_index()
        .rename(columns={"ridge_prediction": f"{experiment}_prediction"})
    )
    _validate_frame(result, experiment)
    expected = X.index[(X.index.year >= first_year) & (X.index.year <= last_year)]
    if not result.index.equals(expected):
        raise ValueError(f"{experiment}: missing or extra test observations.")
    experiment_root.mkdir(parents=True, exist_ok=True)
    result.to_csv(experiment_root / "predictions.csv", index_label="target_timestamp")
    return result


def build_comparison(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align three experiments only after checking all shared fields."""
    if set(frames) != set(EXPERIMENTS):
        raise ValueError(f"Require exactly these experiments: {EXPERIMENTS}")
    for experiment, frame in frames.items():
        _validate_frame(frame, experiment)
    reference = frames["full"]
    comparison = reference.loc[:, list(SHARED_COLUMNS)].copy()
    for experiment in EXPERIMENTS:
        frame = frames[experiment]
        if not frame.index.equals(reference.index):
            raise ValueError(f"{experiment}: timestamps differ from full.")
        for column in SHARED_COLUMNS:
            if column in ("forecast_origin_timestamp", "test_year"):
                if not frame[column].equals(reference[column]):
                    raise ValueError(f"{experiment}: shared {column} differs from full.")
            else:
                _require_close(frame[column], reference[column], f"{experiment} shared {column}")
        comparison[f"{experiment}_prediction"] = frame[f"{experiment}_prediction"]
    return comparison


def evaluate_comparison(
    comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return pooled and annual metrics with each common baseline included once."""

    def evaluate(frame: pd.DataFrame) -> pd.DataFrame:
        records = []
        for experiment in EXPERIMENTS:
            metrics = forecasting.compute_metrics(frame, model_name=experiment)
            selected = metrics.iloc[[0]].assign(
                experiment=experiment, n_features=EXPECTED_FEATURE_COUNTS[experiment]
            )
            records.append(selected)
            if experiment == "full":
                baselines = metrics.iloc[1:].copy()
                baselines["experiment"] = ["zero", "training_mean"]
                baselines["n_features"] = 0
                records.append(baselines)
        return pd.concat(records, ignore_index=True)

    pooled = evaluate(comparison)
    annual = pd.concat(
        [
            evaluate(frame).assign(test_year=int(year))
            for year, frame in comparison.groupby("test_year", sort=True)
        ],
        ignore_index=True,
    )
    return pooled, annual


def aggregate_daily_losses(
    comparison: pd.DataFrame,
    experiments: tuple[str, ...] = EXPERIMENTS,
) -> pd.DataFrame:
    """Sum interval losses by trading day; preserve unequal day lengths."""
    for experiment in experiments:
        _validate_frame(comparison, experiment)
    losses = pd.DataFrame(index=comparison.index)
    losses["n_obs"] = 1
    losses["target_sum_squares"] = comparison["actual_return"] ** 2
    for experiment in experiments:
        losses[f"{experiment}_sse"] = (
            comparison["actual_return"] - comparison[f"{experiment}_prediction"]
        ) ** 2
    daily = losses.groupby(comparison.index.normalize()).sum().sort_index()
    daily.index.name = "date"
    if daily["n_obs"].sum() != len(comparison):
        raise ValueError("Daily aggregation did not preserve all observations.")
    compute_statistics(daily, experiments, ())  # Validate sums and denominator.
    return daily


def compute_statistics(
    daily_losses: pd.DataFrame,
    experiments: tuple[str, ...] = EXPERIMENTS,
    pairs: tuple[tuple[str, str], ...] = COMPARISON_PAIRS,
) -> pd.Series:
    """Calculate pooled R2 and paired differences in decimal units."""
    if not experiments or len(set(experiments)) != len(experiments):
        raise ValueError("Require distinct experiment names.")
    if any(left not in experiments or right not in experiments for left, right in pairs):
        raise ValueError("Comparison pairs must use the supplied experiments.")
    columns = ["n_obs", "target_sum_squares", *(f"{e}_sse" for e in experiments)]
    values = daily_losses.loc[:, columns].to_numpy(dtype=float)
    if (
        len(values) == 0
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values[:, 0] <= 0).any()
    ):
        raise ValueError("Require finite nonnegative daily sums and positive counts.")
    totals = values.sum(axis=0)
    denominator = totals[1]
    if not np.isfinite(totals).all() or denominator <= 0:
        raise ValueError("Require finite total losses and a positive target sum of squares.")
    statistics = {
        f"{experiment}_r2": 1 - totals[i + 2] / denominator
        for i, experiment in enumerate(experiments)
    }
    for left, right in pairs:
        statistics[f"{left}_minus_{right}"] = statistics[f"{left}_r2"] - statistics[f"{right}_r2"]
    return pd.Series(statistics, dtype=float)


def run_bootstrap(
    daily_losses: pd.DataFrame,
    block_lengths: tuple[int, ...] = BLOCK_LENGTHS,
    n_replicates: int = N_REPLICATES,
    seed: int = RANDOM_SEED,
    *,
    experiments: tuple[str, ...] = EXPERIMENTS,
    pairs: tuple[tuple[str, str], ...] = COMPARISON_PAIRS,
) -> pd.DataFrame:
    """Resample paired circular day blocks without refitting any models."""
    compute_statistics(daily_losses, experiments, pairs)
    if (
        not isinstance(daily_losses.index, pd.DatetimeIndex)
        or daily_losses.index.hasnans
        or not daily_losses.index.is_unique
        or not daily_losses.index.is_monotonic_increasing
    ):
        raise ValueError("Original daily losses must have a sorted, unique datetime index.")
    if isinstance(n_replicates, bool) or not isinstance(n_replicates, int) or n_replicates < 2:
        raise ValueError("Require an integer n_replicates >= 2.")
    if not block_lengths or len(set(block_lengths)) != len(block_lengths):
        raise ValueError("Require distinct block lengths.")
    if any(
        isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= len(daily_losses)
        for days in block_lengths
    ):
        raise ValueError("Each block length must be an integer between 1 and n_days.")
    rng = np.random.default_rng(seed)
    records = []
    for block_days in block_lengths:
        print(f"Bootstrap: {block_days}-day blocks, {n_replicates:,} draws...", flush=True)
        for replicate in range(n_replicates):
            positions = sample_block_indices(len(daily_losses), block_days, rng)
            statistics = compute_statistics(daily_losses.iloc[positions], experiments, pairs)
            records.append(
                {"block_days": block_days, "replicate": replicate, **statistics.to_dict()}
            )
    return pd.DataFrame(records)


def summarize_intervals(
    point_estimates: pd.Series,
    bootstrap_draws: pd.DataFrame,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Return pointwise percentile intervals, retaining decimal statistic units."""
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one.")
    if (
        point_estimates.empty
        or not point_estimates.index.is_unique
        or not np.isfinite(point_estimates.to_numpy(dtype=float)).all()
        or bootstrap_draws.empty
    ):
        raise ValueError("Require finite point estimates and nonempty bootstrap draws.")
    if bootstrap_draws.duplicated(["block_days", "replicate"]).any():
        raise ValueError("Duplicate bootstrap replicate labels.")
    tail = (1 - confidence_level) / 2
    records = []
    for block_days, group in bootstrap_draws.groupby("block_days", sort=True):
        if len(group) < 2:
            raise ValueError("Each block length requires at least two draws.")
        for statistic, estimate in point_estimates.items():
            draws = group[statistic].to_numpy(dtype=float)
            if not np.isfinite(draws).all():
                raise ValueError(f"Nonfinite bootstrap draws: {statistic}")
            lower, upper = np.quantile(draws, [tail, 1 - tail])
            records.append(
                {
                    "block_days": int(block_days),
                    "statistic": statistic,
                    "estimate": float(estimate),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "confidence_level": confidence_level,
                    "n_replicates": len(draws),
                }
            )
    return pd.DataFrame(records)


def plot_results(
    annual_metrics: pd.DataFrame,
    intervals: pd.DataFrame,
    output_dir: Path,
    block_days: int = PRIMARY_BLOCK_DAYS,
) -> None:
    """Save annual accuracy and the two incremental-performance intervals."""
    output_dir.mkdir(parents=True, exist_ok=True)
    annual_figure = Figure(figsize=(9, 5), layout="constrained")
    FigureCanvasAgg(annual_figure)
    ax = annual_figure.subplots()
    for experiment in EXPERIMENTS:
        rows = annual_metrics.loc[annual_metrics["experiment"] == experiment].sort_values(
            "test_year"
        )
        if rows.empty or rows["test_year"].duplicated().any():
            raise ValueError(f"Missing or duplicated annual metrics for {experiment}.")
        ax.plot(rows["test_year"], 100 * rows["r2_oos"], marker="o", label=LABELS[experiment])
    years = sorted(annual_metrics["test_year"].unique())
    ax.set_xticks(years[:: max(1, len(years) // 9)])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Test year", ylabel="Out-of-sample R² (%)", title="Ridge feature ablation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    annual_figure.savefig(output_dir / "annual_r2.png", dpi=200)
    annual_figure.clear()

    names = [f"{left}_minus_{right}" for left, right in COMPARISON_PAIRS]
    selected = intervals.loc[
        (intervals["block_days"] == block_days) & intervals["statistic"].isin(names)
    ]
    if len(selected) != len(names) or selected["statistic"].duplicated().any():
        raise ValueError("Require exactly one interval for each primary comparison.")
    rows = selected.set_index("statistic").loc[names]
    if rows["confidence_level"].nunique() != 1:
        raise ValueError("Intervals must share one confidence level.")
    figure = Figure(figsize=(9, 3.6), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    positions = np.arange(len(names))
    # Percentile bounds need not contain the original estimate; draw them separately.
    ax.hlines(positions, 100 * rows["ci_lower"], 100 * rows["ci_upper"], linewidth=2)
    ax.scatter(100 * rows["estimate"], positions, zorder=3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(positions, ["Full − Market only", "Full − Without characteristics"])
    ax.set_ylim(len(names) - 0.5, -0.5)
    level = float(rows["confidence_level"].iloc[0])
    ax.set(
        xlabel="Difference in out-of-sample R² (percentage points)",
        title=f"{level:.0%} pointwise bootstrap intervals · {block_days}-day blocks",
    )
    ax.grid(axis="x", alpha=0.25)
    figure.savefig(output_dir / f"incremental_r2_intervals_{block_days}day.png", dpi=200)
    figure.clear()


def save_results(
    comparison: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    annual_metrics: pd.DataFrame,
    daily_losses: pd.DataFrame,
    point_estimates: pd.Series,
    bootstrap_draws: pd.DataFrame,
    intervals: pd.DataFrame,
    settings: dict,
    output_dir: Path,
) -> None:
    """Save predictions, metrics, bootstrap results and settings to this study's directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "predictions.csv", index_label="target_timestamp")
    daily_losses.to_csv(output_dir / "daily_losses.csv", index_label="date")
    tables = {
        "pooled_metrics": pooled_metrics,
        "annual_metrics": annual_metrics,
        "point_estimates": point_estimates.rename_axis("statistic").reset_index(name="estimate"),
        "bootstrap_draws": bootstrap_draws,
        "confidence_intervals": intervals,
    }
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Run 2004 by default; --last-year 2020 requests the complete study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-year", type=int, default=FIRST_TEST_YEAR)
    parser.add_argument("--last-year", type=int, default=LAST_TEST_YEAR)
    args = parser.parse_args()
    if not 2004 <= args.first_year <= args.last_year <= 2020:
        parser.error("Require 2004 <= first-year <= last-year <= 2020.")

    root = find_project_root(Path.cwd())
    panel_path = root / "data" / "processed" / "forecast_panel.parquet"
    results_root = root / "results" / "forecasting"
    output_root = results_root / f"feature_ablation_{args.first_year}_{args.last_year}"
    X, y, metadata = forecasting.load_forecast_panel(
        panel_path, start_year=forecasting.START_YEAR, test_year=args.last_year
    )
    # Validate references and all feature sets before launching either reduced fit.
    feature_sets = {name: select_feature_columns(X.columns, name) for name in EXPERIMENTS}
    frames = {
        "full": load_full_predictions(results_root, X, y, metadata, args.first_year, args.last_year)
    }
    with threadpool_limits(limits=1, user_api="blas"):
        for experiment in FIT_EXPERIMENTS:
            frames[experiment] = run_experiment(
                X, y, metadata, panel_path, output_root, experiment, args.first_year, args.last_year
            )
    comparison = build_comparison(frames)
    pooled, annual = evaluate_comparison(comparison)
    daily = aggregate_daily_losses(comparison)
    points = compute_statistics(daily)
    draws = run_bootstrap(daily)
    intervals = summarize_intervals(points, draws)
    settings = {
        "first_test_year": args.first_year,
        "last_test_year": args.last_year,
        "history_start_year": forecasting.START_YEAR,
        "train_fraction": forecasting.TRAIN_FRACTION,
        "panel_path": str(panel_path),
        "full_reference_root": str(results_root),
        "feature_sets": feature_sets,
        "feature_counts": EXPECTED_FEATURE_COUNTS,
        "ridge": {
            "lambda_grid": np.exp(np.linspace(-1, 8, 100)).tolist(),
            "alpha": "n_train * lambda",
            "solver": "cholesky",
            "fit_intercept": False,
            "scaling": "Training-only StandardScaler for features and target",
            "selection": "Minimum validation MSE in return units; smaller lambda breaks exact ties",
            "final_fit": "Training data only; no training-validation pooling",
        },
        "n_observations": len(comparison),
        "n_trading_days": len(daily),
        "n_replicates": N_REPLICATES,
        "block_lengths": list(BLOCK_LENGTHS),
        "primary_block_days": PRIMARY_BLOCK_DAYS,
        "seed": RANDOM_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "comparison_pairs": COMPARISON_PAIRS,
        "statistic_units": "Decimal R2 and R2 differences; multiply by 100 for display",
        "bootstrap": "Paired circular trading-day blocks; pointwise percentile intervals",
        "scope": "Follow-up historical analysis; intervals conditional on saved forecasts",
        "reference_checks": (
            "Saved rows, targets, origins, features, splits, baselines, validation grid and alpha"
        ),
    }
    save_results(comparison, pooled, annual, daily, points, draws, intervals, settings, output_root)
    plot_results(annual, intervals, output_root)
    print("\nPooled feature-ablation results:")
    print(pooled.to_string(index=False))
    print("\nPrimary bootstrap intervals (decimal units):")
    print(intervals.loc[intervals["block_days"] == PRIMARY_BLOCK_DAYS].to_string(index=False))
    if args.first_year == args.last_year:
        print("\nSingle-year evaluation; pooled results cover this year only.")
    print(f"\nSaved results to: {output_root.resolve()}")


if __name__ == "__main__":
    main()
