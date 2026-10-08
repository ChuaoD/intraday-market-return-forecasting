"""Paired circular trading-day bootstrap of saved forecast accuracy.

Daily sums retain all intraday errors and their observation weights. All
models share each resample; fitted models remain fixed. R2 and differences
are stored as decimals and multiplied by 100 only for display.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hf_factor_zoo.data_processing import find_project_root

MODELS = ("ridge", "lasso", "elastic_net")
MODEL_PAIRS = (
    ("ridge", "lasso"),
    ("ridge", "elastic_net"),
    ("lasso", "elastic_net"),
)
N_REPLICATES = 5_000
PRIMARY_BLOCK_DAYS = 5
BLOCK_LENGTHS = (5, 10, 20)
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42


def load_comparison(path: Path) -> pd.DataFrame:
    """Load and validate finite, aligned predictions for the 2004-2020 evaluation.

    Timestamps are timezone-naive exchange-local times, not UTC. Their calendar
    dates identify the trading days used for block resampling.
    """
    comparison = pd.read_csv(
        path,
        parse_dates=["target_timestamp"],
        index_col="target_timestamp",
    ).sort_index()

    required_columns = [
        "actual_return",
        *(f"{model}_prediction" for model in MODELS),
    ]

    missing_columns = set(required_columns) - set(comparison.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if comparison.empty:
        raise ValueError("Comparison table is empty.")

    if not isinstance(comparison.index, pd.DatetimeIndex):
        raise ValueError("target_timestamp was not parsed as datetime.")

    if comparison.index.hasnans:
        raise ValueError("Target timestamps contain missing values.")

    if not comparison.index.is_unique:
        raise ValueError("Target timestamps contain duplicates.")

    values = comparison[required_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Returns or predictions contain missing/infinite values.")

    expected_years = set(range(2004, 2021))
    if set(comparison.index.year) != expected_years:
        raise ValueError("Expected observations covering every year from 2004-2020.")

    return comparison


def aggregate_daily_losses(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return daily observation counts, squared-target sums and model error sums.

    Sums preserve pooled R2 when trading days have different observation counts;
    daily mean losses or daily R2 values would change those weights.
    """
    # One row per forecast interval.
    interval_losses = pd.DataFrame(index=comparison.index)
    interval_losses["n_obs"] = 1

    actual = comparison["actual_return"]
    interval_losses["target_sum_squares"] = actual**2

    for model in MODELS:
        prediction = comparison[f"{model}_prediction"]
        interval_losses[f"{model}_sse"] = (actual - prediction) ** 2

    trading_dates = comparison.index.normalize()

    daily_losses = interval_losses.groupby(trading_dates).sum().sort_index()
    daily_losses.index.name = "date"

    # Check that aggregation preserved every observation.
    if daily_losses["n_obs"].sum() != len(comparison):
        raise ValueError("Daily aggregation lost observations.")

    if (daily_losses["n_obs"] <= 0).any():
        raise ValueError("Each trading day must contain observations.")

    values = daily_losses.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Daily losses contain invalid values.")

    if (values < 0).any():
        raise ValueError("Daily counts and squared sums must be nonnegative.")

    return daily_losses


def compute_statistics(daily_losses: pd.DataFrame) -> pd.Series:
    """Return three pooled R2 values and three paired R2 differences in decimal units.

    Repeated rows in resampled daily tables retain their full weight. The
    denominator is the combined squared-target sum for those same sampled rows.
    """
    denominator = daily_losses["target_sum_squares"].sum()

    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("The sum of squared actual returns must be finite and positive.")

    statistics = {}

    # Calculate each model's pooled R2.
    for model in MODELS:
        total_sse = daily_losses[f"{model}_sse"].sum()

        if not np.isfinite(total_sse) or total_sse < 0:
            raise ValueError(f"{model}: invalid total squared error.")

        statistics[f"{model}_r2"] = 1 - total_sse / denominator

    # Calculate paired differences.
    for model_a, model_b in MODEL_PAIRS:
        statistics[f"{model_a}_minus_{model_b}"] = (
            statistics[f"{model_a}_r2"] - statistics[f"{model_b}_r2"]
        )

    return pd.Series(statistics, dtype=float)


def sample_block_indices(
    n_days: int,
    block_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample exactly n_days positions in circular blocks of observed trading days.

    Starting positions are drawn with replacement. Blocks wrap at the final day;
    concatenated positions are truncated to n_days. Duplicates are intentional.
    Wrapping is a resampling convention, not an actual historical adjacency.
    """
    if (
        isinstance(n_days, (bool, np.bool_))
        or not isinstance(n_days, (int, np.integer))
        or n_days <= 0
    ):
        raise ValueError("n_days must be a positive integer.")

    if (
        isinstance(block_days, (bool, np.bool_))
        or not isinstance(block_days, (int, np.integer))
        or not 1 <= block_days <= n_days
    ):
        raise ValueError("block_days must be an integer between 1 and n_days.")

    # Enough blocks to obtain at least n_days positions.
    n_blocks = (n_days + block_days - 1) // block_days

    # Draw block starts with replacement.
    starts = rng.integers(
        low=0,
        high=n_days,
        size=n_blocks,
    )

    # Positions relative to each block's start: 0, 1, ..., block_days - 1.
    offsets = np.arange(block_days)

    # Build consecutive blocks, wrapping around the end of the dataset.
    blocks = (starts[:, None] + offsets[None, :]) % n_days

    # Flatten the blocks and keep exactly n_days positions.
    return blocks.ravel()[:n_days]


def run_bootstrap(
    daily_losses: pd.DataFrame,
    block_days: int,
    n_replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return paired statistics for n_replicates draws at one block length.

    One position sample selects all models together. Positional indexing retains
    repeated days; membership filtering would incorrectly discard repetitions.
    """
    n_days = len(daily_losses)

    if n_days == 0:
        raise ValueError("daily_losses must not be empty.")

    if (
        isinstance(block_days, (bool, np.bool_))
        or not isinstance(block_days, (int, np.integer))
        or not 1 <= block_days <= n_days
    ):
        raise ValueError("block_days must be an integer between 1 and n_days.")

    if (
        isinstance(n_replicates, (bool, np.bool_))
        or not isinstance(n_replicates, (int, np.integer))
        or n_replicates < 2
    ):
        raise ValueError("n_replicates must be an integer of at least 2.")

    records = []

    for replicate in range(n_replicates):
        positions = sample_block_indices(
            n_days=n_days,
            block_days=block_days,
            rng=rng,
        )

        # Select the same days for all models, retaining repeated draws.
        sampled_losses = daily_losses.iloc[positions]

        statistics = compute_statistics(sampled_losses)

        if not np.isfinite(statistics.to_numpy()).all():
            raise ValueError(f"Replicate {replicate} produced nonfinite statistics.")

        records.append(
            {
                "block_days": block_days,
                "replicate": replicate,
                **statistics.to_dict(),
            }
        )

    return pd.DataFrame(records)


def summarize_intervals(
    point_estimates: pd.Series,
    bootstrap_draws: pd.DataFrame,
    confidence_level: float,
) -> pd.DataFrame:
    """Return original estimates, percentile bounds and bootstrap standard errors.

    Intervals for differences use the paired difference draws directly. These
    are pointwise intervals, without multiplicity or model-refitting adjustments.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    statistic_names = point_estimates.index.tolist()
    required = {"block_days", "replicate", *statistic_names}

    missing = required - set(bootstrap_draws.columns)
    if missing:
        raise ValueError(f"Missing bootstrap columns: {sorted(missing)}")

    if bootstrap_draws.empty:
        raise ValueError("Bootstrap draws are empty.")

    if not np.isfinite(point_estimates.to_numpy(dtype=float)).all():
        raise ValueError("Point estimates must be finite.")

    tail = (1 - confidence_level) / 2
    records = []

    for block_days, group in bootstrap_draws.groupby("block_days"):
        if len(group) < 2:
            raise ValueError("Each block length needs at least two draws.")

        for statistic in statistic_names:
            values = group[statistic].to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise ValueError(f"{statistic}: nonfinite bootstrap values.")

            lower, upper = np.quantile(values, [tail, 1 - tail])

            records.append(
                {
                    "statistic": statistic,
                    "block_days": int(block_days),
                    "estimate": float(point_estimates[statistic]),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "bootstrap_std_error": float(values.std(ddof=1)),
                    "n_replicates": len(values),
                    "confidence_level": confidence_level,
                }
            )

    return pd.DataFrame(records)


def plot_bootstrap_intervals(
    intervals: pd.DataFrame,
    output_dir: Path,
    block_days: int = PRIMARY_BLOCK_DAYS,
) -> None:
    """Save a two-panel pointwise CI figure from decimal-unit summaries.

    Dots are original-sample estimates; lines are percentile intervals. Plot
    endpoints directly because percentile intervals need not contain the dot.
    This function can also be called on saved confidence_intervals.csv without
    repeating the bootstrap or fitting any models.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    names = [f"{model}_r2" for model in MODELS] + [f"{a}_minus_{b}" for a, b in MODEL_PAIRS]
    selected = intervals.loc[intervals["block_days"] == block_days].copy()
    if selected["statistic"].duplicated().any() or set(selected["statistic"]) != set(names):
        raise ValueError("Expected one interval per model and pair for the chosen block length.")
    selected = selected.set_index("statistic").loc[names]
    columns = ["estimate", "ci_lower", "ci_upper"]
    if not np.isfinite(selected[columns].to_numpy(dtype=float)).all():
        raise ValueError("Plot estimates and interval endpoints must be finite.")
    if (selected["ci_lower"] > selected["ci_upper"]).any():
        raise ValueError("Interval lower bounds must not exceed upper bounds.")
    if selected["confidence_level"].nunique() != 1 or selected["n_replicates"].nunique() != 1:
        raise ValueError("All plotted intervals must share confidence level and repetition count.")

    confidence = float(selected["confidence_level"].iloc[0])
    repetitions = int(selected["n_replicates"].iloc[0])
    # Convert only the plotting copy; the stored summaries remain in decimal units.
    selected[columns] = selected[columns] * 100
    panels = (
        (
            names[:3],
            ["Ridge", "Lasso", "Elastic Net"],
            "Accuracy versus zero-return forecasts",
            "Out-of-sample $R^2$ (%)",
            "#246A92",
        ),
        (
            names[3:],
            ["Ridge − Lasso", "Ridge − Elastic Net", "Lasso − Elastic Net"],
            "Pairwise accuracy differences",
            "Difference in $R^2$ (percentage points)",
            "#96502A",
        ),
    )

    fig = Figure(figsize=(13, 5.4), facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.26, top=0.76, wspace=0.65)
    fig.suptitle("Bootstrap uncertainty in forecast accuracy", fontsize=18, y=0.96)
    fig.text(
        0.5,
        0.875,
        f"{block_days}-trading-day blocks · {confidence:.0%} pointwise percentile intervals",
        ha="center",
        fontsize=11,
        color="#555555",
    )

    for ax, (statistics, labels, title, xlabel, color) in zip(axes, panels, strict=True):
        values = selected.loc[statistics]
        positions = np.arange(len(statistics))
        ax.axvline(0, color="#777777", linestyle="--", linewidth=1.2, zorder=1)
        ax.hlines(
            positions, values["ci_lower"], values["ci_upper"], color=color, linewidth=2.3, zorder=3
        )
        ax.scatter(values["ci_lower"], positions, marker="|", s=100, color=color, zorder=3)
        ax.scatter(values["ci_upper"], positions, marker="|", s=100, color=color, zorder=3)
        ax.scatter(values["estimate"], positions, s=55, color=color, zorder=4)
        ax.set_yticks(positions, labels, fontsize=11)
        ax.set_ylim(2.65, -0.65)
        lower = min(0.0, float(values[columns].min().min()))
        upper = max(0.0, float(values[columns].max().max()))
        padding = 0.09 * (upper - lower) if upper > lower else 0.01
        ax.set_xlim(lower - padding, upper + padding)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_title(title, fontsize=12, pad=18, loc="left")
        ax.set_xlabel(xlabel, fontsize=11, labelpad=12)
        ax.grid(axis="x", color="#E8E8E8", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, pad=8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#BBBBBB")

    fig.text(
        0.10,
        0.105,
        "Dots: original-sample estimates. Lines: bootstrap intervals. Dashed line: zero.",
        fontsize=10,
        color="#555555",
    )
    fig.text(
        0.10,
        0.055,
        f"{repetitions:,} paired circular block resamples; conditional on saved forecasts; "
        "no model refitting.",
        fontsize=10,
        color="#555555",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"bootstrap_intervals_{block_days}day.{extension}", dpi=200)


def save_results(
    daily_losses: pd.DataFrame,
    point_estimates: pd.Series,
    bootstrap_draws: pd.DataFrame,
    intervals: pd.DataFrame,
    settings: dict[str, object],
    output_dir: Path,
) -> None:
    """Save daily sums, point estimates, bootstrap draws, intervals and run settings.

    Statistics remain decimals; multiplying by 100 gives R2 percentages or
    percentage-point differences between models.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_losses.to_csv(
        output_dir / "daily_losses.csv",
        index_label="date",
    )

    (
        point_estimates.rename_axis("statistic")
        .reset_index(name="estimate")
        .to_csv(output_dir / "point_estimates.csv", index=False)
    )

    bootstrap_draws.to_csv(
        output_dir / "bootstrap_draws.csv",
        index=False,
    )

    intervals.to_csv(
        output_dir / "confidence_intervals.csv",
        index=False,
    )

    with (output_dir / "settings.json").open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2, allow_nan=False)


def main() -> None:
    """Bootstrap the saved comparison at all configured block lengths.

    The generator advances across block lengths from one fixed seed. Original
    pooled metrics are checked before resampling; all configured intervals are
    reported rather than selecting the most favorable block length.
    """
    project_root = find_project_root(Path(__file__).resolve())
    results_root = project_root / "results" / "forecasting"

    comparison_dir = results_root / "comparison_2004_2020"
    input_path = comparison_dir / "predictions.csv"
    output_dir = results_root / "bootstrap_2004_2020"

    # Load and summarize the original evaluation sample.
    comparison = load_comparison(input_path)
    daily_losses = aggregate_daily_losses(comparison)

    # Catch incorrect grouping, including grouping by day-of-month.
    expected_dates = comparison.index.normalize().unique().sort_values()

    if not isinstance(daily_losses.index, pd.DatetimeIndex) or not daily_losses.index.equals(
        expected_dates
    ):
        raise ValueError("Daily aggregation must preserve full trading dates.")

    point_estimates = compute_statistics(daily_losses)

    # Verify that aggregation reproduces the original pooled R2.
    saved_metrics = pd.read_csv(
        comparison_dir / "pooled_metrics.csv",
        index_col="model",
    )

    for model in MODELS:
        np.testing.assert_allclose(
            point_estimates[f"{model}_r2"],
            saved_metrics.loc[f"{model}_prediction", "r2_oos"],
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"{model}: pooled R2 does not match.",
        )

    # Create the generator once and advance it across all runs.
    rng = np.random.default_rng(RANDOM_SEED)
    draw_frames = []

    for block_days in BLOCK_LENGTHS:
        print(
            f"Starting {N_REPLICATES:,} repetitions with {block_days}-day blocks...",
            flush=True,
        )

        draws = run_bootstrap(
            daily_losses=daily_losses,
            block_days=block_days,
            n_replicates=N_REPLICATES,
            rng=rng,
        )
        draw_frames.append(draws)

    bootstrap_draws = pd.concat(draw_frames, ignore_index=True)

    intervals = summarize_intervals(
        point_estimates=point_estimates,
        bootstrap_draws=bootstrap_draws,
        confidence_level=CONFIDENCE_LEVEL,
    )

    settings = {
        "method": "paired_circular_block_bootstrap",
        "input_path": str(input_path.resolve()),
        "seed": RANDOM_SEED,
        "block_lengths": list(BLOCK_LENGTHS),
        "primary_block_days": PRIMARY_BLOCK_DAYS,
        "n_replicates_per_block_length": N_REPLICATES,
        "confidence_level": CONFIDENCE_LEVEL,
        "start_date": str(daily_losses.index.min().date()),
        "end_date": str(daily_losses.index.max().date()),
        "n_days": len(daily_losses),
        "n_obs": int(daily_losses["n_obs"].sum()),
        "stored_units": "decimal R2 and decimal R2 differences",
        "interval_type": "pointwise percentile",
        "models_refitted": False,
    }

    save_results(
        daily_losses=daily_losses,
        point_estimates=point_estimates,
        bootstrap_draws=bootstrap_draws,
        intervals=intervals,
        settings=settings,
        output_dir=output_dir,
    )

    # Convert a copy for display; saved results remain in decimal units.
    plot_bootstrap_intervals(intervals, output_dir, block_days=PRIMARY_BLOCK_DAYS)

    display = intervals.copy()
    value_columns = ["estimate", "ci_lower", "ci_upper", "bootstrap_std_error"]
    display[value_columns] = display[value_columns] * 100

    display["unit"] = np.where(
        display["statistic"].str.contains("_minus_"),
        "percentage points",
        "%",
    )

    display_columns = [
        "statistic",
        "block_days",
        "estimate",
        "ci_lower",
        "ci_upper",
        "unit",
    ]

    for block_days in BLOCK_LENGTHS:
        label = "Primary" if block_days == PRIMARY_BLOCK_DAYS else "Sensitivity"
        print(f"\n{label}: {block_days}-day blocks")
        print(
            display.loc[
                display["block_days"] == block_days,
                display_columns,
            ].to_string(index=False, float_format="{:.6f}".format)
        )

    print(f"\nSaved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
