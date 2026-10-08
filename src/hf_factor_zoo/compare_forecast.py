"""Compare aligned, saved Ridge, Lasso and Elastic Net forecasts.

Produces pooled and annual accuracy, forecast similarity and an annual plot.
This module evaluates existing predictions without fitting models.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hf_factor_zoo.data_processing import find_project_root
from hf_factor_zoo.forecasting import compute_metrics

MODELS = ("ridge", "lasso", "elastic_net")
FIRST_TEST_YEAR = 2004
LAST_TEST_YEAR = 2020
PREDICTION_COLUMNS = tuple(f"{model}_prediction" for model in MODELS)
SHARED_COLUMNS = (
    "actual_return",
    "forecast_origin_timestamp",
    "test_year",
    "zero_prediction",
    "training_mean_prediction",
)
COMBINATIONS = ((0, 1), (0, 2), (1, 2))


def load_predictions(
    results_root: Path,
    first_year: int,
    last_year: int,
) -> dict[str, pd.DataFrame]:
    """Load pooled predictions and verify shared targets, timestamps and baselines.

    Each model must predict exactly the same rows. Missing observations are not
    dropped to force an intersection between mismatched files.
    """
    frames = {}

    for model in MODELS:
        path = results_root / f"{model}_{first_year}_{last_year}" / "predictions.csv"

        frame = pd.read_csv(
            path,
            parse_dates=["target_timestamp", "forecast_origin_timestamp"],
            index_col="target_timestamp",
        ).sort_index()

        required_columns = set(SHARED_COLUMNS) | {f"{model}_prediction"}
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{model}: missing columns {missing_columns}")
        frames[model] = frame

    reference = frames[MODELS[0]]

    for model, frame in frames.items():
        # All models must forecast the same observations exactly once.
        assert frame.index.is_unique, f"{model}: duplicate timestamps"
        assert frame.index.equals(reference.index), f"{model}: target timestamps differ"

        # Compare datetime and integer columns exactly.
        for column in ("forecast_origin_timestamp", "test_year"):
            pd.testing.assert_series_equal(
                frame[column],
                reference[column],
                obj=f"{model}: {column}",
            )

        # Compare floating-point columns with a tight tolerance.
        for column in (
            "actual_return",
            "zero_prediction",
            "training_mean_prediction",
        ):
            np.testing.assert_allclose(
                frame[column].to_numpy(),
                reference[column].to_numpy(),
                rtol=0,
                atol=1e-12,
                equal_nan=False,
                err_msg=f"{model}: {column} differs",
            )
    return frames


def build_comparison(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine prediction columns on the common index verified by load_predictions."""
    reference = frames[MODELS[0]]
    comparison = reference[list(SHARED_COLUMNS)].copy()

    for model, frame in frames.items():
        column = f"{model}_prediction"
        comparison[column] = frame[column]
    return comparison


def evaluate_comparison(
    comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return pooled and annual metrics, including each common baseline once.

    Pooled R2 uses the combined squared-error sums, not the mean of annual R2.
    """

    pooled_metrics = pd.concat(
        [compute_metrics(comparison, model_name=model) for model in MODELS],
        ignore_index=True,
    ).drop_duplicates(subset="model")

    annual_records = []

    for year, year_frame in comparison.groupby("test_year"):
        for model in MODELS:
            metrics = compute_metrics(year_frame, model_name=model)
            annual_records.append(metrics.assign(test_year=year))

    annual_metrics = pd.concat(
        annual_records,
        ignore_index=True,
    ).drop_duplicates(subset=["test_year", "model"])

    return pooled_metrics, annual_metrics


def compare_prediction_values(
    comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return forecast correlations and pairwise RMS differences in return units and bps.

    These describe similarity between forecasts, not statistical significance.
    """
    prediction_columns = [f"{model}_prediction" for model in MODELS]
    forecast_correlations = comparison[prediction_columns].corr()
    prediction_differences = []
    for i, j in COMBINATIONS:
        model_i = MODELS[i]
        model_j = MODELS[j]
        prediction_i = comparison[f"{model_i}_prediction"]
        prediction_j = comparison[f"{model_j}_prediction"]
        rms_difference = np.sqrt(np.mean((prediction_i - prediction_j) ** 2))
        rms_difference_bps = 10_000 * rms_difference
        prediction_differences.append(
            {
                "model_i": model_i,
                "model_j": model_j,
                "rms_difference": rms_difference,
                "rms_difference_bps": rms_difference_bps,
            }
        )
    return forecast_correlations, pd.DataFrame(prediction_differences)


def plot_annual_r2(annual_metrics: pd.DataFrame, output_path: Path) -> None:
    """Save annual model R2 in percent against a zero-return reference."""
    labels = {
        "ridge": "Ridge",
        "lasso": "Lasso",
        "elastic_net": "Elastic Net",
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    for model in MODELS:
        # Select this model's annual results, ordered chronologically.
        model_metrics = annual_metrics.loc[
            annual_metrics["model"] == f"{model}_prediction"
        ].sort_values("test_year")

        ax.plot(
            model_metrics["test_year"],
            model_metrics["r2_oos_percent"],
            label=labels[model],
            marker="o",
            markersize=4,
            linewidth=1.5,
        )

    # Zero corresponds to the zero-return forecast benchmark.
    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    years = sorted(annual_metrics["test_year"].unique())

    ax.set_xticks(years)
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_xlabel("Test year")
    ax.set_ylabel(r"Out-of-sample $R^2$ (%)")
    ax.set_title("Annual forecasting performance")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_comparison(
    comparison: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    annual_metrics: pd.DataFrame,
    forecast_correlations: pd.DataFrame,
    prediction_differences: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save aligned predictions, summary tables and a plot in the comparison directory."""
    # Create the output directory if it does not exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the timestamp index for observation-level predictions.
    comparison.to_csv(
        output_dir / "predictions.csv",
        index_label="target_timestamp",
    )

    # These tables already contain their identifiers as columns.
    pooled_metrics.to_csv(
        output_dir / "pooled_metrics.csv",
        index=False,
    )

    annual_metrics.to_csv(
        output_dir / "annual_metrics.csv",
        index=False,
    )

    # Preserve row labels so each correlation is identifiable.
    forecast_correlations.to_csv(
        output_dir / "forecast_correlations.csv",
        index_label="model",
    )

    prediction_differences.to_csv(
        output_dir / "prediction_differences.csv",
        index=False,
    )

    plot_annual_r2(
        annual_metrics,
        output_dir / "annual_r2.png",
    )


def main() -> None:
    """Evaluate the saved 2004-2020 model runs and print their accuracy and similarity."""
    project_root = find_project_root(Path.cwd())
    results_root = project_root / "results" / "forecasting"
    output_dir = results_root / "comparison_2004_2020"
    frames = load_predictions(results_root, first_year=2004, last_year=2020)
    comparison = build_comparison(frames)
    pooled_metrics, annual_metrics = evaluate_comparison(comparison)
    forecast_correlations, prediction_differences = compare_prediction_values(comparison)
    save_comparison(
        comparison,
        pooled_metrics,
        annual_metrics,
        forecast_correlations,
        prediction_differences,
        output_dir,
    )

    print("\nPooled out-of-sample metrics:")
    print(
        pooled_metrics.to_string(
            index=False,
            formatters={
                "mse": "{:.6e}".format,
                "r2_oos": "{:.8f}".format,
                "r2_oos_percent": "{:.6f}".format,
            },
        )
    )

    print("\nForecast correlations:")
    print(
        forecast_correlations.to_string(
            float_format="{:.6f}".format,
        )
    )

    print("\nPairwise prediction differences:")
    print(
        prediction_differences.to_string(
            index=False,
            formatters={
                "rms_difference": "{:.6e}".format,
                "rms_difference_bps": "{:.6f}".format,
            },
        )
    )

    print(f"\nSaved comparison results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
