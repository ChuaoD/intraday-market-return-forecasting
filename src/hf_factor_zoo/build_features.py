"""Build the forecasting panel."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hf_factor_zoo import features
from hf_factor_zoo.data_processing import find_project_root


def load_canonical_returns(path: Path) -> pd.DataFrame:
    """Load the 15-minute panel and check the assumptions used below."""
    if not path.is_file():
        raise FileNotFoundError(f"Return panel not found: {path}")

    returns = pd.read_parquet(path)
    if returns.shape[1] != 272:
        raise ValueError(f"Expected 272 portfolios, found {returns.shape[1]}.")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Return panel must use a DatetimeIndex.")
    if not returns.index.is_monotonic_increasing or not returns.index.is_unique:
        raise ValueError("Return timestamps must be sorted and unique.")
    if returns.isna().any().any():
        raise ValueError("Return panel contains missing values.")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Return panel contains infinite values.")
    return returns


def build_metadata(
    source_path: Path,
    forecast_path: Path,
    source_panel: pd.DataFrame,
    forecast_panel: pd.DataFrame,
    validation: dict[str, object],
    market_column: str,
) -> dict[str, object]:
    """Describe how the saved forecasting panel was constructed."""
    overnight_rows = int((source_panel.index.time == features.OVERNIGHT_SLOT).sum())
    intraday_candidates = len(source_panel) - overnight_rows

    return {
        "source": {
            "file": source_path.name,
            "rows": len(source_panel),
            "columns": source_panel.shape[1],
        },
        "output": {
            "file": forecast_path.name,
            "rows": len(forecast_panel),
            "feature_columns": 816,
        },
        "return_type": "decimal simple return",
        "target": f"target__{market_column}",
        "features": {
            "lag15": "previous non-overnight 15-minute return",
            "avg60": "mean of previous four non-overnight returns",
            "prevday": "mean of previous trading day's non-overnight returns",
            "rolling_windows_cross_days": True,
        },
        "row_counts": {
            "source": len(source_panel),
            "overnight_removed": overnight_rows,
            "intraday_candidates": intraday_candidates,
            "initialization_rows_removed": (intraday_candidates - len(forecast_panel)),
            "forecast_panel": len(forecast_panel),
        },
        "validation": validation,
    }


def build_feature_artifacts(
    input_path: Path,
    forecast_path: Path,
    metadata_path: Path,
    market_column: str = features.DEFAULT_MARKET_COLUMN,
) -> dict[str, object]:
    """Build, validate, and save the feature panel."""
    simple_returns_15m = load_canonical_returns(input_path)

    # Restrict the history BEFORE computing any lags or daily averages.
    simple_returns_15m = simple_returns_15m.loc[
        (simple_returns_15m.index >= pd.Timestamp("2000-01-01"))
        & (simple_returns_15m.index < pd.Timestamp("2021-01-01"))
    ].copy()

    forecast_panel = features.build_forecast_panel(
        simple_returns_15m,
        market_column=market_column,
    )
    validation = features.validate_forecast_panel(
        forecast_panel,
        market_column=market_column,
    )

    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    forecast_panel.to_parquet(forecast_path)
    metadata = build_metadata(
        input_path,
        forecast_path,
        simple_returns_15m,
        forecast_panel,
        validation,
        market_column,
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    """Build the default artifacts from paths relative to the project root."""
    project_root = find_project_root(Path.cwd())
    metadata = build_feature_artifacts(
        input_path=(project_root / "data" / "processed" / "simple_returns_15min.parquet"),
        forecast_path=(project_root / "data" / "processed" / "forecast_panel.parquet"),
        metadata_path=(project_root / "data" / "processed" / "forecast_panel.metadata.json"),
    )

    print(
        f"Saved {metadata['output']['rows']:,} rows with "
        f"{metadata['output']['feature_columns']} features."
    )


if __name__ == "__main__":
    main()
