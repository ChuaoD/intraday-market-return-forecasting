"""Aggregate 1-minute returns to 15-min returns."""

from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root(start: Path) -> Path:
    """Find the nearest ancestor containing the project configuration."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate

    raise FileNotFoundError("Could not locate project root containing pyproject.toml.")


def resample_log_returns(
    log_returns_1m: pd.DataFrame,
    frequency: str = "15min",
) -> pd.DataFrame:
    """Aggregate one-minute log returns into lower-frequency log returns."""
    # log_returns_1m: log(1 + r)
    return (
        log_returns_1m.assign(date=log_returns_1m.index.date)
        .groupby(
            [
                "date",
                pd.Grouper(
                    freq=frequency,
                    closed="right",
                    label="right",
                ),
            ]
        )
        .sum(min_count=1)
        .reset_index("date", drop=True)
    )


def resample_simple_returns(
    simple_returns_1m: pd.DataFrame,
    frequency: str = "15min",
) -> pd.DataFrame:
    """Aggregate one-minute simple returns into lower-frequency simple returns."""
    log_returns_1m = np.log1p(simple_returns_1m)
    log_returns_15m = resample_log_returns(log_returns_1m, frequency)
    simple_returns_15m = np.expm1(log_returns_15m)
    return simple_returns_15m


def get_resampled_month_log_returns(input_parquet):
    """Validate one monthly simple-return file and return 15-minute log returns."""
    returns_1m = pd.read_parquet(input_parquet)
    assert not returns_1m.isna().any().any(), f"{input_parquet} contains missing returns."
    ff6_group = returns_1m.columns[returns_1m.columns.str.startswith("ff__")]
    ind48_group = returns_1m.columns[returns_1m.columns.str.startswith("ind48__")]
    cha218_group = returns_1m.columns[returns_1m.columns.str.startswith(("cz__", "jkp__"))]
    assert [len(group) for group in (ff6_group, ind48_group, cha218_group)] == [6, 48, 218]
    log_returns_1m = returns_1m.apply(np.log1p)  # log returns
    log_returns_15m = resample_log_returns(log_returns_1m, "15min")
    return log_returns_15m


def save_resampled_month_simple_returns(input_parquet_list, output_path):
    """Combine monthly aggregates and convert log sums back to simple returns."""
    dfs = []
    for data_file in input_parquet_list:
        if data_file.is_file():
            log_returns = get_resampled_month_log_returns(data_file)
            dfs.append(log_returns)
    log_returns_15m = pd.concat(dfs, axis=0).sort_index()
    assert log_returns_15m.index.is_monotonic_increasing
    assert log_returns_15m.index.is_unique
    simple_returns_15m = np.expm1(log_returns_15m)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    simple_returns_15m.to_parquet(output_path)


def main():
    """Build the 15-minute simple-return panel from the monthly source files."""
    PROJECT_ROOT = find_project_root(Path.cwd())
    RETURNS_DATA_DIR = PROJECT_ROOT / "data" / "simple_returns_1min"
    monthly_files = sorted(RETURNS_DATA_DIR.glob("*.parquet"))

    output_path = PROJECT_ROOT / "data" / "processed" / "simple_returns_15min.parquet"

    save_resampled_month_simple_returns(monthly_files, output_path)


if __name__ == "__main__":
    main()
