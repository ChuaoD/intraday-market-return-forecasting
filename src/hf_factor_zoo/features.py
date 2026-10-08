"""Feature construction for intraday market-return forecasting."""

from datetime import time

import numpy as np
import pandas as pd

DEFAULT_MARKET_COLUMN = "ff__mkt"
OVERNIGHT_SLOT = time(9, 30)
FEATURE_HORIZONS = ("lag15", "avg60", "prevday")


def remove_overnight_returns(
    returns_15m: pd.DataFrame,
    overnight_slot: time = OVERNIGHT_SLOT,
) -> pd.DataFrame:
    """Remove the 09:30 close-to-open return from each trading day."""
    return returns_15m.loc[returns_15m.index.time != overnight_slot]


def build_lag15_features(intraday_returns: pd.DataFrame) -> pd.DataFrame:
    """Use the previous observed intraday return as the 15-minute feature."""
    return intraday_returns.shift(1).add_prefix("lag15__")


def build_avg60_features(
    intraday_returns: pd.DataFrame,
    window: int = 4,
) -> pd.DataFrame:
    """Average the previous four intraday returns across the global sequence."""
    return (
        intraday_returns.shift(1)
        .rolling(window=window, min_periods=window)
        .mean()
        .add_prefix("avg60__")
    )


def build_previous_day_features(
    intraday_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Assign the previous trading day's intraday average to today's rows."""
    trading_dates = intraday_returns.index.normalize()
    daily_average = intraday_returns.groupby(trading_dates).mean()
    previous_daily_average = daily_average.shift(1)

    # Reindex repeats yesterday's one-row average for every target today.
    previous_day_features = previous_daily_average.reindex(trading_dates)
    previous_day_features.index = intraday_returns.index
    return previous_day_features.add_prefix("prevday__")


def build_forecast_panel(
    simple_returns_15m: pd.DataFrame,
    market_column: str = DEFAULT_MARKET_COLUMN,
    overnight_slot: time = OVERNIGHT_SLOT,
) -> pd.DataFrame:
    """Combine the target, timing fields, and 816 lagged predictors."""
    intraday_returns = remove_overnight_returns(
        simple_returns_15m,
        overnight_slot,
    )
    intraday_returns.index.name = "target_timestamp"

    lag15 = build_lag15_features(intraday_returns)
    avg60 = build_avg60_features(intraday_returns)
    prevday = build_previous_day_features(intraday_returns)
    forecast_panel = pd.concat([lag15, avg60, prevday], axis=1)

    forecast_panel[f"target__{market_column}"] = intraday_returns[market_column]
    forecast_panel["forecast_origin_timestamp"] = intraday_returns.index - pd.Timedelta(minutes=15)
    forecast_panel["trading_date"] = intraday_returns.index.normalize()

    minutes_after_open = (
        intraday_returns.index.hour * 60 + intraday_returns.index.minute - (9 * 60 + 30)
    )
    forecast_panel["slot_id"] = minutes_after_open // 15

    # Only the first trading day lacks a previous-day feature.
    return forecast_panel.dropna()


def validate_forecast_panel(
    forecast_panel: pd.DataFrame,
    expected_portfolios: int = 272,
    expected_horizons: int = 3,
    market_column: str = DEFAULT_MARKET_COLUMN,
) -> dict[str, object]:
    """Run the core structural and timing checks on the forecast panel.

    The function raises ValueError when a check fails and otherwise returns a
    small JSON-compatible summary for the metadata file.
    """
    if forecast_panel.empty:
        raise ValueError("Forecast panel is empty.")
    if not isinstance(forecast_panel.index, pd.DatetimeIndex):
        raise ValueError("Forecast panel must use a DatetimeIndex.")
    if forecast_panel.index.name != "target_timestamp":
        raise ValueError("Forecast panel index must be named target_timestamp.")
    if not forecast_panel.index.is_monotonic_increasing or not forecast_panel.index.is_unique:
        raise ValueError("Target timestamps must be sorted and unique.")
    if not forecast_panel.columns.is_unique:
        raise ValueError("Forecast panel contains duplicate columns.")

    target_column = f"target__{market_column}"
    required = {
        target_column,
        "forecast_origin_timestamp",
        "trading_date",
        "slot_id",
    }
    missing = required.difference(forecast_panel.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if expected_horizons != len(FEATURE_HORIZONS):
        raise ValueError(
            f"Expected {len(FEATURE_HORIZONS)} feature horizons, not {expected_horizons}."
        )

    # Every horizon must contain the same 272 portfolios in the same order.
    blocks: dict[str, list[str]] = {}
    for horizon in FEATURE_HORIZONS:
        prefix = f"{horizon}__"
        columns = [column for column in forecast_panel.columns if column.startswith(prefix)]
        if len(columns) != expected_portfolios:
            raise ValueError(
                f"{horizon} has {len(columns)} columns; expected {expected_portfolios}."
            )
        blocks[horizon] = [column.removeprefix(prefix) for column in columns]

    if any(blocks[horizon] != blocks["lag15"] for horizon in FEATURE_HORIZONS[1:]):
        raise ValueError("The three feature blocks use different portfolios.")

    feature_columns = [
        f"{horizon}__{portfolio}" for horizon in FEATURE_HORIZONS for portfolio in blocks["lag15"]
    ]
    if len(feature_columns) != expected_portfolios * expected_horizons:
        raise ValueError("Unexpected total number of feature columns.")

    values = forecast_panel[[target_column, *feature_columns]]
    if values.isna().any().any():
        raise ValueError("Target or feature values contain missing data.")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Target or feature values contain infinity.")

    targets = forecast_panel.index
    expected_origins = targets - pd.Timedelta(minutes=15)
    actual_origins = pd.DatetimeIndex(forecast_panel["forecast_origin_timestamp"])
    if not actual_origins.equals(expected_origins):
        raise ValueError("Forecast origins must be 15 minutes before targets.")

    expected_dates = targets.normalize()
    actual_dates = pd.DatetimeIndex(forecast_panel["trading_date"])
    if not actual_dates.equals(expected_dates):
        raise ValueError("trading_date does not match target_timestamp.")

    minutes_after_open = targets.hour * 60 + targets.minute - (9 * 60 + 30)
    expected_slots = np.asarray(minutes_after_open // 15)
    actual_slots = forecast_panel["slot_id"].to_numpy()
    if (
        expected_slots.min() < 1
        or expected_slots.max() > 26
        or not np.array_equal(actual_slots, expected_slots)
    ):
        raise ValueError("slot_id does not match the target time.")

    return {
        "passed": True,
        "rows": int(len(forecast_panel)),
        "feature_columns": int(len(feature_columns)),
        "first_target": targets[0].isoformat(sep=" "),
        "last_target": targets[-1].isoformat(sep=" "),
        "trading_days": int(expected_dates.nunique()),
        "missing_values": 0,
        "infinite_values": 0,
    }
