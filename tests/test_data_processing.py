"""Aggregation of one-minute returns into 15-minute returns."""

import numpy as np
import pandas as pd

from hf_factor_zoo.data_processing import resample_log_returns, resample_simple_returns


def one_minute_index(day: str, last: str = "10:00") -> pd.DatetimeIndex:
    """The 09:30 overnight row followed by one-minute rows 09:31 through `last`."""
    return pd.date_range(f"{day} 09:30", f"{day} {last}", freq="1min")


def test_bins_are_right_closed_and_labeled_by_interval_end():
    index = one_minute_index("2004-01-02")
    # Minute k after 09:30 has log return k, so bin sums are easy to check.
    log_returns = pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)

    result = resample_log_returns(log_returns)["x"]

    # (09:15, 09:30] holds only the overnight row; (09:30, 09:45] holds 09:31-09:45.
    assert result[pd.Timestamp("2004-01-02 09:30")] == 0
    assert result[pd.Timestamp("2004-01-02 09:45")] == sum(range(1, 16))
    assert result[pd.Timestamp("2004-01-02 10:00")] == sum(range(16, 31))
    assert len(result) == 3


def test_aggregation_never_crosses_trading_dates():
    first = one_minute_index("2004-01-02", last="16:00")
    second = one_minute_index("2004-01-05")
    index = first.append(second)
    log_returns = pd.DataFrame({"x": 1.0}, index=index)

    result = resample_log_returns(log_returns)["x"]

    # The next day's overnight row is its own observation, not part of 16:00.
    assert result[pd.Timestamp("2004-01-02 16:00")] == 15
    assert result[pd.Timestamp("2004-01-05 09:30")] == 1
    assert result.index.is_monotonic_increasing and result.index.is_unique


def test_simple_returns_compound_within_each_interval():
    index = one_minute_index("2004-01-02")
    rng = np.random.default_rng(1)
    simple = pd.DataFrame({"x": rng.normal(scale=0.001, size=len(index))}, index=index)

    result = resample_simple_returns(simple)["x"]

    minutes = simple.loc["2004-01-02 09:31":"2004-01-02 09:45", "x"]
    expected = np.prod(1 + minutes) - 1
    np.testing.assert_allclose(result[pd.Timestamp("2004-01-02 09:45")], expected, rtol=1e-12)
