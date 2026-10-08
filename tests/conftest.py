"""Synthetic data shared by the tests.

The real input data cannot be redistributed, so every test builds small panels
with the same clock as the supplied data: one 09:30 overnight row followed by
26 regular-session 15-minute intervals (09:45, ..., 16:00) per trading day.
"""

import numpy as np
import pandas as pd
import pytest

PORTFOLIOS = ("ff__mkt", "ind48__banks", "cz__accruals")


def session_timestamps(
    days: pd.DatetimeIndex,
    include_overnight: bool = True,
) -> pd.DatetimeIndex:
    """Return 09:30 (optional) and 09:45-16:00 timestamps for every day."""
    slots = [pd.Timedelta(hours=9, minutes=30 + 15 * k) for k in range(1, 27)]
    if include_overnight:
        slots = [pd.Timedelta(hours=9, minutes=30), *slots]
    return pd.DatetimeIndex([day + slot for day in days for slot in slots])


def make_returns_15m(
    n_days: int = 6,
    start: str = "2003-12-29",
    portfolios: tuple[str, ...] = PORTFOLIOS,
    seed: int = 0,
) -> pd.DataFrame:
    """Random 15-minute simple returns, including the 09:30 overnight row."""
    days = pd.bdate_range(start, periods=n_days)
    index = session_timestamps(days)
    rng = np.random.default_rng(seed)
    values = rng.normal(scale=0.001, size=(len(index), len(portfolios)))
    return pd.DataFrame(values, index=index, columns=list(portfolios))


@pytest.fixture
def returns_15m() -> pd.DataFrame:
    return make_returns_15m()
