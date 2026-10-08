"""Feature timing: every predictor must use only returns observed before its target.

The tests use synthetic data, so they run without the proprietary inputs.
"""

import numpy as np
import pandas as pd
import pytest

from hf_factor_zoo import features
from tests.conftest import PORTFOLIOS, make_returns_15m

FEATURE_COLUMNS = [
    f"{horizon}__{portfolio}" for horizon in features.FEATURE_HORIZONS for portfolio in PORTFOLIOS
]


def build(returns: pd.DataFrame) -> pd.DataFrame:
    return features.build_forecast_panel(returns)


def test_remove_overnight_returns_drops_only_0930(returns_15m):
    intraday = features.remove_overnight_returns(returns_15m)

    assert (intraday.index.time != features.OVERNIGHT_SLOT).all()
    assert len(intraday) == len(returns_15m) - returns_15m.index.normalize().nunique()


@pytest.mark.parametrize(
    "target",
    [
        "2003-12-30 09:45",  # First usable day: every input comes from the previous day
        "2003-12-30 10:00",  # avg60 mixes three previous-day and one same-day return
        "2003-12-30 10:45",  # avg60 entirely within the day
        "2003-12-30 16:00",  # Last interval of the day
        "2004-01-02 09:45",  # Year boundary
    ],
)
def test_features_match_manual_reconstruction(returns_15m, target):
    panel = build(returns_15m)
    intraday = features.remove_overnight_returns(returns_15m)
    target = pd.Timestamp(target)
    position = intraday.index.get_loc(target)

    previous_date = intraday.index.normalize().unique()
    previous_date = previous_date[previous_date < target.normalize()].max()
    previous_day = intraday.loc[intraday.index.normalize() == previous_date]

    for portfolio in PORTFOLIOS:
        assert panel.at[target, f"lag15__{portfolio}"] == intraday[portfolio].iloc[position - 1]
        np.testing.assert_allclose(
            panel.at[target, f"avg60__{portfolio}"],
            intraday[portfolio].iloc[position - 4 : position].mean(),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            panel.at[target, f"prevday__{portfolio}"],
            previous_day[portfolio].mean(),
            rtol=1e-12,
        )

    assert panel.at[target, "target__ff__mkt"] == intraday.at[target, "ff__mkt"]


@pytest.mark.parametrize("cut", ["2003-12-31 09:45", "2003-12-31 12:30", "2004-01-02 16:00"])
def test_changing_current_or_future_returns_never_changes_earlier_features(cut):
    """Perturbation test for look-ahead bias.

    Replace every return from `cut` onward (including overnight rows) with new
    random values. Features of every target up to and including `cut` must be
    unchanged, because they may depend only on returns before their target.
    """
    original = make_returns_15m(seed=0)
    perturbed = original.copy()
    later = perturbed.index >= pd.Timestamp(cut)
    perturbed.loc[later] = make_returns_15m(seed=99).loc[later]

    before = build(original)
    after = build(perturbed)
    up_to_cut = before.index <= pd.Timestamp(cut)

    pd.testing.assert_frame_equal(
        before.loc[up_to_cut, FEATURE_COLUMNS],
        after.loc[up_to_cut, FEATURE_COLUMNS],
    )
    # The target at `cut` itself did change, so the test is not vacuous.
    assert (
        before.at[pd.Timestamp(cut), "target__ff__mkt"]
        != after.at[pd.Timestamp(cut), "target__ff__mkt"]
    )


def test_overnight_returns_are_never_used(returns_15m):
    modified = returns_15m.copy()
    overnight = modified.index.time == features.OVERNIGHT_SLOT
    modified.loc[overnight] = 0.5

    pd.testing.assert_frame_equal(build(returns_15m), build(modified))


def test_panel_timing_columns_and_initialization(returns_15m):
    panel = build(returns_15m)

    # The first day has no previous-day average, so it is dropped entirely.
    assert panel.index[0] == pd.Timestamp("2003-12-30 09:45")
    assert len(panel) == (returns_15m.index.normalize().nunique() - 1) * 26
    assert (panel["forecast_origin_timestamp"] == panel.index - pd.Timedelta(minutes=15)).all()
    assert panel["slot_id"].between(1, 26).all()
    assert not panel.isna().any().any()


def test_validate_forecast_panel_accepts_a_correct_panel(returns_15m):
    summary = features.validate_forecast_panel(
        build(returns_15m), expected_portfolios=len(PORTFOLIOS)
    )

    assert summary["passed"] is True
    assert summary["feature_columns"] == 3 * len(PORTFOLIOS)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda panel: panel.assign(
            forecast_origin_timestamp=panel["forecast_origin_timestamp"] + pd.Timedelta(minutes=1)
        ),
        lambda panel: panel.assign(**{"lag15__ff__mkt": np.nan}),
        lambda panel: panel.iloc[::-1],
        lambda panel: panel.drop(columns="prevday__cz__accruals"),
    ],
    ids=["late_origin", "missing_value", "unsorted", "missing_feature"],
)
def test_validate_forecast_panel_rejects_corrupted_panels(returns_15m, corrupt):
    panel = corrupt(build(returns_15m))
    with pytest.raises(ValueError):
        features.validate_forecast_panel(panel, expected_portfolios=len(PORTFOLIOS))
