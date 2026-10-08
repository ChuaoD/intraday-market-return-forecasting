"""Trading rules, costs, bootstrap of daily sums and regime breakdowns."""

import numpy as np
import pandas as pd
import pytest

from hf_factor_zoo import strategy_evaluation as strategy
from tests.conftest import session_timestamps


def make_frame(
    first_day: str = "2008-01-02",
    n_days: int = 60,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic Full and market-only forecasts with the columns used by the module."""
    days = pd.bdate_range(first_day, periods=n_days)
    index = session_timestamps(days, include_overnight=False)
    rng = np.random.default_rng(seed)
    actual = rng.normal(scale=0.001, size=len(index))
    frame = pd.DataFrame(
        {
            "actual_return": actual,
            "test_year": index.year,
            "full_prediction": 0.2 * actual + rng.normal(scale=0.0005, size=len(index)),
            "market_only_prediction": 0.1 * actual + rng.normal(scale=0.0005, size=len(index)),
        },
        index=index,
    )
    frame["trading_date"] = frame.index.normalize()
    frame["slot_id"] = np.asarray(
        (frame.index.hour * 60 + frame.index.minute - (9 * 60 + 30)) // 15
    )
    return frame


def test_trailing_references_use_only_previous_days():
    prediction = make_frame()["full_prediction"]
    references = strategy.trailing_forecast_references(prediction)
    days = references.index

    # The first day has no history.
    assert references.iloc[0].isna().all()

    # Day 30's threshold equals the quantile of the previous 20 days' forecasts.
    day = days[30]
    window_days = days[30 - strategy.LOOKBACK_DAYS : 30]
    window = prediction.abs()[prediction.index.normalize().isin(window_days)]
    np.testing.assert_allclose(references.at[day, "q90"], np.quantile(window, 0.9))
    np.testing.assert_allclose(references.at[day, "mean_abs"], window.mean())


def test_changing_todays_forecasts_does_not_change_todays_threshold():
    prediction = make_frame()["full_prediction"]
    days = prediction.index.normalize().unique()
    perturbed = prediction.copy()
    perturbed[prediction.index.normalize() >= days[30]] *= 10

    before = strategy.trailing_forecast_references(prediction)
    after = strategy.trailing_forecast_references(perturbed)

    pd.testing.assert_frame_equal(before.loc[: days[30]], after.loc[: days[30]])
    assert not before.loc[days[31]].equals(after.loc[days[31]])


def test_ex_ante_volatility_ignores_the_same_day():
    returns = make_frame(n_days=15)["actual_return"]
    days = returns.index.normalize().unique()
    shocked = returns.copy()
    shocked[returns.index.normalize() == days[10]] *= 50

    before = strategy.ex_ante_volatility(returns)
    after = strategy.ex_ante_volatility(shocked)

    # Day 10 is unchanged; the shock first appears in day 11's estimate.
    assert before[days[10]] == after[days[10]]
    assert after[days[11]] > before[days[11]]
    assert before.iloc[: strategy.VOL_LOOKBACK_DAYS].isna().all()


def test_positions_follow_the_documented_rules():
    frame = make_frame()
    positions = strategy.build_positions(frame)

    assert list(positions.columns) == strategy.strategy_names()
    assert (positions["long_only"] == 1).all()
    np.testing.assert_array_equal(positions["full_sign"], np.sign(frame["full_prediction"]))
    for q in strategy.THRESHOLD_QUANTILES:
        column = positions[f"full_sign_q{round(100 * q)}"]
        assert column.isin([-1.0, 0.0, 1.0]).all()
    # Threshold and linear rules hold no position on the first day (no history).
    first_day = frame["trading_date"] == frame["trading_date"].iloc[0]
    assert (positions.loc[first_day, ["full_sign_q90", "full_linear"]] == 0).all().all()
    # Higher thresholds trade less often.
    assert (positions["full_sign_q90"] != 0).mean() < (positions["full_sign_q50"] != 0).mean()


def test_turnover_and_pnl_for_a_hand_worked_day():
    index = pd.DatetimeIndex(
        ["2008-01-02 09:45", "2008-01-02 10:00", "2008-01-02 10:15", "2008-01-02 10:30"]
    )
    frame = pd.DataFrame(
        {"actual_return": [0.001, -0.002, 0.003, 0.004], "trading_date": index.normalize()},
        index=index,
    )
    positions = pd.DataFrame({"rule": [1.0, 1.0, -1.0, 0.0]}, index=index)

    gross, turnover, exposure = strategy.strategy_daily_returns(frame, positions)

    # Open 1, hold 0, flip 2, close 1, nothing left to close at 16:00: 4 units.
    assert turnover["rule"].iloc[0] == 4
    np.testing.assert_allclose(gross["rule"].iloc[0], 0.001 - 0.002 - 0.003 + 0)
    assert exposure["rule"].iloc[0] == 3


def test_open_positions_are_closed_at_the_end_of_each_day():
    frame = make_frame(n_days=3)
    positions = pd.DataFrame({"rule": 1.0}, index=frame.index)

    _, turnover, _ = strategy.strategy_daily_returns(frame, positions)

    # Buy at the open, sell at the close: two units per day.
    assert (turnover["rule"] == 2).all()


@pytest.mark.parametrize("cost_bp", [0.0, 0.5, 2.0])
def test_sharpe_from_daily_moments_matches_direct_calculation(cost_bp):
    frame = make_frame()
    positions = strategy.build_positions(frame)
    gross, turnover, exposure = strategy.strategy_daily_returns(frame, positions)
    totals = strategy.strategy_daily_sums(gross, turnover, exposure).sum().to_frame().T
    totals["n_days"] = len(gross)

    annual_return, annual_vol, sharpe = strategy.strategy_performance(totals, "full_sign", cost_bp)

    net = gross["full_sign"] - cost_bp / 10_000 * turnover["full_sign"]
    expected = net.mean() / net.std(ddof=1) * np.sqrt(strategy.TRADING_DAYS_PER_YEAR)
    np.testing.assert_allclose(float(sharpe.iloc[0]), expected, rtol=1e-9)
    np.testing.assert_allclose(float(annual_return.iloc[0]), 100 * 252 * net.mean(), rtol=1e-9)


def test_resampled_totals_equal_sums_of_the_sampled_days():
    daily = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) ** 2})

    totals = strategy.resample_daily_totals(daily, 3, 4, np.random.default_rng(7))

    # Replay the same generator to recover the sampled days, repeats included.
    rng = np.random.default_rng(7)
    for replicate in range(4):
        positions = strategy.sample_block_indices(10, 3, rng)
        expected = daily.iloc[positions].sum()
        np.testing.assert_allclose(totals.iloc[replicate], expected)


def test_pipeline_breakdowns_add_up_to_the_pooled_difference():
    # One block of days in each sample period (2004-2007, 2008-2010, 2011-2020).
    frame = pd.concat(
        [
            make_frame("2005-01-03", 40, seed=1),
            make_frame("2008-01-02", 40, seed=2),
            make_frame("2011-01-03", 40, seed=3),
        ]
    )
    # Market history starts earlier so every test day has ex-ante volatility.
    history = pd.concat([make_frame("2004-11-01", 20, seed=4), frame])["actual_return"]

    vol_regime, _ = strategy.assign_volatility_regimes(frame, strategy.ex_ante_volatility(history))
    period = strategy.assign_periods(frame)
    positions = strategy.build_positions(frame)
    gross, turnover, exposure = strategy.strategy_daily_returns(frame, positions)
    daily = strategy.build_daily_sums(frame, vol_regime, period, gross, turnover, exposure)
    point = daily.sum().to_frame().T
    draws = {5: strategy.resample_daily_totals(daily, 5, 20, np.random.default_rng(0))}

    y2 = (frame["actual_return"] ** 2).sum()
    sse = {
        model: ((frame["actual_return"] - frame[f"{model}_prediction"]) ** 2).sum()
        for model in strategy.MODELS
    }
    pooled_difference = 100 * (sse["market_only"] - sse["full"]) / y2

    for statistics in (
        strategy.time_of_day_statistics,
        strategy.volatility_statistics,
        strategy.volatility_period_statistics,
    ):
        table = strategy.summarize(statistics, point, draws)
        contributions = table.loc[table["statistic"] == "contribution_pp", "estimate"]
        np.testing.assert_allclose(contributions.sum(), pooled_difference, rtol=1e-9)
        assert (table["ci_lower_5d"] <= table["ci_upper_5d"]).all()

    ceiling = strategy.summarize(strategy.sharpe_ceiling_statistics, point, draws)
    r2 = ceiling.loc[
        (ceiling["model"] == "full") & (ceiling["statistic"] == "r2_oos_percent"), "estimate"
    ]
    np.testing.assert_allclose(r2.iloc[0], 100 * (1 - sse["full"] / y2), rtol=1e-9)
