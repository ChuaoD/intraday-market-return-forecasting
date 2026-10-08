"""Trading evaluation of the saved Full and market-only Ridge forecasts.

Run after market_sensitivity has completed for 2004-2020:
    uv run python -m hf_factor_zoo.strategy_evaluation

Covers README Section 6, without refitting any model:
    6.1 Realized returns by forecast decile, and hit rates.
    6.2 A sign rule's P&L, Sharpe ratio and break-even trading cost.
    6.3 Rules that trade only strong forecasts.
    6.4 Realized Sharpe ratios against the ceiling implied by R2.
    6.5 Full's advantage over market-only by volatility regime and time of day.

The traded asset is the market portfolio, a proxy for SPY or index futures.
Positions are held from each forecast time to the end of its 15-minute
interval and closed at the end of every day. Fills are assumed at the forecast
time with no delay.

Every statistic is computed from daily sums. The point estimates and all
bootstrap draws (paired circular blocks of trading days) therefore share the
same code and the same resampled days.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from hf_factor_zoo.bootstrap_forecast import sample_block_indices
from hf_factor_zoo.data_processing import find_project_root

FIRST_TEST_YEAR = 2004
LAST_TEST_YEAR = 2020
TARGET_COLUMN = "target__ff__mkt"

# Full Ridge and the expanded-grid market-only Ridge from market_sensitivity.
MODELS = ("full", "market_only")
BENCHMARK = "market_only"
MODEL_LABELS = {
    "full": "Full (816 predictors)",
    "market_only": "Market only (expanded grid)",
}

TRADING_DAYS_PER_YEAR = 252
N_DECILES = 10

# Trading rules use only forecasts from the previous LOOKBACK_DAYS trading days.
LOOKBACK_DAYS = 20
THRESHOLD_QUANTILES = (0.5, 0.75, 0.9)
# One-way cost per unit of notional traded; a round trip costs twice this.
COST_GRID_BP = (0.0, 0.25, 0.5, 1.0, 2.0)

# Ex-ante volatility: average realized variance over the previous five days.
VOL_LOOKBACK_DAYS = 5
VOL_LABELS = ("low", "medium", "high")
PERIODS = {
    "2004-2007": (2004, 2007),
    "2008-2010": (2008, 2010),
    "2011-2020": (2011, 2020),
}

N_REPLICATES = 5_000
PRIMARY_BLOCK_DAYS = 5
BLOCK_LENGTHS = (5, 10, 20)
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

Statistics = Callable[[pd.DataFrame], pd.DataFrame]


def load_predictions(path: Path) -> pd.DataFrame:
    """Load aligned Full and market-only forecasts with timing fields.

    Adds trading_date and slot_id (1 = interval ending 09:45, 26 = ending 16:00).
    """
    comparison = pd.read_csv(
        path,
        parse_dates=["target_timestamp"],
        index_col="target_timestamp",
    ).sort_index()

    required_columns = [
        "actual_return",
        "test_year",
        *(f"{model}_prediction" for model in MODELS),
    ]
    missing_columns = set(required_columns) - set(comparison.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if not comparison.index.is_unique:
        raise ValueError("Target timestamps contain duplicates.")

    values = comparison[required_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Returns or predictions contain missing/infinite values.")

    expected_years = set(range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1))
    if set(comparison.index.year) != expected_years:
        raise ValueError("Expected observations covering every test year.")

    frame = comparison[required_columns].copy()
    frame["trading_date"] = frame.index.normalize()

    minutes_after_open = frame.index.hour * 60 + frame.index.minute - (9 * 60 + 30)
    frame["slot_id"] = np.asarray(minutes_after_open // 15)
    if frame["slot_id"].min() < 1 or frame["slot_id"].max() > 26:
        raise ValueError("Targets must be regular-session intervals.")

    return frame


def load_market_history(panel_path: Path, frame: pd.DataFrame) -> pd.Series:
    """Load intraday market returns from 2000 onward and check them against the targets.

    The test-period targets must equal the panel's market column; the earlier
    rows supply volatility history for the first test days.
    """
    panel = pd.read_parquet(panel_path, columns=[TARGET_COLUMN])
    market_returns = panel[TARGET_COLUMN].sort_index()

    aligned = market_returns.reindex(frame.index)
    if aligned.isna().any():
        raise ValueError("Panel market returns do not cover every test target.")

    np.testing.assert_allclose(
        aligned.to_numpy(),
        frame["actual_return"].to_numpy(),
        rtol=0,
        atol=1e-12,
        err_msg="Saved actual returns differ from the forecast panel.",
    )
    return market_returns


def assign_forecast_deciles(frame: pd.DataFrame, model: str) -> pd.Series:
    """Rank each forecast into deciles within its test year (1 = most negative).

    Ranking within years keeps every year equally represented in each decile,
    so the extreme deciles are not dominated by high-volatility years. Cutoffs
    use the year's forecasts, not its returns; this is a descriptive sort.
    """
    return frame.groupby("test_year")[f"{model}_prediction"].transform(
        lambda forecasts: pd.qcut(forecasts, N_DECILES, labels=False) + 1
    )


def ex_ante_volatility(market_returns: pd.Series) -> pd.Series:
    """Annualized market volatility known before each trading day opens.

    Daily realized variance is the sum of squared intraday 15-minute returns.
    Day d uses the mean over days d-5, ..., d-1, so it contains no day-d data.
    """
    trading_dates = market_returns.index.normalize()
    daily_rv = (market_returns**2).groupby(trading_dates).sum()

    prior_rv = daily_rv.shift(1).rolling(VOL_LOOKBACK_DAYS, min_periods=VOL_LOOKBACK_DAYS).mean()

    return np.sqrt(TRADING_DAYS_PER_YEAR * prior_rv).rename("ex_ante_vol")


def assign_volatility_regimes(
    frame: pd.DataFrame,
    volatility: pd.Series,
) -> tuple[pd.Series, np.ndarray]:
    """Label each test day low/medium/high by terciles of ex-ante volatility.

    The volatility measure is known at the open, but the tercile cutoffs use
    all test days. The split is descriptive, not an implementable filter.
    """
    test_dates = frame["trading_date"].unique()
    day_volatility = volatility.reindex(test_dates)
    if day_volatility.isna().any():
        raise ValueError("Ex-ante volatility is missing for some test days.")

    day_regime, cutoffs = pd.qcut(
        day_volatility,
        q=len(VOL_LABELS),
        labels=VOL_LABELS,
        retbins=True,
    )

    # Repeat each day's label on its intervals.
    regime = day_regime.astype(str).reindex(frame["trading_date"]).set_axis(frame.index)
    return regime.rename("vol_regime"), cutoffs


def assign_periods(frame: pd.DataFrame) -> pd.Series:
    """Label each observation with its sample period (before, during, after 2008-2010)."""
    period = pd.Series(index=frame.index, dtype=object, name="period")
    for label, (first_year, last_year) in PERIODS.items():
        period[frame["test_year"].between(first_year, last_year)] = label
    if period.isna().any():
        raise ValueError("Every test year must belong to a period.")
    return period


def trailing_forecast_references(prediction: pd.Series) -> pd.DataFrame:
    """Summarize |forecast| over the previous LOOKBACK_DAYS trading days, for each day.

    Returns the quantiles used as thresholds and the mean used to scale the
    linear rule. Only earlier days' forecasts are used, so every value is known
    before the day opens. The first test day has no history and gets NaN.
    """
    magnitude = prediction.abs().to_numpy()
    day_codes, trading_dates = pd.factorize(prediction.index.normalize())

    # Row position where each day starts; rows are sorted by time.
    day_starts = np.searchsorted(day_codes, np.arange(len(trading_dates)))

    records = []
    for day in range(len(trading_dates)):
        first_row = day_starts[max(0, day - LOOKBACK_DAYS)]
        window = magnitude[first_row : day_starts[day]]

        record = {"mean_abs": np.nan}
        record.update({f"q{round(100 * q)}": np.nan for q in THRESHOLD_QUANTILES})
        if len(window) > 0:
            record["mean_abs"] = window.mean()
            for q in THRESHOLD_QUANTILES:
                record[f"q{round(100 * q)}"] = np.quantile(window, q)
        records.append(record)

    return pd.DataFrame(records, index=trading_dates)


def build_positions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the position in the market for every strategy and interval.

    long_only:       +1 in every interval (benchmark).
    <model>_sign:    sign of the forecast, always in the market.
    <model>_sign_qN: sign of the forecast, but only when |forecast| exceeds
                     the Nth percentile of recent |forecast|; otherwise 0.
    <model>_linear:  forecast divided by the recent mean |forecast|, so the
                     average absolute position is about one.
    """
    positions = pd.DataFrame(index=frame.index)
    positions["long_only"] = 1.0

    for model in MODELS:
        prediction = frame[f"{model}_prediction"]
        references = trailing_forecast_references(prediction)
        # Repeat each day's reference values on that day's intervals.
        daily_reference = references.reindex(frame["trading_date"]).set_axis(frame.index)

        direction = np.sign(prediction)
        positions[f"{model}_sign"] = direction

        for q in THRESHOLD_QUANTILES:
            threshold = daily_reference[f"q{round(100 * q)}"]
            # A missing threshold (first day) compares False: no position.
            strong = prediction.abs() > threshold
            positions[f"{model}_sign_q{round(100 * q)}"] = direction.where(strong, 0.0)

        positions[f"{model}_linear"] = (prediction / daily_reference["mean_abs"]).fillna(0.0)

    if not np.isfinite(positions.to_numpy()).all():
        raise ValueError("Positions contain invalid values.")
    return positions


def strategy_names() -> list[str]:
    """Strategy columns in the order produced by build_positions."""
    names = ["long_only"]
    for model in MODELS:
        names.append(f"{model}_sign")
        names.extend(f"{model}_sign_q{round(100 * q)}" for q in THRESHOLD_QUANTILES)
        names.append(f"{model}_linear")
    return names


def grouped_daily_sums(
    values: pd.DataFrame,
    groups: list[pd.Series],
    prefix: str,
) -> pd.DataFrame:
    """Sum interval values by trading day and group; one column per group and value.

    Columns are named "<prefix>__<group>__<value>". A group that is absent on a
    day contributes zero to that day's sums.
    """
    trading_dates = values.index.normalize().rename("date")
    sums = values.groupby([trading_dates, *groups]).sum()

    group_levels = list(range(1, len(groups) + 1))
    wide = sums.unstack(group_levels, fill_value=0.0)
    wide.columns = [
        "__".join([prefix, *(str(level) for level in group), value])
        for value, *group in wide.columns
    ]
    return wide


def accuracy_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Interval count, squared target and each model's squared error."""
    actual = frame["actual_return"]
    values = pd.DataFrame(index=frame.index)
    values["n"] = 1.0
    values["y2"] = actual**2
    for model in MODELS:
        values[f"sse_{model}"] = (actual - frame[f"{model}_prediction"]) ** 2
    return values


def decile_daily_sums(frame: pd.DataFrame) -> pd.DataFrame:
    """Daily count, return sum and forecast sum within each model's forecast decile."""
    blocks = []
    for model in MODELS:
        values = pd.DataFrame(
            {
                "n": 1.0,
                "return": frame["actual_return"],
                "forecast": frame[f"{model}_prediction"],
            },
            index=frame.index,
        )
        deciles = assign_forecast_deciles(frame, model).rename("decile")
        blocks.append(grouped_daily_sums(values, [deciles], prefix=f"decile__{model}"))
    return pd.concat(blocks, axis=1)


def hit_daily_sums(frame: pd.DataFrame) -> pd.DataFrame:
    """Daily counts of correctly predicted signs, for each model and for always-long.

    Intervals with a zero return have no direction and are excluded.
    """
    actual = frame["actual_return"]
    eligible = actual != 0

    values = pd.DataFrame(index=frame.index)
    values["hit__long_only__eligible"] = eligible.astype(float)
    values["hit__long_only__correct"] = (actual > 0).astype(float)

    for model in MODELS:
        prediction = frame[f"{model}_prediction"]
        model_eligible = eligible & (prediction != 0)
        correct = model_eligible & (np.sign(prediction) == np.sign(actual))
        values[f"hit__{model}__eligible"] = model_eligible.astype(float)
        values[f"hit__{model}__correct"] = correct.astype(float)

    return values.groupby(frame["trading_date"]).sum()


def strategy_daily_returns(
    frame: pd.DataFrame,
    positions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return daily gross P&L, turnover and absolute exposure for every strategy.

    Gross P&L of an interval is position * realized return (decimal units of
    notional). Turnover is |position change|: the first interval of a day opens
    from flat, and the last position is closed at the end of the day.
    """
    trading_dates = frame["trading_date"]
    actual = frame["actual_return"]

    gross = positions.mul(actual, axis=0)

    # Position held in the previous interval of the same day; flat at the open.
    previous = positions.groupby(trading_dates).shift(1).fillna(0.0)
    trades = (positions - previous).abs()

    # Close every position after the last observed interval of each day.
    dates = trading_dates.to_numpy()
    is_last_of_day = np.append(dates[1:] != dates[:-1], True)
    trades = trades + positions.abs().mul(is_last_of_day, axis=0)

    daily_gross = gross.groupby(trading_dates).sum()
    daily_turnover = trades.groupby(trading_dates).sum()
    daily_exposure = positions.abs().groupby(trading_dates).sum()
    return daily_gross, daily_turnover, daily_exposure


def strategy_daily_sums(
    daily_gross: pd.DataFrame,
    daily_turnover: pd.DataFrame,
    daily_exposure: pd.DataFrame,
) -> pd.DataFrame:
    """Daily moments from which gross and net Sharpe ratios follow for any cost.

    Net daily P&L is pnl - c * turnover, so its sum of squares is
    sum(pnl^2) - 2c sum(pnl * turnover) + c^2 sum(turnover^2).
    """
    columns = {}
    for strategy in daily_gross.columns:
        pnl = daily_gross[strategy]
        turnover = daily_turnover[strategy]
        columns[f"strategy__{strategy}__pnl"] = pnl
        columns[f"strategy__{strategy}__pnl_sq"] = pnl**2
        columns[f"strategy__{strategy}__turnover"] = turnover
        columns[f"strategy__{strategy}__turnover_sq"] = turnover**2
        columns[f"strategy__{strategy}__pnl_x_turnover"] = pnl * turnover
        columns[f"strategy__{strategy}__exposure"] = daily_exposure[strategy]
    return pd.DataFrame(columns)


def build_daily_sums(
    frame: pd.DataFrame,
    vol_regime: pd.Series,
    period: pd.Series,
    daily_gross: pd.DataFrame,
    daily_turnover: pd.DataFrame,
    daily_exposure: pd.DataFrame,
) -> pd.DataFrame:
    """Combine every section's daily sums into one table indexed by trading date."""
    accuracy = accuracy_values(frame)
    slot = frame["slot_id"].rename("slot")

    blocks = [
        grouped_daily_sums(accuracy, [pd.Series("pooled", index=frame.index)], "all"),
        decile_daily_sums(frame),
        hit_daily_sums(frame),
        strategy_daily_sums(daily_gross, daily_turnover, daily_exposure),
        grouped_daily_sums(accuracy, [slot], "slot"),
        grouped_daily_sums(accuracy, [vol_regime], "vol"),
        grouped_daily_sums(accuracy, [period, vol_regime], "volperiod"),
    ]
    daily = pd.concat(blocks, axis=1).sort_index()
    daily.index.name = "date"
    daily["n_days"] = 1.0

    expected_dates = frame["trading_date"].unique()
    if not daily.index.equals(pd.DatetimeIndex(expected_dates)):
        raise ValueError("Daily sums must contain every test day exactly once.")
    if daily.isna().any().any() or not np.isfinite(daily.to_numpy()).all():
        raise ValueError("Daily sums contain invalid values.")
    if daily["all__pooled__n"].sum() != len(frame):
        raise ValueError("Daily aggregation lost observations.")
    return daily


def resample_daily_totals(
    daily: pd.DataFrame,
    block_days: int,
    n_replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return column totals of `daily` for each circular block resample.

    Each draw counts how often every day was sampled; totals are those counts
    times the daily sums, so repeated days keep their full weight.
    """
    values = daily.to_numpy(dtype=float)
    n_days = len(daily)
    totals = np.empty((n_replicates, values.shape[1]))

    for replicate in range(n_replicates):
        positions = sample_block_indices(n_days=n_days, block_days=block_days, rng=rng)
        day_counts = np.bincount(positions, minlength=n_days)
        totals[replicate] = day_counts @ values

    return pd.DataFrame(totals, columns=daily.columns)


def summarize(
    statistic_function: Statistics,
    point_totals: pd.DataFrame,
    draw_totals: dict[int, pd.DataFrame],
    confidence_level: float = CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Evaluate a statistic on the original sample and on every bootstrap draw.

    statistic_function maps totals (one row per sample) to statistics (one
    column per statistic, labelled by a MultiIndex). Returns one row per
    statistic with the estimate and percentile bounds for each block length.
    """
    tail = (1 - confidence_level) / 2
    estimate = statistic_function(point_totals).iloc[0]
    table = estimate.rename("estimate").to_frame()

    for block_days, totals in draw_totals.items():
        draws = statistic_function(totals)
        if not draws.columns.equals(estimate.index):
            raise ValueError("Bootstrap statistics do not match the point estimates.")
        table[f"ci_lower_{block_days}d"] = draws.quantile(tail).to_numpy()
        table[f"ci_upper_{block_days}d"] = draws.quantile(1 - tail).to_numpy()

    return table.reset_index()


def statistics_frame(columns: dict[tuple, pd.Series], names: list[str]) -> pd.DataFrame:
    """Assemble statistics keyed by tuples into a frame with named MultiIndex columns."""
    frame = pd.DataFrame(columns)
    frame.columns = pd.MultiIndex.from_tuples(list(columns), names=names)
    return frame


def decile_statistics(totals: pd.DataFrame) -> pd.DataFrame:
    """Mean forecast and mean realized return (bp) per decile, plus the top-minus-bottom spread.

    A rising pattern across deciles means larger forecasts are followed by
    larger realized returns on average.
    """
    columns = {}
    for model in MODELS:
        mean_return = {}
        for decile in range(1, N_DECILES + 1):
            prefix = f"decile__{model}__{decile}"
            n_obs = totals[f"{prefix}__n"]
            mean_return[decile] = 10_000 * totals[f"{prefix}__return"] / n_obs
            columns[(model, str(decile), "mean_forecast_bp")] = (
                10_000 * totals[f"{prefix}__forecast"] / n_obs
            )
            columns[(model, str(decile), "mean_return_bp")] = mean_return[decile]

        columns[(model, "10-1", "mean_return_bp")] = mean_return[N_DECILES] - mean_return[1]

    return statistics_frame(columns, names=["model", "decile", "statistic"])


def hit_rate_statistics(totals: pd.DataFrame) -> pd.DataFrame:
    """Share of intervals with the correct sign, and the edge over always-long.

    Always-long's hit rate is the share of positive returns: a model must beat
    it to show directional skill beyond the market's upward drift.
    """

    def hit_rate(name: str) -> pd.Series:
        return totals[f"hit__{name}__correct"] / totals[f"hit__{name}__eligible"]

    columns = {("long_only", "hit_rate_percent"): 100 * hit_rate("long_only")}
    for model in MODELS:
        columns[(model, "hit_rate_percent")] = 100 * hit_rate(model)
        columns[(model, "edge_over_long_only_pp")] = 100 * (hit_rate(model) - hit_rate("long_only"))

    return statistics_frame(columns, names=["model", "statistic"])


def strategy_performance(
    totals: pd.DataFrame,
    strategy: str,
    cost_bp: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Annualized net return (%), volatility (%) and Sharpe ratio from daily moments.

    Uses the sample variance of daily net P&L and sqrt(252) annualization.
    Returns are simple sums of daily P&L on unit notional, not compounded.
    """
    prefix = f"strategy__{strategy}"
    cost = cost_bp / 10_000
    n_days = totals["n_days"]

    net_sum = totals[f"{prefix}__pnl"] - cost * totals[f"{prefix}__turnover"]
    net_sum_sq = (
        totals[f"{prefix}__pnl_sq"]
        - 2 * cost * totals[f"{prefix}__pnl_x_turnover"]
        + cost**2 * totals[f"{prefix}__turnover_sq"]
    )
    mean = net_sum / n_days
    variance = (net_sum_sq / n_days - mean**2) * n_days / (n_days - 1)

    annual_return = 100 * TRADING_DAYS_PER_YEAR * mean
    annual_vol = 100 * np.sqrt(TRADING_DAYS_PER_YEAR * variance)
    return annual_return, annual_vol, annual_return / annual_vol


def strategy_summary_statistics(totals: pd.DataFrame) -> pd.DataFrame:
    """Gross performance, turnover, exposure and break-even cost for every strategy.

    Break-even cost = total gross P&L / total turnover: the one-way cost per
    unit traded (bp) at which net P&L is exactly zero.
    """
    columns = {}
    for strategy in strategy_names():
        prefix = f"strategy__{strategy}"
        annual_return, annual_vol, sharpe = strategy_performance(totals, strategy, cost_bp=0.0)

        columns[(strategy, "gross_annual_return_pct")] = annual_return
        columns[(strategy, "gross_annual_vol_pct")] = annual_vol
        columns[(strategy, "gross_sharpe")] = sharpe
        columns[(strategy, "turnover_per_day")] = totals[f"{prefix}__turnover"] / totals["n_days"]
        columns[(strategy, "mean_abs_position")] = (
            totals[f"{prefix}__exposure"] / totals["all__pooled__n"]
        )
        columns[(strategy, "break_even_cost_bp")] = (
            10_000 * totals[f"{prefix}__pnl"] / totals[f"{prefix}__turnover"]
        )

    return statistics_frame(columns, names=["strategy", "statistic"])


def cost_sensitivity_statistics(totals: pd.DataFrame) -> pd.DataFrame:
    """Net annual return, volatility and Sharpe ratio at each cost in COST_GRID_BP."""
    columns = {}
    for strategy in strategy_names():
        for cost_bp in COST_GRID_BP:
            annual_return, annual_vol, sharpe = strategy_performance(totals, strategy, cost_bp)
            columns[(strategy, cost_bp, "net_annual_return_pct")] = annual_return
            columns[(strategy, cost_bp, "net_annual_vol_pct")] = annual_vol
            columns[(strategy, cost_bp, "net_sharpe")] = sharpe

    return statistics_frame(columns, names=["strategy", "cost_bp", "statistic"])


def maximum_drawdown(daily_pnl: pd.DataFrame) -> pd.Series:
    """Largest peak-to-trough fall in cumulative P&L, in percent of unit notional."""
    cumulative = daily_pnl.cumsum()
    drawdown = cumulative.cummax().clip(lower=0) - cumulative
    return 100 * drawdown.max()


def sharpe_ceiling_statistics(totals: pd.DataFrame) -> pd.DataFrame:
    """Compare realized Sharpe ratios with the Campbell-Thompson ceiling implied by R2.

    If the forecast equals the conditional mean and conditional variance is
    constant, a position proportional to the forecast has per-interval Sharpe
    sqrt(R2 / (1 - R2)), where R2 is measured against a zero forecast. Scaling
    by sqrt(intervals per year) annualizes it. Realized forecasts are noisy and
    volatility varies, so this is an approximate upper reference, before costs.
    """
    intervals_per_year = TRADING_DAYS_PER_YEAR * totals["all__pooled__n"] / totals["n_days"]

    columns = {}
    for model in MODELS:
        r2 = 1 - totals[f"all__pooled__sse_{model}"] / totals["all__pooled__y2"]
        ceiling = np.sqrt(r2.clip(lower=0) / (1 - r2) * intervals_per_year)

        columns[(model, "r2_oos_percent")] = 100 * r2
        columns[(model, "sharpe_ceiling")] = ceiling
        columns[(model, "realized_sharpe_linear")] = strategy_performance(
            totals, f"{model}_linear", cost_bp=0.0
        )[2]
        columns[(model, "realized_sharpe_sign")] = strategy_performance(
            totals, f"{model}_sign", cost_bp=0.0
        )[2]

    return statistics_frame(columns, names=["model", "statistic"])


def group_accuracy_statistics(
    totals: pd.DataFrame,
    prefix: str,
    groups: list[tuple[str, ...]],
    names: list[str],
) -> pd.DataFrame:
    """Within-group R2 and additive contributions to the pooled R2 difference.

    contribution_pp = 100 * (SSE_benchmark - SSE_full) / pooled sum(y^2), so
    contributions over all groups sum to the pooled R2 difference. Within-group
    R2 divides by the group's own sum(y^2). target_energy_share is the group's
    share of pooled sum(y^2), i.e. its weight in pooled R2.
    """
    pooled_y2 = totals["all__pooled__y2"]
    columns = {}

    for group in groups:
        key = "__".join([prefix, *group])
        y2 = totals[f"{key}__y2"]
        sse_full = totals[f"{key}__sse_full"]
        sse_benchmark = totals[f"{key}__sse_{BENCHMARK}"]

        r2_full = 1 - sse_full / y2
        r2_benchmark = 1 - sse_benchmark / y2

        columns[(*group, "n_obs")] = totals[f"{key}__n"]
        columns[(*group, "target_energy_share_pct")] = 100 * y2 / pooled_y2
        columns[(*group, "r2_full_pct")] = 100 * r2_full
        columns[(*group, "r2_benchmark_pct")] = 100 * r2_benchmark
        columns[(*group, "r2_difference_pp")] = 100 * (r2_full - r2_benchmark)
        columns[(*group, "contribution_pp")] = 100 * (sse_benchmark - sse_full) / pooled_y2

    return statistics_frame(columns, names=[*names, "statistic"])


def time_of_day_statistics(totals: pd.DataFrame) -> pd.DataFrame:
    """Full versus market-only accuracy for each 15-minute slot of the session."""
    groups = [(str(slot),) for slot in range(1, 27)]
    return group_accuracy_statistics(totals, "slot", groups, names=["slot_id"])


def volatility_statistics(totals: pd.DataFrame) -> pd.DataFrame:
    """Full versus market-only accuracy by ex-ante volatility tercile."""
    groups = [(label,) for label in VOL_LABELS]
    return group_accuracy_statistics(totals, "vol", groups, names=["vol_regime"])


def volatility_period_statistics(totals: pd.DataFrame) -> pd.DataFrame:
    """Full versus market-only accuracy by sample period and volatility tercile.

    The key question: does the high-volatility advantage persist after 2010?
    """
    groups = [(period, label) for period in PERIODS for label in VOL_LABELS]
    return group_accuracy_statistics(totals, "volperiod", groups, names=["period", "vol_regime"])


def interval_bounds(table: pd.DataFrame, block_days: int = PRIMARY_BLOCK_DAYS) -> np.ndarray:
    """Error-bar half-widths (2 x n) from percentile bounds around the estimate."""
    lower = table["estimate"] - table[f"ci_lower_{block_days}d"]
    upper = table[f"ci_upper_{block_days}d"] - table["estimate"]
    return np.vstack([lower.to_numpy(), upper.to_numpy()])


def style_axis(ax) -> None:
    """Shared minimal axis styling."""
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save_figure(figure: Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    figure.clear()


def plot_deciles(deciles: pd.DataFrame, output_path: Path) -> None:
    """Mean realized return per forecast decile, with bootstrap intervals."""
    figure = Figure(figsize=(12, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, len(MODELS), sharey=True)

    for ax, model, color in zip(axes, MODELS, ("#246A92", "#d58d36"), strict=True):
        rows = deciles.loc[
            (deciles["model"] == model)
            & (deciles["decile"] != "10-1")
            & (deciles["statistic"] == "mean_return_bp")
        ]
        forecasts = deciles.loc[
            (deciles["model"] == model) & (deciles["statistic"] == "mean_forecast_bp")
        ]
        positions = np.arange(1, N_DECILES + 1)
        ax.bar(
            positions,
            rows["estimate"],
            yerr=interval_bounds(rows),
            color=color,
            alpha=0.85,
            capsize=3,
            label="Mean realized return",
        )
        ax.plot(
            positions,
            forecasts["estimate"],
            color="black",
            marker="o",
            markersize=4,
            linewidth=1,
            label="Mean forecast",
        )
        style_axis(ax)
        ax.set_xticks(positions)
        ax.set_xlabel("Forecast decile within test year (1 = most negative)")
        ax.set_title(MODEL_LABELS[model], loc="left")
    axes[0].set_ylabel("Next 15-minute market return (bp)")
    axes[0].legend(loc="upper left")
    figure.suptitle(
        f"Realized returns by forecast decile, 2004-2020 "
        f"({PRIMARY_BLOCK_DAYS}-day block bootstrap 95% intervals)"
    )
    save_figure(figure, output_path)


def plot_cumulative_pnl(daily_gross: pd.DataFrame, output_path: Path) -> None:
    """Cumulative gross P&L of the main strategies, in percent of unit notional."""
    lines = {
        "long_only": ("Long only (session)", "#999999", "-"),
        "full_sign": ("Full: sign", "#246A92", "-"),
        "full_sign_q90": ("Full: sign, top 10% |forecast|", "#439581", "-"),
        "full_linear": ("Full: linear", "#5477a4", "--"),
        "market_only_sign": ("Market only: sign", "#d58d36", "-"),
    }
    figure = Figure(figsize=(11, 5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()

    for strategy, (label, color, linestyle) in lines.items():
        cumulative = 100 * daily_gross[strategy].cumsum()
        ax.plot(cumulative.index, cumulative, label=label, color=color, linestyle=linestyle)

    style_axis(ax)
    ax.set_ylabel("Cumulative gross P&L (% of notional)")
    ax.set_title("Gross performance before transaction costs (sums of daily P&L)", loc="left")
    ax.legend()
    save_figure(figure, output_path)


def plot_cost_sensitivity(point_totals: pd.DataFrame, output_path: Path) -> None:
    """Net Sharpe ratio against one-way cost, for each trading rule (point estimates)."""
    costs = np.linspace(0, 2, 81)
    lines = {
        "full_sign": ("Full: sign (always in)", "#246A92", "-"),
        "full_sign_q50": ("Full: |forecast| > trailing median", "#5477a4", "-"),
        "full_sign_q75": ("Full: |forecast| > trailing 75th pct", "#439581", "-"),
        "full_sign_q90": ("Full: |forecast| > trailing 90th pct", "#2f6b3a", "-"),
        "full_linear": ("Full: linear", "#246A92", ":"),
        "market_only_sign": ("Market only: sign", "#d58d36", "--"),
        "long_only": ("Long only (session)", "#999999", "--"),
    }
    figure = Figure(figsize=(10, 5.5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()

    for strategy, (label, color, linestyle) in lines.items():
        sharpe = [
            float(strategy_performance(point_totals, strategy, cost)[2].iloc[0]) for cost in costs
        ]
        ax.plot(costs, sharpe, label=label, color=color, linestyle=linestyle)

    style_axis(ax)
    ax.set_xlabel("One-way cost per unit of notional traded (bp)")
    ax.set_ylabel("Annualized net Sharpe ratio")
    ax.set_title("How much trading cost can each rule absorb?", loc="left")
    ax.legend(fontsize=9)
    save_figure(figure, output_path)


def plot_time_of_day(slots: pd.DataFrame, output_path: Path) -> None:
    """Within-slot R2 for both models, and each slot's contribution to Full's advantage."""
    pivot = slots.set_index(["slot_id", "statistic"])
    slot_ids = [str(slot) for slot in range(1, 27)]
    positions = np.arange(1, 27)
    clock_labels = [
        (pd.Timestamp("2000-01-01 09:30") + pd.Timedelta(minutes=15 * slot)).strftime("%H:%M")
        for slot in positions
    ]

    figure = Figure(figsize=(12, 7), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1, sharex=True)

    for offset, statistic, label, color in (
        (-0.2, "r2_full_pct", MODEL_LABELS["full"], "#246A92"),
        (0.2, "r2_benchmark_pct", MODEL_LABELS[BENCHMARK], "#d58d36"),
    ):
        values = pivot.xs(statistic, level="statistic").loc[slot_ids, "estimate"]
        axes[0].bar(positions + offset, values, width=0.4, label=label, color=color)
    axes[0].set_ylabel("Within-slot $R^2$ (%)")
    axes[0].set_title("Forecast accuracy by time of day (interval end time)", loc="left")
    axes[0].legend()

    contributions = pivot.xs("contribution_pp", level="statistic").loc[slot_ids]
    axes[1].bar(
        positions,
        contributions["estimate"],
        yerr=interval_bounds(contributions),
        color="#439581",
        capsize=2,
    )
    axes[1].set_ylabel("Contribution to pooled $R^2$\ndifference (pp)")
    axes[1].set_title(
        "Full − market only; contributions sum to the pooled difference "
        f"({PRIMARY_BLOCK_DAYS}-day block 95% intervals)",
        loc="left",
    )

    for ax in axes:
        style_axis(ax)
    axes[1].set_xticks(positions, clock_labels, rotation=60)
    save_figure(figure, output_path)


def plot_volatility_regimes(
    regimes: pd.DataFrame,
    regime_periods: pd.DataFrame,
    output_path: Path,
) -> None:
    """Contribution of each volatility tercile to Full's advantage, overall and by period."""
    colors = {"low": "#9ecae1", "medium": "#5477a4", "high": "#08306b"}
    figure = Figure(figsize=(12, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2, sharey=True, width_ratios=[1, 2.2])

    pooled = regimes.loc[regimes["statistic"] == "contribution_pp"].set_index("vol_regime")
    pooled = pooled.loc[list(VOL_LABELS)]
    axes[0].bar(
        np.arange(len(VOL_LABELS)),
        pooled["estimate"],
        yerr=interval_bounds(pooled),
        color=[colors[label] for label in VOL_LABELS],
        capsize=3,
    )
    axes[0].set_xticks(np.arange(len(VOL_LABELS)), [f"{label} vol" for label in VOL_LABELS])
    axes[0].set_ylabel("Contribution to pooled $R^2$\ndifference (pp)")
    axes[0].set_title("All years", loc="left")

    by_period = regime_periods.loc[regime_periods["statistic"] == "contribution_pp"]
    width = 0.26
    for index, label in enumerate(VOL_LABELS):
        rows = (
            by_period.loc[by_period["vol_regime"] == label].set_index("period").loc[list(PERIODS)]
        )
        axes[1].bar(
            np.arange(len(PERIODS)) + (index - 1) * width,
            rows["estimate"],
            yerr=interval_bounds(rows),
            width=width,
            color=colors[label],
            capsize=2,
            label=f"{label} vol",
        )
    axes[1].set_xticks(np.arange(len(PERIODS)), list(PERIODS))
    axes[1].set_title("By sample period", loc="left")
    axes[1].legend()

    for ax in axes:
        style_axis(ax)
    figure.suptitle(
        "Where does Full beat market only? Terciles of ex-ante volatility "
        f"({PRIMARY_BLOCK_DAYS}-day block 95% intervals)"
    )
    save_figure(figure, output_path)


def print_table(title: str, table: pd.DataFrame, block_days: int = PRIMARY_BLOCK_DAYS) -> None:
    """Print estimates with the primary-block interval."""
    columns = [
        column for column in table.columns if not column.startswith(("ci_lower", "ci_upper"))
    ]
    columns += [f"ci_lower_{block_days}d", f"ci_upper_{block_days}d"]
    print(f"\n{title}")
    print(table[columns].to_string(index=False, float_format="{:.4f}".format))


def run_evaluation(project_root: Path, n_replicates: int = N_REPLICATES) -> Path:
    """Build daily sums, bootstrap them once, and save every section's tables and figures."""
    results_root = project_root / "results" / "forecasting"
    suffix = f"{FIRST_TEST_YEAR}_{LAST_TEST_YEAR}"
    input_path = results_root / f"market_sensitivity_{suffix}" / "predictions.csv"
    panel_path = project_root / "data" / "processed" / "forecast_panel.parquet"
    output_dir = results_root / f"strategy_{suffix}"

    # Inputs and labels known at each forecast origin.
    frame = load_predictions(input_path)
    market_returns = load_market_history(panel_path, frame)
    vol_regime, vol_cutoffs = assign_volatility_regimes(frame, ex_ante_volatility(market_returns))
    period = assign_periods(frame)
    positions = build_positions(frame)

    # Daily P&L of every strategy, then the combined table of daily sums.
    daily_gross, daily_turnover, daily_exposure = strategy_daily_returns(frame, positions)
    daily = build_daily_sums(frame, vol_regime, period, daily_gross, daily_turnover, daily_exposure)

    # The original sample is the special case where every day is drawn once.
    point_totals = daily.sum().to_frame().T

    # Check: pooled R2 from daily sums reproduces the saved full-sample metric.
    saved_pooled = pd.read_csv(
        input_path.parent / "pooled_metrics.csv",
        index_col="model",
    )
    for model in MODELS:
        r2 = 1 - point_totals[f"all__pooled__sse_{model}"] / point_totals["all__pooled__y2"]
        np.testing.assert_allclose(
            float(r2.iloc[0]),
            saved_pooled.loc[f"{model}_prediction", "r2_oos"],
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"{model}: pooled R2 does not match the saved metric.",
        )

    # One generator advances across block lengths, as in bootstrap_forecast.
    rng = np.random.default_rng(RANDOM_SEED)
    draw_totals = {}
    for block_days in BLOCK_LENGTHS:
        print(f"Resampling {n_replicates:,} draws with {block_days}-day blocks...", flush=True)
        draw_totals[block_days] = resample_daily_totals(daily, block_days, n_replicates, rng)

    # Every table below uses the same resampled days.
    deciles = summarize(decile_statistics, point_totals, draw_totals)
    hit_rates = summarize(hit_rate_statistics, point_totals, draw_totals)
    strategies = summarize(strategy_summary_statistics, point_totals, draw_totals)
    costs = summarize(cost_sensitivity_statistics, point_totals, draw_totals)
    ceiling = summarize(sharpe_ceiling_statistics, point_totals, draw_totals)
    slots = summarize(time_of_day_statistics, point_totals, draw_totals)
    regimes = summarize(volatility_statistics, point_totals, draw_totals)
    regime_periods = summarize(volatility_period_statistics, point_totals, draw_totals)

    # Path-dependent statistics have no daily-sum form; report point estimates only.
    drawdowns = maximum_drawdown(daily_gross).rename("estimate")
    strategies = pd.concat(
        [
            strategies,
            drawdowns.rename_axis("strategy")
            .reset_index()
            .assign(statistic="gross_max_drawdown_pct"),
        ],
        ignore_index=True,
    ).sort_values(["strategy", "statistic"], kind="stable")

    # Contributions must add up to the pooled difference within every breakdown.
    pooled_difference = 100 * float(
        (
            (point_totals[f"all__pooled__sse_{BENCHMARK}"] - point_totals["all__pooled__sse_full"])
            / point_totals["all__pooled__y2"]
        ).iloc[0]
    )
    for name, table in (("slots", slots), ("regimes", regimes), ("periods", regime_periods)):
        total = table.loc[table["statistic"] == "contribution_pp", "estimate"].sum()
        np.testing.assert_allclose(
            total, pooled_difference, rtol=1e-9, err_msg=f"{name}: contributions do not add up."
        )

    # Save tables, daily P&L and figures.
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "deciles.csv": deciles,
        "hit_rates.csv": hit_rates,
        "strategy_summary.csv": strategies,
        "cost_sensitivity.csv": costs,
        "sharpe_ceiling.csv": ceiling,
        "time_of_day.csv": slots,
        "volatility_regimes.csv": regimes,
        "volatility_regimes_by_period.csv": regime_periods,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)

    daily_output = pd.concat(
        [daily_gross.add_suffix("__gross_pnl"), daily_turnover.add_suffix("__turnover")],
        axis=1,
    )
    daily_output.to_csv(output_dir / "daily_pnl.csv", index_label="date")

    plot_deciles(deciles, output_dir / "forecast_deciles.png")
    plot_cumulative_pnl(daily_gross, output_dir / "cumulative_pnl.png")
    plot_cost_sensitivity(point_totals, output_dir / "cost_sensitivity.png")
    plot_time_of_day(slots, output_dir / "time_of_day.png")
    plot_volatility_regimes(regimes, regime_periods, output_dir / "volatility_regimes.png")

    settings = {
        "input_path": str(input_path.resolve()),
        "models": list(MODELS),
        "benchmark": BENCHMARK,
        "n_obs": len(frame),
        "n_days": len(daily),
        "trading_days_per_year": TRADING_DAYS_PER_YEAR,
        "lookback_days_for_thresholds_and_scaling": LOOKBACK_DAYS,
        "threshold_quantiles": list(THRESHOLD_QUANTILES),
        "cost_grid_bp_one_way": list(COST_GRID_BP),
        "vol_lookback_days": VOL_LOOKBACK_DAYS,
        "vol_tercile_cutoffs_annualized": [float(cutoff) for cutoff in vol_cutoffs[1:-1]],
        "periods": {label: list(years) for label, years in PERIODS.items()},
        "seed": RANDOM_SEED,
        "block_lengths": list(BLOCK_LENGTHS),
        "primary_block_days": PRIMARY_BLOCK_DAYS,
        "n_replicates_per_block_length": n_replicates,
        "confidence_level": CONFIDENCE_LEVEL,
        "models_refitted": False,
        "positions_closed_overnight": True,
        "execution": "fill at forecast origin, zero latency, cost per unit notional traded",
    }
    with (output_dir / "settings.json").open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2, allow_nan=False)

    # Console summary of the headline numbers.
    print_table(
        "Forecast deciles (Full, mean realized return in bp)",
        deciles.loc[(deciles["model"] == "full") & (deciles["statistic"] == "mean_return_bp")],
    )
    print_table("Hit rates", hit_rates)
    print_table(
        "Strategy summary",
        strategies.loc[
            strategies["statistic"].isin(["gross_sharpe", "break_even_cost_bp", "turnover_per_day"])
        ],
    )
    print_table("Sharpe ceiling implied by R2", ceiling)
    print_table(
        "Volatility terciles",
        regimes.loc[
            regimes["statistic"].isin(
                ["target_energy_share_pct", "r2_difference_pp", "contribution_pp"]
            )
        ],
    )
    print_table(
        "Volatility terciles by period",
        regime_periods.loc[regime_periods["statistic"] == "contribution_pp"],
    )
    print(f"\nPooled Full − market-only difference: {pooled_difference:.6f} pp")
    print(f"Saved strategy evaluation to: {output_dir.resolve()}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-replicates",
        type=int,
        default=N_REPLICATES,
        help="Bootstrap draws per block length (use a small value for a quick check).",
    )
    args = parser.parse_args()
    run_evaluation(find_project_root(Path.cwd()), n_replicates=args.n_replicates)


if __name__ == "__main__":
    main()
