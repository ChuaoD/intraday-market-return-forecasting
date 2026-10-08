"""Circular block sampling and daily-sum bootstrap statistics."""

import numpy as np
import pandas as pd
import pytest

from hf_factor_zoo import bootstrap_forecast as bootstrap
from tests.conftest import session_timestamps


def make_comparison(n_days: int = 30, seed: int = 0) -> pd.DataFrame:
    """Synthetic aligned predictions for the three models."""
    index = session_timestamps(pd.bdate_range("2004-01-02", periods=n_days), False)
    rng = np.random.default_rng(seed)
    actual = rng.normal(scale=0.001, size=len(index))
    frame = pd.DataFrame({"actual_return": actual}, index=index)
    for scale, model in zip((0.3, 0.2, 0.1), bootstrap.MODELS, strict=True):
        frame[f"{model}_prediction"] = scale * actual + rng.normal(scale=0.001, size=len(index))
    return frame


def test_blocks_are_consecutive_days_that_wrap_around():
    rng = np.random.default_rng(0)
    positions = bootstrap.sample_block_indices(n_days=23, block_days=5, rng=rng)

    assert len(positions) == 23
    assert positions.min() >= 0 and positions.max() < 23
    # Within each block, the next day follows the previous one modulo n_days.
    for block in np.split(positions[:20], 4):
        assert (np.diff(block) % 23 == 1).all()


@pytest.mark.parametrize(
    ("n_days", "block_days"),
    [(0, 1), (10, 0), (10, 11), (10, 2.5)],
)
def test_invalid_block_arguments_raise(n_days, block_days):
    with pytest.raises(ValueError):
        bootstrap.sample_block_indices(n_days, block_days, np.random.default_rng(0))


def test_daily_sums_reproduce_pooled_r2():
    comparison = make_comparison()
    daily = bootstrap.aggregate_daily_losses(comparison)

    statistics = bootstrap.compute_statistics(daily)

    y2 = (comparison["actual_return"] ** 2).sum()
    for model in bootstrap.MODELS:
        sse = ((comparison["actual_return"] - comparison[f"{model}_prediction"]) ** 2).sum()
        np.testing.assert_allclose(statistics[f"{model}_r2"], 1 - sse / y2, rtol=1e-12)
    np.testing.assert_allclose(
        statistics["ridge_minus_lasso"],
        statistics["ridge_r2"] - statistics["lasso_r2"],
    )


def test_bootstrap_is_reproducible_with_a_fixed_seed():
    daily = bootstrap.aggregate_daily_losses(make_comparison())

    first = bootstrap.run_bootstrap(daily, 5, 20, np.random.default_rng(42))
    second = bootstrap.run_bootstrap(daily, 5, 20, np.random.default_rng(42))

    pd.testing.assert_frame_equal(first, second)
