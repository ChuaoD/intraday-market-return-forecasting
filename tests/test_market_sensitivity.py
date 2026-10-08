"""Additive annual decomposition of pooled R2 differences."""

import numpy as np
import pandas as pd

from hf_factor_zoo.market_sensitivity import annual_contributions
from tests.conftest import session_timestamps


def test_annual_contributions_sum_to_the_pooled_difference():
    days = pd.bdate_range("2008-12-01", "2009-01-30")
    index = session_timestamps(days, include_overnight=False)
    rng = np.random.default_rng(0)
    actual = rng.normal(scale=0.001, size=len(index))
    comparison = pd.DataFrame(
        {"actual_return": actual, "test_year": index.year},
        index=index,
    )
    for name, scale in (
        ("full", 0.3),
        ("market_only", 0.1),
        ("market_original", 0.15),
        ("no_characteristics", 0.25),
    ):
        comparison[f"{name}_prediction"] = scale * actual + rng.normal(scale=0.001, size=len(index))

    result = annual_contributions(comparison)
    full_vs_market = result.loc[result["comparison"] == "full_minus_market_only"]

    def r2(column: str) -> float:
        sse = ((comparison["actual_return"] - comparison[column]) ** 2).sum()
        return 1 - sse / (comparison["actual_return"] ** 2).sum()

    pooled_difference_pp = 100 * (r2("full_prediction") - r2("market_only_prediction"))
    np.testing.assert_allclose(full_vs_market["contribution_pp"].sum(), pooled_difference_pp)
    assert set(full_vs_market["test_year"]) == {2008, 2009}
