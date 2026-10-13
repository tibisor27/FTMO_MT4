"""Tests for the post-analysis: bootstrap coverage, paired comparison, sizing, prop-firm rules."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.analysis import (  # noqa: E402
    bootstrap_sharpe, compare_variants, drawdown_table, prop_firm_pass,
    vol_target_strategy, yearly_table,
)


def series(mu, sd, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(rng.normal(mu, sd, n), index=idx, name="ret")


def test_bootstrap_interval_brackets_the_point_estimate_and_flags_noise():
    r = series(0.0, 0.01, 1500, seed=1)          # true Sharpe = 0
    b = bootstrap_sharpe(r, n_boot=2000)
    assert b["lo95"] < b["sharpe"] < b["hi95"]
    assert b["p_sharpe_le_0"] > 0.05             # cannot rule out zero

    good = series(0.0015, 0.01, 2500, seed=2)    # true Sharpe ~ 2.4
    bg = bootstrap_sharpe(good, n_boot=2000)
    assert bg["lo95"] > 0 and bg["p_sharpe_le_0"] < 0.01


def test_paired_comparison_detects_a_real_difference_and_ignores_a_shared_shock():
    rng = np.random.default_rng(3)
    n = 1500
    common = rng.normal(0, 0.01, n)              # the same market shock in both
    idx = pd.bdate_range("2018-01-01", periods=n)
    a = pd.Series(common + 0.0008, index=idx)    # a has the edge
    b = pd.Series(common, index=idx)
    c = compare_variants(a, b, n_boot=2000)
    assert c["diff"] > 0 and c["lo95"] > 0 and c["p_diff_le_0"] < 0.05
    # identical series -> no difference
    c0 = compare_variants(b, b.copy(), n_boot=500)
    assert abs(c0["diff"]) < 1e-9


def test_vol_target_strategy_hits_the_target_and_respects_the_cap():
    r = series(0.0005, 0.02, 1200, seed=4)
    s = vol_target_strategy(r, target_daily=0.005, lookback=20, max_leverage=4.0)
    s = s[s != 0]
    assert 0.003 < s.std(ddof=1) < 0.008          # roughly on target
    lev = (s / r.reindex(s.index)).replace([np.inf, -np.inf], np.nan).dropna()
    assert lev.max() <= 4.0 + 1e-9
    # the first `lookback` days cannot be sized
    assert (vol_target_strategy(r, 0.005, 20).iloc[:20] == 0).all()


def test_prop_firm_no_edge_is_near_the_geometric_odds_and_edge_helps():
    flat = series(0.0, 0.005, 1500, seed=5)
    p0 = prop_firm_pass(flat, target=0.06, trailing_dd=0.04, daily_loss=0.02,
                        scale_to_daily_vol=0.005, n_sims=4000)
    assert 0.10 < p0["p_pass"] < 0.45             # a coin flip weighted by 4/(4+6)

    edge = series(0.0007, 0.005, 1500, seed=6)    # Sharpe ~ 2.2
    p1 = prop_firm_pass(edge, target=0.06, trailing_dd=0.04, daily_loss=0.02,
                        scale_to_daily_vol=0.005, n_sims=4000)
    assert p1["p_pass"] > p0["p_pass"] + 0.15


def test_prop_firm_smaller_size_raises_pass_probability_for_a_real_edge():
    edge = series(0.0006, 0.006, 1500, seed=7)
    big = prop_firm_pass(edge, scale_to_daily_vol=0.02, n_sims=4000)
    small = prop_firm_pass(edge, scale_to_daily_vol=0.005, n_sims=4000)
    assert small["p_pass"] > big["p_pass"]


def test_drawdown_and_yearly_tables_are_consistent_with_the_series():
    idx = pd.bdate_range("2020-01-01", periods=260)
    r = pd.Series(0.001, index=idx)
    r.iloc[100:110] = -0.02                       # one clean drawdown episode
    dd = drawdown_table(r, top=3)
    assert len(dd) >= 1 and dd.iloc[0]["depth_pct"] < -15
    y = yearly_table(r)
    assert list(y.index) == [2020] and y.loc[2020, "days"] == 260
