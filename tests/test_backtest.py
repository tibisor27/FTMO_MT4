"""Tests for the backtest engine on fabricated bars: fills, stops, costs, sizing, no look-ahead."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import backtest, session_vwap  # noqa: E402
from src.events.noise_area import noise_area_bands, noise_area_signal  # noqa: E402
from tests.test_events import make_days  # noqa: E402

TZ = "America/New_York"


def _flat_signal(df, values: dict) -> pd.Series:
    """Signal series that is NaN everywhere except at the given positional indices."""
    s = pd.Series(np.nan, index=df.index)
    for i, v in values.items():
        s.iloc[i] = v
    return s


def test_entry_is_next_bar_open_and_exit_is_session_close():
    df = make_days(1, seed=10)
    o = df["open"].to_numpy().copy(); c = df["close"].to_numpy().copy()
    o[:] = 100.0; c[:] = 100.0
    o[31] = 100.0          # entry fill (bar after the signal at bar 30)
    c[-1] = 102.0          # session close
    df["open"], df["close"] = o, c
    res = backtest(df, _flat_signal(df, {30: 1.0}), exit_rule="close", cost_bps=0.0, vol_lookback=1)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t.t_in == df.index[31] and t.t_out == df.index[-1]
    assert t.price_in == 100.0 and t.price_out == 102.0
    assert t.exit_reason == "eod"
    assert abs(t.ret_gross - 0.02) < 1e-12
    assert abs(res.daily["ret"].iloc[0] - 0.02) < 1e-12


def test_short_side_and_cost_are_charged_per_round_trip():
    df = make_days(1, seed=11)
    o = df["open"].to_numpy().copy(); c = df["close"].to_numpy().copy()
    o[:] = 100.0; c[:] = 100.0; c[-1] = 99.0
    df["open"], df["close"] = o, c
    res = backtest(df, _flat_signal(df, {30: -1.0}), exit_rule="close", cost_bps=10.0, vol_lookback=1)
    t = res.trades.iloc[0]
    assert abs(t.ret_gross - 0.01) < 1e-12          # short gains when price falls
    assert abs(t.ret_net - (0.01 - 0.001)) < 1e-12  # 10 bps round trip


def test_stop_fills_on_the_next_bar_not_the_trigger_bar():
    df = make_days(1, seed=12)
    o = df["open"].to_numpy().copy(); c = df["close"].to_numpy().copy()
    o[:] = 100.0; c[:] = 100.0
    c[40] = 98.0     # trigger bar: below the lower band
    o[41] = 97.0     # the fill must happen here, worse than the trigger close
    df["open"], df["close"] = o, c
    bands = pd.DataFrame({"ub": 101.0, "lb": 99.0}, index=df.index)
    res = backtest(df, _flat_signal(df, {30: 1.0, 40: -1.0}), bands, exit_rule="opposite_band",
                   cost_bps=0.0, vol_lookback=1)
    stop = res.trades[res.trades["exit_reason"] == "stop"].iloc[0]
    assert stop.t_out == df.index[41] and stop.price_out == 97.0


def test_opposite_band_reverses_but_never_re_enters_the_same_side_in_one_bar():
    df = make_days(1, seed=13)
    o = df["open"].to_numpy().copy(); c = df["close"].to_numpy().copy()
    o[:] = 100.0; c[:] = 100.0; c[40] = 98.0
    df["open"], df["close"] = o, c
    bands = pd.DataFrame({"ub": 101.0, "lb": 99.0}, index=df.index)
    res = backtest(df, _flat_signal(df, {30: 1.0, 40: -1.0}), bands, exit_rule="opposite_band",
                   cost_bps=0.0, vol_lookback=1)
    assert list(res.trades["side"]) == [1, -1]          # long stopped out, flipped short
    assert list(res.trades["exit_reason"]) == ["stop", "eod"]

    # a stop that does NOT flip the side must leave us flat for that bar
    res2 = backtest(df, _flat_signal(df, {30: 1.0, 40: 1.0}), bands, exit_rule="current_band",
                    cost_bps=0.0, vol_lookback=1)
    assert len(res2.trades) == 1 and res2.trades.iloc[0].exit_reason == "stop"


def test_vol_targeting_scales_and_caps_leverage_without_look_ahead():
    df = make_days(40, seed=14)
    sig = pd.Series(np.nan, index=df.index)
    for day in pd.Index(df.index.tz_convert(TZ).date).unique():
        pos = np.flatnonzero(np.asarray(df.index.tz_convert(TZ).date) == day)
        sig.iloc[pos[30]] = 1.0
    res = backtest(df, sig, exit_rule="close", cost_bps=0.0, vol_target=0.02, vol_lookback=14, max_leverage=4.0)
    lev = res.daily.loc[res.daily["n_trades"] > 0, "leverage"]
    assert (lev <= 4.0 + 1e-9).all() and (lev > 0).all()
    # the first 14 sessions cannot be sized (no volatility history) -> no trades there
    assert res.daily.index.min() > pd.Index(df.index.tz_convert(TZ).date).unique()[13]


def test_flat_days_are_counted_in_the_sharpe():
    df = make_days(20, seed=15)
    sig = pd.Series(np.nan, index=df.index)
    pos = np.flatnonzero(np.asarray(df.index.tz_convert(TZ).date) == pd.Index(df.index.tz_convert(TZ).date).unique()[18])
    sig.iloc[pos[30]] = 1.0
    res = backtest(df, sig, exit_rule="close", cost_bps=0.0, vol_lookback=1)
    assert res.stats["trades"] == 1
    assert res.stats["days"] > 1                      # flat days are in the daily series
    assert res.stats["days_traded_pct"] < 20


def test_vwap_is_a_session_cumulative_average():
    df = make_days(2, seed=16)
    dates = np.asarray(df.index.tz_convert(TZ).date)
    v = session_vwap(df, dates)
    tp = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy()
    assert abs(v[0] - tp[0]) < 1e-9                    # first bar of a session
    assert abs(v[390] - tp[390]) < 1e-9                # resets on the next session


def test_noise_area_signal_matches_its_bands_and_has_no_lookahead():
    df = make_days(20, seed=17)
    sig = noise_area_signal(df, lookback=14)
    bands = noise_area_bands(df, lookback=14)
    dec = sig.dropna()
    assert len(dec) > 0
    b = bands.loc[dec.index]
    expected = np.where(df.loc[dec.index, "close"] > b["ub"], 1.0,
                        np.where(df.loc[dec.index, "close"] < b["lb"], -1.0, 0.0))
    assert np.array_equal(dec.to_numpy(), expected)
    # nothing decided during the 14-day warm-up
    first_day = pd.Index(dec.index.tz_convert(TZ).date).min()
    assert first_day > pd.Index(df.index.tz_convert(TZ).date).unique()[13]
