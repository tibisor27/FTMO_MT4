"""
Leakage and accounting guarantees. These are the tests that must never go red.

The headline one is `test_future_shuffle_leaves_the_past_untouched`: destroy every bar after a cut
date and the P&L before that date must be identical bit for bit. Any look-ahead anywhere in the
pipeline - bands, signal, sizing, fills - shows up here, because a leak means future information
reached a past decision.

The rest pin down the execution model so a future refactor cannot quietly make it optimistic.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import backtest, session_vwap  # noqa: E402
from src.events.noise_area import _bands, noise_area_bands, noise_area_signal  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "USATECHIDXUSD_1m.parquet"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="raw parquet not present")


def synthetic(ndays=40, spike_day=20, spike=0.10):
    """Flat sessions at 100, except one day that closes `spike` higher."""
    blocks = []
    for day in pd.bdate_range("2024-01-02", periods=ndays):
        blocks.append(pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=390, freq="1min"))
    idx = (pd.DatetimeIndex(np.concatenate([b.values for b in blocks]))
           .tz_localize("America/New_York").tz_convert("UTC"))
    px = np.full(len(idx), 100.0)
    f = pd.DataFrame({"open": px.copy(), "high": px.copy(), "low": px.copy(),
                      "close": px.copy(), "volume": 1.0}, index=idx)
    sel = np.repeat(np.arange(ndays), 390) == spike_day
    f.loc[sel, ["close", "high"]] = 100.0 * (1 + spike)
    f.loc[sel, "open"] = 100.0
    return f


def test_sigma_never_sees_the_current_day():
    f = synthetic()
    ub = _bands(f, 14, 1.0, "America/New_York", "09:30", "16:00").groupby("date")["ub"].last()
    # the spike day itself is priced off the 14 flat days before it
    assert ub.iloc[20] == pytest.approx(100.0, abs=1e-9)
    # the day after, sigma = spike/14 and the reference is the gapped-up previous close
    assert ub.iloc[21] == pytest.approx(110.0 * (1 + 0.10 / 14), rel=1e-9)
    # 14 days later the spike leaves the window entirely
    assert ub.iloc[35] == pytest.approx(100.0, abs=1e-9)


@needs_data
def test_future_shuffle_leaves_the_past_untouched():
    from src.data.intraday import load_intraday
    df = load_intraday(str(DATA))
    cut = pd.Timestamp("2022-01-03", tz="UTC")
    scrambled = df.copy()
    post = scrambled.index >= cut
    cols = ["open", "high", "low", "close", "volume"]
    blk = scrambled.loc[post, cols].to_numpy()
    scrambled.loc[post, cols] = blk[np.random.default_rng(1).permutation(len(blk))]

    out = []
    for frame in (df, scrambled):
        sig = noise_area_signal(frame, lookback=14, vm=1.3)
        bnd = noise_area_bands(frame, lookback=14, vm=1.3)
        daily = backtest(frame, sig, bnd, exit_rule="opposite_band", cost_bps=0.65).daily["ret"]
        daily.index = pd.to_datetime(daily.index)
        out.append(daily[daily.index < cut.tz_localize(None)])

    assert len(out[0]) == len(out[1])
    np.testing.assert_array_equal(out[0].to_numpy(), out[1].to_numpy())


@needs_data
def test_fills_land_on_the_next_bar_and_costs_are_charged_once():
    from src.data.intraday import load_intraday
    df = load_intraday(str(DATA))
    sig = noise_area_signal(df, lookback=14, vm=1.3)
    bnd = noise_area_bands(df, lookback=14, vm=1.3)
    r = backtest(df, sig, bnd, exit_rule="opposite_band", cost_bps=0.65)
    tr = r.trades
    t_in = pd.to_datetime(tr["t_in"], utc=True)
    t_out = pd.to_datetime(tr["t_out"], utc=True)

    assert np.allclose(tr["price_in"], df["open"].reindex(t_in).to_numpy())
    stop = tr["exit_reason"] == "stop"
    assert np.allclose(tr.loc[stop, "price_out"], df["open"].reindex(t_out[stop]).to_numpy())
    eod = tr["exit_reason"] == "eod"
    assert np.allclose(tr.loc[eod, "price_out"], df["close"].reindex(t_out[eod]).to_numpy())
    assert (t_out > t_in).all()

    # every fill is strictly later than the decision bar that produced it
    dec = sig.dropna().index.to_numpy()
    pos = np.clip(np.searchsorted(dec, t_in.to_numpy(), side="left") - 1, 0, len(dec) - 1)
    gap = (t_in.to_numpy().astype("datetime64[m]") - dec.astype("datetime64[m]")[pos]).astype(int)
    assert (gap > 0).all()

    implied_bps = (tr["ret_gross"].sum() - tr["ret_net"].sum()) / len(tr) * 1e4
    assert implied_bps == pytest.approx(0.65, abs=0.02)
    assert tr["pnl"].sum() == pytest.approx(r.daily["ret"].sum(), abs=1e-9)


@needs_data
def test_vwap_is_volume_weighted_not_a_time_average():
    from src.data.intraday import load_intraday, rth_mask
    df = load_intraday(str(DATA))
    d = df.loc[rth_mask(df.index, "America/New_York", "09:30", "16:00").values]
    dates = np.asarray(d.index.tz_convert("America/New_York").date)
    tp = ((d["high"] + d["low"] + d["close"]) / 3).to_numpy()
    vwap = session_vwap(d, dates)
    twap = (pd.Series(tp).groupby(dates).cumsum().to_numpy()
            / (pd.Series(1, index=range(len(tp))).groupby(dates).cumcount() + 1).to_numpy())
    assert np.nanmean(np.abs(vwap - twap) / tp) * 1e4 > 0.5


@needs_data
def test_flipping_the_signal_flips_the_sign_of_the_edge():
    from src.data.intraday import load_intraday
    df = load_intraday(str(DATA))
    sig = noise_area_signal(df, lookback=14, vm=1.3)
    bnd = noise_area_bands(df, lookback=14, vm=1.3)
    fwd = backtest(df, sig, bnd, exit_rule="close", cost_bps=0.65).stats["sharpe"]
    inv = backtest(df, -sig, bnd, exit_rule="close", cost_bps=0.65).stats["sharpe"]
    assert fwd > 0 > inv
