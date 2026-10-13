"""Tests on fabricated 1-minute bars: each one pins down what 'correct' means for a generator."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.events.macro_sweep import macro_sweep_events  # noqa: E402
from src.events.noise_area import noise_area_events  # noqa: E402
from src.event_study import dedup_events, forward_returns, run_event_study  # noqa: E402

TZ = "America/New_York"


def make_days(n_days: int, start="2024-01-08", price=100.0, drift_per_min=0.0, seed=0) -> pd.DataFrame:
    """Flat, low-noise RTH sessions (390 bars/day, 9:30-16:00 New York), index in UTC."""
    rng = np.random.default_rng(seed)
    frames = []
    p = price
    for d in pd.bdate_range(start, periods=n_days):
        idx = pd.date_range(f"{d.date()} 09:30", periods=390, freq="1min", tz=TZ)
        steps = drift_per_min + rng.normal(0, 0.01, 390)
        c = p + np.cumsum(steps)
        o = np.r_[p, c[:-1]]
        h = np.maximum(o, c) + 0.005
        l = np.minimum(o, c) - 0.005
        frames.append(pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1.0}, index=idx))
        p = c[-1]
    df = pd.concat(frames)
    df.index = df.index.tz_convert("UTC")
    df.index.name = "ts"
    return df


# ------------------------------------------------------------------ noise area
def test_noise_area_fires_long_on_a_trend_day_and_only_after_lookback():
    df = make_days(16)  # 14 lookback + 1 + the trend day
    # last day: strong uptrend of +1% by 10:30 -> way outside a band built from ~0.01-noise days
    last = df.index.tz_convert(TZ).date == df.index.tz_convert(TZ).date.max()
    n = last.sum()
    ramp = np.linspace(0, 3.0, n)
    for col in ("open", "high", "low", "close"):
        df.loc[last, col] = df.loc[last, col].to_numpy() + ramp
    ev, elig = noise_area_events(df, lookback=14)
    last_day = ev[ev["t0"].dt.tz_convert(TZ).dt.date == pd.Timestamp("2024-01-29").date()]
    assert len(last_day) >= 1
    first = last_day.iloc[0]
    assert first.side == 1 and first.kind == "long"
    assert first.t0.tz_convert(TZ).strftime("%H:%M") == "10:00"
    # no events on the 14 warm-up days
    assert ev["t0"].dt.tz_convert(TZ).dt.date.min() > pd.Timestamp("2024-01-08").date() + pd.Timedelta(days=13)
    # eligible times are only decision times
    assert set(elig.tz_convert(TZ).minute.unique()) <= {0, 30}


def test_noise_area_flat_days_produce_no_events():
    df = make_days(20, seed=1)
    ev, _ = noise_area_events(df, lookback=14, vm=3.0)
    assert len(ev) == 0


# ------------------------------------------------------------------ macro sweep
def _sweep_day() -> pd.DataFrame:
    """One session; a swing high at 100.30 formed at 09:40, swept at 09:52 (macro window), rejected, iFVG."""
    df = make_days(1, seed=2)
    c = df["close"].to_numpy().copy()
    o = df["open"].to_numpy().copy()
    h = df["high"].to_numpy().copy()
    l = df["low"].to_numpy().copy()
    # bars 0..30 flat at 100 with tiny noise, bar 10 = swing high 100.30 (n=5 either side lower)
    c[:] = 100.0; o[:] = 100.0; h[:] = 100.02; l[:] = 99.98
    h[10] = 100.30; c[10] = 100.25; o[10] = 100.0
    # impulse up into the level at bars 20-22 creating a bullish FVG: low[22] > high[20]
    o[20], c[20], h[20], l[20] = 100.00, 100.08, 100.10, 99.99
    o[21], c[21], h[21], l[21] = 100.08, 100.20, 100.22, 100.07
    o[22], c[22], h[22], l[22] = 100.20, 100.32, 100.34, 100.15  # bar 22 = 09:52 macro window; high 100.34 >= 100.30+0.02, closes above
    # bar 23 closes back below the level -> sweep confirmed; bar 24 closes below FVG bottom (100.10) -> iFVG
    o[23], c[23], h[23], l[23] = 100.28, 100.26, 100.36, 100.20
    o[24], c[24], h[24], l[24] = 100.26, 100.05, 100.27, 100.03
    df["open"], df["close"], df["high"], df["low"] = o, c, h, l
    return df


def test_macro_sweep_ifvg_detects_the_handmade_short():
    df = _sweep_day()
    ev, elig = macro_sweep_events(df, swing_n=5, sweep_min=0.02, reject_bars=3, fvg_lookback=15, inversion_bars=10, require_ifvg=True)
    assert len(ev) == 1
    e = ev.iloc[0]
    assert e.side == -1 and e.kind == "ifvg"
    assert e.t0 == df.index[24]
    assert e.sweep_ts == df.index[23]
    assert abs(e.level - 100.30) < 1e-9 and abs(e.fvg_bottom - 100.10) < 1e-9
    # every eligible bar is inside a macro window
    mins = elig.tz_convert(TZ).minute
    assert ((mins >= 50) | (mins < 10)).all()


def test_macro_sweep_control_fires_at_the_sweep_bar():
    df = _sweep_day()
    ev, _ = macro_sweep_events(df, swing_n=5, sweep_min=0.02, reject_bars=3, require_ifvg=False)
    assert len(ev) == 1 and ev.iloc[0].kind == "sweep" and ev.iloc[0].t0 == df.index[23]


def test_macro_sweep_outside_window_is_ignored():
    df = _sweep_day()
    ev, _ = macro_sweep_events(df, swing_n=5, sweep_min=0.02, macro_before=1, macro_after=1, require_ifvg=False)
    assert len(ev) == 0


# ------------------------------------------------------------------ event study
def test_forward_returns_known_values_and_session_cap():
    df = make_days(1, seed=3)
    df.loc[:, "open"] = 100.0
    df.loc[:, "close"] = 100.0
    df.iloc[5, df.columns.get_loc("close")] = 101.0  # bar t0+5min (index 5); entry = open of bar 1
    t0 = df.index[0]
    ev = pd.DataFrame({"t0": [t0, df.index[384]], "side": [1, -1]})
    R, Z = forward_returns(df, ev, [5, 30], entry="next_open")
    assert abs(R.loc[0, "5"] - np.log(101 / 100) * 1e4) < 1e-6
    assert np.isnan(R.loc[1, "30"])  # 30 min after 15:54 crosses the close -> excluded
    assert abs(R.loc[1, "5"]) < 1e-9


def test_dedup_keeps_first_of_cluster():
    t = pd.to_datetime(["2024-01-08 15:00", "2024-01-08 15:05", "2024-01-08 15:45", "2024-01-09 15:00"], utc=True)
    ev = pd.DataFrame({"t0": t, "side": [1, -1, 1, 1]})
    out = dedup_events(ev, 30)
    assert list(out["t0"]) == [t[0], t[2], t[3]]


def test_run_event_study_on_noise_has_no_edge():
    df = make_days(30, seed=4)
    rng = np.random.default_rng(5)
    t0 = df.index[rng.choice(len(df) - 200, 150, replace=False)]
    ev = pd.DataFrame({"t0": t0, "side": rng.choice([1, -1], 150), "strategy": "rnd"})
    res = run_event_study(df, ev, df.index, horizons=[5, 30], min_gap_min=1, n_perm=200)
    assert abs(res.summary.loc["30", "t"]) < 3
    assert res.summary.loc["30", "p_perm"] > 0.01
