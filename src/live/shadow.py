"""
Shadow backtest: replay the session and diff it against what the bot actually did.

This is the piece almost nobody builds and the one that pays for itself the first week. Every
evening it answers three questions that nothing else can:

  1. Did the bot decide what the backtest would have decided? A mismatch is a BUG - wrong warm-up,
     wrong timezone, a stale bar, a missed wake-up. Find it on demo, not on a funded account.
  2. What did execution actually cost? Comparing each fill against the next bar's open turns
     `cost_bps` from an assumption into a measurement of your own broker.
  3. Did anything fire inside a blackout window? That one ends accounts.

Divergence is expected to be small but never zero. Set a threshold now, in writing, and treat a
breach as a stop-trading event rather than something to explain away later.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.live.clock import Blackout, SessionClock
from src.live.runner import decide

log = logging.getLogger("live.shadow")


@dataclass
class Diff:
    bar: pd.Timestamp
    kind: str          # "match" | "missing" | "extra" | "wrong_side"
    expected: int
    actual: int
    note: str = ""


def replay(bars: pd.DataFrame, cfg: dict, day: str) -> pd.DataFrame:
    """What the strategy WOULD have done on `day`, decision bar by decision bar."""
    s, st = cfg["session"], cfg["strategy"]
    clock = SessionClock(tz=s["tz"], rth_start=s["rth_start"], rth_end=s["rth_end"],
                         decision_minutes=tuple(s["decision_minutes"]),
                         first_decision=s["first_decision"], last_decision=s["last_decision"],
                         eod_flatten=s["eod_flatten"],
                         act_offset_min=int(s.get("act_offset_min", 1)))
    loc = bars.index.tz_convert(clock.tz)
    same_day = (pd.Series(loc.date, index=bars.index) == pd.Timestamp(day).date()).to_numpy()
    # RTH only: the CFD prints around the clock, but the strategy only exists inside the session
    in_rth = np.array([clock.in_session(t) for t in bars.index])
    todays = bars.index[same_day & in_rth]
    rows, pos = [], 0
    for ts in todays:
        act = ts + pd.Timedelta(minutes=clock.act_offset_min)
        if clock.decision_bar(act) is None:
            continue
        d = decide(bars, ts, pos, lookback=int(st["lookback"]), vm=float(st["vm"]),
                   exit_rule=st["exit_rule"], tz=clock.tz, rth_start=clock.rth_start,
                   rth_end=clock.rth_end, entries_allowed=clock.entries_allowed(act))
        rows.append({"bar": ts, "close": d.close, "ub": d.ub, "lb": d.lb,
                     "current": d.current, "target": d.target, "reason": d.reason})
        pos = d.target
    if pos != 0 and len(todays):
        rows.append({"bar": todays[-1], "close": float(bars.loc[todays[-1], "close"]),
                     "ub": np.nan, "lb": np.nan, "current": pos, "target": 0, "reason": "eod"})
    return pd.DataFrame(rows)


def compare(expected: pd.DataFrame, decisions_csv: str | Path, day: str) -> list[Diff]:
    p = Path(decisions_csv)
    if not p.exists():
        return [Diff(pd.NaT, "missing", 0, 0, f"fara jurnal de decizii la {p}")]
    live = pd.read_csv(p)
    live["bar"] = pd.to_datetime(live["bar"], utc=True)
    live = live[live["bar"].dt.tz_convert("America/New_York").dt.date == pd.Timestamp(day).date()]
    out: list[Diff] = []
    exp = expected.copy()
    if not exp.empty:
        exp["bar"] = pd.to_datetime(exp["bar"], utc=True)
    seen = set()
    for _, e in exp.iterrows():
        if str(e.get("reason", "")) == "eod":
            continue          # the end-of-day flatten is a fill, not a decision; checked separately
        m = live[live["bar"] == e["bar"]]
        if m.empty:
            out.append(Diff(e["bar"], "missing", int(e["target"]), 0,
                            "botul nu a procesat aceasta bara"))
            continue
        seen.add(e["bar"])
        a = int(m.iloc[0]["target"])
        out.append(Diff(e["bar"], "match" if a == int(e["target"]) else "wrong_side",
                        int(e["target"]), a, str(m.iloc[0]["reason"])))
    for _, a in live.iterrows():
        if a["bar"] not in seen:
            out.append(Diff(a["bar"], "extra", 0, int(a["target"]),
                            "botul a actionat pe o bara pe care backtestul nu o considera decizie"))
    return out


def slippage(fills_csv: str | Path, bars: pd.DataFrame, day: str) -> pd.DataFrame:
    """
    Realised slippage per fill, in basis points, signed so positive = worse than the backtest.

    The backtest assumes the open of the bar after the decision. Anything else is what execution
    really cost you, and it is the honest input to `cost_bps`.
    """
    p = Path(fills_csv)
    if not p.exists():
        return pd.DataFrame()
    f = pd.read_csv(p)
    f["ts"] = pd.to_datetime(f["ts"], utc=True)
    f = f[f["ts"].dt.tz_convert("America/New_York").dt.date == pd.Timestamp(day).date()]
    f = f[(f["lots"] > 0) & (f["kind"] != "noop")]
    if f.empty:
        return pd.DataFrame()
    idx = bars.index
    rows = []
    for _, r in f.iterrows():
        i = idx.searchsorted(r["ts"].floor("min"), side="left")
        if i >= len(idx):
            continue
        ref = float(bars["open"].iloc[i])
        side = int(r["side"])
        slip = (float(r["price"]) - ref) * side / ref * 1e4
        rows.append({"ts": r["ts"], "side": side, "lots": r["lots"], "fill": r["price"],
                     "referinta_backtest": ref, "slippage_bps": round(slip, 3), "kind": r["kind"]})
    return pd.DataFrame(rows)


def ended_flat(fills_csv: str | Path, day: str) -> bool | None:
    """Did the day close with no open position? None when there is nothing to judge."""
    p = Path(fills_csv)
    if not p.exists() or p.stat().st_size == 0:
        return None
    f = pd.read_csv(p)
    if f.empty or "ts" not in f.columns:
        return None
    f["ts"] = pd.to_datetime(f["ts"], utc=True)
    f = f[f["ts"].dt.tz_convert("America/New_York").dt.date == pd.Timestamp(day).date()]
    f = f[f["kind"] != "noop"]
    if f.empty:
        return None
    return abs(float((f["side"] * f["lots"]).sum())) < 1e-9


def report(day: str, diffs: list[Diff], slip: pd.DataFrame, blackout: Blackout | None = None,
           fills_csv: str | Path | None = None) -> str:
    ok = sum(1 for d in diffs if d.kind == "match")
    bad = [d for d in diffs if d.kind != "match"]
    L = [f"# Reconciliere · {day}", "",
         f"- decizii comparate: **{len(diffs)}**",
         f"- potrivite: **{ok}**",
         f"- divergente: **{len(bad)}**"]
    if not slip.empty:
        L += [f"- fill-uri: **{len(slip)}** · slippage median "
              f"**{slip['slippage_bps'].median():.2f} bps** · cel mai prost "
              f"**{slip['slippage_bps'].max():.2f} bps**"]
    L.append("")
    if bad:
        L += ["## ⚠️ Divergente", "", "| bara | tip | asteptat | real | nota |", "|---|---|---|---|---|"]
        for d in bad:
            L.append(f"| {d.bar} | {d.kind} | {d.expected:+d} | {d.actual:+d} | {d.note} |")
        L.append("")
    else:
        L += ["✅ Botul a decis exact ce ar fi decis backtestul.", ""]

    if fills_csv:
        flat = ended_flat(fills_csv, day)
        if flat is True:
            L += ["✅ Ziua s-a inchis fara pozitie deschisa.", ""]
        elif flat is False:
            L += ["## 🚨 ZIUA NU S-A INCHIS PLATA — expunere peste noapte, contrar strategiei", ""]

    if blackout is not None and fills_csv and Path(fills_csv).exists():
        f = pd.read_csv(fills_csv)
        f["ts"] = pd.to_datetime(f["ts"], utc=True)
        hits = [(t, blackout.blocked(t)) for t in f["ts"] if blackout.blocked(t)]
        if hits:
            L += ["## 🚨 FILL-URI IN FEREASTRA DE STIRI — incalcare de regula", ""]
            for t, name in hits:
                L.append(f"- {t} · {name}")
            L.append("")
        else:
            L += ["✅ Niciun fill in fereastra de blackout.", ""]

    if not slip.empty:
        L += ["## Slippage pe fill", "", slip.to_markdown(index=False), "",
              f"**Cost implicit dus-intors: {2*slip['slippage_bps'].median():.2f} bps.** "
              "Compara-l cu `cost_bps` din backtest.", ""]
    return "\n".join(L)
