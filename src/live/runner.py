"""
The live loop.

One rule governs this file: the signal is IMPORTED, never reimplemented. `_bands` is the same
function the backtest calls, so there is exactly one definition of what the band is. If you ever
find yourself writing `sigma = ...` in here, stop - that is how live quietly stops matching the
backtest.

Sequence at each acting moment (HH:01, HH:31):

    guards -> blackout -> fetch bars -> bands -> target position -> reconcile -> delta order

`decide()` is pure: bars in, target side out. It is what the tests exercise, and it is what the
shadow backtest replays. Everything with a side effect lives in `Runner.step()`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.events.noise_area import _bands
from src.live.broker import Broker, Position
from src.live.clock import Blackout, SessionClock
from src.live.guards import Guards, size_lots
from src.live.state import LiveState

log = logging.getLogger("live.runner")

EXIT_RULES = ("close", "opposite_band", "current_band", "band_vwap")


@dataclass
class Decision:
    bar: pd.Timestamp
    close: float
    ub: float
    lb: float
    current: int
    target: int
    reason: str


def session_vwap_upto(day_bars: pd.DataFrame) -> float:
    tp = (day_bars["high"] + day_bars["low"] + day_bars["close"]) / 3.0
    v = day_bars["volume"] if "volume" in day_bars.columns else pd.Series(1.0, index=day_bars.index)
    v = v.fillna(0.0)
    if float(v.sum()) <= 0:
        return float(tp.mean())
    return float((tp * v).sum() / v.sum())


#pentru fiecare bara de decizie, decide ce pozitie ar trebui sa fie
def decide(bars: pd.DataFrame, bar_ts: pd.Timestamp, current: int, *, lookback: int, vm: float,
           exit_rule: str, tz: str, rth_start: str, rth_end: str,
           entries_allowed: bool = True) -> Decision:
    """Pure: what position should we hold after the bar stamped `bar_ts` closed?"""
    if exit_rule not in EXIT_RULES:
        raise ValueError(f"exit_rule necunoscut: {exit_rule}")
    b = _bands(bars, lookback, vm, tz, rth_start, rth_end)
    key = pd.Timestamp(bar_ts)
    if key.tz is None:
        key = key.tz_localize("UTC")
    if key not in b.index:
        return Decision(key, np.nan, np.nan, np.nan, current, current, "bara lipseste din benzi")
    row = b.loc[key]
    if not np.isfinite(row["ub"]) or not np.isfinite(row["lb"]):
        return Decision(key, float(row["close"]), np.nan, np.nan, current, current,
                        "benzi NaN (incalzire incompleta)")

    close, ub, lb = float(row["close"]), float(row["ub"]), float(row["lb"])
    signal = 1 if close > ub else (-1 if close < lb else 0)

    target, reason = current, "fara schimbare"
    if signal != 0 and signal != current:
        if entries_allowed:
            target = signal
            reason = "intrare" if current == 0 else "inversare"
        else:
            reason = "semnal ignorat: intrarile sunt inchise (regula de gap trading)"
    elif current != 0:
        day = bars.loc[bars.index <= key]
        day = day.loc[day.index.tz_convert(tz).date == key.tz_convert(tz).date()]
        stop_hit = False
        if exit_rule == "opposite_band":
            stop_hit = (current == 1 and close < lb) or (current == -1 and close > ub)
        elif exit_rule == "current_band":
            stop_hit = (current == 1 and close < ub) or (current == -1 and close > lb)
        elif exit_rule == "band_vwap":
            vw = session_vwap_upto(day)
            stop_hit = (current == 1 and close < max(ub, vw)) or (current == -1 and close > min(lb, vw))
        if stop_hit:
            target, reason = 0, f"stop ({exit_rule})"
    return Decision(key, close, ub, lb, current, target, reason)


class Runner:
    def __init__(self, cfg: dict, broker: Broker, root: Path | None = None):
        self.cfg = cfg
        self.broker = broker
        self.root = Path(root or Path(__file__).resolve().parents[2])
        s = cfg["session"]
        self.clock = SessionClock(tz=s["tz"], rth_start=s["rth_start"], rth_end=s["rth_end"],
                                  decision_minutes=tuple(s["decision_minutes"]),
                                  first_decision=s["first_decision"],
                                  last_decision=s["last_decision"],
                                  eod_flatten=s["eod_flatten"],
                                  act_offset_min=int(s.get("act_offset_min", 1)))
        self.blackout = Blackout.from_csv(self.root / cfg["news"]["events_csv"],
                                          int(cfg["news"]["blackout_minutes"]))
        g = cfg["guards"]
        self.guards = Guards(max_daily_loss_pct=g["max_daily_loss_pct"],
                             max_total_loss_pct=g["max_total_loss_pct"],
                             max_orders_per_day=int(g["max_orders_per_day"]),
                             max_lots=float(g["max_lots"]),
                             halt_file=str(self.root / g["halt_file"]))
        self.state = LiveState.load(self.root / cfg["paths"]["state"])
        self.state.path = str(self.root / cfg["paths"]["state"])
        self.symbol = cfg["symbol"]

    # ---------------- logging of intent, not just of orders ----------------
    def _record(self, d: Decision, lots: float, note: str) -> None:
        p = self.root / self.cfg["paths"]["decisions"]
        p.parent.mkdir(parents=True, exist_ok=True)
        new = (not p.exists()) or p.stat().st_size == 0   # an existing but EMPTY file still needs a header
        with p.open("a", encoding="utf-8") as f:
            if new:
                f.write("bar,close,ub,lb,current,target,lots,reason,note\n")
            f.write(f"{d.bar},{d.close},{d.ub},{d.lb},{d.current},{d.target},"
                    f"{lots},{d.reason},{note}\n")

    def _record_fill(self, fill) -> None:
        p = self.root / self.cfg["paths"]["fills"]
        p.parent.mkdir(parents=True, exist_ok=True)
        new = (not p.exists()) or p.stat().st_size == 0
        with p.open("a", encoding="utf-8") as f:
            if new:
                f.write("ts,side,lots,price,kind,comment\n")
            f.write(f"{fill.ts},{fill.side},{fill.lots},{fill.price},{fill.kind},{fill.comment}\n")

    def _sync(self) -> Position:
        pos = self.broker.position(self.symbol)
        self.state.adopt(pos.side, pos.lots)
        return pos

    def _flatten(self, why: str) -> None:
        for fill in self.broker.close_all(self.symbol):
            self._record_fill(fill)
            log.info("flatten (%s): %s", why, fill)
        self.state.believed_side, self.state.believed_lots = 0, 0.0
        self.state.save()

    # ---------------- one tick of the loop ----------------
    def step(self, now: pd.Timestamp) -> str:
        now = pd.Timestamp(now)
        if now.tz is None:
            now = now.tz_localize("UTC")

        if not self.broker.is_connected():
            self.state.halt("terminalul nu e conectat")
            return "deconectat"

        equity = self.broker.equity()
        local_date = str(self.clock.local(now).date())
        if self.state.roll_day(local_date, equity):
            self.state.save()

        if self.state.halted:
            return f"oprit: {self.state.halt_reason}"

        pos = self._sync()

        reason = self.guards.check(equity=equity, day_start_equity=self.state.day_start_equity,
                                   initial_balance=self.state.initial_balance,
                                   orders_today=self.state.orders_today)
        if reason:
            if not pos.flat:
                self._flatten(f"guard: {reason}")
            self.state.halt(reason)
            return f"oprit: {reason}"

        blocked = self.blackout.blocked(now)
        if blocked:
            self.state.save()
            return f"blackout: {blocked}"

        if self.clock.is_flatten_time(now):
            if not pos.flat:
                self._flatten("sfarsit de sesiune")
                return "flatten EOD"
            return "flat, sfarsit de sesiune"

        bar = self.clock.decision_bar(now)
        if bar is None:
            return "nu e moment de decizie"
        bar_utc = bar.tz_convert("UTC")
        if self.state.seen(bar_utc):
            return "bara deja procesata"

        st = self.cfg["strategy"]
        need = int(st.get("warmup_days", 25)) * 1440
        bars = self.broker.bars(self.symbol, need)
        d = decide(bars, bar_utc, pos.side, lookback=int(st["lookback"]), vm=float(st["vm"]),
                   exit_rule=st["exit_rule"], tz=self.clock.tz,
                   rth_start=self.clock.rth_start, rth_end=self.clock.rth_end,
                   entries_allowed=self.clock.entries_allowed(now))

        note = ""
        lots = 0.0
        if d.target != d.current:
            if d.target == 0:
                self._flatten(d.reason)
                note = "inchis"
            else:
                r = self.cfg["risk"]
                lots = self.guards.clamp_lots(size_lots(
                    equity=equity, price=d.close, contract_size=float(r["contract_size"]),
                    strategy_daily_vol=float(r["strategy_daily_vol"]),
                    target_daily_vol=float(r["target_daily_vol"]),
                    lot_step=float(r["lot_step"]), min_lot=float(r["min_lot"]),
                    max_lots=float(self.guards.max_lots)))
                if lots <= 0:
                    note = "marime 0 dupa rotunjire - nu trimit nimic"
                    log.warning("%s (echitate %.0f, pret %.1f)", note, equity, d.close)
                else:
                    fill = self.broker.market_order(self.symbol, d.target, lots, d.reason)
                    self._record_fill(fill)
                    self.state.orders_today += 1
                    self.state.believed_side, self.state.believed_lots = d.target, lots
                    note = f"ordin {d.target:+d} {lots}"
                    log.info("%s @ %.2f (%s)", note, fill.price, d.reason)

        self.state.mark(bar_utc)
        self.state.save()
        self._record(d, lots, note)
        return f"{d.reason} -> {d.target:+d} {note}".strip()


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
