"""
Persisted state, so a restart is boring.

The hard rule: on start, whatever this file says, the BROKER is the truth. The file only records
what the bot believes it has already done today, so a restart at 11:15 does not re-enter a position
it opened at 10:01. Every field that is per-day resets automatically when the trading date changes.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger("live.state")


@dataclass
class LiveState:
    trade_date: str = ""
    processed_bars: list[str] = field(default_factory=list)
    believed_side: int = 0
    believed_lots: float = 0.0
    day_start_equity: float = 0.0
    initial_balance: float = 0.0
    orders_today: int = 0
    halted: bool = False
    halt_reason: str = ""
    path: str = "state/live_state.json"

    # ---------- persistence ----------
    @classmethod
    def load(cls, path: str | Path) -> "LiveState":
        p = Path(path)
        if not p.exists():
            return cls(path=str(p))
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.error("stare corupta la %s (%s) - pornesc de la zero, dar NU deschid nimic "
                      "pana nu citesc pozitia reala din cont", p, e)
            return cls(path=str(p))
        raw.pop("path", None)
        known = {f for f in cls.__dataclass_fields__ if f != "path"}
        return cls(path=str(p), **{k: v for k, v in raw.items() if k in known})

    def save(self) -> None:
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(p)                      # atomic: a crash mid-write never leaves half a file

    # ---------- day handling ----------
    def roll_day(self, local_date: str, equity: float) -> bool:
        if self.trade_date == local_date:
            return False
        log.info("zi noua %s (era %s) - resetez contoarele zilnice", local_date, self.trade_date or "-")
        self.trade_date = local_date
        self.processed_bars = []
        self.orders_today = 0
        self.day_start_equity = equity
        if self.initial_balance <= 0:
            self.initial_balance = equity
        self.halted = False
        self.halt_reason = ""
        return True

    def seen(self, bar_ts: pd.Timestamp) -> bool:
        return str(pd.Timestamp(bar_ts)) in self.processed_bars

    def mark(self, bar_ts: pd.Timestamp) -> None:
        s = str(pd.Timestamp(bar_ts))
        if s not in self.processed_bars:
            self.processed_bars.append(s)

    def adopt(self, side: int, lots: float) -> None:
        """Trust the broker over our own memory."""
        if (side, round(lots, 4)) != (self.believed_side, round(self.believed_lots, 4)):
            log.warning("pozitia reala (%+d, %.2f) difera de starea salvata (%+d, %.2f) - "
                        "adopt realitatea", side, lots, self.believed_side, self.believed_lots)
        self.believed_side, self.believed_lots = side, lots

    def halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason
        log.error("OPRIRE: %s", reason)
        self.save()
