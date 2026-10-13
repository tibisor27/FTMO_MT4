"""
When the bot is allowed to act.

Two things live here and nowhere else:

  1. The decision schedule. The backtest decides on the bar stamped HH:00 / HH:30 and fills at the
     OPEN of the next bar. Live, that means: at HH:01:00 read the just-completed HH:00 bar and send
     a market order immediately. Acting at HH:00:00 would read an unfinished bar - the single
     easiest way to make live disagree with the backtest.

  2. The blackout windows. FTMO forbids opening OR closing anything within 2 minutes of a listed
     macro release on a funded Standard account, and an SL/TP firing inside that window is itself a
     breach. The default margin here is wider than the rule for that reason.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger("live.clock")


def _hm(s: str) -> int:
    return int(s[:2]) * 60 + int(s[3:])


@dataclass
class SessionClock:
    tz: str = "America/New_York"
    rth_start: str = "09:30"
    rth_end: str = "16:00"
    decision_minutes: tuple[int, ...] = (0, 30)
    first_decision: str = "10:00"
    last_decision: str = "13:30"     # prudent reading of the gap-trading rule
    eod_flatten: str = "15:57"
    act_offset_min: int = 1          # act one minute after the decision bar closes

    #converteste timestampul in New_York si il retunreaza
    def local(self, ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(self.tz)

    #verifica daca minutul curent e in orele de tranzactionare
    def in_session(self, ts: pd.Timestamp) -> bool:
        t = self.local(ts)
        if t.dayofweek >= 5:
            return False
        m = t.hour * 60 + t.minute
        return _hm(self.rth_start) <= m < _hm(self.rth_end)

    #retunreaza timestampul barelor de decizie(ora la care botul ia decizia)
    def decision_bar(self, ts: pd.Timestamp) -> pd.Timestamp | None:
        """
        If `ts` is an acting moment, return the timestamp of the decision BAR it acts on.

        Acting at 10:01:xx acts on the bar stamped 10:00. Returns None at any other minute.

        Plafonul de sus e `eod_flatten`, NU `last_decision`. Dupa `last_decision` botul
        trebuie sa poata in continuare sa IASA dintr-o pozitie - doar sa nu mai deschida.
        Cine decide asta e `entries_allowed()`, pe care `decide()` il primeste ca parametru.
        Daca plafonul ar sta aici, `decide()` nu s-ar mai apela deloc si stopul de banda
        nu s-ar putea declansa intre 13:31 si inchiderea fortata.
        """
        t = self.local(ts)
        if t.dayofweek >= 5:
            return None

        #.floor("min") truncheaza secundele si milisecundele
        #iau fix bara de decizie (xx:00 sau xx:30) care e bara de acum un minut(eu procesez mereu bara de dupa xx:00 sau xx:30)
        bar = t.floor("min") - pd.Timedelta(minutes=self.act_offset_min)
        if bar.minute not in self.decision_minutes:
            return None
        m = bar.hour * 60 + bar.minute
        if m < _hm(self.first_decision):
            return None
        if m >= _hm(self.eod_flatten):          # dupa asta inchide oricum tot
            return None
        if not (_hm(self.rth_start) <= m < _hm(self.rth_end)):
            return None
        return bar

    #verifica daca minutul curent e in intervalul de incheiere a sesiunii(15:57-16:00)
    def is_flatten_time(self, ts: pd.Timestamp) -> bool:
        t = self.local(ts)
        if t.dayofweek >= 5:
            return False
        return (t.hour * 60 + t.minute) >= _hm(self.eod_flatten) and \
               (t.hour * 60 + t.minute) < _hm(self.rth_end)

    def entries_allowed(self, ts: pd.Timestamp) -> bool:
        """Past `last_decision` the bot may still exit, but must not open anything new."""
        t = self.local(ts)
        return (t.hour * 60 + t.minute) <= _hm(self.last_decision) + self.act_offset_min


@dataclass
class Blackout:
    """Restricted-event windows, loaded from a CSV the trader maintains."""
    windows: list[tuple[pd.Timestamp, pd.Timestamp, str]]
    margin_min: int = 5

    @classmethod
    def from_csv(cls, path: str | Path, margin_min: int = 5) -> "Blackout":
        p = Path(path)
        if not p.exists():
            log.warning("fara calendar de evenimente la %s - NU exista blackout activ", p)
            return cls([], margin_min)
        d = pd.read_csv(p)
        col = "datetime_utc" if "datetime_utc" in d.columns else d.columns[0]
        ts = pd.to_datetime(d[col], utc=True)
        name = d["event"] if "event" in d.columns else pd.Series(["restricted"] * len(d))
        w = [(t - pd.Timedelta(minutes=margin_min), t + pd.Timedelta(minutes=margin_min), str(n))
             for t, n in zip(ts, name)]
        log.info("calendar incarcat: %d evenimente, marja +/-%d min", len(w), margin_min)
        return cls(w, margin_min)

    def blocked(self, ts: pd.Timestamp) -> str | None:
        ts = pd.Timestamp(ts)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        for lo, hi, name in self.windows:
            if lo <= ts <= hi:
                return name
        return None
