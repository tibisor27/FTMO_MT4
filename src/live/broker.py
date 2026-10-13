"""
Broker adapter. One protocol, two implementations.

`MT5Broker` talks to a running MetaTrader 5 terminal through the `MetaTrader5` package, which is
Windows-only - so it is imported lazily and this module stays importable on macOS and Linux.
`PaperBroker` replays a DataFrame and fills at the next bar's open, which is exactly the backtest's
assumption; it is what the tests and the dry runs use.

Everything below returns tz-aware UTC timestamps. The terminal reports its own server time
(GMT+2 +DST for MT5), so converting once, here, is the only place that conversion is allowed to
happen. Every timezone bug in a live trading system starts with that conversion happening twice.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

log = logging.getLogger("live.broker")


@dataclass
class Position:
    side: int = 0            # +1 long, -1 short, 0 flat
    lots: float = 0.0
    price_open: float = 0.0
    ticket: int | None = None

    @property
    def flat(self) -> bool:
        return self.side == 0 or self.lots <= 0


@dataclass
class Fill:
    ts: pd.Timestamp
    side: int
    lots: float              # volumul EXECUTAT, nu cel cerut
    price: float
    kind: str                # "entry" | "exit" | "reverse"
    comment: str = ""
    requested: float = 0.0   # ce am cerut; lots != requested = executie partiala

    @property
    def partial(self) -> bool:
        return self.requested > 0 and abs(self.lots - self.requested) > 1e-9


@dataclass(frozen=True)
class SymbolSpec:
    """Ce spune TERMINALUL despre simbol. Nimic din asta nu se ia din fisierul de configurare."""
    contract_size: float
    lot_step: float
    min_lot: float
    max_lot: float
    digits: int = 2


class OrderRejected(RuntimeError):
    """Ordin respins. `fatal` = nu are rost sa reincerci."""

    def __init__(self, msg: str, retcode: int | None = None, fatal: bool = True):
        super().__init__(msg)
        self.retcode, self.fatal = retcode, fatal


class Broker(Protocol):
    server_utc_offset_s: int

    def connect(self) -> None: ...
    def symbol_spec(self, symbol: str) -> "SymbolSpec": ...
    def is_connected(self) -> bool: ...
    def equity(self) -> float: ...
    def balance(self) -> float: ...
    def bars(self, symbol: str, count: int) -> pd.DataFrame: ...
    def position(self, symbol: str) -> Position: ...
    def market_order(self, symbol: str, side: int, lots: float, comment: str = "") -> Fill: ...
    def close_all(self, symbol: str) -> list[Fill]: ...
    def tick(self, symbol: str) -> tuple[float, float]: ...


# --------------------------------------------------------------------------- paper
class PaperBroker:
    """Deterministic simulator over a bar frame. Fills at the open of the bar AFTER `now`."""

    def __init__(self, frame: pd.DataFrame, start_equity: float = 50_000.0,
                 contract_size: float = 2.0, cost_bps: float = 1.5):
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise TypeError("PaperBroker needs a tz-aware DatetimeIndex")
        self.frame = frame.sort_index()
        self._equity = start_equity
        self._balance = start_equity
        self.contract_size = contract_size
        self.cost = cost_bps / 1e4
        self.now: pd.Timestamp = self.frame.index[0]
        self._pos = Position()
        self.fills: list[Fill] = []
        self._connected = True
        self.server_utc_offset_s = 0          # datele de hartie sunt deja in UTC

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        return SymbolSpec(contract_size=self.contract_size, lot_step=0.01,
                          min_lot=0.01, max_lot=100.0)

    # ---- clock control (tests drive this) ----
    def set_now(self, ts: pd.Timestamp) -> None:
        self.now = pd.Timestamp(ts).tz_convert("UTC")

    def _next_open(self) -> tuple[pd.Timestamp, float]:
        # side="left": an order sent AT 11:01:00 fills at the open of the 11:01 bar, which is
        # exactly what the backtest assumes. side="right" would push it to 11:02 and make the
        # paper broker one bar more pessimistic than the model it is meant to mirror.
        i = self.frame.index.searchsorted(self.now, side="left")
        if i >= len(self.frame):
            i = len(self.frame) - 1
        return self.frame.index[i], float(self.frame["open"].iloc[i])

    # ---- protocol ----
    def connect(self) -> None:
        self._connected = True

    def is_connected(self) -> bool:
        return self._connected

    def equity(self) -> float:
        return self._equity

    def balance(self) -> float:
        return self._balance

    def bars(self, symbol: str, count: int) -> pd.DataFrame:
        return self.frame.loc[self.frame.index <= self.now].tail(count).copy()

    def position(self, symbol: str) -> Position:
        return self._pos

    def tick(self, symbol: str) -> tuple[float, float]:
        _, px = self._next_open()
        return px, px

    def _settle(self, exit_price: float) -> None:
        p = self._pos
        if p.flat:
            return
        gross = (exit_price / p.price_open - 1.0) * p.side
        net = gross - 2 * self.cost
        pnl = net * p.lots * self.contract_size * p.price_open
        self._balance += pnl
        self._equity = self._balance
        self._pos = Position()

    def market_order(self, symbol: str, side: int, lots: float, comment: str = "") -> Fill:
        ts, px = self._next_open()
        kind = "entry"
        if not self._pos.flat:
            if self._pos.side == side:
                return Fill(ts, side, 0.0, px, "noop", "already on this side", requested=0.0)
            self._settle(px)
            kind = "reverse"
        if side != 0 and lots > 0:
            self._pos = Position(side=side, lots=lots, price_open=px, ticket=len(self.fills) + 1)
        f = Fill(ts, side, lots, px, kind, comment, requested=lots)
        self.fills.append(f)
        return f

    def close_all(self, symbol: str) -> list[Fill]:
        if self._pos.flat:
            return []
        ts, px = self._next_open()
        side, lots = self._pos.side, self._pos.lots
        self._settle(px)
        f = Fill(ts, -side, lots, px, "exit", "close_all", requested=lots)
        self.fills.append(f)
        return [f]


# --------------------------------------------------------------------------- coduri
# Numerice intentionat: modulul MetaTrader5 exista doar pe Windows, iar testele nu-l au.
DONE_CODES = frozenset({10009,      # TRADE_RETCODE_DONE        - executat
                        10010})     # TRADE_RETCODE_DONE_PARTIAL - executat PARTIAL
RETRY_CODES = frozenset({10004,     # REQUOTE     - pretul s-a miscat intre citire si trimitere
                         10006,     # REJECT      - respins, de regula tot de pret
                         10021,     # PRICE_OFF   - fara cotatii in acest moment
                         10024,     # TOO_MANY_REQUESTS
                         10031})    # CONNECTION  - fara legatura cu serverul
RETCODE_NAMES = {10004: "REQUOTE", 10006: "REJECT", 10009: "DONE", 10010: "DONE_PARTIAL",
                 10013: "INVALID_REQUEST", 10014: "INVALID_VOLUME", 10015: "INVALID_PRICE",
                 10016: "INVALID_STOPS", 10018: "MARKET_CLOSED", 10019: "NO_MONEY",
                 10021: "PRICE_OFF", 10024: "TOO_MANY_REQUESTS", 10026: "SERVER_DISABLES_AT",
                 10027: "CLIENT_DISABLES_AT", 10031: "CONNECTION"}


class MT5Broker:
    """
    Thin wrapper over the MetaTrader5 terminal API.

    The terminal must already be running and logged in; this does not launch it. `deviation` is in
    points and is the maximum slippage the terminal may accept on a market order - set it wide enough
    that orders do not get rejected at 10:00, and narrow enough that a bad print does not fill.
    """

    def __init__(self, login: int | None = None, password: str | None = None,
                 server: str | None = None, magic: int = 20260913, deviation: int = 20,
                 filling: str = "IOC"):
        self.login, self.password, self.server = login, password, server
        self.magic, self.deviation, self.filling = magic, deviation, filling
        self._mt5 = None
        self.server_utc_offset_s: int | None = None   # masurat la connect(), NU ghicit
        self.order_retries = 2
        self._spec: dict[str, SymbolSpec] = {}

    @property
    def mt5(self):
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5  # noqa: N813
            except ImportError as e:                                  # pragma: no cover
                raise RuntimeError(
                    "pachetul MetaTrader5 nu e disponibil (merge doar pe Windows). "
                    "Foloseste PaperBroker pentru dezvoltare."
                ) from e
            self._mt5 = mt5
        return self._mt5

    def connect(self) -> None:
        kw = {}
        if self.login:
            kw = {"login": int(self.login), "password": self.password, "server": self.server}
        if not self.mt5.initialize(**kw):
            raise RuntimeError(f"MT5 initialize a esuat: {self.mt5.last_error()}")
        log.info("MT5 conectat: %s", self.mt5.account_info()._asdict().get("server"))

    # ---------------- fusul orar al serverului ----------------
    def measure_server_offset(self, symbol: str, tz: str = "America/New_York",
                              rth_start: str = "09:30") -> int:
        """
        Decalajul dintre ceasul serverului si UTC, in secunde.

        MT5 ne da epoci in ora SERVERULUI. Daca le etichetam ca UTC, taiem alta sesiune
        si nimic nu da eroare. Deci il masuram in DOUA feluri care nu depind unul de altul
        si cerem sa fie de acord:

          A. epoca ultimului tick minus ceasul nostru, rotunjita la sfert de ora
          B. minutul din zi cu cel mai mare volum trebuie sa fie deschiderea cash (09:30 NY)

        Daca difera, ridicam. Un decalaj gresit e cea mai scumpa eroare tacuta din tot sistemul.
        """
        import time as _time

        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None or not getattr(tick, "time", 0):
            raise RuntimeError(f"fara tick pentru {symbol}: {self.mt5.last_error()}")
        a = int(round((float(tick.time) - _time.time()) / 900.0)) * 900

        r = self.mt5.copy_rates_from_pos(symbol, self.mt5.TIMEFRAME_M1, 0, 20000)
        b = None
        if r is not None and len(r) > 2000:
            d = pd.DataFrame(r)
            vol = "tick_volume" if "tick_volume" in d.columns else "real_volume"
            srv = pd.to_datetime(d["time"], unit="s", utc=True)
            mod = srv.dt.hour * 60 + srv.dt.minute
            busiest = int(d.groupby(mod)[vol].median().idxmax())
            # ce minut-din-zi UTC inseamna rth_start in tz, pentru ultima zi din serie
            day = srv.iloc[-1].tz_convert(tz).normalize()
            want = (day + pd.Timedelta(hours=int(rth_start[:2]), minutes=int(rth_start[3:]))
                    ).tz_convert("UTC")
            b = int(round(((busiest - (want.hour * 60 + want.minute)) % 1440) / 15.0)) * 15 * 60
            if b > 12 * 3600:
                b -= 24 * 3600

        if b is not None and abs(a - b) > 900:
            raise RuntimeError(
                f"decalajul serverului nu se confirma: tick spune {a/3600:+.2f}h, "
                f"volumul spune {b/3600:+.2f}h. NU pornesc pe un fus orar nesigur.")
        self.server_utc_offset_s = a
        log.info("decalaj server->UTC: %+.2f h (tick) / %s (volum)", a / 3600,
                 f"{b/3600:+.2f} h" if b is not None else "nemasurat")
        return a

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        """Specificatia reala, din terminal. Singura sursa de adevar pentru marimea contractului."""
        if symbol in self._spec:
            return self._spec[symbol]
        i = self.mt5.symbol_info(symbol)
        if i is None:
            raise RuntimeError(f"simbolul {symbol} nu exista in Market Watch: {self.mt5.last_error()}")
        sp = SymbolSpec(contract_size=float(i.trade_contract_size),
                        lot_step=float(i.volume_step), min_lot=float(i.volume_min),
                        max_lot=float(i.volume_max), digits=int(i.digits))
        self._spec[symbol] = sp
        log.info("spec %s: contract_size=%s lot_step=%s min=%s max=%s",
                 symbol, sp.contract_size, sp.lot_step, sp.min_lot, sp.max_lot)
        return sp

    def is_connected(self) -> bool:
        try:
            return self.mt5.terminal_info() is not None and self.mt5.account_info() is not None
        except Exception:
            return False

    def equity(self) -> float:
        return float(self.mt5.account_info().equity)

    def balance(self) -> float:
        return float(self.mt5.account_info().balance)

    def bars(self, symbol: str, count: int) -> pd.DataFrame:
        r = self.mt5.copy_rates_from_pos(symbol, self.mt5.TIMEFRAME_M1, 0, int(count))
        if r is None or len(r) == 0:
            raise RuntimeError(f"fara bare pentru {symbol}: {self.mt5.last_error()}")
        if self.server_utc_offset_s is None:
            raise RuntimeError("decalajul serverului nu a fost masurat - cheama "
                               "measure_server_offset() inainte de prima citire de bare")
        d = pd.DataFrame(r)
        # MT5 stampileaza barele in ora SERVERULUI, ca epoca unix. Scadem decalajul masurat
        # la conectare si abia apoi etichetam UTC. Conversia se face o singura data, aici.
        d.index = pd.to_datetime(d["time"].astype("int64") - int(self.server_utc_offset_s),
                                 unit="s", utc=True)
        d = d.rename(columns={"tick_volume": "volume"})
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in d.columns]
        return d[keep].astype(float)

    def position(self, symbol: str) -> Position:
        ps = self.mt5.positions_get(symbol=symbol)
        if not ps:
            return Position()
        mine = [p for p in ps if p.magic == self.magic] or list(ps)
        net = sum((1 if p.type == self.mt5.POSITION_TYPE_BUY else -1) * p.volume for p in mine)
        if abs(net) < 1e-9:
            return Position()
        first = mine[0]
        return Position(side=1 if net > 0 else -1, lots=abs(net),
                        price_open=float(first.price_open), ticket=int(first.ticket))

    def tick(self, symbol: str) -> tuple[float, float]:
        t = self.mt5.symbol_info_tick(symbol)
        if t is None:
            raise RuntimeError(f"fara tick pentru {symbol}")
        return float(t.bid), float(t.ask)

    def _fill_mode(self, symbol: str):
        info = self.mt5.symbol_info(symbol)
        modes = {"IOC": self.mt5.ORDER_FILLING_IOC, "FOK": self.mt5.ORDER_FILLING_FOK,
                 "RETURN": self.mt5.ORDER_FILLING_RETURN}
        want = modes.get(self.filling.upper(), self.mt5.ORDER_FILLING_IOC)
        if info is not None and hasattr(info, "filling_mode"):
            for name, m in modes.items():
                if info.filling_mode & (1 << list(modes).index(name)):
                    return want
        return want

    def _send(self, symbol: str, side: int, lots: float, comment: str, reduce_ticket=None) -> Fill:
        bid, ask = self.tick(symbol)
        price = ask if side > 0 else bid
        req = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": self.mt5.ORDER_TYPE_BUY if side > 0 else self.mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": int(self.deviation),
            "magic": int(self.magic),
            "comment": comment[:31],
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._fill_mode(symbol),
        }
        if reduce_ticket is not None:
            req["position"] = int(reduce_ticket)
        last = None
        for attempt in range(self.order_retries + 1):
            if attempt:                       # pretul s-a miscat - il reluam inainte de reincercare
                bid, ask = self.tick(symbol)
                req["price"] = ask if side > 0 else bid
            res = self.mt5.order_send(req)
            if res is None:
                last = OrderRejected(f"order_send a intors None: {self.mt5.last_error()}",
                                     None, fatal=False)
                log.warning("%s (incercarea %d)", last, attempt + 1)
                continue
            rc = int(res.retcode)
            if rc in DONE_CODES:
                got, want = float(res.volume), float(lots)
                if abs(got - want) > 1e-9:
                    log.warning("EXECUTIE PARTIALA: cerut %.2f, executat %.2f (retcode %d) - "
                                "pozitia ramane mai mica decat a cerut politica", want, got, rc)
                return Fill(pd.Timestamp.now(tz="UTC"), side, got, float(res.price),
                            "entry" if reduce_ticket is None else "exit", comment, requested=want)
            fatal = rc not in RETRY_CODES
            last = OrderRejected(f"ordin respins: {rc} {RETCODE_NAMES.get(rc, '?')} "
                                 f"{getattr(res, 'comment', '')} {self.mt5.last_error()}",
                                 rc, fatal=fatal)
            if fatal:
                raise last
            log.warning("%s (incercarea %d din %d)", last, attempt + 1, self.order_retries + 1)
        raise last if last else OrderRejected("ordin esuat fara raspuns", None, fatal=False)

    def market_order(self, symbol: str, side: int, lots: float, comment: str = "") -> Fill:
        cur = self.position(symbol)
        if not cur.flat and cur.side == side:
            return Fill(pd.Timestamp.now(tz="UTC"), side, 0.0, 0.0, "noop",
                        "already on this side", requested=0.0)
        if not cur.flat:
            self.close_all(symbol)
        return self._send(symbol, side, lots, comment)

    def close_all(self, symbol: str) -> list[Fill]:
        out = []
        for p in (self.mt5.positions_get(symbol=symbol) or []):
            if p.magic != self.magic:
                continue
            side = -1 if p.type == self.mt5.POSITION_TYPE_BUY else 1
            out.append(self._send(symbol, side, p.volume, "flatten", reduce_ticket=p.ticket))
        return out
