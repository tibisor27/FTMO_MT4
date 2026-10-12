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
    lots: float
    price: float
    kind: str                # "entry" | "exit" | "reverse"
    comment: str = ""


class Broker(Protocol):
    def connect(self) -> None: ...
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
                return Fill(ts, side, 0.0, px, "noop", "already on this side")
            self._settle(px)
            kind = "reverse"
        if side != 0 and lots > 0:
            self._pos = Position(side=side, lots=lots, price_open=px, ticket=len(self.fills) + 1)
        f = Fill(ts, side, lots, px, kind, comment)
        self.fills.append(f)
        return f

    def close_all(self, symbol: str) -> list[Fill]:
        if self._pos.flat:
            return []
        ts, px = self._next_open()
        side, lots = self._pos.side, self._pos.lots
        self._settle(px)
        f = Fill(ts, -side, lots, px, "exit", "close_all")
        self.fills.append(f)
        return [f]


# --------------------------------------------------------------------------- MetaTrader 5
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
        d = pd.DataFrame(r)
        # MT5 stamps bars in SERVER time expressed as a unix epoch; converting once, here.
        d.index = pd.to_datetime(d["time"], unit="s", utc=True)
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
        res = self.mt5.order_send(req)
        if res is None or res.retcode != self.mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"ordin respins: {res.retcode if res else 'None'} "
                               f"{getattr(res, 'comment', '')} {self.mt5.last_error()}")
        return Fill(pd.Timestamp.now(tz="UTC"), side, float(res.volume), float(res.price),
                    "entry" if reduce_ticket is None else "exit", comment)

    def market_order(self, symbol: str, side: int, lots: float, comment: str = "") -> Fill:
        cur = self.position(symbol)
        if not cur.flat and cur.side == side:
            return Fill(pd.Timestamp.now(tz="UTC"), side, 0.0, 0.0, "noop", "already on this side")
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
