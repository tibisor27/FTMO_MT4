"""
The things that stop the bot before the prop firm does.

Every limit here is set INSIDE the firm's limit, never equal to it. A rule you hit at exactly the
firm's threshold is a rule you have already broken, because their equity snapshot and yours are not
the same number at the same instant. The margin is the point.

`check()` returns None when everything is fine, or a string reason. A reason means: flatten, halt,
and do not trade again today.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("live.guards")


@dataclass
class Guards:
    max_daily_loss_pct: float = 0.05      # limita firmei; vezi nota din config
    max_total_loss_pct: float = 0.10      # limita firmei
    max_orders_per_day: int = 20
    max_lots: float = 20.0                # plafon absolut de siguranta, nu plafon de risc
    max_notional_mult: float = 5.0        # plafonul care chiar leaga: notional <= x * echitate
    halt_file: str = "HALT"

    def check(self, *, equity: float, day_start_equity: float, initial_balance: float,
              orders_today: int) -> str | None:
        if Path(self.halt_file).exists():
            return f"fisierul de oprire '{self.halt_file}' exista"
        if day_start_equity > 0:
            dd = (day_start_equity - equity) / day_start_equity
            if dd >= self.max_daily_loss_pct:
                return (f"pierdere zilnica {dd*100:.2f}% >= limita proprie "
                        f"{self.max_daily_loss_pct*100:.2f}%")
        if initial_balance > 0:
            tot = (initial_balance - equity) / initial_balance
            if tot >= self.max_total_loss_pct:
                return (f"pierdere totala {tot*100:.2f}% >= limita proprie "
                        f"{self.max_total_loss_pct*100:.2f}%")
        if orders_today >= self.max_orders_per_day:
            return f"{orders_today} ordine azi >= plafonul {self.max_orders_per_day}"
        return None

    def clamp_lots(self, lots: float) -> float:
        if lots > self.max_lots:
            log.warning("marime %.2f taiata la plafonul %.2f", lots, self.max_lots)
        return min(lots, self.max_lots)


def size_lots(*, equity: float, price: float, contract_size: float,
              strategy_daily_vol: float, target_daily_vol: float,
              lot_step: float = 0.01, min_lot: float = 0.01, max_lots: float = 20.0,
              max_notional_mult: float | None = 5.0) -> float:
    """
    Translate a volatility target into lots.

    leverage = target / the strategy's OWN daily volatility measured at 1x in the backtest.
    That second number is a constant you update when you re-run the backtest - it is not estimated
    live, because estimating it live introduces a feedback loop between sizing and realised vol.

    `max_notional_mult` e un DISJUNCTOR, nu un instrument de risc. Prinde o tinta de
    volatilitate absurda sau un `strategy_daily_vol` ramas la zero dupa o editare gresita.
    NU prinde un `contract_size` gresit: acolo noționalul calculat ramane corect si doar
    numarul de loturi e gresit. Pentru asta exista verificarea contra `symbol_info()`
    de la pornire, in runner. Doua pericole diferite, doua aparari diferite.
    """
    if min(price, contract_size, strategy_daily_vol) <= 0:
        return 0.0
    lev = target_daily_vol / strategy_daily_vol
    if max_notional_mult is not None and lev > max_notional_mult:
        log.error("levier cerut %.2f peste disjunctorul de notional %.2f - taiat. "
                  "Daca nu te asteptai la asta, opreste si verifica tinta de volatilitate.",
                  lev, max_notional_mult)
        lev = float(max_notional_mult)
    notional = equity * lev
    lots = notional / (contract_size * price)
    lots = round(lots / lot_step) * lot_step
    if lots > max_lots:
        # max_lots e un numar ABSOLUT intr-o lume de marimi relative: la un cont mic nu se
        # atinge niciodata, la unul mare incepe sa taie. Daca se declanseaza, vrei sa stii.
        log.warning("marime %.2f taiata la plafonul absolut max_lots=%.2f "
                    "(= %.0f%% din ce cerea politica). Plafonul asta nu scaleaza cu contul.",
                    lots, max_lots, 100.0 * max_lots / lots)
        lots = max_lots
    return round(lots, 4) if lots >= min_lot else 0.0
