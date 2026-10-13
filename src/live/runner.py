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
from src.live.broker import Broker, OrderRejected, Position
from src.live.clock import Blackout, SessionClock
from src.live.guards import Guards, size_lots
from src.live.sizing import SizingPolicy, trading_days_left
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
        self.guards.max_notional_mult = float(g.get("max_notional_mult", 5.0))
        self.state = LiveState.load(self.root / cfg["paths"]["state"])
        self.state.path = str(self.root / cfg["paths"]["state"])
        self.symbol = cfg["symbol"]

        # cate treziri la rand fara terminal inainte sa consideram ca e o cadere reala
        self.max_conn_fails = int(cfg.get("session", {}).get("max_conn_fails", 6))
        self.conn_fails = 0

        # politica de marime
        sz = cfg.get("sizing", {}) or {}
        self.sizing_mode = str(sz.get("mode", "fixed")).lower()
        self.policy: SizingPolicy | None = None
        self.horizon_days = int(sz.get("horizon_days", 126))
        self.phase_targets = {1: float(sz.get("phase1_target_pct", 0.10)),
                              2: float(sz.get("phase2_target_pct", 0.05))}
        if self.sizing_mode == "policy":
            self.policy = SizingPolicy.from_csv(self.root / sz["policy_csv"])
            if self.horizon_days > self.policy.horizon:
                raise ValueError(f"orizontul din config ({self.horizon_days}) depaseste "
                                 f"tabelul ({self.policy.horizon} zile)")
            if self.state.phase not in self.policy.table:
                self.state.phase = int(sz.get("phase", 1))
        self.spec = None

    # ---------------- verificarea de dinainte de prima bara ----------------
    def preflight(self) -> None:
        """
        Confrunta fisierul de configurare cu ce spune TERMINALUL. Se cheama o data, la pornire,
        inainte de orice ordin. Daca marimea contractului difera, nu pornim: nicio garda de mai
        jos nu poate prinde un multiplicator gresit, pentru ca ea vede acelasi numar gresit.
        """
        self.spec = self.broker.symbol_spec(self.symbol)
        r = self.cfg["risk"]
        bad = []
        for key, real in (("contract_size", self.spec.contract_size),
                          ("lot_step", self.spec.lot_step),
                          ("min_lot", self.spec.min_lot)):
            want = float(r[key])
            if abs(want - real) > 1e-9:
                bad.append(f"{key}: config {want} vs terminal {real}")
        if bad:
            raise RuntimeError("specificatia simbolului nu se potriveste cu configurarea:\n  "
                               + "\n  ".join(bad)
                               + "\nCorecteaza configs/live/*.yaml dupa terminal, nu invers.")
        self._check_state_matches_account()
        log.info("preflight OK: %s contract_size=%s lot_step=%s min_lot=%s",
                 self.symbol, self.spec.contract_size, self.spec.lot_step, self.spec.min_lot)

    def _check_state_matches_account(self, tol: float = 0.30) -> None:
        """
        Starea salvata descrie un PROGRAM, nu un cont oarecare: `initial_balance` si
        `phase_start_balance` sunt reperele fata de care se masoara si pierderea totala, si
        pozitia in tabelul de marime. Daca pornesti pe un cont de alta marime fara sa stergi
        fisierul de stare, amandoua raman la valorile vechi - garda te-ar opri instantaneu cu
        "pierdere totala 80%", iar politica ar citi din coltul in care renunta. Esec tacut si
        foarte derutant, deci il facem zgomotos.
        """
        bal = float(self.broker.balance())
        if bal <= 0:
            return
        for name, ref in (("initial_balance", self.state.initial_balance),
                          ("phase_start_balance", self.state.phase_start_balance)):
            if ref and abs(bal - ref) / ref > tol:
                raise RuntimeError(
                    f"starea salvata nu se potriveste cu contul: {name}={ref:,.2f} dar contul "
                    f"are balanta {bal:,.2f} ({abs(bal-ref)/ref:.0%} diferenta).\n"
                    f"Daca ai schimbat contul, goleste {self.state.path} si reporneste "
                    f"(programul incepe de la zero: zile ramase, faza, repere)."
                )

    # ---------------- cat de mare tranzactionam azi ----------------
    def vol_target(self, equity: float, now: pd.Timestamp) -> tuple[float, dict]:
        """Tinta de volatilitate zilnica (fractie, nu procente) plus contextul, pentru jurnal."""
        if self.sizing_mode != "policy" or self.policy is None:
            return float(self.cfg["risk"]["target_daily_vol"]), {"mode": "fixed"}
        today = str(self.clock.local(now).date())
        if not self.state.program_start:
            self.state.start_program(today, self.broker.balance())
        base = self.state.phase_start_balance or self.broker.balance()
        eq_rel = (equity / base - 1.0) if base > 0 else 0.0
        days = trading_days_left(self.state.program_start, today, self.horizon_days)
        pct = self.policy.vol_target_pct(self.state.phase, days, eq_rel)
        return pct / 100.0, {"mode": "policy", "phase": self.state.phase,
                             "days_left": days, "eq_rel": eq_rel, "vol_pct": pct}

    def _maybe_advance_phase(self, equity: float) -> None:
        base = self.state.phase_start_balance
        tgt = self.phase_targets.get(self.state.phase)
        if self.policy is None or not base or tgt is None:
            return
        if self.state.phase + 1 in self.policy.table and equity >= base * (1.0 + tgt):
            self.state.advance_phase(equity)
            self.state.save()

    def _adopt_orders_today(self) -> None:
        """Reconstruieste contorul de ordine din jurnalul de executii: o repornire nu are voie
        sa-l piarda, altfel plafonul zilnic numara in minus."""
        p = self.root / self.cfg["paths"]["fills"]
        if not p.exists() or p.stat().st_size == 0:
            return
        try:
            d = pd.read_csv(p)
            if d.empty or "ts" not in d.columns:
                return
            ts = pd.to_datetime(d["ts"], utc=True, errors="coerce")
            day = pd.Series(self.clock.local(t).date() if pd.notna(t) else None for t in ts)
            n = int((day.astype(str) == self.state.trade_date).sum())
            if n > self.state.orders_today:
                log.info("contorul de ordine refacut din jurnal: %d (era %d)",
                         n, self.state.orders_today)
                self.state.orders_today = n
        except Exception:                                   # pragma: no cover
            log.exception("nu pot reface contorul de ordine din %s", p)

    # ---------------- logging of intent, not just of orders ----------------
    def _record(self, d: Decision, lots: float, note: str, ctx: dict | None = None) -> None:
        ctx = ctx or {}
        p = self.root / self.cfg["paths"]["decisions"]
        p.parent.mkdir(parents=True, exist_ok=True)
        new = (not p.exists()) or p.stat().st_size == 0   # an existing but EMPTY file still needs a header
        with p.open("a", encoding="utf-8") as f:
            if new:
                f.write("bar,close,ub,lb,current,target,lots,vol_target_pct,phase,"
                        "days_left,eq_rel,reason,note\n")
            f.write(f"{d.bar},{d.close},{d.ub},{d.lb},{d.current},{d.target},{lots},"
                    f"{ctx.get('vol_pct','')},{ctx.get('phase','')},{ctx.get('days_left','')},"
                    f"{ctx.get('eq_rel','')},{d.reason},{note}\n")

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

        # ---- 1. legatura. O cadere scurta NU e motiv de oprire a zilei. ----
        if not self.broker.is_connected():
            self.conn_fails += 1
            if self.conn_fails < self.max_conn_fails:
                log.warning("terminalul nu raspunde (%d/%d) - reincerc",
                            self.conn_fails, self.max_conn_fails)
                return f"deconectat ({self.conn_fails}/{self.max_conn_fails})"
            try:                                   # cadere sustinuta: INCHIDEM, apoi oprim
                if not self.broker.position(self.symbol).flat:
                    self._flatten("deconectare sustinuta")
            except Exception:                      # pragma: no cover
                log.exception("NU am putut inchide pozitia la deconectare - ramane deschisa, "
                              "verifica manual terminalul")
            self.state.halt(f"terminalul nu e conectat ({self.conn_fails} incercari)")
            return "oprit: deconectat"
        self.conn_fails = 0

        equity = self.broker.equity()
        local_date = str(self.clock.local(now).date())
        if self.state.roll_day(local_date, equity):
            self._adopt_orders_today()
            self.state.save()

        # ---- 2. pozitia reala, INAINTE de orice ramura care poate iesi din functie ----
        pos = self._sync()

        # ---- 3. inchiderea de final. Necondiționata: nici oprirea de ieri, nici o fereastra
        #         de stiri nu au voie sa lase o pozitie deschisa peste noapte. ----
        if self.clock.is_flatten_time(now):
            if not pos.flat:
                b = self.blackout.blocked(now)
                if b:
                    log.error("inchid la final DESI suntem in fereastra restrictionata (%s); "
                              "o pozitie peste noapte ar fi mai rea - noteaza incidentul", b)
                self._flatten("sfarsit de sesiune")
                return "flatten EOD"
            return "flat, sfarsit de sesiune"

        if self.state.halted:
            return f"oprit: {self.state.halt_reason}"

        # ---- 4. garzile ----
        reason = self.guards.check(equity=equity, day_start_equity=self.state.day_start_equity,
                                   initial_balance=self.state.initial_balance,
                                   orders_today=self.state.orders_today)
        if reason:
            if not pos.flat:
                self._flatten(f"guard: {reason}")
            self.state.halt(reason)
            return f"oprit: {reason}"

        # ---- 5. blackout: blocheaza actiuni NOI, nu inchiderea de final (deja trecuta) ----
        blocked = self.blackout.blocked(now)
        if blocked:
            self.state.save()
            return f"blackout: {blocked}"

        self._maybe_advance_phase(equity)
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
        tvol, ctx = self.vol_target(equity, now)
        if d.target != d.current:
            if d.target == 0:
                self._flatten(d.reason)
                note = "inchis"
            else:
                r = self.cfg["risk"]
                spec = self.spec
                lots = size_lots(
                    equity=equity, price=d.close,
                    contract_size=float(spec.contract_size if spec else r["contract_size"]),
                    strategy_daily_vol=float(r["strategy_daily_vol"]),
                    target_daily_vol=tvol,
                    lot_step=float(spec.lot_step if spec else r["lot_step"]),
                    min_lot=float(spec.min_lot if spec else r["min_lot"]),
                    max_lots=float(self.guards.max_lots),
                    max_notional_mult=float(self.guards.max_notional_mult))
                if lots <= 0:
                    note = "marime 0 dupa rotunjire - nu trimit nimic"
                    log.warning("%s (echitate %.0f, pret %.1f, tinta %.3f%%/zi)",
                                note, equity, d.close, tvol * 100)
                else:
                    try:
                        fill = self.broker.market_order(self.symbol, d.target, lots, d.reason)
                    except OrderRejected as e:
                        # O bara ratata nu e o zi pierduta. Fatal = oprim, altfel mergem mai departe.
                        note = f"ordin respins: {e}"
                        log.error("%s", note)
                        if e.fatal:
                            self.state.halt(f"ordin respins fatal: {e}")
                        self.state.mark(bar_utc)
                        self.state.save()
                        self._record(d, 0.0, note, ctx)
                        return note
                    self._record_fill(fill)
                    self.state.orders_today += 1
                    # marimea CREZUTA vine din executie, nu din cerere
                    self.state.believed_side = d.target if fill.lots > 0 else 0
                    self.state.believed_lots = float(fill.lots)
                    note = f"ordin {d.target:+d} {fill.lots}"
                    if fill.partial:
                        note += f" (PARTIAL din {fill.requested})"
                    log.info("%s @ %.2f (%s) tinta %.2f%%/zi", note, fill.price, d.reason,
                             tvol * 100)

        self.state.mark(bar_utc)
        self.state.save()
        self._record(d, lots, note, ctx)
        return f"{d.reason} -> {d.target:+d} {note}".strip()


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
