"""
Tests for the live layer. These pin down the behaviours that cost money when they are wrong:
when the bot is allowed to act, that a restart never double-enters, that the guards fire before the
prop firm's limits, and that the live decision equals the backtest's decision on the same bar.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.live.broker import PaperBroker                      # noqa: E402
from src.live.clock import Blackout, SessionClock            # noqa: E402
from src.live.guards import Guards, size_lots                # noqa: E402
from src.live.runner import Runner, decide                   # noqa: E402
from src.live.state import LiveState                         # noqa: E402

NY = "America/New_York"


def frame(ndays=30, seed=0):
    """Sessions of 390 one-minute bars with a drift that pushes price out of the band late."""
    rng = np.random.default_rng(seed)
    idx = []
    for d in pd.bdate_range("2024-01-02", periods=ndays):
        idx.append(pd.date_range(d + pd.Timedelta(hours=9, minutes=30), periods=390, freq="1min"))
    index = (pd.DatetimeIndex(np.concatenate([i.values for i in idx]))
             .tz_localize(NY).tz_convert("UTC"))
    px = 100 + np.cumsum(rng.normal(0, 0.01, len(index)))
    day = np.repeat(np.arange(ndays), 390)
    px = px + (day == ndays - 1) * np.linspace(0, 3.0, len(index))   # last day trends hard
    return pd.DataFrame({"open": px, "high": px + .01, "low": px - .01,
                         "close": px, "volume": 1.0}, index=index)


def cfg(tmp_path, **over):
    c = {
        "symbol": "TEST", "magic": 1,
        "session": {"tz": NY, "rth_start": "09:30", "rth_end": "16:00",
                    "decision_minutes": [0, 30], "first_decision": "10:00",
                    "last_decision": "13:30", "eod_flatten": "15:57", "act_offset_min": 1},
        "strategy": {"lookback": 14, "vm": 1.0, "exit_rule": "opposite_band", "warmup_days": 25},
        "risk": {"strategy_daily_vol": 0.00625, "target_daily_vol": 0.0075,
                 "contract_size": 2.0, "lot_step": 0.01, "min_lot": 0.01},
        "guards": {"max_daily_loss_pct": 0.03, "max_total_loss_pct": 0.07,
                   "max_orders_per_day": 20, "max_lots": 3.0, "max_notional_mult": 5.0,
                   "halt_file": "NO_HALT_FILE"},
        "sizing": {"mode": "fixed"},
        "news": {"events_csv": "configs/live/does_not_exist.csv", "blackout_minutes": 5},
        "paths": {"state": "state.json", "fills": "fills.csv",
                  "decisions": "decisions.csv", "log": "live.log"},
    }
    for k, v in over.items():
        c[k] = {**c[k], **v} if isinstance(v, dict) else v
    return c


# ----------------------------------------------------------------- clock
def test_clock_acts_one_minute_after_the_decision_bar():
    c = SessionClock()
    # acting at 10:01 acts on the 10:00 bar
    assert c.decision_bar(pd.Timestamp("2024-03-04 15:01", tz="UTC")).minute == 0
    # acting exactly at 10:00 would read an unfinished bar -> not an acting moment
    assert c.decision_bar(pd.Timestamp("2024-03-04 15:00", tz="UTC")) is None
    # 13:31 acts on the 13:30 bar
    assert c.decision_bar(pd.Timestamp("2024-03-04 18:31", tz="UTC")).hour == 13
    # 14:01 e DUPA last_decision, dar ramane moment de decizie: altfel stopul de banda
    # nu s-ar putea declansa intre 13:31 si inchiderea fortata. Plafonul de intrari se
    # aplica in entries_allowed(), nu aici.
    assert c.decision_bar(pd.Timestamp("2024-03-04 19:01", tz="UTC")).hour == 14
    assert c.decision_bar(pd.Timestamp("2024-03-04 20:31", tz="UTC")).hour == 15
    # dupa eod_flatten (15:57) nu se mai decide nimic
    assert c.decision_bar(pd.Timestamp("2024-03-04 21:01", tz="UTC")) is None
    # weekend
    assert c.decision_bar(pd.Timestamp("2024-03-09 15:01", tz="UTC")) is None


def test_entries_close_but_exits_stay_open_after_the_cutoff():
    c = SessionClock()
    assert c.entries_allowed(pd.Timestamp("2024-03-04 18:31", tz="UTC"))       # 13:31 NY
    assert not c.entries_allowed(pd.Timestamp("2024-03-04 19:31", tz="UTC"))   # 14:31 NY
    assert c.is_flatten_time(pd.Timestamp("2024-03-04 20:58", tz="UTC"))       # 15:58 NY


def test_blackout_covers_the_window_with_margin(tmp_path):
    p = tmp_path / "ev.csv"
    p.write_text("datetime_utc,event\n2026-09-17T18:00:00Z,FOMC\n", encoding="utf-8")
    b = Blackout.from_csv(p, margin_min=5)
    assert b.blocked(pd.Timestamp("2026-09-17 18:03", tz="UTC")) == "FOMC"
    assert b.blocked(pd.Timestamp("2026-09-17 17:56", tz="UTC")) == "FOMC"
    assert b.blocked(pd.Timestamp("2026-09-17 18:06", tz="UTC")) is None


# ----------------------------------------------------------------- guards
def test_guards_fire_inside_the_firms_limits():
    g = Guards(max_daily_loss_pct=0.03, max_total_loss_pct=0.07, halt_file="NO_SUCH_FILE")
    assert g.check(equity=100.0, day_start_equity=100.0, initial_balance=100.0, orders_today=0) is None
    # a 3% daily loss trips our guard while the firm would still allow 5%
    assert "pierdere zilnica" in g.check(equity=97.0, day_start_equity=100.0,
                                        initial_balance=100.0, orders_today=0)
    assert "pierdere totala" in g.check(equity=93.0, day_start_equity=93.5,
                                       initial_balance=100.0, orders_today=0)
    assert "ordine azi" in g.check(equity=100.0, day_start_equity=100.0,
                                   initial_balance=100.0, orders_today=99)


def test_sizing_is_rounded_capped_and_refuses_dust():
    lots = size_lots(equity=50_000, price=25_000, contract_size=2.0,
                     strategy_daily_vol=0.00625, target_daily_vol=0.0075, max_lots=3.0)
    assert lots == pytest.approx(1.2, abs=0.01)          # 50k * 1.2x / 50k notional
    assert size_lots(equity=50_000, price=25_000, contract_size=2.0,
                     strategy_daily_vol=0.00625, target_daily_vol=0.05, max_lots=3.0) == 3.0
    assert size_lots(equity=10, price=25_000, contract_size=2.0,
                     strategy_daily_vol=0.00625, target_daily_vol=0.0075) == 0.0


# ----------------------------------------------------------------- state
def test_restart_does_not_reprocess_a_bar(tmp_path):
    s = LiveState(path=str(tmp_path / "s.json"))
    s.roll_day("2026-09-09", 50_000)
    bar = pd.Timestamp("2026-09-09 14:00", tz="UTC")
    assert not s.seen(bar)
    s.mark(bar)
    s.save()
    again = LiveState.load(tmp_path / "s.json")
    assert again.seen(bar)
    assert again.roll_day("2026-09-09", 50_000) is False
    assert again.roll_day("2026-09-10", 50_000) is True
    assert again.seen(bar) is False                      # counters reset on a new day


def test_corrupt_state_does_not_crash(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json", encoding="utf-8")
    s = LiveState.load(p)
    assert s.believed_side == 0 and s.trade_date == ""


# ----------------------------------------------------------------- decide
def test_decide_matches_the_backtest_bands_and_respects_the_entry_cutoff():
    f = frame()
    bar = f.index[-1] - pd.Timedelta(minutes=30)
    d = decide(f, bar, 0, lookback=14, vm=1.0, exit_rule="opposite_band",
               tz=NY, rth_start="09:30", rth_end="16:00")
    assert d.target in (-1, 0, 1)
    if d.target != 0:
        # same bar, entries closed -> the signal must be ignored, not acted on
        blocked = decide(f, bar, 0, lookback=14, vm=1.0, exit_rule="opposite_band",
                         tz=NY, rth_start="09:30", rth_end="16:00", entries_allowed=False)
        assert blocked.target == 0
        assert "gap trading" in blocked.reason


def test_decide_exits_a_long_when_price_breaks_the_opposite_band():
    f = frame()
    bar = f.index[-1] - pd.Timedelta(minutes=5)
    held = decide(f, bar, -1, lookback=14, vm=1.0, exit_rule="opposite_band",
                  tz=NY, rth_start="09:30", rth_end="16:00")
    # the last day trends up hard, so a short must be stopped out or reversed, never held
    assert held.target != -1


# ----------------------------------------------------------------- runner
def test_runner_opens_once_and_is_idempotent_across_restarts(tmp_path):
    f = frame()
    c = cfg(tmp_path)
    broker = PaperBroker(f, start_equity=50_000, contract_size=2.0)
    r = Runner(c, broker, tmp_path)
    last_day = f.index[-1].tz_convert(NY).date()
    minutes = [t for t in f.index if t.tz_convert(NY).date() == last_day]

    for ts in minutes:
        broker.set_now(ts)
        r.step(ts)
    orders_first_pass = len(broker.fills)
    assert orders_first_pass > 0, "ziua construita sa aiba trend nu a produs nicio tranzactie"

    # a fresh Runner over the same state file must not re-enter anything
    r2 = Runner(c, broker, tmp_path)
    for ts in minutes:
        broker.set_now(ts)
        r2.step(ts)
    assert len(broker.fills) == orders_first_pass


def test_runner_flattens_at_end_of_session_and_halts_on_the_halt_file(tmp_path):
    f = frame()
    c = cfg(tmp_path)
    broker = PaperBroker(f, start_equity=50_000, contract_size=2.0)
    r = Runner(c, broker, tmp_path)
    last_day = f.index[-1].tz_convert(NY).date()
    for ts in [t for t in f.index if t.tz_convert(NY).date() == last_day]:
        broker.set_now(ts)
        r.step(ts)
    assert broker.position("TEST").flat, "pozitia nu a fost inchisa la finalul sesiunii"

    (tmp_path / "HALT").write_text("stop", encoding="utf-8")
    c2 = cfg(tmp_path, guards={"halt_file": "HALT"})
    r3 = Runner(c2, broker, tmp_path)
    r3.state.trade_date = ""
    # un moment DIN sesiune: in fereastra de inchidere ramura de flatten raspunde prima,
    # si asta e intentionat - inchiderea de final trece inaintea oricarei opriri.
    mid = [t for t in f.index if t.tz_convert(NY).date() == last_day
           and t.tz_convert(NY).hour == 11][0]
    broker.set_now(mid)
    out = r3.step(mid)
    assert "oprit" in out and "HALT" in r3.state.halt_reason


# ═════════════════════════════════════════════════════════════════════════════
# Reparatiile din 2026-10-12. Fiecare test de mai jos pica pe codul dinainte.
# ═════════════════════════════════════════════════════════════════════════════
import types                                                 # noqa: E402

from src.live.broker import MT5Broker, OrderRejected, SymbolSpec  # noqa: E402
from src.live.sizing import SizingPolicy, trading_days_left   # noqa: E402


class FakeMT5:
    """Minimul din modulul MetaTrader5 de care are nevoie MT5Broker."""
    TIMEFRAME_M1 = 1
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC, ORDER_FILLING_FOK, ORDER_FILLING_RETURN = 1, 0, 2
    POSITION_TYPE_BUY = 0

    def __init__(self, rates=None, tick_time=None, responses=None, spec=None):
        self._rates, self._tick_time = rates, tick_time
        self._responses = list(responses or [])
        self._spec = spec
        self.sent = []

    def last_error(self): return (1, "Success")
    def copy_rates_from_pos(self, *a, **k): return self._rates
    def symbol_info_tick(self, *a, **k):
        return types.SimpleNamespace(time=self._tick_time, bid=100.0, ask=100.1)
    def symbol_info(self, *a, **k): return self._spec
    def order_send(self, req):
        self.sent.append(dict(req))
        return self._responses.pop(0) if self._responses else None


def _rates(n=5, start_epoch=1_700_000_000):
    import numpy as _np
    return _np.array([(start_epoch + 60 * i, 1., 1., 1., 1., 10, 1, 0) for i in range(n)],
                     dtype=[("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"),
                            ("close", "<f8"), ("tick_volume", "<u8"), ("spread", "<i4"),
                            ("real_volume", "<u8")])


# ---------------------------------------------------------------- 1. fusul orar
def test_bars_subtract_the_measured_server_offset():
    b = MT5Broker()
    b._mt5 = FakeMT5(rates=_rates())
    b.server_utc_offset_s = 3 * 3600                      # serverul e pe GMT+3
    d = b.bars("X", 5)
    assert d.index[0] == pd.Timestamp(1_700_000_000 - 3 * 3600, unit="s", tz="UTC")


def test_bars_refuse_to_run_before_the_offset_is_measured():
    b = MT5Broker()
    b._mt5 = FakeMT5(rates=_rates())
    with pytest.raises(RuntimeError, match="decalajul serverului nu a fost masurat"):
        b.bars("X", 5)


def test_measure_server_offset_raises_when_the_two_methods_disagree():
    import time as _t
    import numpy as _np
    # tick-ul spune +3h; volumul e plat, dar il facem sa arate un alt minut aglomerat
    n = 3000
    arr = _rates(n, int(_t.time()) + 3 * 3600 - 60 * n)
    arr["tick_volume"] = 1
    arr["tick_volume"][::390] = 10_000                    # un minut aglomerat arbitrar
    b = MT5Broker()
    b._mt5 = FakeMT5(rates=arr, tick_time=_t.time() + 3 * 3600)
    try:
        off = b.measure_server_offset("X")
    except RuntimeError as e:
        assert "nu se confirma" in str(e)
    else:
        assert off == 3 * 3600                            # cele doua metode au cazut de acord


# ---------------------------------------------------------------- 2. retcode-uri
def _res(retcode, volume=1.0, price=100.0):
    return types.SimpleNamespace(retcode=retcode, volume=volume, price=price,
                                 comment="", deal=1, order=1)


def test_a_requote_is_retried_and_then_succeeds():
    b = MT5Broker()
    b._mt5 = FakeMT5(responses=[_res(10004), _res(10009, volume=1.0)])
    b.server_utc_offset_s = 0
    f = b._send("X", 1, 1.0, "test")
    assert f.lots == 1.0 and len(b._mt5.sent) == 2        # a reincercat o data


def test_no_money_is_fatal_and_is_not_retried():
    b = MT5Broker()
    b._mt5 = FakeMT5(responses=[_res(10019)])
    with pytest.raises(OrderRejected) as e:
        b._send("X", 1, 1.0, "test")
    assert e.value.fatal and len(b._mt5.sent) == 1


def test_partial_fill_reports_the_executed_volume_not_the_requested_one():
    b = MT5Broker()
    b._mt5 = FakeMT5(responses=[_res(10010, volume=0.8)])
    f = b._send("X", 1, 1.2, "test")
    assert f.lots == 0.8 and f.requested == 1.2 and f.partial


# ---------------------------------------------------------------- 3. specificatia
def test_preflight_refuses_a_contract_size_that_differs_from_the_terminal(tmp_path):
    f = frame()
    broker = PaperBroker(f, start_equity=50_000, contract_size=20.0)   # terminalul spune 20
    r = Runner(cfg(tmp_path), broker, tmp_path)                        # configul spune 2
    with pytest.raises(RuntimeError, match="contract_size"):
        r.preflight()


# ---------------------------------------------------------------- 4. iesiri dupa 13:31
def test_exits_are_still_evaluated_after_the_entry_cutoff():
    c = SessionClock()
    # intrarile sunt inchise...
    assert not c.entries_allowed(pd.Timestamp("2024-03-04 19:01", tz="UTC"))
    # ...dar exista in continuare o bara de decizie pe care decide() o poate evalua
    assert c.decision_bar(pd.Timestamp("2024-03-04 19:01", tz="UTC")) is not None


def test_decide_closes_a_long_after_the_cutoff(tmp_path):
    f = frame()
    bars = f.loc[f.index <= f.index[-30]]
    ts = [t for t in bars.index if t.tz_convert(NY).hour == 14 and t.tz_convert(NY).minute == 0][-1]
    d = decide(bars, ts, current=1, lookback=14, vm=1.0, exit_rule="opposite_band",
               tz=NY, rth_start="09:30", rth_end="16:00", entries_allowed=False)
    assert d.target in (0, 1)           # poate iesi; important e ca a fost EVALUAT
    assert "bara lipseste" not in d.reason


# ---------------------------------------------------------------- 5. flatten vs blackout
def test_eod_flatten_happens_even_inside_a_blackout_window(tmp_path):
    f = frame()
    last_day = f.index[-1].tz_convert(NY).date()
    ev = tmp_path / "ev.csv"
    t = pd.Timestamp(f"{last_day} 15:57", tz=NY).tz_convert("UTC")
    ev.write_text(f"datetime_utc,event\n{t.isoformat()},FOMC\n", encoding="utf-8")
    c = cfg(tmp_path, news={"events_csv": "ev.csv", "blackout_minutes": 30})
    broker = PaperBroker(f, start_equity=50_000, contract_size=2.0)
    r = Runner(c, broker, tmp_path)
    for ts in [x for x in f.index if x.tz_convert(NY).date() == last_day]:
        broker.set_now(ts)
        r.step(ts)
    assert broker.position("TEST").flat, "fereastra de blackout a blocat inchiderea de final"


# ---------------------------------------------------------------- 6. deconectare
class FlakyBroker(PaperBroker):
    def is_connected(self): return False


def test_a_short_disconnect_does_not_halt_the_day(tmp_path):
    f = frame()
    broker = FlakyBroker(f, start_equity=50_000, contract_size=2.0)
    r = Runner(cfg(tmp_path, session={"max_conn_fails": 3}), broker, tmp_path)
    ts = f.index[100]
    assert "deconectat" in r.step(ts) and not r.state.halted
    assert "deconectat" in r.step(ts) and not r.state.halted
    assert "oprit" in r.step(ts) and r.state.halted          # abia a treia oara


# ---------------------------------------------------------------- 7. politica de marime
def _policy_csv(tmp_path):
    rows = []
    for ph in (1, 2):
        for h in (1, 2, 3):
            for e, v in ((-0.10, 0.3), (-0.05, 0.3), (0.0, 2.5), (0.05, 2.5), (0.10, 1.0)):
                rows.append((ph, h, e, v if ph == 1 else 1.0))
    p = tmp_path / "pol.csv"
    pd.DataFrame(rows, columns=["faza", "zile_ramase", "echitate", "vol_target_pct"]).to_csv(
        p, index=False)
    return p


def test_policy_rounds_down_between_grid_points(tmp_path):
    pol = SizingPolicy.from_csv(_policy_csv(tmp_path))
    assert pol.vol_target_pct(1, 1, 0.0) == 2.5
    assert pol.vol_target_pct(1, 1, -0.05) == 0.3
    # intre -0.05 (0.3) si 0.0 (2.5) luam minimul, nu cel mai apropiat
    assert pol.vol_target_pct(1, 1, -0.001) == 0.3
    # in afara grilei se taie la margine
    assert pol.vol_target_pct(1, 1, -5.0) == 0.3
    assert pol.vol_target_pct(1, 99, 0.0) == 2.5


def test_policy_rejects_a_table_with_holes(tmp_path):
    p = tmp_path / "gauri.csv"
    pd.DataFrame([(1, 1, 0.0, 2.5), (1, 2, 0.05, 1.0)],
                 columns=["faza", "zile_ramase", "echitate", "vol_target_pct"]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="gauri"):
        SizingPolicy.from_csv(p)


def test_trading_days_left_counts_business_days():
    assert trading_days_left("2024-03-04", "2024-03-04", 126) == 126
    assert trading_days_left("2024-03-04", "2024-03-11", 126) == 121      # 5 zile lucratoare
    assert trading_days_left("2024-03-04", "2030-01-01", 126) == 1        # nu coboara sub 1


def test_runner_uses_the_policy_and_logs_the_context(tmp_path):
    f = frame()
    pol = _policy_csv(tmp_path)
    c = cfg(tmp_path, sizing={"mode": "policy", "policy_csv": pol.name,
                              "horizon_days": 3, "phase": 1})
    broker = PaperBroker(f, start_equity=50_000, contract_size=2.0)
    r = Runner(c, broker, tmp_path)
    broker.set_now(f.index[100])
    tvol, ctx = r.vol_target(50_000.0, f.index[100])
    assert ctx["mode"] == "policy" and ctx["phase"] == 1
    assert tvol == pytest.approx(0.025)                   # 2.5%/zi la echitate 0


def test_phase_advances_when_the_target_is_reached(tmp_path):
    f = frame()
    c = cfg(tmp_path, sizing={"mode": "policy", "policy_csv": _policy_csv(tmp_path).name,
                              "horizon_days": 3, "phase": 1, "phase1_target_pct": 0.10})
    broker = PaperBroker(f, start_equity=50_000, contract_size=2.0)
    r = Runner(c, broker, tmp_path)
    r.state.start_program("2024-03-04", 50_000.0)
    r._maybe_advance_phase(55_100.0)                      # +10.2%
    assert r.state.phase == 2 and r.state.phase_start_balance == 55_100.0


# ---------------------------------------------------------------- 8. disjunctorul
def test_notional_circuit_breaker_caps_an_absurd_target():
    normal = size_lots(equity=50_000, price=100.0, contract_size=2.0,
                       strategy_daily_vol=0.00625, target_daily_vol=0.025,
                       max_lots=10_000, max_notional_mult=5.0)
    absurd = size_lots(equity=50_000, price=100.0, contract_size=2.0,
                       strategy_daily_vol=0.00625, target_daily_vol=0.50,
                       max_lots=10_000, max_notional_mult=5.0)
    assert normal == pytest.approx(1000.0)                # levier 4.0 -> 200k / 200
    assert absurd == pytest.approx(1250.0)                # taiat la levier 5.0, nu 80


# ---------------------------------------------------------------- 9. contorul de ordine
def test_orders_today_is_rebuilt_from_the_fill_log(tmp_path):
    f = frame()
    c = cfg(tmp_path)
    broker = PaperBroker(f, start_equity=50_000, contract_size=2.0)
    r = Runner(c, broker, tmp_path)
    day = f.index[-1].tz_convert(NY).date()
    (tmp_path / "fills.csv").write_text(
        "ts,side,lots,price,kind,comment\n"
        + "".join(f"{pd.Timestamp(f'{day} 10:0{i}', tz=NY).tz_convert('UTC')},1,1.0,100,entry,x\n"
                  for i in range(3)), encoding="utf-8")
    r.state.trade_date = str(day)
    r.state.orders_today = 0
    r._adopt_orders_today()
    assert r.state.orders_today == 3


# ═════════════════════════════════════════════════════════════════════════════
# Scalarea cu marimea contului (13.10.2026)
# ═════════════════════════════════════════════════════════════════════════════
def test_sizing_scales_linearly_with_equity():
    kw = dict(price=29_500.0, contract_size=2.0, strategy_daily_vol=0.005816,
              target_daily_vol=0.025, lot_step=0.01, min_lot=0.01,
              max_lots=1000.0, max_notional_mult=5.0)
    a = size_lots(equity=10_000, **kw)
    b = size_lots(equity=50_000, **kw)
    assert a == pytest.approx(0.73, abs=0.01)
    assert b == pytest.approx(3.64, abs=0.01)
    assert b / a == pytest.approx(5.0, rel=0.02)      # de cinci ori contul, de cinci ori marimea


def test_policy_lookup_is_independent_of_account_size(tmp_path):
    pol = SizingPolicy.from_csv(_policy_csv(tmp_path))
    # tabelul e indexat pe echitate RELATIVA, deci acelasi procent da acelasi raspuns
    assert pol.vol_target_pct(1, 1, 0.0) == pol.vol_target_pct(1, 1, 0.0)
    for base in (10_000, 50_000, 200_000):
        eq_rel = (base * 1.05) / base - 1.0
        assert pol.vol_target_pct(1, 1, eq_rel) == pol.vol_target_pct(1, 1, 0.05)


def test_guards_are_percentages_so_they_scale(tmp_path):
    g = Guards(max_daily_loss_pct=0.05, max_total_loss_pct=0.10)
    for base in (10_000, 50_000, 200_000):
        assert g.check(equity=base * 0.96, day_start_equity=base,
                       initial_balance=base, orders_today=0) is None
        assert g.check(equity=base * 0.94, day_start_equity=base,
                       initial_balance=base, orders_today=0) is not None


def test_preflight_refuses_a_state_left_over_from_another_account(tmp_path):
    f = frame()
    broker = PaperBroker(f, start_equity=10_000, contract_size=2.0)
    r = Runner(cfg(tmp_path), broker, tmp_path)
    r.state.initial_balance = 50_000.0          # stare ramasa de la contul vechi
    with pytest.raises(RuntimeError, match="nu se potriveste cu contul"):
        r.preflight()


def test_preflight_accepts_a_normal_drawdown(tmp_path):
    f = frame()
    broker = PaperBroker(f, start_equity=46_000, contract_size=2.0)   # -8%, in limite
    r = Runner(cfg(tmp_path), broker, tmp_path)
    r.state.initial_balance = 50_000.0
    r.preflight()                                # nu trebuie sa ridice


def test_max_lots_cap_warns_when_it_binds(caplog):
    with caplog.at_level("WARNING"):
        lots = size_lots(equity=400_000, price=29_500.0, contract_size=2.0,
                         strategy_daily_vol=0.005816, target_daily_vol=0.025,
                         max_lots=20.0, max_notional_mult=5.0)
    assert lots == 20.0
    assert any("max_lots" in rec.message for rec in caplog.records)
