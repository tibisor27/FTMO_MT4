"""
Noise Area intraday momentum — bands, signal and event generator.

Implements the entry logic of Zarattini, Aziz, Barbon (2024/2025), "Beat the Market":
    sigma[d, m]  = mean over the previous `lookback` days of |Close[d-i, m] / Open[d-i] - 1|
    UB[d, m]     = max(Open[d], PrevClose[d]) * (1 + vm * sigma[d, m])
    LB[d, m]     = min(Open[d], PrevClose[d]) * (1 - vm * sigma[d, m])

Three public functions, all built on the same `_bands` computation (single source of truth):

    noise_area_bands(df, ...)   -> DataFrame[ub, lb] on every RTH bar        (for exits/backtest)
    noise_area_signal(df, ...)  -> Series +1/-1/0 at decision bars, NaN else (for the backtest)
    noise_area_events(df, ...)  -> (events, eligible)                        (for the event study)

`signal` is a STATE (is price outside the band right now?), `events` are the TRANSITIONS of that
state. The event study measures the drift after fresh signals; the backtest needs the state, because
after a tight stop you may re-enter later the same day while the state is still on.

No look-ahead: sigma for day d is a mean over days d-1 ... d-lookback (`shift(1)` before rolling).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.intraday import rth_mask

logger = logging.getLogger(__name__)


#functia asta returneaza benzile din rth
#df final va avea indexul in UTC, dar toata logica de bussines din el ii facuta pe "America/New_York"
def _bands(
    df: pd.DataFrame,
    lookback: int,
    vm: float,
    tz: str,
    rth_start: str,
    rth_end: str,
) -> pd.DataFrame:
    """Upper/lower Noise Area boundaries for every RTH bar, plus the session bookkeeping columns. RTH = Regular trading hours"""
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise TypeError("Contract violated: df needs a tz-aware DatetimeIndex")

    #creez masca cu minutele valide pt strategie pt df de fata
    mask = rth_mask(df.index, tz, rth_start, rth_end)
    d = df.loc[mask.values, ["open", "high", "low", "close"]].copy()
    if d.empty:
        raise ValueError("No bars inside regular trading hours; check tz/rth settings")

    local = d.index.tz_convert(tz)
    #adauga coloana de data pentru a grupa mai jos datele(open) in functie de zi
    d["date"] = local.date

    #iau primul minut al sesiunii ca referinta (in s0 -> 9:30 -> 570)
    s0 = int(rth_start[:2]) * 60 + int(rth_start[3:])
    #d["mod"] = numarul de minute trecute de 9:30 
    d["mod"] = (local.hour * 60 + local.minute) - s0  # minute-of-session, 0 = first bar

    #grupez datele pe zi, si iau din fiecare zi prima valoare de open
    # prima valoare din acest df_rth care e de fapt open al sesiunii(09:30)
    day_open = d.groupby("date")["open"].first()

    #iau din fiecare zi ultima valoare de close (16:00)
    day_close = d.groupby("date")["close"].last()
    
    #am nevoie de prev_close pt a stii in ziua urmatoare de la ce valoare sa inceapa banda corect
    prev_close = day_close.shift(1)

    #calaculez miscarea relativa(impartire) nu absoluta(scadere)
    #impartirea imi da un ratio iar eu vreau procent => fac -1.0 si obtin procentajul
    #.map() face ca ptfiecare valoare din coloana "date" sa mapeze indexu "date" cu valoarea din "day_open" de la indexul corespondent
    d["move"] = (d["close"] / d["date"].map(day_open) - 1.0).abs()

    #creaza un df de tipul matrice 2D cu valorile liniilor si coloanelor dorite
    #am neovie de acst tabel pt ca pt vreau sa fac media a ultimelor 14 miscari dintr un anume minut
    pivot = d.pivot_table(index="date", columns="mod", values="move")

    #calculeaza media ultimelor 14 miscari pt feicare minut din zi
    #WARNING! Daca un minut lipseste(NaN) automat media din care face parte va fi tot NaN
    #pt ca min_periods = 14 cere exact 14 valori la calcul! Deci grupul celor 14 minute din medie
    #va avea mereu NaN
    # Exista zile scurte de tranzactionare => minute NaN => Media grupului din care fac parte (14 zile) = NaN
    sigma = pivot.shift(1).rolling(lookback, min_periods=lookback).mean()  # days x minutes

    ref_hi = np.maximum(day_open, prev_close)
    ref_lo = np.minimum(day_open, prev_close)

    #sigma = valori procentuale => adunam 1 pentru a obine ratia 
    #iar ratia o multiplicam cu pretul real si obtinem o valoarea exacta a benzii (nu procentuala sau ratie)
    ub = sigma.mul(vm).add(1.0).mul(ref_hi, axis=0)
    lb = (1.0 - sigma.mul(vm)).mul(ref_lo, axis=0)

    #transforma un tabel 2D intr unul 1D (Varianta mai rapida folosing numpy pt compilatorul de C din spate)
    # in pandas s ar fi facut mai usor
    #fiecare data din d["date"] are un index de la 0-73
    #pt feicare index din ub adauga toate datele din d["date"] corespondente 
    # array([ 0,  0,  0, ..., 73, 73, 73], shape=(28320,))
    ri = ub.index.get_indexer(d["date"].to_numpy())
    ci = ub.columns.get_indexer(d["mod"].to_numpy())
    ok = (ri >= 0) & (ci >= 0)
    ris, cis = np.clip(ri, 0, None), np.clip(ci, 0, None)

    def _pick(frame: pd.DataFrame) -> np.ndarray:
        v = frame.to_numpy()[ris, cis]
        return np.where(ok, v, np.nan)

    out = pd.DataFrame(
        {
            "ub": _pick(ub),
            "lb": _pick(lb),
            "close": d["close"].to_numpy(),
            "date": d["date"].to_numpy(),
            "mod": d["mod"].to_numpy(),
            "ref_open": d["date"].map(day_open).to_numpy(),
            "ref_prev_close": d["date"].map(prev_close).to_numpy(),
        },
        index=d.index,
    )
    return out


def noise_area_bands(
    df: pd.DataFrame,
    lookback: int = 14,
    vm: float = 1.0,
    tz: str = "America/New_York",
    rth_start: str = "09:30",
    rth_end: str = "16:00",
    **_ignored,
) -> pd.DataFrame:
    """Public: DataFrame[ub, lb] indexed by RTH bars (NaN during the warm-up days)."""
    return _bands(df, lookback, vm, tz, rth_start, rth_end)[["ub", "lb"]]


#aplica benzile din rth pe intreg df si retunreaza un pandas series de lungimea df-ului cu semnalul +1, -1 sau 0
def noise_area_signal(
    df: pd.DataFrame,
    lookback: int = 14,
    vm: float = 1.0,
    decision_minutes: tuple[int, ...] = (0, 30),
    first_decision: str = "10:00",
    tz: str = "America/New_York",
    rth_start: str = "09:30",
    rth_end: str = "16:00",
    **_ignored,
) -> pd.Series:
    """Public: +1 above the band / -1 below / 0 inside, at decision bars only; NaN elsewhere."""
    b = _bands(df, lookback, vm, tz, rth_start, rth_end)
    local = b.index.tz_convert(tz)
    fd = int(first_decision[:2]) * 60 + int(first_decision[3:])
    is_decision = local.minute.isin(decision_minutes) & ((local.hour * 60 + local.minute) >= fd)

    sig = pd.Series(np.nan, index=df.index, name="signal")
    #np.where(conditie, val if true, val if false)
    #daca ii peste linie punem 1, daca ii sub linie punem -1, in rest punem 0
    v = np.where(b["close"] > b["ub"], 1.0, np.where(b["close"] < b["lb"], -1.0, 0.0))

    #fara asta candelele cu NaN ar fi automat 0
    v = np.where(b["ub"].isna().to_numpy(), np.nan, v)  # warm-up days stay NaN
    sig.loc[b.index[is_decision]] = v[is_decision]
    return sig


def noise_area_events(
    df: pd.DataFrame,
    lookback: int = 14,
    vm: float = 1.0,
    decision_minutes: tuple[int, ...] = (0, 30),
    first_decision: str = "10:00",
    tz: str = "America/New_York",
    rth_start: str = "09:30",
    rth_end: str = "16:00",
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Entry events = transitions of the signal state, per session (a reversal is a new event)."""
    b = _bands(df, lookback, vm, tz, rth_start, rth_end)
    local = b.index.tz_convert(tz)
    fd = int(first_decision[:2]) * 60 + int(first_decision[3:])
    is_decision = local.minute.isin(decision_minutes) & ((local.hour * 60 + local.minute) >= fd)

    dec = b[is_decision].dropna(subset=["ub", "lb"]).copy()

    events = []
    for _date, day in dec.groupby("date", sort=True):
        pos = 0
        for ts, row in day.iterrows():  #.iterrows() retunreaza perechi (index, Series) pt feicare rand
            if row.close > row.ub and pos != 1:
                events.append((ts, 1, "reverse_long" if pos == -1 else "long", row.ub, row.ref_open, row.ref_prev_close))
                pos = 1
            elif row.close < row.lb and pos != -1:
                events.append((ts, -1, "reverse_short" if pos == 1 else "short", row.lb, row.ref_open, row.ref_prev_close))
                pos = -1

    ev = pd.DataFrame(events, columns=["t0", "side", "kind", "level", "ref_open", "ref_prev_close"])
    ev["t0"] = pd.to_datetime(ev["t0"], utc=True) if len(ev) else pd.DatetimeIndex([], tz="UTC")
    ev["strategy"] = "noise_area"
    eligible = dec.index.tz_convert("UTC")
    logger.info("noise_area: %d events over %d decision times (%d days)", len(ev), len(dec), dec["date"].nunique())
    return ev.reset_index(drop=True), eligible
