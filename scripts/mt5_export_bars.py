"""
Exporta bare M1 din terminalul MT5 intr-un parquet pe care il poate citi `load_intraday`.

E varianta minimala, de depanare, a lui `python -m pipeline ingest/silver`: fara Bronze, fara
Silver, fara verificari de calitate - doar ca sa ai bare pentru reconciliere in ziua in care
pipeline-ul inca nu e pe masina de live. Pentru rutina zilnica foloseste pipeline-ul.

Fara dependente din repo (doar MetaTrader5 + pandas + pyarrow).

    python scripts/mt5_export_bars.py --symbol US100.cash --days 45 \
        --out data/live_bars/US100.cash_1m.parquet

Iesire: index DatetimeIndex in UTC ADEVARAT (decalajul serverului masurat si scazut),
coloane open, high, low, close, volume.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd


def server_offset_seconds(mt5, symbol: str, rth_start_utc_check: bool = True) -> int:
    """
    Decalajul ceasului serverului fata de UTC, in secunde, masurat in doua feluri independente.

    A. epoca ultimului tick minus ceasul nostru
    B. minutul din zi cu cel mai mare volum trebuie sa fie deschiderea cash de la New York

    Daca nu sunt de acord, ridicam: un decalaj gresit nu da nicio eroare, doar muta sesiunea.
    """
    t = mt5.symbol_info_tick(symbol)
    if t is None or not getattr(t, "time", 0):
        raise SystemExit(f"fara tick pentru {symbol}: {mt5.last_error()}")
    a = int(round((float(t.time) - time.time()) / 900.0)) * 900

    if rth_start_utc_check:
        r = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 20000)
        if r is not None and len(r) > 2000:
            d = pd.DataFrame(r)
            vol = "tick_volume" if "tick_volume" in d.columns else "real_volume"
            srv = pd.to_datetime(d["time"], unit="s", utc=True)
            mod = srv.dt.hour * 60 + srv.dt.minute
            busiest = int(d.groupby(mod)[vol].median().idxmax())
            day = srv.iloc[-1].tz_convert("America/New_York").normalize()
            want = (day + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC")
            b = int(round(((busiest - (want.hour * 60 + want.minute)) % 1440) / 15.0)) * 15 * 60
            if b > 12 * 3600:
                b -= 24 * 3600
            if abs(a - b) > 900:
                raise SystemExit(
                    f"decalajul serverului nu se confirma: tick spune {a/3600:+.2f}h, "
                    f"volumul spune {b/3600:+.2f}h. NU scriu un fisier pe un fus orar nesigur.")
            print(f"decalaj server -> UTC: {a/3600:+.2f} h (tick) / {b/3600:+.2f} h (volum)")
            return a
    print(f"decalaj server -> UTC: {a/3600:+.2f} h (doar din tick)")
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="US100.cash")
    ap.add_argument("--days", type=int, default=45,
                    help="cate zile calendaristice in urma (implicit 45, acopera incalzirea de 25 de zile)")
    ap.add_argument("--out", default="data/live_bars/US100.cash_1m.parquet")
    a = ap.parse_args()

    import MetaTrader5 as mt5                                  # noqa: N813
    if not mt5.initialize():
        raise SystemExit(f"MT5 initialize a esuat: {mt5.last_error()}")
    try:
        off = server_offset_seconds(mt5, a.symbol)
        count = int(a.days) * 1440
        r = mt5.copy_rates_from_pos(a.symbol, mt5.TIMEFRAME_M1, 0, count)
        if r is None or len(r) == 0:
            raise SystemExit(f"fara bare pentru {a.symbol}: {mt5.last_error()}")
        d = pd.DataFrame(r)
        d.index = pd.to_datetime(d["time"].astype("int64") - off, unit="s", utc=True)
        d.index.name = "ts"
        d = d.rename(columns={"tick_volume": "volume"})
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in d.columns]
        d = d[keep].astype("float64").sort_index()
        d = d[~d.index.duplicated(keep="last")]

        from pathlib import Path
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(out)
        print(f"\nscris {len(d):,} bare in {out}")
        print(f"  interval : {d.index[0]}  ->  {d.index[-1]}  (UTC)")
        ny = d.index.tz_convert("America/New_York")
        print(f"  New York : {ny[0]}  ->  {ny[-1]}")
        print(f"  zile cu bare RTH: {len(set(ny[(ny.hour*60+ny.minute >= 570) & (ny.hour*60+ny.minute < 960)].date))}")
        print("\nVerificare: prima bara a sesiunii de ieri trebuie sa fie la 09:30 New York.")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
