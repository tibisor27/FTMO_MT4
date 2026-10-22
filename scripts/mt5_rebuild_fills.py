"""
Reconstruieste live_fills.csv din ISTORICUL DE TRANZACTII al terminalului MT5.

De ce exista: `order_send` intoarce, la multi brokeri, `result.price == 0.0` pentru ordine la
piata - pretul real al executiei sta in DEAL, nu in raspunsul la ordin. Daca botul a rulat cu o
versiune de broker.py fara cautarea in istoric, jurnalul tau are preturi 0.0 si slippage-ul nu se
poate calcula. Terminalul insa stie tot. Scriptul asta ia adevarul de acolo.

E INTENTIONAT fara dependente din repo (doar MetaTrader5 + pandas), ca sa poata fi copiat singur
pe masina de live.

    python scripts/mt5_rebuild_fills.py --symbol US100.cash --magic 20260913 \
        --from 2026-10-20 --to 2026-10-23 --out state/live_fills_rebuilt.csv

Nu suprascrie nimic: scrie intr-un fisier nou si iti spune ce difera fata de cel existent.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone

import pandas as pd


def server_offset_seconds(mt5, symbol: str) -> int:
    """Cat e ceasul serverului fata de UTC, in secunde, rotunjit la sfert de ora."""
    t = mt5.symbol_info_tick(symbol)
    if t is None or not getattr(t, "time", 0):
        raise SystemExit(f"fara tick pentru {symbol}: {mt5.last_error()}")
    return int(round((float(t.time) - time.time()) / 900.0)) * 900


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="US100.cash")
    ap.add_argument("--magic", type=int, default=20260913)
    ap.add_argument("--from", dest="dfrom", required=True, help="prima zi, YYYY-MM-DD (UTC)")
    ap.add_argument("--to", dest="dto", required=True, help="ultima zi INCLUSIV, YYYY-MM-DD (UTC)")
    ap.add_argument("--out", default="state/live_fills_rebuilt.csv")
    ap.add_argument("--compare", default="state/live_fills.csv")
    ap.add_argument("--all-magics", action="store_true", help="nu filtra dupa magic")
    a = ap.parse_args()

    import MetaTrader5 as mt5                                  # noqa: N813
    if not mt5.initialize():
        raise SystemExit(f"MT5 initialize a esuat: {mt5.last_error()}")
    try:
        off = server_offset_seconds(mt5, a.symbol)
        print(f"decalaj server -> UTC: {off/3600:+.2f} h")

        # Fereastra o largim cu doua zile in fiecare parte: argumentele lui history_deals_get sunt
        # interpretate inconsecvent intre build-uri, deci cerem mai mult si filtram noi dupa.
        lo = datetime.fromisoformat(a.dfrom).replace(tzinfo=timezone.utc) - timedelta(days=2)
        hi = datetime.fromisoformat(a.dto).replace(tzinfo=timezone.utc) + timedelta(days=3)
        deals = mt5.history_deals_get(lo.replace(tzinfo=None), hi.replace(tzinfo=None))
        if deals is None:
            raise SystemExit(f"history_deals_get a intors None: {mt5.last_error()}")
        print(f"tranzactii brute in fereastra: {len(deals)}")

        rows = []
        for d in deals:
            if d.symbol != a.symbol:
                continue
            if not a.all_magics and d.magic != a.magic:
                continue
            if d.type not in (0, 1):                           # 0 = BUY, 1 = SELL; restul = taxe
                continue
            ts = pd.Timestamp(int(d.time) - off, unit="s", tz="UTC")
            # entry = deschide/mareste, exit/reverse = reduce/inchide
            kind = {0: "entry", 1: "exit", 2: "reverse"}.get(int(d.entry), f"entry_{d.entry}")
            rows.append({
                "ts": ts,
                "side": 1 if d.type == 0 else -1,
                "lots": float(d.volume),
                "price": float(d.price),
                "kind": kind,
                "comment": (d.comment or "").strip(),
                "deal": int(d.ticket),
                "order": int(d.order),
                "position": int(d.position_id),
                "commission": float(d.commission),
                "swap": float(d.swap),
                "profit": float(d.profit),
            })

        if not rows:
            print("nicio tranzactie potrivita. Verifica --symbol si --magic "
                  "(sau ruleaza cu --all-magics).")
            return 1

        f = pd.DataFrame(rows).sort_values("ts")
        lo_d, hi_d = pd.Timestamp(a.dfrom, tz="UTC"), pd.Timestamp(a.dto, tz="UTC") + timedelta(days=1)
        f = f[(f["ts"] >= lo_d) & (f["ts"] < hi_d)].reset_index(drop=True)

        from pathlib import Path
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        f.to_csv(out, index=False)
        print(f"\nscris {len(f)} executii in {out}\n")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(f[["ts", "side", "lots", "price", "kind", "deal", "commission", "swap", "profit"]]
                  .to_string(index=False))

        old = Path(a.compare)
        if old.exists() and old.stat().st_size:
            o = pd.read_csv(old)
            o["ts"] = pd.to_datetime(o["ts"], utc=True)
            o = o[(o["ts"] >= lo_d) & (o["ts"] < hi_d)]
            print(f"\n--- fata de {old} ({len(o)} randuri in aceeasi fereastra) ---")
            zero = int((o["price"] == 0).sum()) if "price" in o.columns else 0
            if zero:
                print(f"  {zero} randuri cu pret 0.0 in jurnalul botului -> acum au pret real")
            if len(o) != len(f):
                print(f"  ATENTIE: numar diferit de executii ({len(o)} vs {len(f)})")
        print("\nCand esti multumit:  copy state\\live_fills.csv state\\live_fills.bak  "
              "&&  copy " + str(out).replace("/", "\\") + " state\\live_fills.csv")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
