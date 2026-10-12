"""
Run the strategy live (or on a demo account, which is where it belongs first).

    python scripts/run_live.py --config configs/live/ftmo_nasdaq.yaml --dry-run
    python scripts/run_live.py --config configs/live/ftmo_nasdaq.yaml --login 123 --server FTMO-Demo

--dry-run uses a PaperBroker fed from data/raw/, so you can watch a whole session replay in seconds
without touching a terminal. That is the first thing to run, on any OS.

Stop it at any time by creating a file named HALT in the project root: the bot flattens and stops.
"""
from __future__ import annotations

import argparse
import getpass
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.live.broker import MT5Broker, PaperBroker   # noqa: E402
from src.live.runner import Runner, load_config      # noqa: E402


def setup_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(name)-14s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(path, encoding="utf-8")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/live/ftmo_nasdaq.yaml")
    ap.add_argument("--dry-run", action="store_true", help="PaperBroker peste data/raw/")
    ap.add_argument("--data", default="data/raw/USATECHIDXUSD_1m.parquet")
    ap.add_argument("--day", default=None, help="dry-run: ce zi sa reia (implicit ultima)")
    ap.add_argument("--login", type=int, default=None)
    ap.add_argument("--server", default=None)
    ap.add_argument("--poll", type=float, default=5.0, help="secunde intre verificari")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    setup_log(ROOT / cfg["paths"]["log"])
    log = logging.getLogger("live")

    if args.dry_run:
        from src.data.intraday import load_intraday
        df = load_intraday(str(ROOT / args.data))
        tz = cfg["session"]["tz"]
        day = pd.Timestamp(args.day).date() if args.day else df.index[-1].tz_convert(tz).date()
        start = pd.Timestamp(day, tz=tz) - pd.Timedelta(days=40)
        df = df.loc[df.index >= start.tz_convert("UTC")]
        broker = PaperBroker(df, contract_size=float(cfg["risk"]["contract_size"]))
        runner = Runner(cfg, broker, ROOT)
        loc = df.index.tz_convert(tz)
        minutes = df.index[(pd.Series(loc.date, index=df.index) == day).to_numpy()]
        log.info("DRY RUN pe %s · %d minute", day, len(minutes))
        for ts in minutes:
            broker.set_now(ts)
            out = runner.step(ts)
            if out not in ("nu e moment de decizie", "bara deja procesata"):
                log.info("%s  %s", ts.tz_convert(tz).strftime("%H:%M"), out)
        log.info("echitate finala: %.2f (start %.2f)", broker.equity(), 50000.0)
        return

    password = None
    if args.login:
        password = getpass.getpass("parola MT5: ")
    broker = MT5Broker(login=args.login, password=password, server=args.server,
                       magic=int(cfg["magic"]))
    broker.connect()
    runner = Runner(cfg, broker, ROOT)
    log.info("pornit pe %s · simbol %s · ultima decizie %s",
             cfg["session"]["tz"], cfg["symbol"], cfg["session"]["last_decision"])
    last = None
    while True:
        now = pd.Timestamp.now(tz="UTC").floor("s")
        try:
            out = runner.step(now)
            if out != last and out not in ("nu e moment de decizie", "bara deja procesata"):
                log.info("%s", out)
                last = out
            if runner.state.halted:
                log.error("oprit definitiv pentru azi: %s", runner.state.halt_reason)
                break
        except Exception:                                  # pragma: no cover
            log.exception("eroare in bucla - incerc sa inchid tot")
            try:
                runner._flatten("exceptie in bucla")
            finally:
                runner.state.halt("exceptie in bucla")
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
