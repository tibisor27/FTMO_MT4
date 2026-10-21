"""
Compare what the bot did against what the backtest says it should have done.

    python scripts/reconcile_day.py --day 2026-09-09
    python scripts/reconcile_day.py --day 2026-09-09 --data data/raw/USATECHIDXUSD_1m.parquet

Run it after every session. Writes results/live/reconcile_<day>.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.intraday import load_intraday          # noqa: E402
from src.live.clock import Blackout                  # noqa: E402
from src.live.runner import load_config              # noqa: E402
from src.live.shadow import compare, replay, report, slippage  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/live/ftmo_nasdaq.yaml")
    ap.add_argument("--day", required=True)
    ap.add_argument("--data", default="data/raw/USATECHIDXUSD_1m.parquet")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    bars = load_intraday(str(ROOT / args.data))
    exp = replay(bars, cfg, args.day)
    diffs = compare(exp, ROOT / cfg["paths"]["decisions"], args.day)
    slip = slippage(ROOT / cfg["paths"]["fills"], bars, args.day)
    bl = Blackout.from_csv(ROOT / cfg["news"]["events_csv"], int(cfg["news"]["blackout_minutes"]))
    md = report(args.day, diffs, slip, bl, ROOT / cfg["paths"]["fills"])
    out = ROOT / "results" / "live" / f"reconcile_{args.day}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nscris in {out}")
    bad = [d for d in diffs if d.kind != "match"]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
