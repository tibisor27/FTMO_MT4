"""
Loader for intraday OHLCV files (1-minute bars) used by the event-study harness.

Contract OUT:
    DataFrame with a tz-aware DatetimeIndex (UTC), strictly increasing, no duplicates,
    columns: open, high, low, close, volume (float64).

Accepts:
    - parquet/csv with a 'ts' or 'timestamp' column (UTC or tz-aware), e.g. the NAS100 export
    - parquet/csv already indexed by a DatetimeIndex
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED = ["open", "high", "low", "close"]


def load_intraday(path: str | Path, tz: str = "UTC") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Intraday file not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # --- index ---
    #se asigura ca indexul e DatetimeIndex si converteste in UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        ts_col = next((c for c in ("ts", "timestamp", "datetime", "date") if c in df.columns), None)
        if ts_col is None:
            raise ValueError("No DatetimeIndex and no 'ts'/'timestamp' column found")
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df.set_index(ts_col)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(tz)
    df.index.name = "ts"

    # --- schema ---
    #valideaza ca avem shema corecta a datelor
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Critical error: missing columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = float("nan")
    df = df[REQUIRED + ["volume"]].astype("float64")

    # --- content ---
    n0 = len(df)

    #index.duplicated cauta toate randurile duplicate(pe primele le pastreaza ca si False)
    #~inverteaza valoarea(true devine false si invers)
    
    df = df[~df.index.duplicated(keep="first")].sort_index()
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    if bad.any():
        logger.warning("%d bars violate high>=max(o,c) / low<=min(o,c); dropping them", int(bad.sum()))
        df = df[~bad]
    df = df.dropna(subset=REQUIRED)
    logger.info("Loaded %s: %d bars (%d dropped), %s -> %s", path.name, len(df), n0 - len(df), df.index.min(), df.index.max())
    return df


#returneaza un Series cu indexul dataframeului in care minutele din sesiunea dorita sunt true(fara weeked)
def rth_mask(index: pd.DatetimeIndex, tz: str = "America/New_York", start: str = "09:30", end: str = "16:00") -> pd.Series:
    """Boolean mask for regular trading hours [start, end) in the given timezone."""
    local = index.tz_convert(tz)
    minutes = local.hour * 60 + local.minute
    s = int(start[:2]) * 60 + int(start[3:])
    e = int(end[:2]) * 60 + int(end[3:])
    return pd.Series((minutes >= s) & (minutes < e) & (local.dayofweek < 5), index=index)
