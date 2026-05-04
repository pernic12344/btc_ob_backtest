"""Fetch BTCUSDT perpetual daily klines from Binance USDT-M futures and cache to CSV."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

BASE_URL = "https://fapi.binance.com"
KLINES_PATH = "/fapi/v1/klines"

CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "btcusdt_1d.csv")


def _fetch_chunk(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1500) -> list:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    r = requests.get(BASE_URL + KLINES_PATH, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1d",
                 start: str = "2024-05-04", end: str = "2026-05-04",
                 use_cache: bool = True) -> pd.DataFrame:
    """Return OHLCV DataFrame indexed by date (UTC)."""
    if use_cache and os.path.exists(CACHE_PATH):
        df = pd.read_csv(CACHE_PATH, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        s = pd.Timestamp(start, tz="UTC").normalize().tz_localize(None)
        e = pd.Timestamp(end, tz="UTC").normalize().tz_localize(None)
        if df.index.min() <= s and df.index.max() >= e - pd.Timedelta(days=2):
            return df.loc[s:e]

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

    all_rows: list = []
    cursor = start_ms
    while cursor < end_ms:
        chunk = _fetch_chunk(symbol, interval, cursor, end_ms)
        if not chunk:
            break
        all_rows.extend(chunk)
        last_open_ms = chunk[-1][0]
        next_cursor = last_open_ms + 24 * 60 * 60 * 1000
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.1)

    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df[["date", "open", "high", "low", "close", "volume"]].drop_duplicates("date").set_index("date").sort_index()

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.reset_index().to_csv(CACHE_PATH, index=False)
    return df


if __name__ == "__main__":
    df = fetch_klines(use_cache=False)
    print(f"Fetched {len(df)} rows: {df.index.min().date()} → {df.index.max().date()}")
    print(df.head())
    print(df.tail())
