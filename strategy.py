"""Order block detection (body engulfing) and 3-bar swing fractal detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

FRACTAL_N = 3


@dataclass
class Signal:
    signal_date: pd.Timestamp      # date of engulfing candle (D)
    direction: str                 # "long" or "short"
    ob_high: float                 # OB candle (D-1) high (wick incl)
    ob_low: float                  # OB candle (D-1) low (wick incl)
    entry_date: pd.Timestamp       # D+1 — entry executes at this date's open
    tp2_price: Optional[float]     # nearest swing high (long) / low (short) before D-1, or None


def is_bullish(o: float, c: float) -> bool:
    return c > o


def is_bearish(o: float, c: float) -> bool:
    return c < o


def find_swing_high_before(df: pd.DataFrame, idx: int, n: int = FRACTAL_N) -> Optional[float]:
    """Most recent swing high (3-bar fractal) STRICTLY before df.index[idx]."""
    highs = df["high"].values
    for i in range(idx - n, n - 1, -1):
        if highs[i] > max(highs[i - n:i]) and highs[i] > max(highs[i + 1:i + n + 1]):
            return float(highs[i])
    return None


def find_swing_low_before(df: pd.DataFrame, idx: int, n: int = FRACTAL_N) -> Optional[float]:
    lows = df["low"].values
    for i in range(idx - n, n - 1, -1):
        if lows[i] < min(lows[i - n:i]) and lows[i] < min(lows[i + 1:i + n + 1]):
            return float(lows[i])
    return None


def detect_signals(df: pd.DataFrame) -> list[Signal]:
    """Scan daily OHLCV; emit signals at engulfing-bar D, executable at D+1 open.

    Bearish OB (SHORT): D-1 bullish, D bearish, D body engulfs D-1 body.
    Bullish OB (LONG):  D-1 bearish, D bullish, D body engulfs D-1 body.
    """
    signals: list[Signal] = []
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    dates = df.index

    for d in range(1, len(df) - 1):  # need D+1 to exist for entry
        po, pc = o[d - 1], c[d - 1]
        co, cc = o[d], c[d]
        ph, pl = h[d - 1], l[d - 1]

        direction: Optional[str] = None
        # SHORT: prev bullish, curr bearish, curr body engulfs prev body
        if is_bullish(po, pc) and is_bearish(co, cc):
            if co >= pc and cc <= po:
                direction = "short"
        # LONG: prev bearish, curr bullish, curr body engulfs prev body
        elif is_bearish(po, pc) and is_bullish(co, cc):
            if co <= pc and cc >= po:
                direction = "long"

        if direction is None:
            continue

        if direction == "long":
            tp2 = find_swing_high_before(df, d - 1)
        else:
            tp2 = find_swing_low_before(df, d - 1)

        signals.append(Signal(
            signal_date=dates[d],
            direction=direction,
            ob_high=float(ph),
            ob_low=float(pl),
            entry_date=dates[d + 1],
            tp2_price=tp2,
        ))

    return signals


if __name__ == "__main__":
    from data_fetcher import fetch_klines
    df = fetch_klines()
    sigs = detect_signals(df)
    print(f"{len(sigs)} signals total")
    for s in sigs[:10]:
        print(s)
