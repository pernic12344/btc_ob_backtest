"""Single-position simulator. Conservative vs Optimistic same-bar SL/TP handling."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd

from strategy import Signal

NOTIONAL = 40_000.0
START_CAPITAL = 10_000.0
FEE_RATE = 0.0004           # 0.04% taker, applied to closed notional
MAX_LEVERAGE = 50.0         # equity * MAX_LEVERAGE >= NOTIONAL required to enter


@dataclass
class Trade:
    entry_date: pd.Timestamp
    direction: str
    entry_price: float
    sl: float
    tp1: float
    tp2: Optional[float]
    qty_total: float                       # BTC units = NOTIONAL / entry_price
    # filled later
    tp1_date: Optional[pd.Timestamp] = None
    tp1_price: Optional[float] = None
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""                  # "sl" | "tp2" | "bep" | "reverse" | "eod"
    pnl: float = 0.0
    fees: float = 0.0
    equity_after: float = 0.0
    skipped_tp2_invalid: bool = False


def _sl_distance(entry: float, sl: float, direction: str) -> float:
    return (sl - entry) if direction == "short" else (entry - sl)
    # positive number for normal setups; we'll take abs in PnL math


def _compute_tp1(entry: float, sl: float, direction: str) -> float:
    risk = abs(entry - sl)
    return entry - risk if direction == "short" else entry + risk


def _pnl_long(entry: float, exit_p: float, qty: float) -> float:
    return (exit_p - entry) * qty


def _pnl_short(entry: float, exit_p: float, qty: float) -> float:
    return (entry - exit_p) * qty


def _close_pnl(entry: float, exit_p: float, qty: float, direction: str) -> float:
    return _pnl_long(entry, exit_p, qty) if direction == "long" else _pnl_short(entry, exit_p, qty)


def run_backtest(df: pd.DataFrame, signals: list[Signal], mode: str = "conservative") -> tuple[list[Trade], pd.Series]:
    """Run sim. Returns (trades, equity_curve_series).

    mode: "conservative" -> SL wins same-bar tie. "optimistic" -> TP wins.
    """
    assert mode in ("conservative", "optimistic")

    sig_by_entry: dict[pd.Timestamp, Signal] = {s.entry_date: s for s in signals}
    trades: list[Trade] = []
    equity = START_CAPITAL
    equity_series: dict[pd.Timestamp, float] = {}

    open_trade: Optional[Trade] = None
    qty_remaining: float = 0.0
    tp1_done: bool = False
    current_sl: float = 0.0   # may move to BEP after TP1

    dates = df.index
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values

    def close_position(date: pd.Timestamp, price: float, reason: str) -> None:
        nonlocal open_trade, qty_remaining, tp1_done, current_sl, equity
        assert open_trade is not None
        pnl = _close_pnl(open_trade.entry_price, price, qty_remaining, open_trade.direction)
        fee = price * qty_remaining * FEE_RATE
        equity += pnl - fee
        open_trade.pnl += pnl
        open_trade.fees += fee
        open_trade.exit_date = date
        open_trade.exit_price = price
        open_trade.exit_reason = reason
        open_trade.equity_after = equity
        trades.append(open_trade)
        open_trade = None
        qty_remaining = 0.0
        tp1_done = False

    def open_new(sig: Signal, entry_price: float, date: pd.Timestamp) -> None:
        nonlocal open_trade, qty_remaining, tp1_done, current_sl, equity
        if equity * MAX_LEVERAGE < NOTIONAL:
            return  # cannot afford
        risk = abs(entry_price - (sig.ob_high if sig.direction == "short" else sig.ob_low))
        if risk == 0:
            return
        # TP2 must be on the favorable side
        tp2 = sig.tp2_price
        skip_tp2 = False
        if tp2 is not None:
            if sig.direction == "long" and tp2 <= entry_price:
                tp2 = None; skip_tp2 = True
            if sig.direction == "short" and tp2 >= entry_price:
                tp2 = None; skip_tp2 = True
        qty = NOTIONAL / entry_price
        sl = sig.ob_high if sig.direction == "short" else sig.ob_low
        tp1 = _compute_tp1(entry_price, sl, sig.direction)
        entry_fee = entry_price * qty * FEE_RATE
        equity -= entry_fee
        open_trade = Trade(
            entry_date=date, direction=sig.direction, entry_price=entry_price,
            sl=sl, tp1=tp1, tp2=tp2, qty_total=qty, fees=entry_fee,
            skipped_tp2_invalid=skip_tp2,
        )
        qty_remaining = qty
        tp1_done = False
        current_sl = sl

    def partial_tp1(date: pd.Timestamp) -> None:
        """Close half at TP1, move stop to BEP."""
        nonlocal qty_remaining, tp1_done, current_sl, equity
        assert open_trade is not None
        half = open_trade.qty_total / 2.0
        price = open_trade.tp1
        pnl = _close_pnl(open_trade.entry_price, price, half, open_trade.direction)
        fee = price * half * FEE_RATE
        equity += pnl - fee
        open_trade.pnl += pnl
        open_trade.fees += fee
        open_trade.tp1_date = date
        open_trade.tp1_price = price
        qty_remaining -= half
        tp1_done = True
        current_sl = open_trade.entry_price  # BEP

    for i, date in enumerate(dates):
        bar_o, bar_h, bar_l = o[i], h[i], l[i]

        # 1) Check for entry — entry happens at THIS bar's open if signal's entry_date == date
        if date in sig_by_entry:
            sig = sig_by_entry[date]
            # Reverse close if opposite direction
            if open_trade is not None and open_trade.direction != sig.direction:
                close_position(date, bar_o, "reverse")
            # Same direction with open trade -> ignore signal
            if open_trade is None:
                open_new(sig, bar_o, date)
                # Entry happens at bar open. Continue to intra-bar SL/TP check below.

        # 2) Check SL / TP fills on the same bar (skip the very first bar of entry? No — entry is at open, then high/low can hit)
        if open_trade is None:
            equity_series[date] = equity
            continue

        # Determine candidate fills on this bar based on price range
        d = open_trade.direction
        sl = current_sl
        tp1 = open_trade.tp1
        tp2 = open_trade.tp2

        sl_hit = (bar_l <= sl) if d == "long" else (bar_h >= sl)
        tp1_hit = (not tp1_done) and ((bar_h >= tp1) if d == "long" else (bar_l <= tp1))
        tp2_hit = tp1_done and (tp2 is not None) and ((bar_h >= tp2) if d == "long" else (bar_l <= tp2))

        # Resolve
        if not tp1_done:
            # Phase A: only sl and tp1 in play
            if sl_hit and tp1_hit:
                if mode == "conservative":
                    close_position(date, sl, "sl")
                else:
                    partial_tp1(date)
                    # After TP1, stop is BEP. BEP could also be hit same bar if range covers entry.
                    bep = open_trade.entry_price
                    bep_hit = (bar_l <= bep) if d == "long" else (bar_h >= bep)
                    if bep_hit:
                        close_position(date, bep, "bep")
            elif sl_hit:
                close_position(date, sl, "sl")
            elif tp1_hit:
                partial_tp1(date)
                # After TP1, stop moves to BEP. Re-check BEP / TP2 on this same bar.
                if open_trade is not None:
                    bep = open_trade.entry_price
                    bep_hit = (bar_l <= bep) if d == "long" else (bar_h >= bep)
                    tp2_hit_now = (tp2 is not None) and ((bar_h >= tp2) if d == "long" else (bar_l <= tp2))
                    if bep_hit and tp2_hit_now:
                        if mode == "conservative":
                            close_position(date, bep, "bep")
                        else:
                            close_position(date, tp2, "tp2")
                    elif bep_hit:
                        close_position(date, bep, "bep")
                    elif tp2_hit_now:
                        close_position(date, tp2, "tp2")
        else:
            # Phase B: stop is BEP, target is TP2 (if any)
            bep = open_trade.entry_price
            bep_hit = (bar_l <= bep) if d == "long" else (bar_h >= bep)
            tp2_hit_now = (tp2 is not None) and ((bar_h >= tp2) if d == "long" else (bar_l <= tp2))
            if bep_hit and tp2_hit_now:
                if mode == "conservative":
                    close_position(date, bep, "bep")
                else:
                    close_position(date, tp2, "tp2")
            elif bep_hit:
                close_position(date, bep, "bep")
            elif tp2_hit_now:
                close_position(date, tp2, "tp2")

        equity_series[date] = equity

    # Close any dangling position at last close
    if open_trade is not None:
        last_close = float(df["close"].iloc[-1])
        last_date = dates[-1]
        close_position(last_date, last_close, "eod")
        equity_series[last_date] = equity

    eq = pd.Series(equity_series).sort_index()
    return trades, eq


if __name__ == "__main__":
    from data_fetcher import fetch_klines
    from strategy import detect_signals
    df = fetch_klines()
    sigs = detect_signals(df)
    for mode in ("conservative", "optimistic"):
        trades, eq = run_backtest(df, sigs, mode=mode)
        print(f"{mode}: {len(trades)} trades, final equity ${eq.iloc[-1]:,.2f}")
