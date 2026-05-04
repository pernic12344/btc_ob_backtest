"""Per-trade candlestick charts + overview map of all entries on BTC price."""
from __future__ import annotations

import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.patches import Rectangle

from data_fetcher import fetch_klines
from strategy import detect_signals
from backtester import run_backtest

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "trade_charts")


def draw_candles(ax, sub: pd.DataFrame) -> None:
    """Simple OHLC candlestick drawer on a matplotlib axes."""
    width = 0.6
    for ts, row in sub.iterrows():
        x = mdates.date2num(ts)
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = "#26a69a" if c >= o else "#ef5350"
        ax.vlines(x, l, h, color=color, linewidth=1)
        rect_low = min(o, c)
        rect_h = max(abs(c - o), 1e-9)
        ax.add_patch(Rectangle((x - width / 2, rect_low), width, rect_h,
                               facecolor=color, edgecolor=color))


def plot_one_trade(df: pd.DataFrame, trade: pd.Series, idx: int, out_path: str) -> None:
    entry_dt = pd.Timestamp(trade["entry_date"])
    exit_dt = pd.Timestamp(trade["exit_date"])
    # Pad window: 8 days before signal (= entry - 1), 4 days after exit
    sig_dt = entry_dt - pd.Timedelta(days=1)  # the engulfing bar D
    ob_dt = entry_dt - pd.Timedelta(days=2)   # the OB bar D-1
    pad_start = sig_dt - pd.Timedelta(days=8)
    pad_end = exit_dt + pd.Timedelta(days=4)
    sub = df.loc[pad_start:pad_end].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    draw_candles(ax, sub)

    # OB rectangle (D-1 high/low spans the trade duration)
    ob_high = float(sub.loc[ob_dt, "high"]) if ob_dt in sub.index else None
    ob_low = float(sub.loc[ob_dt, "low"]) if ob_dt in sub.index else None
    if ob_high is not None and ob_low is not None:
        x0 = mdates.date2num(ob_dt)
        x1 = mdates.date2num(pad_end)
        ax.add_patch(Rectangle((x0, ob_low), x1 - x0, ob_high - ob_low,
                               facecolor="#888", alpha=0.15, edgecolor="#444",
                               linestyle="--", linewidth=0.8, label="OB zone"))

    # Levels
    entry = float(trade["entry_price"])
    sl = float(trade["sl"])
    tp1 = float(trade["tp1"])
    tp2 = trade["tp2"] if pd.notna(trade["tp2"]) else None
    ax.axhline(entry, color="#1976d2", linewidth=1.4, linestyle="-", label=f"Entry {entry:,.1f}")
    ax.axhline(sl, color="#c62828", linewidth=1.4, linestyle="--", label=f"SL {sl:,.1f}")
    ax.axhline(tp1, color="#2e7d32", linewidth=1.0, linestyle=":", label=f"TP1 {tp1:,.1f}")
    if tp2 is not None:
        ax.axhline(float(tp2), color="#558b2f", linewidth=1.0, linestyle=":", label=f"TP2 {tp2:,.1f}")

    # Markers
    ax.scatter([mdates.date2num(entry_dt)], [entry], marker="^" if trade["direction"] == "long" else "v",
               color="#1976d2", s=120, zorder=5, label="Entry")
    exit_price = float(trade["exit_price"])
    exit_color = "#2e7d32" if trade["net_pnl"] > 0 else "#c62828"
    ax.scatter([mdates.date2num(exit_dt)], [exit_price], marker="X", color=exit_color, s=140, zorder=5,
               label=f"Exit ({trade['exit_reason']})")
    if pd.notna(trade.get("tp1_date")) and pd.notna(trade.get("tp1_price")):
        ax.scatter([mdates.date2num(pd.Timestamp(trade["tp1_date"]))], [float(trade["tp1_price"])],
                   marker="o", color="#2e7d32", s=80, zorder=5, label="TP1 hit (50% off)")

    # Title with trade summary
    pnl = trade["net_pnl"]
    eq = trade["equity_after"]
    title = (f"#{idx:02d}  {trade['direction'].upper()}  "
             f"{entry_dt.date()} → {exit_dt.date()}  "
             f"PnL ${pnl:+,.0f}  →  Equity ${eq:,.0f}")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Price (USDT)")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(sub) // 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_overview(df: pd.DataFrame, trades: pd.DataFrame, out_path: str) -> None:
    """All entries plotted on the BTC price line, colored by win/loss."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df["close"], color="#444", linewidth=0.8, label="BTC close")

    for _, t in trades.iterrows():
        x = pd.Timestamp(t["entry_date"])
        y = float(t["entry_price"])
        win = t["net_pnl"] > 0
        color = "#2e7d32" if win else "#c62828"
        marker = "^" if t["direction"] == "long" else "v"
        ax.scatter([x], [y], marker=marker, color=color, s=70, edgecolor="white", linewidth=0.6, zorder=5)

    # Legend handles
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2e7d32", markersize=10, label="Long win"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#c62828", markersize=10, label="Long loss"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#2e7d32", markersize=10, label="Short win"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#c62828", markersize=10, label="Short loss"),
    ]
    ax.legend(handles=handles, loc="upper left")
    ax.set_title("All trades on BTCUSDT (Conservative mode)")
    ax.set_ylabel("Price (USDT)")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = fetch_klines()
    signals = detect_signals(df)
    trades, _ = run_backtest(df, signals, mode="conservative")
    from metrics import trades_to_df
    tdf = trades_to_df(trades)

    # Overview
    overview_path = os.path.join(os.path.dirname(__file__), "results", "trades_overview.png")
    plot_overview(df, tdf, overview_path)
    print(f"[overview] {overview_path}")

    # Per-trade
    for i, row in tdf.iterrows():
        out = os.path.join(OUT_DIR, f"trade_{i:02d}_{row['direction']}_{pd.Timestamp(row['entry_date']).date()}.png")
        plot_one_trade(df, row, i, out)
    print(f"[per-trade] {len(tdf)} charts → {OUT_DIR}/")


if __name__ == "__main__":
    main()
