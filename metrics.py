"""Performance metrics for a list of trades + equity curve."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from backtester import Trade, START_CAPITAL


def compute_metrics(trades: list[Trade], equity: pd.Series) -> dict[str, Any]:
    if not trades:
        return {
            "trades": 0, "final_equity": float(equity.iloc[-1]) if len(equity) else START_CAPITAL,
            "total_return_pct": 0.0, "mdd_pct": 0.0, "win_rate_pct": 0.0,
            "profit_factor": float("nan"), "avg_r": 0.0,
            "long_trades": 0, "long_win_rate_pct": 0.0,
            "short_trades": 0, "short_win_rate_pct": 0.0,
            "total_fees": 0.0,
        }

    final_eq = float(equity.iloc[-1])
    total_return = (final_eq / START_CAPITAL - 1.0) * 100.0

    # Max drawdown on equity curve
    peak = equity.cummax()
    dd = (equity - peak) / peak
    mdd = float(dd.min()) * 100.0

    pnls = [t.pnl - t.fees for t in trades]  # net per-trade
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / len(trades) * 100.0
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    # R-multiple per trade: realized PnL / initial risk in $
    rs = []
    for t in trades:
        risk_per_unit = abs(t.entry_price - t.sl)
        risk_dollar = risk_per_unit * t.qty_total
        if risk_dollar > 0:
            rs.append((t.pnl - t.fees) / risk_dollar)
    avg_r = sum(rs) / len(rs) if rs else 0.0

    longs = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]
    long_wins = [t for t in longs if (t.pnl - t.fees) > 0]
    short_wins = [t for t in shorts if (t.pnl - t.fees) > 0]
    long_wr = (len(long_wins) / len(longs) * 100.0) if longs else 0.0
    short_wr = (len(short_wins) / len(shorts) * 100.0) if shorts else 0.0

    total_fees = sum(t.fees for t in trades)

    # Liquidation risk check: 4x leverage initial → liquidation around ~25% adverse from entry.
    # Flag trades where adverse excursion exceeded ~20% (proxy: SL distance / entry > 20%)
    flagged = [t for t in trades if abs(t.entry_price - t.sl) / t.entry_price > 0.20]

    return {
        "trades": len(trades),
        "final_equity": round(final_eq, 2),
        "total_return_pct": round(total_return, 2),
        "mdd_pct": round(mdd, 2),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 2) if pf != float("inf") else None,
        "avg_r": round(avg_r, 3),
        "long_trades": len(longs),
        "long_win_rate_pct": round(long_wr, 2),
        "short_trades": len(shorts),
        "short_win_rate_pct": round(short_wr, 2),
        "total_fees": round(total_fees, 2),
        "liq_risk_trades": len(flagged),
    }


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    rows = [asdict(t) for t in trades]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["net_pnl"] = df["pnl"] - df["fees"]
    return df
