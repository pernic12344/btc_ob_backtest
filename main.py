"""Run end-to-end backtest: fetch data → detect signals → run both modes → report."""
from __future__ import annotations

import json
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_fetcher import fetch_klines
from strategy import detect_signals
from backtester import run_backtest, START_CAPITAL
from metrics import compute_metrics, trades_to_df

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def sanity_check(df: pd.DataFrame) -> list[str]:
    issues = []
    # Missing daily bars
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing = full_range.difference(df.index)
    if len(missing) > 0:
        issues.append(f"누락된 일봉 {len(missing)}개: {[d.date() for d in missing[:5]]}")
    # Outlier candles (±20% range)
    rng_pct = (df["high"] - df["low"]) / df["low"]
    outliers = df[rng_pct > 0.20]
    if len(outliers) > 0:
        issues.append(f"이상치(단일봉 range >20%) {len(outliers)}개: " +
                      ", ".join(f"{d.date()} {p*100:.1f}%" for d, p in zip(outliers.index, rng_pct[rng_pct > 0.20])))
    return issues


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("BTCUSDT 오더블럭 전략 백테스트")
    print("=" * 60)

    df = fetch_klines()
    print(f"\n[데이터] {len(df)}개 일봉, {df.index.min().date()} → {df.index.max().date()}")
    issues = sanity_check(df)
    if issues:
        print("\n[데이터 정합성 보고]")
        for s in issues:
            print(f"  - {s}")

    signals = detect_signals(df)
    long_sigs = [s for s in signals if s.direction == "long"]
    short_sigs = [s for s in signals if s.direction == "short"]
    print(f"\n[신호] 총 {len(signals)}개 (long {len(long_sigs)}, short {len(short_sigs)})")

    summary = {
        "data_period": [str(df.index.min().date()), str(df.index.max().date())],
        "data_bars": len(df),
        "signals_total": len(signals),
        "signals_long": len(long_sigs),
        "signals_short": len(short_sigs),
        "start_capital": START_CAPITAL,
        "modes": {},
    }

    eq_curves: dict[str, pd.Series] = {}
    for mode in ("conservative", "optimistic"):
        trades, eq = run_backtest(df, signals, mode=mode)
        m = compute_metrics(trades, eq)
        summary["modes"][mode] = m
        eq_curves[mode] = eq
        # Save trades CSV
        tdf = trades_to_df(trades)
        tdf.to_csv(os.path.join(RESULTS_DIR, f"trades_{mode}.csv"), index=False)

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'지표':<20} {'Conservative':>18} {'Optimistic':>18}")
    print("-" * 60)
    keys = [
        ("trades", "트레이드 수"),
        ("final_equity", "최종 자본 ($)"),
        ("total_return_pct", "총 수익률 (%)"),
        ("mdd_pct", "MDD (%)"),
        ("win_rate_pct", "승률 (%)"),
        ("profit_factor", "Profit Factor"),
        ("avg_r", "평균 R-multiple"),
        ("long_trades", "Long 트레이드"),
        ("long_win_rate_pct", "Long 승률 (%)"),
        ("short_trades", "Short 트레이드"),
        ("short_win_rate_pct", "Short 승률 (%)"),
        ("total_fees", "총 수수료 ($)"),
        ("liq_risk_trades", "고위험 트레이드(>20% 손절거리)"),
    ]
    for k, label in keys:
        cv = summary["modes"]["conservative"][k]
        ov = summary["modes"]["optimistic"][k]
        cv_s = f"{cv:,.2f}" if isinstance(cv, float) else str(cv)
        ov_s = f"{ov:,.2f}" if isinstance(ov, float) else str(ov)
        print(f"{label:<20} {cv_s:>18} {ov_s:>18}")
    print("=" * 60)

    # Equity curve chart
    fig, ax = plt.subplots(figsize=(12, 6))
    for mode, eq in eq_curves.items():
        ax.plot(eq.index, eq.values, label=mode.capitalize(), linewidth=1.5)
    ax.axhline(START_CAPITAL, color="gray", linestyle="--", linewidth=0.8, label=f"Start ${START_CAPITAL:,.0f}")
    ax.set_title("BTCUSDT Order Block Strategy — Equity Curve")
    ax.set_ylabel("Equity ($)")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "equity_curve.png")
    fig.savefig(chart_path, dpi=120)
    print(f"\n[차트] {chart_path}")

    # Save summary JSON
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[요약] {summary_path}")
    print(f"[트레이드] {RESULTS_DIR}/trades_conservative.csv, trades_optimistic.csv")


if __name__ == "__main__":
    main()
