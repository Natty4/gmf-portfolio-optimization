import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import os
from .utils import annualize_return, annualize_vol, sharpe_ratio


def backtest(
    weights: dict,
    returns: pd.DataFrame,
    start: str,
    end: str,
    rebalance: Optional[str] = None,
) -> pd.Series:
    """Backtest portfolio strategy and return cumulative returns."""
    weights = pd.Series(weights)
    returns = returns.loc[start:end]

    if rebalance is None:
        port_ret = (returns * weights).sum(axis=1)
        cum = (1 + port_ret).cumprod()
        return cum
    else:
        # Periodic rebalancing
        cum = pd.Series(index=returns.index, dtype=float)
        current_value = 1.0
        last_date = returns.index[0]
        for period_end, chunk in returns.groupby(pd.Grouper(freq=rebalance)):
            if len(chunk) == 0:
                continue
            port_ret = (chunk * weights).sum(axis=1)
            growth = (1 + port_ret).prod()
            current_value *= growth
            start_value = (
                cum.loc[last_date]
                if last_date in cum.index and not np.isnan(cum.loc[last_date])
                else 1.0
            )
            cum.loc[chunk.index] = np.linspace(start_value, current_value, len(chunk))
            last_date = chunk.index[-1]
        cum.ffill(inplace=True)
        cum.iloc[0] = 1.0
        return cum


def evaluate_series(cum: pd.Series) -> dict:
    """Evaluate portfolio performance metrics."""
    daily_ret = cum.pct_change().dropna()
    m = daily_ret.mean()
    s = daily_ret.std()
    return {
        "total_return": float(cum.iloc[-1] - 1),
        "ann_return": float(annualize_return(m)),
        "ann_vol": float(annualize_vol(s)),
        "sharpe": float(sharpe_ratio(m, s)),
    }


def run_backtest(
    strategy_weights: dict,
    benchmark_weights: dict,
    returns_daily: pd.DataFrame,
    bt_start: str,
    bt_end: str,
    artifacts_dir: str = "artifacts",
) -> tuple[dict, dict, pd.Series, pd.Series]:
    """Run backtest comparison between strategy and benchmark."""
    # Strategy backtest
    strat_cum = backtest(
        strategy_weights, returns_daily, bt_start, bt_end, rebalance=None
    )

    # Benchmark backtest
    bench_cum = backtest(
        benchmark_weights, returns_daily, bt_start, bt_end, rebalance=None
    )

    # Align indices
    idx = strat_cum.index.intersection(bench_cum.index)
    strat_cum = strat_cum.loc[idx]
    bench_cum = bench_cum.loc[idx]

    # Plot cumulative performance
    fig = plt.figure(figsize=(12, 6))
    plt.plot(strat_cum.index, strat_cum.values, label="Strategy (Max Sharpe)")
    plt.plot(bench_cum.index, bench_cum.values, label="Benchmark 60/40")
    plt.title("Backtest: Strategy vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Value (Start=1.0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "backtest_cumulative.png"))
    plt.close(fig)

    # Calculate metrics
    strat_metrics = evaluate_series(strat_cum)
    bench_metrics = evaluate_series(bench_cum)

    # Save metrics
    pd.DataFrame(
        [strat_metrics, bench_metrics], index=["Strategy", "Benchmark60/40"]
    ).to_csv(os.path.join(artifacts_dir, "backtest_metrics.csv"))

    # Save cumulative returns for further analysis
    strat_cum.to_csv(os.path.join(artifacts_dir, "strategy_cumulative_returns.csv"))
    bench_cum.to_csv(os.path.join(artifacts_dir, "benchmark_cumulative_returns.csv"))

    return strat_metrics, bench_metrics, strat_cum, bench_cum
