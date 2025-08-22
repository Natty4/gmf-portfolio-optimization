import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from .utils import sharpe_ratio, annualize_return, annualize_vol


def adf_test(series: pd.Series) -> dict:
    """Perform Augmented Dickey-Fuller test for stationarity."""
    res = adfuller(series.dropna(), autolag="AIC")
    return {
        "adf_stat": res[0],
        "p_value": res[1],
        "lags_used": res[2],
        "n_obs": res[3],
        "crit_values": res[4],
    }


def historical_var(pct_series: pd.Series, alpha: float = 0.05) -> float:
    """Calculate Historical Value-at-Risk."""
    q = pct_series.dropna().quantile(alpha)
    return abs(q)


def outlier_days(
    pct_series: pd.Series, z_thresh: float = 3.0, lookback: int = 21
) -> pd.DataFrame:
    """Identify outlier days using Z-score method."""
    rolling_m = pct_series.rolling(lookback).mean()
    rolling_s = pct_series.rolling(lookback).std()
    z = (pct_series - rolling_m) / rolling_s
    mask = z.abs() > z_thresh
    return pd.DataFrame({"return": pct_series, "zscore": z})[mask]


def run_eda(
    adj_df: pd.DataFrame,
    log_ret: pd.DataFrame,
    pct_ret: pd.DataFrame,
    artifacts_dir: str = "artifacts",
) -> None:
    """Perform exploratory data analysis."""
    # Plot prices
    fig1 = plt.figure(figsize=(12, 6))
    adj_df.plot(ax=plt.gca(), title="Adjusted Close — TSLA / BND / SPY")
    plt.xlabel("Date")
    plt.ylabel("Price")
    fig1.savefig(
        os.path.join(artifacts_dir, "prices_timeseries.png"), bbox_inches="tight"
    )
    plt.close(fig1)

    # Plot daily % returns
    fig2 = plt.figure(figsize=(12, 6))
    (pct_ret * 100).plot(
        ax=plt.gca(), alpha=0.8, title="Daily % Returns — TSLA / BND / SPY"
    )
    plt.xlabel("Date")
    plt.ylabel("Daily Return (%)")
    fig2.savefig(os.path.join(artifacts_dir, "daily_returns.png"), bbox_inches="tight")
    plt.close(fig2)

    # Rolling volatility (21d std)
    roll = pct_ret.rolling(21).std() * np.sqrt(252)
    fig3 = plt.figure(figsize=(12, 6))
    roll.plot(ax=plt.gca(), title="Annualized Rolling Volatility (21d)")
    plt.xlabel("Date")
    plt.ylabel("Volatility (annualized)")
    fig3.savefig(
        os.path.join(artifacts_dir, "rolling_volatility.png"), bbox_inches="tight"
    )
    plt.close(fig3)

    # Outliers on TSLA
    tsla_out = outlier_days(pct_ret["TSLA"])
    tsla_out.to_csv(os.path.join(artifacts_dir, "tsla_outliers.csv"))

    # ADF tests (Close vs Returns)
    adf_close = adf_test(adj_df["TSLA"])
    adf_ret = adf_test(log_ret["TSLA"])
    pd.DataFrame([adf_close, adf_ret], index=["TSLA_Close", "TSLA_LogReturn"]).to_csv(
        os.path.join(artifacts_dir, "adf_tests_tsla.csv")
    )

    # Risk metrics
    risk_rows = []
    for c in pct_ret.columns:
        m = pct_ret[c].mean()
        s = pct_ret[c].std()
        sr = sharpe_ratio(m, s, rf_daily=0.0)
        var95 = historical_var(pct_ret[c], alpha=0.05)
        var99 = historical_var(pct_ret[c], alpha=0.01)
        risk_rows.append(
            {
                "asset": c,
                "mean_daily_ret": m,
                "daily_vol": s,
                "ann_return": annualize_return(m),
                "ann_vol": annualize_vol(s),
                "sharpe": sr,
                "VaR_95": var95,
                "VaR_99": var99,
            }
        )
    pd.DataFrame(risk_rows).to_csv(
        os.path.join(artifacts_dir, "risk_metrics.csv"), index=False
    )
