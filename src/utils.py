# src/utils.py
import math
import numpy as np
import pandas as pd


def annualize_return(daily_ret: float, trading_days: int = 252) -> float:
    """Annualize daily return."""
    return (1 + daily_ret) ** trading_days - 1


def annualize_vol(daily_vol: float, trading_days: int = 252) -> float:
    """Annualize daily volatility."""
    return daily_vol * math.sqrt(trading_days)


def sharpe_ratio(
    mean_daily_ret: float, daily_vol: float, rf_daily: float = 0.0
) -> float:
    """Calculate Sharpe ratio."""
    if daily_vol == 0:
        return np.nan
    return (mean_daily_ret - rf_daily) / daily_vol * math.sqrt(252)


def historical_var(pct_series: pd.Series, alpha: float = 0.05) -> float:
    """Calculate Historical Value-at-Risk."""
    q = pct_series.dropna().quantile(alpha)
    return abs(q)


def chronological_split(
    df: pd.DataFrame, train_end: str, test_end: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train and test sets."""
    df = df.copy()
    df_train = df.loc[:train_end]
    df_test = df.loc[pd.to_datetime(train_end) + pd.offsets.Day(1) : test_end]
    return df_train, df_test


def reconstruct_prices_from_log_returns(
    last_price: float, fc_log_rets: np.ndarray
) -> np.ndarray:
    """Reconstruct prices from log returns."""
    prices = [last_price]
    for r in fc_log_rets:
        prices.append(prices[-1] * np.exp(r))
    return np.array(prices[1:])


def residual_bootstrap_intervals(
    actual_log_ret: pd.Series,
    pred_log_ret: np.ndarray,
    n_boot: int = 200,
    alpha: float = 0.05,
):
    """Calculate prediction intervals using residual bootstrap."""
    residuals = actual_log_ret[-1000:].dropna().values - 0  # center at 0
    if len(residuals) < 50:
        residuals = np.random.normal(
            0, np.std(pred_log_ret) if len(pred_log_ret) > 1 else 0.01, size=500
        )

    sims = []
    n_steps = len(pred_log_ret)
    for _ in range(n_boot):
        noise = np.random.choice(residuals, size=n_steps, replace=True)
        sims.append(pred_log_ret + noise)
    sims = np.array(sims)
    lower = np.percentile(sims, 100 * (alpha / 2), axis=0)
    upper = np.percentile(sims, 100 * (1 - alpha / 2), axis=0)
    return lower, upper
