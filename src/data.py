import pandas as pd
import numpy as np
import yfinance as yf
from .utils import chronological_split


def download_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download financial data from Yahoo Finance."""
    print(f"Downloading: {tickers} from {start} to {end} ...")
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )

    # Ensure MultiIndex columns even for single ticker
    if isinstance(data.columns, pd.MultiIndex):
        return data
    else:
        # Single ticker case, build MultiIndex
        data = pd.concat({tickers[0]: data}, axis=1)
        return data


def to_adj_close_frame(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract Adjusted Close for each ticker into a single DataFrame."""
    adj = {}
    for t in tickers:
        # Use Adj Close if available; fallback to Close
        if "Adj Close" in raw[t].columns:
            adj[t] = raw[(t, "Adj Close")]
        else:
            adj[t] = raw[(t, "Close")]

    adj_df = pd.DataFrame(adj)
    adj_df.index = pd.to_datetime(adj_df.index)
    adj_df.sort_index(inplace=True)
    return adj_df


def clean_align(adj_df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then drop remaining NAs; keep common trading calendar."""
    df = adj_df.copy()
    df = df.ffill().bfill()
    df = df.dropna(how="any")
    return df


def compute_returns(df_prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute log returns and percentage returns."""
    log_ret = np.log(df_prices).diff()
    pct_ret = df_prices.pct_change()
    return log_ret, pct_ret
