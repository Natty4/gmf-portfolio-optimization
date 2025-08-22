import numpy as np
import pandas as pd
import pytest

from src.data import clean_align, compute_returns, to_adj_close_frame


def test_clean_align(sample_financial_data):
    """Test data cleaning and alignment."""
    # Add some NaN values
    data_with_nans = sample_financial_data.copy()
    data_with_nans.iloc[5:10] = np.nan

    cleaned = clean_align(data_with_nans)

    assert not cleaned.isna().any().any()
    assert len(cleaned) > 0


def test_compute_returns(sample_financial_data):
    """Test returns computation."""
    log_ret, pct_ret = compute_returns(sample_financial_data)

    assert log_ret.shape == sample_financial_data.shape
    assert pct_ret.shape == sample_financial_data.shape
    assert log_ret.iloc[0].isna().all()  # First row should be NaN due to diff


def test_compute_returns_empty():
    """Test returns computation with empty data."""
    empty_df = pd.DataFrame()
    log_ret, pct_ret = compute_returns(empty_df)

    assert log_ret.empty
    assert pct_ret.empty
