import pytest
import pandas as pd
import numpy as np
from src.data import compute_returns
from src.utils import sharpe_ratio, annualize_vol


def test_integration_data_to_metrics(sample_financial_data):
    """Test integration from data processing to metric calculation."""
    # Compute returns
    log_ret, pct_ret = compute_returns(sample_financial_data)

    # Calculate metrics for each asset
    for asset in pct_ret.columns:
        returns = pct_ret[asset].dropna()
        if len(returns) > 0:
            vol = annualize_vol(returns.std())
            sharpe = sharpe_ratio(returns.mean(), returns.std())

            # Basic sanity checks
            assert vol > 0
            assert not np.isnan(sharpe)


def test_portfolio_optimization_integration(sample_returns_data):
    """Test portfolio optimization integration."""
    from src.portfolio import monte_carlo_optimization

    # Create reasonable expected returns
    expected_rets = pd.Series({"TSLA": 0.15, "SPY": 0.10, "BND": 0.03})

    result = monte_carlo_optimization(expected_rets, sample_returns_data, "test")

    # Verify portfolio properties
    assert result["perf_max_sharpe"]["ret"] >= result["perf_min_vol"]["ret"]
    assert result["perf_max_sharpe"]["vol"] >= result["perf_min_vol"]["vol"]
