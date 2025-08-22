import numpy as np
import pytest

from src.utils import annualize_return, annualize_vol, historical_var, sharpe_ratio


def test_annualize_return():
    """Test annualized return calculation."""
    daily_return = 0.001  # 0.1% daily
    annualized = annualize_return(daily_return)

    # (1 + 0.001)^252 - 1 ≈ 0.286
    assert pytest.approx(annualized, 0.01) == 0.286


def test_annualize_vol():
    """Test annualized volatility calculation."""
    daily_vol = 0.02  # 2% daily volatility
    annualized = annualize_vol(daily_vol)

    # 0.02 * sqrt(252) ≈ 0.317
    assert pytest.approx(annualized, 0.01) == 0.317


def test_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    mean_return = 0.001
    volatility = 0.02
    sharpe = sharpe_ratio(mean_return, volatility)

    expected = (mean_return / volatility) * np.sqrt(252)
    assert pytest.approx(sharpe, 0.01) == expected


def test_sharpe_ratio_zero_volatility():
    """Test Sharpe ratio with zero volatility."""
    sharpe = sharpe_ratio(0.001, 0.0)
    assert np.isnan(sharpe)


def test_historical_var(sample_returns_data):
    """Test historical Value at Risk calculation."""
    returns = sample_returns_data["TSLA"]
    var_95 = historical_var(returns, alpha=0.05)

    # VaR should be positive (representing potential loss)
    assert var_95 > 0
    assert var_95 == abs(returns.quantile(0.05))
