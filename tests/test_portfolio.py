import pytest
import pandas as pd
import numpy as np
from src.portfolio import monte_carlo_optimization, validate_expected_returns

def test_monte_carlo_optimization():
    """Test Monte Carlo portfolio optimization."""
    expected_rets = pd.Series({'TSLA': 0.15, 'SPY': 0.10, 'BND': 0.03})
    
    # Create sample returns data
    np.random.seed(42)
    returns_data = pd.DataFrame({
        'TSLA': np.random.normal(0.001, 0.02, 100),
        'SPY': np.random.normal(0.0005, 0.01, 100),
        'BND': np.random.normal(0.0002, 0.005, 100)
    })
    
    result = monte_carlo_optimization(expected_rets, returns_data, 'test')
    
    assert 'weights_max_sharpe' in result
    assert 'weights_min_vol' in result
    assert 'perf_max_sharpe' in result
    assert 'perf_min_vol' in result
    
    # Weights should sum to approximately 1
    weights_sum = sum(result['weights_max_sharpe'].values())
    assert pytest.approx(weights_sum, 0.01) == 1.0

def test_validate_expected_returns():
    """Test expected returns validation."""
    expected_rets = pd.Series({
        'TSLA': 2.0,  # Too high - should be capped
        'BND': -1.5,  # Too low - should be capped
        'SPY': 0.12   # Valid
    })
    
    validated = validate_expected_returns(expected_rets, max_annual_return=0.5)
    
    assert validated['TSLA'] == 0.5  # Capped from 2.0
    assert validated['BND'] == -0.5  # Capped from -1.5
    assert validated['SPY'] == 0.12  # Unchanged