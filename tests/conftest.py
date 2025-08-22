import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_financial_data():
    """Create sample financial data for testing."""
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    np.random.seed(42)

    data = {
        "TSLA": 100 + np.cumsum(np.random.normal(0.001, 0.02, len(dates))),
        "SPY": 300 + np.cumsum(np.random.normal(0.0005, 0.01, len(dates))),
        "BND": 80 + np.cumsum(np.random.normal(0.0002, 0.005, len(dates))),
    }

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_returns_data():
    """Create sample returns data for testing."""
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    np.random.seed(42)

    returns = pd.DataFrame(
        {
            "TSLA": np.random.normal(0.001, 0.02, len(dates)),
            "SPY": np.random.normal(0.0005, 0.01, len(dates)),
            "BND": np.random.normal(0.0002, 0.005, len(dates)),
        },
        index=dates,
    )

    return returns


@pytest.fixture
def sample_forecast_data():
    """Create sample forecast data for testing."""
    dates = pd.date_range("2021-01-01", "2021-06-30", freq="D")

    return pd.DataFrame(
        {
            "Price_FC": 150 + np.cumsum(np.random.normal(0.001, 0.015, len(dates))),
            "Lower": 140 + np.cumsum(np.random.normal(0.0005, 0.012, len(dates))),
            "Upper": 160 + np.cumsum(np.random.normal(0.0015, 0.018, len(dates))),
        },
        index=dates,
    )
