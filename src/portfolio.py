import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

try:
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import risk_models

    PYPFOPT_AVAILABLE = True
except Exception:
    PYPFOPT_AVAILABLE = False


def optimize_portfolio(
    expected_rets_vec: pd.Series, returns_hist: pd.DataFrame, label: str = "run1"
):
    """Build Efficient Frontier; return dict with weights, plots, and two key portfolios."""
    try:
        if PYPFOPT_AVAILABLE:
            mu = expected_rets_vec.copy()
            S = risk_models.sample_cov(returns_hist)

            # Add small regularization to covariance matrix
            S_reg = S + np.eye(S.shape[0]) * 1e-6

            ef = EfficientFrontier(mu, S_reg)

            try:
                w_max_sharpe = ef.max_sharpe()
                cleaned_ms = ef.clean_weights()
                perf_ms = ef.portfolio_performance(verbose=False)
            except Exception as e:
                print(
                    f"Max Sharpe optimization failed: {e}. Using Monte Carlo fallback."
                )
                return monte_carlo_optimization(mu, returns_hist, label)

            ef = EfficientFrontier(mu, S_reg)
            try:
                w_min_vol = ef.min_volatility()
                cleaned_mv = ef.clean_weights()
                perf_mv = ef.portfolio_performance(verbose=False)
            except Exception as e:
                print(f"Min Vol optimization failed: {e}. Using Monte Carlo fallback.")
                return monte_carlo_optimization(mu, returns_hist, label)

            # Frontier curve sampling
            w_ret, w_risk, w_sharpe = [], [], []
            for targ_ret in np.linspace(mu.min(), mu.max() * 1.5, 80):
                try:
                    ef = EfficientFrontier(mu, S_reg)
                    ef.efficient_return(targ_ret)
                    ret, vol, shrp = ef.portfolio_performance()
                    w_ret.append(ret)
                    w_risk.append(vol)
                    w_sharpe.append(shrp)
                except Exception:
                    continue

            # Plot
            fig = plt.figure(figsize=(10, 6))
            plt.scatter(w_risk, w_ret, s=10, alpha=0.6, label="Frontier (samples)")
            plt.scatter(perf_ms[1], perf_ms[0], marker="*", s=200, label="Max Sharpe")
            plt.scatter(perf_mv[1], perf_mv[0], marker="D", s=120, label="Min Vol")
            plt.xlabel("Volatility (σ)")
            plt.ylabel("Expected Return (μ)")
            plt.title("Efficient Frontier — GMF")
            plt.legend()
            path = os.path.join("artifacts", f"efficient_frontier_{label}.png")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)

            return {
                "weights_max_sharpe": cleaned_ms,
                "perf_max_sharpe": {
                    "ret": perf_ms[0],
                    "vol": perf_ms[1],
                    "sharpe": perf_ms[2],
                },
                "weights_min_vol": cleaned_mv,
                "perf_min_vol": {
                    "ret": perf_mv[0],
                    "vol": perf_mv[1],
                    "sharpe": perf_mv[2],
                },
                "frontier_image": path,
            }
        else:
            return monte_carlo_optimization(expected_rets_vec, returns_hist, label)
    except Exception as e:
        print(f"Portfolio optimization failed: {e}. Using Monte Carlo fallback.")
        return monte_carlo_optimization(expected_rets_vec, returns_hist, label)


def monte_carlo_optimization(
    expected_rets_vec: pd.Series, returns_hist: pd.DataFrame, label: str
):
    """Monte Carlo optimization fallback."""
    mu = expected_rets_vec.values
    cov = returns_hist.cov().values

    # Handle NaN/inf values
    mu = np.nan_to_num(mu, nan=0.1, posinf=0.5, neginf=-0.5)

    if np.isnan(cov).any() or np.isinf(cov).any():
        print("Warning: Invalid values in covariance matrix. Using diagonal matrix.")
        cov = np.eye(len(mu)) * 0.04

    n = len(mu)
    sims = 50000
    weights = np.random.dirichlet(np.ones(n), size=sims)
    port_rets = weights @ mu
    port_vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    sharpe = port_rets / (port_vols + 1e-12)
    sharpe = np.nan_to_num(sharpe, nan=-10, posinf=10, neginf=-10)

    i_ms = np.nanargmax(sharpe)
    i_mv = np.nanargmin(port_vols)

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(port_vols, port_rets, s=1, alpha=0.3)
    plt.scatter(port_vols[i_ms], port_rets[i_ms], marker="*", s=200, label="Max Sharpe")
    plt.scatter(port_vols[i_mv], port_rets[i_mv], marker="D", s=120, label="Min Vol")
    plt.xlabel("Volatility (σ)")
    plt.ylabel("Expected Return (μ)")
    plt.title("Efficient Frontier — Monte Carlo")
    plt.legend()
    path = os.path.join("artifacts", f"efficient_frontier_mc_{label}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    return {
        "weights_max_sharpe": dict(zip(returns_hist.columns, weights[i_ms].round(4))),
        "perf_max_sharpe": {
            "ret": float(port_rets[i_ms]),
            "vol": float(port_vols[i_ms]),
            "sharpe": float(sharpe[i_ms]),
        },
        "weights_min_vol": dict(zip(returns_hist.columns, weights[i_mv].round(4))),
        "perf_min_vol": {
            "ret": float(port_rets[i_mv]),
            "vol": float(port_vols[i_mv]),
            "sharpe": float(sharpe[i_mv]),
        },
        "frontier_image": path,
    }


def validate_expected_returns(
    expected_rets: pd.Series, max_annual_return: float = 0.5
) -> pd.Series:
    """Cap expected returns to reasonable values and handle NaN."""
    validated = expected_rets.copy()
    for asset in validated.index:
        if np.isnan(validated[asset]) or np.isinf(validated[asset]):
            print(
                f"Warning: {asset} has invalid expected return {validated[asset]}. Using default."
            )
            if asset == "TSLA":
                validated[asset] = 0.15
            elif asset == "BND":
                validated[asset] = 0.03
            elif asset == "SPY":
                validated[asset] = 0.10
        elif validated[asset] > max_annual_return:
            print(
                f"Warning: Capping {asset} expected return from {validated[asset]:.2f} to {max_annual_return:.2f}"
            )
            validated[asset] = max_annual_return
        elif validated[asset] < -max_annual_return:
            print(
                f"Warning: Capping {asset} expected return from {validated[asset]:.2f} to {-max_annual_return:.2f}"
            )
            validated[asset] = -max_annual_return
    return validated
