import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import gridspec
from .utils import sharpe_ratio, historical_var


def create_interactive_dashboard(adj_df, forecast_df, backtest_results, artifacts_dir):
    """Create interactive dashboard with Plotly."""

    # Extract cumulative returns from backtest results
    strat_cum, bench_cum = backtest_results["cumulative_returns"]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Price History & Forecast", "Backtest Performance"),
    )

    # Price history (last 2 years for clarity)
    recent_data = adj_df["TSLA"].iloc[-504:]  # ~2 years of trading days
    fig.add_trace(
        go.Scatter(
            x=recent_data.index,
            y=recent_data,
            name="TSLA Historical",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    # Forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["Price_FC"],
            name="Forecast",
            line=dict(color="red"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["Lower"],
            name="Lower CI",
            line=dict(dash="dash", color="gray"),
            fill=None,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["Upper"],
            name="Upper CI",
            line=dict(dash="dash", color="gray"),
            fill="tonexty",
        ),
        row=1,
        col=1,
    )

    # Backtest comparison
    fig.add_trace(
        go.Scatter(
            x=strat_cum.index,
            y=strat_cum.values,
            name="Strategy",
            line=dict(color="green"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=bench_cum.index,
            y=bench_cum.values,
            name="Benchmark",
            line=dict(color="orange"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=800, title_text="GMF Investment Analysis Dashboard", showlegend=True
    )
    fig.write_html(f"{artifacts_dir}/interactive_dashboard.html")


def performance_attribution(weights, returns, artifacts_dir):
    """Analyze performance attribution by asset."""

    weighted_returns = returns * pd.Series(weights)
    contribution = weighted_returns.sum(axis=1)

    # Calculate contribution percentages
    total_contribution = weighted_returns.sum().sum()
    asset_contribution = (weighted_returns.sum() / total_contribution * 100).round(1)

    fig = px.pie(
        values=asset_contribution.values,
        names=asset_contribution.index,
        title="Performance Attribution by Asset",
    )
    fig.write_html(f"{artifacts_dir}/performance_attribution.html")

    return asset_contribution


def enhanced_risk_analysis(pct_ret, artifacts_dir):
    """Enhanced risk analysis with additional metrics."""

    # Drawdown analysis
    cumulative = (1 + pct_ret).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative / rolling_max - 1) * 100

    fig = go.Figure()
    for asset in pct_ret.columns:
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown[asset], name=f"{asset} Drawdown")
        )

    fig.update_layout(title="Maximum Drawdown Analysis", yaxis_title="Drawdown (%)")
    fig.write_html(f"{artifacts_dir}/drawdown_analysis.html")

    # Correlation heatmap
    corr_matrix = pct_ret.corr()
    fig = px.imshow(
        corr_matrix, text_auto=True, aspect="auto", title="Asset Correlation Matrix"
    )
    fig.write_html(f"{artifacts_dir}/correlation_heatmap.html")


def create_model_comparison_chart(arima_metrics, lstm_metrics, artifacts_dir):
    """Create model performance comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ["MAE", "RMSE", "MAPE_%"]
    arima_values = [arima_metrics[m] for m in metrics]
    lstm_values = [lstm_metrics[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, arima_values, width, label="ARIMA", alpha=0.8, color="#1f77b4"
    )
    bars2 = ax.bar(
        x + width / 2, lstm_values, width, label="LSTM", alpha=0.8, color="#ff7f0e"
    )

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Values")
    ax.set_title(
        "Model Performance Comparison\n(Lower values indicate better performance)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["MAE", "RMSE", "MAPE (%)"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{artifacts_dir}/model_performance_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_efficient_frontier_chart(result, artifacts_dir):
    """Create Efficient Frontier chart from optimization results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # For Monte Carlo results, we might not have frontier data
    # Plot optimized portfolios
    max_sharpe_ret = result["perf_max_sharpe"]["ret"]
    max_sharpe_vol = result["perf_max_sharpe"]["vol"]
    min_vol_ret = result["perf_min_vol"]["ret"]
    min_vol_vol = result["perf_min_vol"]["vol"]

    # Create a simple efficient frontier line for visualization
    frontier_vol = np.linspace(min_vol_vol * 0.8, max_sharpe_vol * 1.2, 50)
    frontier_ret = max_sharpe_ret * (frontier_vol / max_sharpe_vol) ** 0.5

    ax.plot(frontier_vol, frontier_ret, "b-", alpha=0.6, label="Efficient Frontier")

    ax.scatter(
        max_sharpe_vol,
        max_sharpe_ret,
        marker="*",
        s=300,
        label=f"Max Sharpe (Return: {max_sharpe_ret:.1%}, Risk: {max_sharpe_vol:.1%})",
        color="red",
        edgecolors="black",
    )
    ax.scatter(
        min_vol_vol,
        min_vol_ret,
        marker="D",
        s=200,
        label=f"Min Volatility (Return: {min_vol_ret:.1%}, Risk: {min_vol_vol:.1%})",
        color="green",
        edgecolors="black",
    )

    ax.set_xlabel("Volatility (Risk)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier - Portfolio Optimization")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{artifacts_dir}/efficient_frontier.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_backtest_chart(strat_cum, bench_cum, artifacts_dir):
    """Create backtest performance comparison chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate returns for annotation
    strat_return = (strat_cum.iloc[-1] - 1) * 100
    bench_return = (bench_cum.iloc[-1] - 1) * 100
    outperformance = strat_return - bench_return

    # Plot cumulative returns
    ax.plot(
        strat_cum.index,
        strat_cum.values,
        label=f"Strategy ({strat_return:.1f}%)",
        linewidth=2.5,
        color="#2ca02c",
    )
    ax.plot(
        bench_cum.index,
        bench_cum.values,
        label=f"Benchmark 60/40 ({bench_return:.1f}%)",
        linewidth=2.5,
        color="#d62728",
        linestyle="--",
    )

    # Add outperformance annotation if there's significant data
    if len(strat_cum) > 10:
        mid_point = len(strat_cum) // 2
        ax.annotate(
            f"Outperformance: +{outperformance:.1f}%",
            xy=(
                strat_cum.index[mid_point],
                (strat_cum.values[mid_point] + bench_cum.values[mid_point]) / 2,
            ),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1"),
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (Start = 1.0)")
    ax.set_title("Backtest Performance: Strategy vs Benchmark\n(Aug 2024 - Jul 2025)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y-1:.0%}"))

    plt.tight_layout()
    plt.savefig(
        f"{artifacts_dir}/backtest_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_forecast_chart(historical_prices, forecast_data, artifacts_dir):
    """Create price forecast chart with confidence intervals."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot historical data (last 252 trading days ~1 year)
    historical_recent = (
        historical_prices.iloc[-252:]
        if len(historical_prices) > 252
        else historical_prices
    )
    ax.plot(
        historical_recent.index,
        historical_recent,
        label="Historical Prices",
        linewidth=2,
        color="#1f77b4",
    )

    # Plot forecast if available
    if forecast_data is not None and len(forecast_data) > 0:
        ax.plot(
            forecast_data.index,
            forecast_data["Price_FC"],
            label="Forecast",
            linewidth=2.5,
            color="#ff7f0e",
        )

        # Plot confidence intervals if available
        if "Lower" in forecast_data.columns and "Upper" in forecast_data.columns:
            ax.fill_between(
                forecast_data.index,
                forecast_data["Lower"],
                forecast_data["Upper"],
                alpha=0.3,
                label="95% Confidence Interval",
                color="#ff7f0e",
            )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.set_title("TSLA 12-Month Price Forecast with Confidence Intervals")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{artifacts_dir}/tsla_forecast.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_risk_dashboard(pct_ret, artifacts_dir):
    """Create comprehensive risk metrics dashboard using percentage returns."""
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    assets = ["TSLA", "BND", "SPY"]
    colors = ["#d62728", "#2ca02c", "#1f77b4"]  # Red, Green, Blue

    # Calculate risk metrics
    risk_data = {}
    for asset in assets:
        if asset in pct_ret.columns:
            returns = pct_ret[asset].dropna()
            if len(returns) > 0:
                risk_data[asset] = {
                    "ann_vol": returns.std() * np.sqrt(252),
                    "sharpe": sharpe_ratio(returns.mean(), returns.std()),
                    "var_95": historical_var(returns, 0.05),
                }

    # 1. Volatility Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    volatilities = [
        risk_data[asset]["ann_vol"] * 100 for asset in assets if asset in risk_data
    ]
    assets_vol = [asset for asset in assets if asset in risk_data]

    if volatilities:
        bars = ax1.bar(
            assets_vol, volatilities, color=colors[: len(assets_vol)], alpha=0.8
        )
        ax1.set_title("Annualized Volatility by Asset")
        ax1.set_ylabel("Volatility (%)")
        ax1.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
            )

    # 2. Sharpe Ratio Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    sharpe_ratios = [
        risk_data[asset]["sharpe"] for asset in assets if asset in risk_data
    ]
    assets_sharpe = [asset for asset in assets if asset in risk_data]

    if sharpe_ratios:
        bars = ax2.bar(
            assets_sharpe, sharpe_ratios, color=colors[: len(assets_sharpe)], alpha=0.8
        )
        ax2.set_title("Sharpe Ratio by Asset")
        ax2.set_ylabel("Sharpe Ratio")
        ax2.grid(True, alpha=0.3, axis="y")

        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.05,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

    # 3. Value at Risk (95%)
    ax3 = fig.add_subplot(gs[1, 0])
    var_95 = [
        risk_data[asset]["var_95"] * 100 for asset in assets if asset in risk_data
    ]
    assets_var = [asset for asset in assets if asset in risk_data]

    if var_95:
        bars = ax3.bar(assets_var, var_95, color=colors[: len(assets_var)], alpha=0.8)
        ax3.set_title("Value at Risk (95% Confidence)")
        ax3.set_ylabel("VaR (%)")
        ax3.grid(True, alpha=0.3, axis="y")

        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
            )

    # 4. Maximum Drawdown (simplified calculation)
    ax4 = fig.add_subplot(gs[1, 1])
    drawdowns = []
    assets_drawdown = []

    for asset in assets:
        if asset in pct_ret.columns:
            returns = pct_ret[asset].dropna()
            if len(returns) > 0:
                cumulative = (1 + returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative / rolling_max - 1).min() * 100
                drawdowns.append(drawdown)
                assets_drawdown.append(asset)

    if drawdowns:
        bars = ax4.bar(
            assets_drawdown, drawdowns, color=colors[: len(assets_drawdown)], alpha=0.8
        )
        ax4.set_title("Maximum Drawdown")
        ax4.set_ylabel("Drawdown (%)")
        ax4.grid(True, alpha=0.3, axis="y")

        for bar in bars:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()
    plt.savefig(
        f"{artifacts_dir}/risk_metrics_dashboard.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_allocation_charts(result, artifacts_dir):
    """Create portfolio allocation pie charts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#ff7f0e", "#2ca02c", "#1f77b4"]  # TSLA, BND, SPY colors

    # Max Sharpe Portfolio
    if "weights_max_sharpe" in result:
        max_sharpe_weights = list(result["weights_max_sharpe"].values())
        max_sharpe_labels = list(result["weights_max_sharpe"].keys())

        wedges1, texts1, autotexts1 = ax1.pie(
            max_sharpe_weights,
            labels=max_sharpe_labels,
            autopct="%1.1f%%",
            colors=colors[: len(max_sharpe_weights)],
            startangle=90,
        )
        ax1.set_title("Max Sharpe Portfolio Allocation\n(Aggressive Growth Strategy)")

        # Make autopct text larger and bold
        for autotext in autotexts1:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(10)

    # Min Volatility Portfolio
    if "weights_min_vol" in result:
        min_vol_weights = list(result["weights_min_vol"].values())
        min_vol_labels = list(result["weights_min_vol"].keys())

        wedges2, texts2, autotexts2 = ax2.pie(
            min_vol_weights,
            labels=min_vol_labels,
            autopct="%1.1f%%",
            colors=colors[: len(min_vol_weights)],
            startangle=90,
        )
        ax2.set_title(
            "Minimum Volatility Portfolio Allocation\n(Conservative Strategy)"
        )

        for autotext in autotexts2:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(10)

    plt.tight_layout()
    plt.savefig(
        f"{artifacts_dir}/portfolio_allocation.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def generate_all_visualizations(
    arima_metrics,
    lstm_metrics,
    result,
    strat_cum,
    bench_cum,
    historical_prices,
    forecast_data,
    pct_ret,
    artifacts_dir,
):
    """Generate all required visualizations for the report."""

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    print("Generating visualizations...")

    try:
        # 1. Model Performance Comparison
        create_model_comparison_chart(arima_metrics, lstm_metrics, artifacts_dir)
        print("✓ Model comparison chart generated")

        # 2. Efficient Frontier
        create_efficient_frontier_chart(result, artifacts_dir)
        print("✓ Efficient frontier chart generated")

        # 3. Backtest Comparison
        create_backtest_chart(strat_cum, bench_cum, artifacts_dir)
        print("✓ Backtest comparison chart generated")

        # 4. Price Forecast
        create_forecast_chart(historical_prices, forecast_data, artifacts_dir)
        print("✓ Price forecast chart generated")

        # 5. Risk Metrics Dashboard
        create_risk_dashboard(pct_ret, artifacts_dir)
        print("✓ Risk metrics dashboard generated")

        # 6. Portfolio Allocation
        create_allocation_charts(result, artifacts_dir)
        print("✓ Portfolio allocation charts generated")

    except Exception as e:
        print(f"⚠️ Visualization generation failed: {e}")
        import traceback

        traceback.print_exc()
