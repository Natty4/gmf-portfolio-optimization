import sys
import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR
sys.path.append(str(SRC_DIR))

import os
import math
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data import download_data, to_adj_close_frame, clean_align, compute_returns
from src.eda import run_eda
from src.models import (
    fit_arima_on_log_returns,
    arima_predict_prices,
    fit_lstm,
    lstm_recursive_forecast,
    calculate_metrics,
)
from src.portfolio import optimize_portfolio, validate_expected_returns
from src.backtest import run_backtest, backtest, evaluate_series
from src.utils import (
    chronological_split,
    reconstruct_prices_from_log_returns,
    residual_bootstrap_intervals,
    annualize_return,
)

# Enhanced reporting
from src.reporting import generate_executive_summary


# Configuration
SEED = 42
np.random.seed(SEED)

TICKERS = ["TSLA", "BND", "SPY"]
START = "2015-07-01"
END = "2025-07-31"
TRAIN_END = "2023-12-31"
TEST_END = "2025-07-31"
FORECAST_HORIZON_DAYS = 252
LSTM_LOOKBACK = 60
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def main():
    """Main function to execute the entire pipeline."""
    # 1) Data
    print("Downloading data...")
    raw = download_data(TICKERS, START, END)
    adj = to_adj_close_frame(raw, TICKERS)
    adj = clean_align(adj)
    log_ret, pct_ret = compute_returns(adj)

    # 2) EDA
    print("Performing EDA...")
    run_eda(adj, log_ret, pct_ret, ARTIFACTS_DIR)

    # 3) Train/Test split (TSLA focus for models)
    adj_train, adj_test = chronological_split(adj[["TSLA"]], TRAIN_END, TEST_END)
    log_ret_train, log_ret_test = chronological_split(
        log_ret[["TSLA"]], TRAIN_END, TEST_END
    )

    # 4) ARIMA
    print("Fitting ARIMA model...")
    arima_model = fit_arima_on_log_returns(log_ret_train["TSLA"])
    steps_test = len(log_ret_test.dropna())
    arima_fc_logret_test, arima_fc_price_test = arima_predict_prices(
        arima_model, log_ret_train["TSLA"], adj_train["TSLA"].iloc[-1], steps_test
    )
    arima_price_series = pd.Series(
        arima_fc_price_test, index=adj_test.index[:steps_test], name="ARIMA_FC_Price"
    )

    # 5) LSTM
    print("Fitting LSTM model...")
    lstm_model, scaler = fit_lstm(
        log_ret_train["TSLA"], lookback=LSTM_LOOKBACK, seed=SEED
    )
    lstm_fc_logret_test = lstm_recursive_forecast(
        lstm_model, scaler, log_ret_train["TSLA"], steps_test, LSTM_LOOKBACK
    )
    lstm_price_test = reconstruct_prices_from_log_returns(
        adj_train["TSLA"].iloc[-1], lstm_fc_logret_test
    )
    lstm_price_series = pd.Series(
        lstm_price_test, index=adj_test.index[:steps_test], name="LSTM_FC_Price"
    )

    # 6) Evaluation on TEST
    y_true = adj_test["TSLA"].iloc[:steps_test]
    arima_metrics = calculate_metrics(y_true, arima_price_series)
    lstm_metrics = calculate_metrics(y_true, lstm_price_series)
    pd.DataFrame([arima_metrics, lstm_metrics], index=["ARIMA", "LSTM"]).to_csv(
        os.path.join(ARTIFACTS_DIR, "model_test_metrics.csv")
    )

    # 7) 12-month Forecast
    winner = "ARIMA" if arima_metrics["RMSE"] <= lstm_metrics["RMSE"] else "LSTM"
    steps_future = FORECAST_HORIZON_DAYS
    last_price = adj["TSLA"].iloc[-1]

    if winner == "ARIMA":
        fc_logret_future = arima_model.predict(n_periods=steps_future)
        fc_price_future = reconstruct_prices_from_log_returns(
            last_price, fc_logret_future
        )
        try:
            fc, conf = arima_model.predict(n_periods=steps_future, return_conf_int=True)
            lower = reconstruct_prices_from_log_returns(last_price, conf[:, 0])
            upper = reconstruct_prices_from_log_returns(last_price, conf[:, 1])
        except Exception:
            lower_log, upper_log = residual_bootstrap_intervals(
                log_ret["TSLA"], fc_logret_future
            )
            lower = reconstruct_prices_from_log_returns(last_price, lower_log)
            upper = reconstruct_prices_from_log_returns(last_price, upper_log)
    else:
        fc_logret_future = lstm_recursive_forecast(
            lstm_model, scaler, log_ret["TSLA"], steps_future, LSTM_LOOKBACK
        )
        fc_price_future = reconstruct_prices_from_log_returns(
            last_price, fc_logret_future
        )
        lower_log, upper_log = residual_bootstrap_intervals(
            log_ret["TSLA"], fc_logret_future
        )
        lower = reconstruct_prices_from_log_returns(last_price, lower_log)
        upper = reconstruct_prices_from_log_returns(last_price, upper_log)

    # Build forecast dataframe
    future_idx = pd.bdate_range(
        start=adj.index[-1] + pd.offsets.BDay(1), periods=steps_future
    )
    fc_df = pd.DataFrame(
        {"Price_FC": fc_price_future, "Lower": lower, "Upper": upper}, index=future_idx
    )
    fc_df.to_csv(os.path.join(ARTIFACTS_DIR, f"tsla_forecast_{winner.lower()}.csv"))

    # Plot forecast
    fig = plt.figure(figsize=(12, 6))
    plt.plot(adj.index[-252:], adj["TSLA"].iloc[-252:], label="History (last 1y)")
    plt.plot(fc_df.index, fc_df["Price_FC"], label=f"Forecast ({winner})")
    plt.fill_between(
        fc_df.index, fc_df["Lower"], fc_df["Upper"], alpha=0.2, label="Confidence Band"
    )
    plt.title("TSLA 12-Month Forecast (with Confidence Intervals)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, f"tsla_forecast_{winner.lower()}.png"))
    plt.close(fig)

    # 8) Portfolio Optimization
    print("Optimizing portfolio...")
    fc_daily_logret = pd.Series(fc_logret_future, index=future_idx)
    tsla_forecast_daily = (
        pct_ret["TSLA"].mean()
        if fc_daily_logret.isna().any()
        else float(fc_daily_logret.mean())
    )
    tsla_historical_daily = pct_ret["TSLA"].mean()
    tsla_exp_daily = 0.5 * tsla_forecast_daily + 0.5 * tsla_historical_daily
    tsla_exp_daily = (
        pct_ret["TSLA"].mean() if np.isnan(tsla_exp_daily) else tsla_exp_daily
    )
    tsla_exp_ann = annualize_return(tsla_exp_daily)

    hist_daily = adj.pct_change().dropna()
    mu_hist_daily = hist_daily.mean()
    bnd_exp_ann = (
        annualize_return(mu_hist_daily["BND"])
        if not np.isnan(mu_hist_daily["BND"])
        else 0.02
    )
    spy_exp_ann = (
        annualize_return(mu_hist_daily["SPY"])
        if not np.isnan(mu_hist_daily["SPY"])
        else 0.08
    )

    exp_ann = pd.Series({"TSLA": tsla_exp_ann, "BND": bnd_exp_ann, "SPY": spy_exp_ann})
    exp_ann = validate_expected_returns(exp_ann, max_annual_return=0.5)
    print(f"Expected annual returns: {exp_ann}")

    result = optimize_portfolio(exp_ann, hist_daily, winner.lower())
    weights_df = pd.DataFrame(
        [
            {**{"portfolio": "MaxSharpe"}, **result["weights_max_sharpe"]},
            {**{"portfolio": "MinVol"}, **result["weights_min_vol"]},
        ]
    )
    weights_df.to_csv(os.path.join(ARTIFACTS_DIR, "optimized_weights.csv"), index=False)

    # 9) Backtest
    print("Running backtest...")
    bt_start = "2024-08-01"
    bt_end = "2025-07-31"
    returns_daily = adj.pct_change().dropna()
    w_ms = result["weights_max_sharpe"]
    if not isinstance(w_ms, dict):
        w_ms = {k: float(v) for k, v in w_ms.items()}
    bench_w = {"SPY": 0.6, "BND": 0.4, "TSLA": 0.0}

    # Run backtest and capture cumulative returns
    strat_metrics, bench_metrics, strat_cum, bench_cum = run_backtest(
        w_ms, bench_w, returns_daily, bt_start, bt_end, ARTIFACTS_DIR
    )

    # Generate comprehensive executive summary
    portfolio_weights = {
        "max_sharpe": dict(result["weights_max_sharpe"]),
        "min_vol": dict(result["weights_min_vol"]),
    }

    backtest_results = {
        "strategy": strat_metrics,
        "benchmark": bench_metrics,
        "cumulative_returns": (strat_cum, bench_cum),
    }

    try:
        report_path = generate_executive_summary(
            arima_metrics,
            lstm_metrics,
            winner,
            portfolio_weights,
            backtest_results,
            ARTIFACTS_DIR,
        )

        if report_path:
            if report_path.endswith(".pdf"):
                print(f"📊 Detailed PDF Report: {report_path}")
            else:
                print(f"📋 Text Report: {report_path}")

    except Exception as e:
        print(f"⚠️  Report generation failed: {e}")
        # Fallback to simple text output
        print("\n" + "=" * 60)
        print("GMF INVESTMENTS - EXECUTIVE SUMMARY")
        print("=" * 60)
        print(f"Selected Model: {winner}")
        print(
            f"LSTM Improvement: {((1-lstm_metrics['RMSE']/arima_metrics['RMSE'])*100):.1f}%"
        )
        print(f"Strategy Return: {strat_metrics['total_return']*100:.1f}%")
        print(f"Benchmark Return: {bench_metrics['total_return']*100:.1f}%")
        print(
            f"Outperformance: {(strat_metrics['total_return']-bench_metrics['total_return'])*100:+.1f}%"
        )
        print("Max Sharpe Portfolio:", dict(result["weights_max_sharpe"]))
        print("Min Vol Portfolio:", dict(result["weights_min_vol"]))

    print(f"📈 Interactive Dashboard: artifacts/interactive_dashboard.html")
    print(f"📉 Risk Analysis: artifacts/drawdown_analysis.html")
    print(f"📊 Performance Attribution: artifacts/performance_attribution.html")

    # Print key takeaways for quick review
    print(f"\n🎯 KEY TAKEAWAYS:")
    print(
        f"   • LSTM improved accuracy by {((1-lstm_metrics['RMSE']/arima_metrics['RMSE'])*100):.1f}%"
    )
    print(
        f"   • Strategy outperformed benchmark by {(strat_metrics['total_return']-bench_metrics['total_return'])*100:+.1f}%"
    )
    print(
        f"   • Max Sharpe portfolio: {portfolio_weights['max_sharpe']['TSLA']*100:.1f}% TSLA"
    )
    print(f"   • Min Vol portfolio: {portfolio_weights['min_vol']['BND']*100:.1f}% BND")

    # Generate comprehensive visualizations
    try:
        from src.visualization import generate_all_visualizations

        generate_all_visualizations(
            arima_metrics=arima_metrics,
            lstm_metrics=lstm_metrics,
            result=result,
            strat_cum=strat_cum,
            bench_cum=bench_cum,
            historical_prices=adj["TSLA"],
            forecast_data=fc_df,
            pct_ret=pct_ret,  # Pass the percentage returns directly
            artifacts_dir=ARTIFACTS_DIR,
        )

    except Exception as e:
        print(f"⚠️ Visualization generation skipped: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
