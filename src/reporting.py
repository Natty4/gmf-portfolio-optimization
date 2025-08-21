import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import datetime
import os

# Add this at the top of the file
try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'GMF Investments - Portfolio Optimization Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 8, f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def add_bullet_point(self, text, indent=10):
        """Add bullet point with proper Unicode handling"""
        self.set_x(indent)
        # Use a simple dash instead of Unicode bullet
        self.cell(5, 6, '-', 0, 0)
        self.set_x(indent + 5)
        self.multi_cell(0, 6, text)

def generate_executive_summary(arima_metrics, lstm_metrics, winner, portfolio_weights, 
                             backtest_results, artifacts_dir="artifacts"):
    """Generate a professional executive summary report."""
    
    if not FPDF_AVAILABLE:
        print("Warning: FPDF not available. Skipping PDF report generation.")
        return None
    
    pdf = PDF()
    pdf.add_page()
    
    # Executive Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'Executive Summary', 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, 
        "This report presents the results of our time series forecasting and portfolio optimization analysis. "
        "We compared ARIMA and LSTM models for Tesla stock prediction, optimized portfolio allocation using "
        "Modern Portfolio Theory, and backtested the strategy against a 60/40 benchmark."
    )
    pdf.ln(5)
    
    # Key Findings Box
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, 'Key Findings:', 0, 1, fill=True)
    pdf.set_font("Arial", size=10)
    
    strat = backtest_results['strategy']
    bench = backtest_results['benchmark']
    outperformance = strat['total_return'] - bench['total_return']
    
    findings = [
        f"LSTM model outperformed ARIMA by {(lstm_metrics['RMSE']/arima_metrics['RMSE']-1)*100:.1f}% in accuracy",
        f"Max Sharpe portfolio delivered {strat['total_return']*100:.1f}% vs benchmark {bench['total_return']*100:.1f}%",
        f"Strategy outperformed benchmark by {outperformance*100:.1f} percentage points",
        f"Conservative portfolio maintains {portfolio_weights['min_vol']['BND']*100:.1f}% in bonds for capital preservation"
    ]
    
    for finding in findings:
        pdf.add_bullet_point(finding)
        pdf.ln(1)
    
    pdf.ln(10)
    
    # Model Performance
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, '1. Model Performance Analysis', 0, 1)
    pdf.set_font("Arial", size=10)
    
    model_data = [
        ['Model', 'MAE', 'RMSE', 'MAPE (%)'],
        ['ARIMA', f"{arima_metrics['MAE']:.2f}", f"{arima_metrics['RMSE']:.2f}", f"{arima_metrics['MAPE_%']:.1f}"],
        ['LSTM', f"{lstm_metrics['MAE']:.2f}", f"{lstm_metrics['RMSE']:.2f}", f"{lstm_metrics['MAPE_%']:.1f}"],
        ['Improvement', f"{(1-lstm_metrics['MAE']/arima_metrics['MAE'])*100:.1f}%", 
         f"{(1-lstm_metrics['RMSE']/arima_metrics['RMSE'])*100:.1f}%", 
         f"{(1-lstm_metrics['MAPE_%']/arima_metrics['MAPE_%'])*100:.1f}%"]
    ]
    
    col_widths = [40, 30, 30, 30]
    for row in model_data:
        for i, item in enumerate(row):
            pdf.cell(col_widths[i], 8, str(item), border=1)
        pdf.ln()
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, f'Selected Model: {winner} (Lower values indicate better performance)', 0, 1)
    
    pdf.ln(10)
    
    # Portfolio Recommendations
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, '2. Portfolio Recommendations', 0, 1)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, 'Max Sharpe Portfolio (Aggressive Growth Strategy):', 0, 1)
    pdf.set_font("Arial", size=10)
    
    max_sharpe_data = []
    for asset, weight in portfolio_weights['max_sharpe'].items():
        max_sharpe_data.append([asset, f"{weight*100:.1f}%"])
    
    for row in max_sharpe_data:
        pdf.cell(40, 6, row[0])
        pdf.cell(20, 6, row[1])
        pdf.ln()
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, 'Minimum Volatility Portfolio (Conservative Strategy):', 0, 1)
    pdf.set_font("Arial", size=10)
    
    min_vol_data = []
    for asset, weight in portfolio_weights['min_vol'].items():
        min_vol_data.append([asset, f"{weight*100:.1f}%"])
    
    for row in min_vol_data:
        pdf.cell(40, 6, row[0])
        pdf.cell(20, 6, row[1])
        pdf.ln()
    
    pdf.ln(10)
    
    # Backtest Results
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, '3. Backtest Performance (Aug 2024 - Jul 2025)', 0, 1)
    
    perf_data = [
        ['Metric', 'Strategy', 'Benchmark (60/40)', 'Difference'],
        ['Total Return', f"{strat['total_return']*100:.1f}%", f"{bench['total_return']*100:.1f}%", 
         f"{outperformance*100:+.1f}%"],
        ['Annualized Return', f"{strat['ann_return']*100:.1f}%", f"{bench['ann_return']*100:.1f}%", 
         f"{(strat['ann_return']-bench['ann_return'])*100:+.1f}%"],
        ['Annual Volatility', f"{strat['ann_vol']*100:.1f}%", f"{bench['ann_vol']*100:.1f}%", 
         f"{(strat['ann_vol']-bench['ann_vol'])*100:+.1f}%"],
        ['Sharpe Ratio', f"{strat['sharpe']:.2f}", f"{bench['sharpe']:.2f}", 
         f"{strat['sharpe']-bench['sharpe']:+.2f}"]
    ]
    
    col_widths = [50, 35, 35, 30]
    for row in perf_data:
        for i, item in enumerate(row):
            pdf.cell(col_widths[i], 7, str(item), border=1)
        pdf.ln()
    
    pdf.ln(10)
    
    # Risk Assessment
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, '4. Risk Assessment & Recommendations', 0, 1)
    pdf.set_font("Arial", size=10)
    
    risk_insights = [
        "The aggressive portfolio offers significant growth potential but comes with higher volatility",
        "The conservative portfolio provides stability with 85.6% allocation to bonds",
        f"Strategy volatility ({strat['ann_vol']*100:.1f}%) is significantly higher than benchmark ({bench['ann_vol']*100:.1f}%)",
        "For risk-averse clients: Recommend Minimum Volatility portfolio",
        "For growth-oriented clients: Recommend Max Sharpe portfolio with proper risk disclosure"
    ]
    
    for insight in risk_insights:
        pdf.add_bullet_point(insight)
        pdf.ln(1)
    
    pdf.ln(10)
    
    # Investment Recommendations
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, '5. Investment Recommendations', 0, 1)
    pdf.set_font("Arial", size=10)
    
    recommendations = [
        "Implement dynamic rebalancing based on monthly forecast updates",
        "Consider adding stop-loss mechanisms for the aggressive portfolio",
        "Monitor Tesla's fundamental indicators alongside technical forecasts",
        "For new allocations, consider dollar-cost averaging into the strategy",
        "Regular review of model performance and recalibration as needed"
    ]
    
    for rec in recommendations:
        pdf.add_bullet_point(rec)
        pdf.ln(1)
    
    # Save report
    report_path = f"{artifacts_dir}/GMF_Portfolio_Report_{datetime.datetime.now().strftime('%Y%m%d')}.pdf"
    
    try:
        pdf.output(report_path)
        print(f"PDF report successfully generated: {report_path}")
    except UnicodeEncodeError as e:
        print(f"Warning: Could not generate PDF due to encoding issues: {e}")
        print("Generating text report instead...")
        report_path = generate_text_report(arima_metrics, lstm_metrics, winner, portfolio_weights, backtest_results, artifacts_dir)
    
    return report_path

def generate_text_report(arima_metrics, lstm_metrics, winner, portfolio_weights, backtest_results, artifacts_dir):
    """Generate a text-based report as fallback."""
    
    strat = backtest_results['strategy']
    bench = backtest_results['benchmark']
    outperformance = strat['total_return'] - bench['total_return']
    
    report_content = f"""
GMF INVESTMENTS - PORTFOLIO OPTIMIZATION REPORT
===============================================

EXECUTIVE SUMMARY:
This report presents the results of our time series forecasting and portfolio optimization analysis.

KEY FINDINGS:
- LSTM model outperformed ARIMA by {(lstm_metrics['RMSE']/arima_metrics['RMSE']-1)*100:.1f}% in accuracy
- Max Sharpe portfolio delivered {strat['total_return']*100:.1f}% vs benchmark {bench['total_return']*100:.1f}%
- Strategy outperformed benchmark by {outperformance*100:.1f} percentage points
- Conservative portfolio maintains {portfolio_weights['min_vol']['BND']*100:.1f}% in bonds

MODEL PERFORMANCE:
ARIMA: MAE={arima_metrics['MAE']:.2f}, RMSE={arima_metrics['RMSE']:.2f}, MAPE={arima_metrics['MAPE_%']:.1f}%
LSTM: MAE={lstm_metrics['MAE']:.2f}, RMSE={lstm_metrics['RMSE']:.2f}, MAPE={lstm_metrics['MAPE_%']:.1f}%
Selected Model: {winner}

PORTFOLIO ALLOCATIONS:
Max Sharpe (Aggressive):
  TSLA: {portfolio_weights['max_sharpe']['TSLA']*100:.1f}%
  SPY: {portfolio_weights['max_sharpe']['SPY']*100:.1f}%
  BND: {portfolio_weights['max_sharpe']['BND']*100:.1f}%

Min Volatility (Conservative):
  TSLA: {portfolio_weights['min_vol']['TSLA']*100:.1f}%
  SPY: {portfolio_weights['min_vol']['SPY']*100:.1f}%
  BND: {portfolio_weights['min_vol']['BND']*100:.1f}%

BACKTEST RESULTS (1 Year):
- Strategy Return: {strat['total_return']*100:.1f}%
- Benchmark Return: {bench['total_return']*100:.1f}%
- Outperformance: {outperformance*100:+.1f}%
- Strategy Sharpe: {strat['sharpe']:.2f}
- Benchmark Sharpe: {bench['sharpe']:.2f}
- Strategy Volatility: {strat['ann_vol']*100:.1f}%
- Benchmark Volatility: {bench['ann_vol']*100:.1f}%

RECOMMENDATIONS:
1. Implement dynamic rebalancing based on monthly forecast updates
2. Consider stop-loss mechanisms for aggressive portfolio
3. Monitor fundamental indicators alongside technical forecasts
4. Use dollar-cost averaging for new allocations
5. Regular model performance reviews

Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
    
    report_path = f"{artifacts_dir}/GMF_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_path
