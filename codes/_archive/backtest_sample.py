from src.agent import DataCollector
from optimizer.PortfolioOptimizer import PortfolioOptimizerModule
import pandas as pd
import matplotlib.pyplot as plt
import logging
from src.config.settings import PORTFOLIOS
from src.agent.DataCollector import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Backtesting and Plotting Functions
# ----------------------------
def extract_etf_tickers(portfolio: dict, key: str = "treasuries") -> list:
    """
    Extracts ETF tickers from a portfolio dictionary.
    """
    tickers = []
    assets = portfolio.get(key, [])
    for asset in assets:
        ticker = asset.get("etf")
        if ticker:
            tickers.append(ticker)
    return tickers

def compute_cumulative_returns(close_prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Computes cumulative returns for a portfolio given daily Close prices and constant weights.
    """
    returns = close_prices.pct_change().dropna()
    port_daily_returns = returns.dot(weights)
    cum_returns = (1 + port_daily_returns).cumprod()
    return cum_returns

def backtest_black_litterman(collector: DataCollector, portfolio: dict, view_dict: dict,
                             full_start: str, full_end: str, backtest_start: str, backtest_end: str,
                             rebalance_freq: str = "W") -> (pd.Series, pd.Series):
    """
    Backtests a Black-Litterman portfolio.
    
    Parameters:
        collector: DataCollector instance.
        portfolio: dict – portfolio structure (used for ticker extraction).
        view_dict: dict – views for Black-Litterman.
        full_start, full_end: str – period to download full ETF data.
        backtest_start, backtest_end: str – period for backtesting (must be within full period).
        rebalance_freq: str – rebalancing frequency (default "W" for weekly).
    
    Returns:
        Tuple of (BL cumulative return series, Benchmark cumulative return series).
    """
    tickers = extract_etf_tickers(portfolio)
    # Download adjusted close panel data.
    all_prices = collector.download_etf_adj_close(tickers, full_start, full_end)
    # Estimate covariance matrix and get the close prices and returns.
    cov_matrix, close_prices, _ = collector.estimate_covariance_matrix(all_prices, method="ledoit_wolf")
    
    # Define rebalancing dates within the backtest period
    all_dates = pd.to_datetime(close_prices.index)
    rebal_dates = all_dates[(all_dates >= pd.to_datetime(backtest_start)) & (all_dates <= pd.to_datetime(backtest_end))]
    rebal_dates = rebal_dates[::7]  # roughly weekly
    
    bl_cum_returns = []
    benchmark_returns = []
    
    # For benchmark: equal weight buy-and-hold starting at first rebalancing date.
    benchmark_weights = pd.Series(1/len(tickers), index=tickers)
    
    for i in range(len(rebal_dates)-1):
        rebalance_date = rebal_dates[i]
        next_date = rebal_dates[i+1]
        lookback_start = (rebalance_date - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
        lookback_end = (rebalance_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        window_prices = close_prices.loc[lookback_start:lookback_end]
        if window_prices.empty:
            continue
        daily_returns = window_prices.pct_change().dropna()
        est_mu = daily_returns.mean() * 252
        # Estimate covariance from the one-year lookback window.
        est_cov, _, _ = collector.estimate_covariance_matrix(window_prices, method="ledoit_wolf")
        optimizer = PortfolioOptimizerModule(fixed_income_portfolio, est_mu, est_cov)
        bl_weights = optimizer.black_litterman_allocation(view_dict=view_dict, tau=0.05, delta=2.5)
        period_prices = close_prices.loc[rebalance_date:next_date]
        if period_prices.empty:
            continue
        bl_cum = compute_cumulative_returns(period_prices, bl_weights)
        bench_cum = compute_cumulative_returns(period_prices, benchmark_weights)
        bl_cum_returns.append(bl_cum.iloc[-1])
        benchmark_returns.append(bench_cum.iloc[-1])
    
    bl_series = pd.Series(bl_cum_returns, index=rebal_dates[1:len(bl_cum_returns)+1])
    bl_compounded = (bl_series + 1).cumprod()
    bench_series = pd.Series(benchmark_returns, index=rebal_dates[1:len(benchmark_returns)+1])
    bench_compounded = (bench_series + 1).cumprod()
    
    return bl_compounded, bench_compounded


# ----------------------------
# Main Backtest Execution
# ----------------------------
if __name__ == "__main__":
    # Define fixed income ETF portfolio structure
    fixed_income_portfolio = PORTFOLIOS['bond']
    # Extract tickers from portfolio
    tickers = extract_etf_tickers(fixed_income_portfolio, key="treasuries")
    logger.info(f"Tickers: {tickers}")
    
    # Define backtest period: Use two years of data, with the second year for rebalancing simulation.
    full_start = "2023-11-01"
    full_end = "2025-03-30"
    # For backtesting, we will update weekly during the second year.
    backtest_start = "2024-11-01"
    backtest_end = "2025-03-30"
    
    # Initialize DataCollector
    collector = DataCollector()
    
    # Define a fixed view dictionary for Black-Litterman (example views)
    view_dict = {"TLT": 0.08, "TLH": 0.09}
    
    # Run backtest: get cumulative returns for BL portfolio and benchmark.
    bl_cum_return, bench_cum_return = backtest_black_litterman(
        collector=collector,
        portfolio=fixed_income_portfolio,
        view_dict=view_dict,
        full_start=full_start,
        full_end=full_end,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        rebalance_freq="W"
    )
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(bl_cum_return.index, bl_cum_return.values, label="Black-Litterman Portfolio")
    plt.plot(bench_cum_return.index, bench_cum_return.values, label="Buy & Hold Benchmark", linestyle="--")
    plt.title("Cumulative Returns: Black-Litterman vs. Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()