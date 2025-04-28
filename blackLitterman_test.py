#!/usr/bin/env python
import pandas as pd
import numpy as np
from numpy.linalg import inv, eigvalsh   # h = Hermitian
import matplotlib.pyplot as plt
import os
from src.agent.DataCollector import *
from dateutil.relativedelta import relativedelta

# ----- Global Parameters -----
LOOKBACK_WEEKS = 10  
ROBUST_LOOKBACK_WEEKS = 260     # ≈ 5 years  (adjust as you like)
UNCERTAINTY_SCALE = 0.5   # Scaling factor for view uncertainty
EPSILON = 1e-4           # Small constant to avoid division by zero
TAU = 0.2             # Scaling parameter for the prior covariance in BL
RISK_AVERSION = 1       # Risk aversion parameter for mean-variance optimization
RISK_FREE_RATE = 0.0

from src.config.settings import PORTFOLIOS  

# Define asset lists from settings
fx_tickers = [entry["etf"] for entry in PORTFOLIOS['fx'].get("currencies", [])]
fi_tickers = [entry["etf"] for entry in PORTFOLIOS['bond'].get("treasuries", [])]
equity_tickers = [entry["etf"] for entry in PORTFOLIOS["equity"].get("sectors", [])]
commodity_tickers = [entry["etf"] for entry in PORTFOLIOS["commodity"].get("sectors", [])]
# -----------------------------------------------
# Data loading function
def load_data(asset_class="fx"):
    """
    Load prediction and actual data files.
    For FX, equity, commodity:
      - Prediction file: data/{asset_class}_weekly_predictions.csv (with columns: date, etf, predicted_return, confidence)
      - Actual file: data/{asset_class}_combined_features_weekly.csv (price data; we compute returns from prices)
    
    For fixed income ("fi"):
      - Prediction file: data/fi_weekly_predictions.csv (with columns: date, etf, predicted_return, confidence)
      - Actual file: data/fi_combined_features_weekly.csv (with columns for each ETF containing the total volume‐weighted return)
    
    Returns:
      predictions, actual_data
    """
    asset_class = asset_class.lower()
    try:
        pred_file   = f"data/{asset_class}_weekly_predictions.csv"
        actual_file = f"data/{asset_class}_combined_features_weekly.csv"
    except:
        raise ValueError("asset_class must be 'fx', 'fi', 'equity', or 'commodity'!")
    
    print("Loading prediction data...")
    predictions = pd.read_csv(pred_file)
    predictions['date'] = pd.to_datetime(predictions['date'])
    
    print("Loading actual data...")
    actual_data = pd.read_csv(actual_file)
    if 'Unnamed: 0' in actual_data.columns:
        actual_data['date'] = pd.to_datetime(actual_data['Unnamed: 0'])
        actual_data = actual_data.drop('Unnamed: 0', axis=1)
    else:
        first_col = actual_data.columns[0]
        actual_data['date'] = pd.to_datetime(actual_data[first_col])
        actual_data = actual_data.drop(first_col, axis=1)
    
    # Always sort actual_data ascending by date
    actual_data = actual_data.sort_values('date')
    return predictions, actual_data

# -----------------------------------------------
# Mean-variance portfolio optimizer
def mean_variance_portfolio(Sigma, expected_returns, risk_aversion=RISK_AVERSION):
    inv_Sigma = inv(Sigma)
    raw_weights = inv_Sigma @ expected_returns
    norm_factor = np.sum(np.abs(raw_weights))
    if norm_factor < EPSILON:
        return np.ones_like(raw_weights) / len(raw_weights)
    return raw_weights / norm_factor

# -----------------------------------------------
# Black-Litterman update function
def black_litterman_update(pi, Sigma, q, confidences, tau=TAU, uncertainty_scale=UNCERTAINTY_SCALE, epsilon=EPSILON):
    omega_diag = np.array([uncertainty_scale * (1 - c) + epsilon for c in confidences])
    Omega = np.diag(omega_diag)
    inv_term = inv(tau * Sigma) + inv(Omega)
    r_bl = inv(inv_term) @ (inv(tau * Sigma) @ pi + inv(Omega) @ q)
    return r_bl

# Maximum Sharpe Portfolio (Tangency Portfolio) Optimizer
def max_sharpe_portfolio(Sigma, expected_returns, risk_free_rate=RISK_FREE_RATE):
    """
    Compute maximum Sharpe (tangency) portfolio weights.
    Under the assumption that r_f is zero (or given), the optimal weights are proportional to:
         w ∝ Σ⁻¹ (expected_returns - r_f)
    Then normalized so that sum(w) = 1.
    """
    adjusted_returns = expected_returns - risk_free_rate
    inv_Sigma = inv(Sigma)
    raw_weights = inv_Sigma @ adjusted_returns
    norm_factor = np.sum(np.abs(raw_weights))
    if norm_factor < EPSILON:
        return np.ones_like(raw_weights) / len(raw_weights)
    return raw_weights / norm_factor

def robust_covariance_estimation(tickers, robust_start_date, end_date):
    """
    Uses DataCollector to download adjusted close prices and estimate a robust covariance matrix.
    
    Parameters:
      tickers: list of tickers (for FX or FI)
      robust_start_date: starting date for robust estimation 
      end_date: end date (current prediction date)
      
    Returns:
      Sigma: robust covariance matrix (numpy array)
      robust_pi: Baseline mean returns (column vector) over the robust period.
    """
    collector  = DataCollector(full_start_date=robust_start_date,
                               target_start_date=robust_start_date,
                               end_date=end_date)
    price = collector.get_etf_adj_close(tickers, robust_start_date, end_date)
    if price.empty:
        raise ValueError("No price data downloaded for robust covariance estimation.")

    # convert to WEEKLY returns before shrinkage
    weekly_ret = price.resample("W-FRI").last().pct_change().dropna()

    lw = LedoitWolf().fit(weekly_ret.values)
    Sigma = lw.covariance_
    pi    = weekly_ret.mean().values.reshape(-1,1)
    return Sigma, pi

# -----------------------------------------------
# Rolling backtest pipeline
def rolling_bl_backtest(predictions, actual_data, asset_list, asset_class="fx", lookback_weeks=LOOKBACK_WEEKS):
    """
    Perform a rolling backtest.
    For each prediction date (week) where at least 'lookback_weeks' of historical actual data is available:
      - For FX, equity, commodity: compute weekly returns from price data using pct_change.
      - For FI: assume the actual data already provides the weekly total volume-weighted return.
    Then use these historical returns to compute baseline expected returns (pi) and covariance (Sigma),
    update expected returns with the BL formula using the prediction (q) and confidence scores,
    solve for portfolio weights, and compute next week's portfolio return.
    
    Also computes an equal-weighted portfolio return for comparison.
    
    Returns:
      results_df: DataFrame with columns: date, bl_portfolio_return, equal_weighted_return.
      weights_history: List of BL weights (as numpy arrays) used each week (for turnover calculation).
    """
    # Pivot predictions to get one row per date with columns for each asset
    pred_pivot = predictions.pivot(index='date', columns='etf', values='predicted_return')
    conf_pivot = predictions.pivot(index='date', columns='etf', values='confidence')
    
    actual_data = actual_data.sort_values('date')
    
    results = []
    weights_history = []
    
    for current_date in pred_pivot.index:
        # Only use historical data strictly before current_date
        hist_data = actual_data[actual_data['date'] < current_date]
        
        if len(hist_data) < lookback_weeks:
            continue
        
        lookback_data = hist_data.tail(lookback_weeks)
        if asset_class == "fi":
            # For FI: assume columns already contain weekly returns
            returns_lookback = lookback_data[asset_list].dropna()
        else:
            # For FX, equity, commodity: compute returns from prices
            returns_lookback = lookback_data[asset_list].pct_change().dropna()
        
        if returns_lookback.empty:
            continue
        
        pi = returns_lookback.mean().values.reshape(-1, 1)
        # Sigma = returns_lookback.cov().values
        # inside rolling_bl_backtest, before BL update
        # rolling_bl_backtest, right after you create Sigma
        Sigma_short = returns_lookback.cov().values
        robust_start_date = (pd.to_datetime(current_date)
                     - pd.Timedelta(weeks=ROBUST_LOOKBACK_WEEKS)).strftime("%Y-%m-%d")
        Sigma_long, robust_pi = robust_covariance_estimation(
                                    asset_list,
                                    robust_start_date,
                                    current_date.strftime("%Y-%m-%d"))
        lambda_ = 0.7
        Sigma = lambda_*Sigma_short + (1-lambda_)*Sigma_long         # or λ-blend of your choice

        # >>> add safety lines here <<<
        Sigma = np.nan_to_num(Sigma)
        Sigma = 0.5*(Sigma + Sigma.T)      # force symmetry
        base   = np.trace(Sigma)/Sigma.shape[0]
        Sigma += (1e-6*base if base>1e-12 else 1e-6) * np.eye(Sigma.shape[0])

        pi = robust_pi
        # pi = returns_lookback.mean().values.reshape(-1, 1)
        # Sigma, pi = robust_covariance_estimation(asset_list, '2015-01-01', '2023-01-01')
        min_eig = eigvalsh(Sigma).min()
        if min_eig < 0:
            Sigma -= 10 * min_eig * np.eye(*Sigma.shape)
        
        # Check predictions: asset_list must be a subset of pred_pivot columns
        if not set(asset_list).issubset(set(pred_pivot.columns)):
            print("Missing predictions for some assets.")
            continue
        
        q = pred_pivot.loc[current_date, asset_list].values.reshape(-1, 1)
        confidences = conf_pivot.loc[current_date, asset_list].values
        
        # Update expected returns via BL
        updated_returns = black_litterman_update(pi, Sigma, q, confidences)
        # bl_weights = max_sharpe_portfolio(Sigma, updated_returns, risk_free_rate=RISK_FREE_RATE)
        bl_weights = mean_variance_portfolio(Sigma, updated_returns, risk_aversion=RISK_AVERSION)
        weights_history.append(bl_weights.flatten())
        
        n = len(asset_list)
        eq_weights = np.ones((n, 1)) / n
        
        # Find next week's date
        future_dates = actual_data[actual_data['date'] > current_date]['date']
        if future_dates.empty:
            continue
        next_date = future_dates.iloc[0]
        
        current_row = actual_data[actual_data['date'] == current_date]
        future_row = actual_data[actual_data['date'] == next_date]
        if current_row.empty or future_row.empty:
            continue
        
        if asset_class == "fi":
            # For fixed income, assume the actual data already provides the weekly return
            next_returns = future_row[asset_list].iloc[0].values
            next_returns = np.array(next_returns)
        else:
            # For FX, equity, commodity: Compute returns from prices: (price_next / price_current - 1)
            next_returns = []
            for asset in asset_list:
                try:
                    price_next = future_row[asset].values[0]
                    price_current = current_row[asset].values[0]
                    ret = price_next / price_current - 1
                except Exception:
                    ret = 0
                next_returns.append(ret)
            next_returns = np.array(next_returns)
            
        next_returns = np.clip(next_returns, -0.5, 0.5)
        
        # Fix: Extract scalar values from matrix multiplication
        bl_port_return = float((bl_weights.T @ next_returns.reshape(-1, 1))[0, 0])
        eq_port_return = float((eq_weights.T @ next_returns.reshape(-1, 1))[0, 0])
        
        results.append({
            'date': current_date,
            'bl_portfolio_return': bl_port_return,
            'equal_weighted_return': eq_port_return
        })
    
    if not results:
        return pd.DataFrame(columns=['date', 'bl_portfolio_return', 'equal_weighted_return']), []
    
    results_df = pd.DataFrame(results)
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df = results_df.sort_values('date').reset_index(drop=True)
    return results_df, weights_history

# -----------------------------------------------
# Performance Metrics Function
def calculate_performance_metrics(returns_series, weights_history=None, risk_free_rate=0):
    """
    Calculate performance metrics from a weekly return series.
    
    Metrics include:
      - Annualized Return (compounded weekly)
      - Annualized Standard Deviation
      - Sharpe Ratio (annualized, assuming a specified risk_free_rate)
      - Maximum Drawdown
      - Average Turnover (if weights_history is provided)
    """
    total_return = (1 + returns_series).prod() - 1
    n_weeks = len(returns_series)
    annual_return = (1 + total_return) ** (52 / n_weeks) - 1
    annual_std = returns_series.std() * np.sqrt(52)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_std if annual_std > 0 else np.nan
    
    cum_returns = (1 + returns_series).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    metrics = {
        'Annualized Return': annual_return,
        'Annualized Std Dev': annual_std,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }
    
    if weights_history is not None and len(weights_history) > 1:
        turnovers = []
        for i in range(1, len(weights_history)):
            prev = weights_history[i-1]
            current = weights_history[i]
            turnover = np.sum(np.abs(current - prev))
            turnovers.append(turnover)
        avg_turnover = np.mean(turnovers)
        metrics['Average Turnover'] = avg_turnover
    else:
        metrics['Average Turnover'] = np.nan
    
    return metrics

# -----------------------------------------------
# Plot cumulative returns
def plot_cumulative_returns(results_df, asset_class):

    results_df['bl_cum_return'] = (1 + results_df['bl_portfolio_return']).cumprod() - 1
    results_df['eq_cum_return'] = (1 + results_df['equal_weighted_return']).cumprod() - 1

    plt.figure(figsize=(12, 6))
    plt.plot(results_df['date'], results_df['bl_cum_return'], label='BL Portfolio', marker='o')
    plt.plot(results_df['date'], results_df['eq_cum_return'], label='Equal-Weighted Portfolio', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title(f'{asset_class} Cumulative Returns: BL Portfolio vs Equal-Weighted Portfolio')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('data/evaluation', exist_ok=True)
    plt.savefig(f'data/evaluation/{asset_class}_cumulative_returns.png')
    plt.show()

# -----------------------------------------------
# Main function
def main():
    # Change asset_class to "fx", "fi", "equity", or "commodity" as needed
    asset_class = "fx"  # or "fi", "equity", "commodity" for other asset classes
    predictions, actual_data = load_data(asset_class=asset_class)
    
    if asset_class == "fx":
        assets = fx_tickers
    elif asset_class == "fi":
        assets = fi_tickers
    elif asset_class == "equity":
        assets = equity_tickers
    elif asset_class == "commodity":
        assets = commodity_tickers
    else:
        raise ValueError("Unsupported asset class")
    
    results_df, weights_history = rolling_bl_backtest(predictions, actual_data, assets, asset_class=asset_class, lookback_weeks=LOOKBACK_WEEKS)
    if results_df.empty:
        print("No backtest results. Check data and lookback period.")
        return
    
    os.makedirs('data/evaluation', exist_ok=True)
    results_df.to_csv(f'data/evaluation/{asset_class}_portfolio_returns_backtest.csv', index=False)
    plot_cumulative_returns(results_df, asset_class)
    
    # Calculate performance metrics for the BL portfolio returns
    bl_returns_series = results_df['bl_portfolio_return']
    metrics = calculate_performance_metrics(bl_returns_series, weights_history, risk_free_rate=0)
    
    print("Performance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("Backtest complete. Portfolio returns and metrics saved, and cumulative return plot generated.")

if __name__ == "__main__":
    main()
