#!/usr/bin/env python
import pandas as pd
import numpy as np
from numpy.linalg import inv, eigvalsh   # h = Hermitian
import matplotlib.pyplot as plt
import os
from src.agent.DataCollector import *
from dateutil.relativedelta import relativedelta

# ----- Global Parameters -----
LOOKBACK_WEEKS = 12
ROBUST_LOOKBACK_WEEKS = 260    # ≈ 5 years  (adjust as needed
UNCERTAINTY_SCALE = 0.8  # Scaling factor for view uncertainty
EPSILON = 1e-6           # Small constant to avoid division by zero
TAU = 0.5            # Scaling parameter for the prior covariance in BL
RISK_AVERSION = 2       # Risk aversion parameter for mean-variance optimization
RISK_FREE_RATE = 0.03
LAMBDA_ = 0.1
MAX_TURNOVER = 0.2
# Mapping for FX instruments
fx_instrument_to_etf = {
    "EUR/USD": "FXE",
    "GBP/USD": "FXB",
    "USD/JPY": "FXY",
    "USD/CHF": "FXF",
    "USD/CAD": "FXC"
}
from src.config.settings import PORTFOLIOS  

# Define asset lists from settings
fx_tickers = [entry["etf"] for entry in PORTFOLIOS['fx'].get("currencies", [])]
fi_tickers = [entry["etf"] for entry in PORTFOLIOS['bond'].get("treasuries", [])]
equity_tickers = [entry["etf"] for entry in PORTFOLIOS["equity"]["sectors"]]
commodity_tickers = [entry["etf"] for entry in PORTFOLIOS["commodity"]["sectors"]]
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
        pred_file   = f"data/predictions/{asset_class}_weekly_predictions.csv"
        actual_file = f"data/features/{asset_class}_combined_features_weekly.csv"
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

def mean_variance_portfolio_long_only(Sigma: np.ndarray, mu: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Long‐only mean‐variance portfolio.
    
    Args:
      Sigma: (N×N) covariance matrix.
      mu:    (N,) vector of expected returns.
      epsilon: small threshold to avoid divide‐by‐zero.
    
    Returns:
      w: (N,) weights ≥0 summing to 1.
    """
    # 1) Raw (unconstrained) weights ∝ Σ⁻¹ μ
    inv_S = np.linalg.inv(Sigma)
    raw   = inv_S @ mu

    # 2) Force long‐only
    clipped = np.clip(raw, 0.0, None)

    # 3) Renormalize to 100% net exposure
    total = clipped.sum()
    if total < epsilon:
        # If no positive signal, default to equal‐weight
        return np.ones_like(clipped) / len(clipped)

    return clipped / total

def mean_variance_portfolio_full_enforced(Sigma, expected_returns, risk_aversion=RISK_AVERSION):
    """
    Σ⁻¹ π  →  raw_weights
    then scale so sum(raw_weights)=1 exactly.
    """
    inv_Sigma   = inv(Sigma)
    raw_weights = inv_Sigma @ expected_returns

    total = raw_weights.sum()
    if abs(total) < EPSILON:
        # if the raw signal is basically zero, fall back to equal weight
        n = len(raw_weights)
        return np.ones((n,1)) / n

    return raw_weights / total

def apply_turnover(prev_w: np.ndarray,
                   target_w: np.ndarray,
                   max_turnover: float = MAX_TURNOVER,
                   eps: float = 1e-8) -> np.ndarray:
    """
    If sum(|Δw|) > max_turnover,
    scale Δw = target_w - prev_w by α = max_turnover / sum(|Δw|)
    to satisfy the cap, else move fully to target.
    Always returns a vector summing to 1.
    """
    delta = target_w - prev_w
    total_turn = np.abs(delta).sum()
    if total_turn <= max_turnover or total_turn < eps:
        return target_w
    alpha = max_turnover / total_turn
    w_new = prev_w + alpha * delta
    # numerical fixup: ensure sums to 1
    return w_new / w_new.sum()
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

import cvxpy as cp

import numpy as np
import cvxpy as cp

def bl_weights_with_turnover(pi, Sigma, q, confidences,
                             prev_w,
                             tau=TAU,
                             uncertainty_scale=UNCERTAINTY_SCALE,
                             risk_aversion=RISK_AVERSION,
                             turnover_penalty=MAX_TURNOVER,
                             reg_eps=1e-6):
    """
    Compute BL posterior returns (μ) via black_litterman_update (as in :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}),
    then solve
        maximize  wᵀμ – (risk_aversion/2) wᵀΣw  – turnover_penalty·‖w–prev_w‖₁
        subject to  ∑w=1,  w≥0.
    Returns new weight vector of shape (n,1).
    """
    # 1) Compute BL posterior returns μ
    mu = black_litterman_update(pi, Sigma, q, confidences,
                                tau=tau,
                                uncertainty_scale=uncertainty_scale).flatten()
    
    # 2) Regularize Σ to ensure positive‐definiteness
    n = Sigma.shape[0]
    Sigma_reg = 0.5*(Sigma + Sigma.T) + reg_eps*np.eye(n)
    
    # 3) Setup CVXPY variables
    w = cp.Variable(n)
    ret_term  = mu @ w
    risk_term = (risk_aversion/2)*cp.quad_form(w, Sigma_reg)
    turn_term = turnover_penalty * cp.norm1(w - prev_w.flatten())
    
    # 4) Define and solve the QP
    objective = cp.Maximize(ret_term - risk_term - turn_term)
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True)
    except cp.error.SolverError:
        # fallback if OSQP fails
        prob.solve(solver=cp.ECOS, verbose=False)
    
    # 5) Handle solver failure
    if w.value is None:
        # if both solvers fail, stick with previous weights
        return prev_w
    return w.value.reshape(-1,1)


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
    # 1) Compute static MV benchmark weights on the very first lookback
    first_date = pred_pivot.index.min()
    first_hist = actual_data[actual_data['date'] < first_date].tail(lookback_weeks)
    if asset_class=="fi":
        ret0 = first_hist[asset_list].dropna()
    else:
        ret0 = first_hist[asset_list].pct_change().dropna()
    pi0    = ret0.mean().values.reshape(-1,1)
    Sigma0 = ret0.cov().values
    prev_w = mean_variance_portfolio(Sigma0, pi0)        # static benchmark

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

        Sigma_short = returns_lookback.cov().values
        robust_start_date = (pd.to_datetime(current_date)
                     - pd.Timedelta(weeks=ROBUST_LOOKBACK_WEEKS)).strftime("%Y-%m-%d")
        Sigma_long, robust_pi = robust_covariance_estimation(
                                    asset_list,
                                    robust_start_date,
                                    current_date.strftime("%Y-%m-%d"))
        lambda_ = LAMBDA_
        Sigma = lambda_*Sigma_short + (1-lambda_)*Sigma_long         # or λ-blend of choice

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
        
        prev_w = mean_variance_portfolio(Sigma, pi)
        # Update expected returns via BL
        updated_returns = black_litterman_update(pi, Sigma, q, confidences)
        # bl_weights = max_sharpe_portfolio(Sigma, updated_returns, risk_free_rate=RISK_FREE_RATE)
        target_bl_weights = mean_variance_portfolio_full_enforced(Sigma, updated_returns, risk_aversion=RISK_AVERSION)
        # bl_weights = mean_variance_portfolio_long_only(Sigma, updated_returns)
        
        bl_weights = apply_turnover(prev_w, target_bl_weights)
        # bl_weights = bl_weights_with_turnover(
        #     pi, Sigma, q, confidences,
        #     prev_w=prev_w,
        #     tau=TAU,
        #     uncertainty_scale=UNCERTAINTY_SCALE,
        #     risk_aversion=RISK_AVERSION,
        #     turnover_penalty=MAX_TURNOVER
        # )

        prev_w = bl_weights
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
        mv_weights_t = mean_variance_portfolio(Sigma, pi)
        print(mv_weights_t)
        mv_port_return = (mv_weights_t.flatten() @ next_returns)
        bl_port_return = float(bl_weights.T @ next_returns.reshape(-1, 1))
        eq_port_return = float(eq_weights.T @ next_returns.reshape(-1, 1))
        # mv_port_return = float(mv_weights.T @ next_returns.reshape(-1, 1))
        
        results.append({
            'date': current_date,
            'bl_portfolio_return': bl_port_return,
            'equal_weighted_return': eq_port_return,
            'mv_portfolio_return': mv_port_return
        })
    
    if not results:
        return pd.DataFrame(columns=['date', 'bl_portfolio_return', 'equal_weighted_return', 'mv_portfolio_return']), []
    
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
    results_df['mv_cum_return'] = (1 + results_df['mv_portfolio_return']).cumprod() - 1
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['date'], results_df['bl_cum_return'], label='BL Portfolio', marker='o')
    plt.plot(results_df['date'], results_df['eq_cum_return'], label='Equal-Weighted Portfolio', linestyle='--')
    plt.plot(results_df['date'], results_df['mv_cum_return'], label='Static MV Benchmark', linestyle='-.')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title(f'{asset_class} Cumulative Returns Comparison')
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
    asset_class = "equity"  
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
    results_df.to_csv('data/evaluation/portfolio_returns_backtest.csv', index=False)
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
