import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----- Global Parameters -----
LOOKBACK_WEEKS = 10        # number of weeks to use for computing the prior and covariance
UNCERTAINTY_SCALE = 0.001     # scaling factor for view uncertainty
EPSILON = 1e-4             # small constant to avoid division by zero

# Mapping from instrument name to Treasury ETF ticker
instrument_to_etf = {
    "Short-Term Treasury": "SHV",
    "1-3 Year Treasury": "SHY",
    "3-7 Year Treasury": "IEI",
    "7-10 Year Treasury": "IEF",
    "10-20 Year Treasury": "TLH",
    "20+ Year Treasury": "TLT"
}

def load_data():
    """
    Load prediction and actual data files.
    Predictions come from 'data/fi_weekly_predictions.csv' and actual Treasury ETF data from
    'data/fi_combined_features_weekly.csv'.
    """
    print("Loading prediction data...")
    predictions = pd.read_csv('data/fi_weekly_predictions.csv')
    predictions['date'] = pd.to_datetime(predictions['date'])
    
    print("Loading actual Treasury ETF data...")
    actual_data = pd.read_csv('data/fi_combined_features_weekly.csv')
    # Ensure the date column is set correctly
    if 'Unnamed: 0' in actual_data.columns:
        actual_data['date'] = pd.to_datetime(actual_data['Unnamed: 0'])
        actual_data = actual_data.drop('Unnamed: 0', axis=1)
    else:
        first_col = actual_data.columns[0]
        actual_data['date'] = pd.to_datetime(actual_data[first_col])
        actual_data = actual_data.drop(first_col, axis=1)
    
    # Convert data to ensure it's numeric
    etf_tickers = list(instrument_to_etf.values())
    for etf in etf_tickers:
        if etf in actual_data.columns:
            actual_data[etf] = pd.to_numeric(actual_data[etf], errors='coerce')
    
    return predictions, actual_data

def prepare_evaluation_data(predictions, actual_data):
    """
    Prepare evaluation data by merging predictions with next week's actual ETF returns.
    We create a "date_only" column (without time) for consistent merging.
    Only the ETF price columns for IEF, IEI, SHV, SHY, TLH, and TLT are used.
    
    Note: actual_data contains daily returns, not prices. We need to compute weekly returns
    by compounding the daily returns.
    """
    print(f"Actual data columns: {actual_data.columns.tolist()}")
    
    treasury_to_etf = {
        "Short-Term Treasury": "SHV",
        "1-3 Year Treasury": "SHY",
        "3-7 Year Treasury": "IEI",
        "7-10 Year Treasury": "IEF",
        "10-20 Year Treasury": "TLH",
        "20+ Year Treasury": "TLT"
    }
    etf_tickers = list(treasury_to_etf.values())
    
    # For each ETF, get the next week's return directly since we are already working with return data
    actual_returns = pd.DataFrame({'date': actual_data['date']})
    for etf in etf_tickers:
        if etf in actual_data.columns:
            # The data already contains returns, so no need to calculate pct_change
            # Get next week's return for each current date
            actual_returns[f'{etf}_return'] = actual_data[etf].shift(-1)
        else:
            print(f"Warning: {etf} column not found in actual data.")
    
    actual_returns = actual_returns.dropna(subset=[f'{etf}_return' for etf in etf_tickers])
    
    # Create date-only columns for merging.
    predictions['date_only'] = predictions['date'].dt.date
    actual_returns['date_only'] = pd.to_datetime(actual_returns['date']).dt.date
    
    merged = pd.merge(predictions, actual_returns[['date_only'] + [f'{etf}_return' for etf in etf_tickers]],
                      on='date_only', how='inner')
    
    def get_actual_return(row):
        etf = treasury_to_etf.get(row['instrument'], None)
        if etf:
            return row[f'{etf}_return']
        else:
            return np.nan
    merged['actual_return'] = merged.apply(get_actual_return, axis=1)
    merged = merged.dropna(subset=['actual_return'])
    return merged

def clean_data_for_metrics(data, columns):
    """Helper function to clean data by removing infinities and NaNs before calculating metrics"""
    clean_data = data.copy()
    for col in columns:
        # Replace inf with NaN first
        clean_data[col] = clean_data[col].replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN values in any of the specified columns
    clean_data = clean_data.dropna(subset=columns)
    return clean_data

def calculate_metrics(eval_df):
    """
    Calculate performance metrics for the predictions.
    """
    required_cols = ['predicted_return', 'actual_return']
    missing_cols = [col for col in required_cols if col not in eval_df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in evaluation data: {missing_cols}")
    
    # Clean data before metrics calculation
    eval_clean = clean_data_for_metrics(eval_df, required_cols)
    
    if len(eval_clean) == 0:
        print("Warning: No valid data after cleaning. Cannot calculate metrics.")
        return {}, {}
    
    metrics = {
        'Mean Squared Error': mean_squared_error(eval_clean['actual_return'], eval_clean['predicted_return']),
        'Root Mean Squared Error': np.sqrt(mean_squared_error(eval_clean['actual_return'], eval_clean['predicted_return'])),
        'Mean Absolute Error': mean_absolute_error(eval_clean['actual_return'], eval_clean['predicted_return']),
        'R-squared': r2_score(eval_clean['actual_return'], eval_clean['predicted_return'])
    }
    
    correlation = eval_clean[['actual_return', 'predicted_return']].corr().iloc[0, 1]
    metrics['Correlation'] = correlation
    
    correct_direction = (np.sign(eval_clean['predicted_return']) == np.sign(eval_clean['actual_return'])).mean()
    metrics['Directional Accuracy'] = correct_direction
    
    eval_clean['weighted_prediction'] = eval_clean['predicted_return'] * eval_clean['confidence']
    weighted_correlation = eval_clean[['actual_return', 'weighted_prediction']].corr().iloc[0, 1]
    metrics['Weighted Correlation'] = weighted_correlation
    
    instrument_metrics = {}
    for instrument in eval_clean['instrument'].unique():
        inst_data = eval_clean[eval_clean['instrument'] == instrument]
        if len(inst_data) > 0:
            instrument_metrics[instrument] = {
                'MSE': mean_squared_error(inst_data['actual_return'], inst_data['predicted_return']),
                'Directional Accuracy': (np.sign(inst_data['predicted_return']) == np.sign(inst_data['actual_return'])).mean(),
                'Correlation': inst_data[['actual_return', 'predicted_return']].corr().iloc[0, 1],
                'Count': len(inst_data)
            }
    
    return metrics, instrument_metrics

def calculate_trading_performance(eval_df):
    """
    Calculate trading strategy performance metrics based on predictions.
    """
    # Clean data for performance calculations
    strategy_df = clean_data_for_metrics(eval_df, ['predicted_return', 'actual_return', 'confidence'])
    
    if len(strategy_df) == 0:
        print("Warning: No valid data after cleaning. Cannot calculate performance.")
        return {}, {}, pd.DataFrame()
    
    strategy_df['dir_strategy_return'] = np.sign(strategy_df['predicted_return']) * strategy_df['actual_return']
    strategy_df['conf_strategy_return'] = np.sign(strategy_df['predicted_return']) * strategy_df['confidence'] * strategy_df['actual_return']
    
    # Avoid division by zero for norm_factor
    pred_abs_mean = strategy_df['predicted_return'].abs().mean()
    norm_factor = pred_abs_mean if pred_abs_mean > 0 else 1.0
    strategy_df['size_strategy_return'] = (strategy_df['predicted_return'] / norm_factor) * strategy_df['actual_return']
    
    strategies = ['dir_strategy_return', 'conf_strategy_return', 'size_strategy_return']
    strategy_performance = {}
    baseline_return = strategy_df['actual_return'].mean()
    
    for strat in strategies:
        strat_returns = strategy_df[strat]
        strategy_performance[strat] = {
            'Mean Return': strat_returns.mean(),
            'Cumulative Return': (1 + strat_returns).prod() - 1,
            'Sharpe Ratio': strat_returns.mean() / strat_returns.std() if strat_returns.std() > 0 else 0,
            'Win Rate': (strat_returns > 0).mean(),
            'vs Baseline': strat_returns.mean() - baseline_return
        }
    
    instrument_strategy_perf = {}
    for instrument in strategy_df['instrument'].unique():
        inst_data = strategy_df[strategy_df['instrument'] == instrument]
        if len(inst_data) > 0:
            instrument_strategy_perf[instrument] = {}
            inst_baseline = inst_data['actual_return'].mean()
            for strat in strategies:
                strat_returns = inst_data[strat]
                instrument_strategy_perf[instrument][strat] = {
                    'Mean Return': strat_returns.mean(),
                    'Cumulative Return': (1 + strat_returns).prod() - 1,
                    'vs Baseline': strat_returns.mean() - inst_baseline
                }
    
    return strategy_performance, instrument_strategy_perf, strategy_df

def create_visualizations(eval_df, strategy_df):
    """
    Create visualizations to analyze prediction performance.
    """
    os.makedirs('data/evaluation', exist_ok=True)
    
    # Scatter plot of predicted vs actual returns by instrument.
    plt.figure(figsize=(10, 8))
    clean_eval_df = clean_data_for_metrics(eval_df, ['predicted_return', 'actual_return'])
    sns.scatterplot(x='predicted_return', y='actual_return',
                    hue='instrument', size='confidence',
                    data=clean_eval_df)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title('Predicted vs Actual Treasury ETF Returns')
    plt.xlabel('Predicted Return')
    plt.ylabel('Actual Return')
    plt.tight_layout()
    plt.savefig('data/evaluation/predicted_vs_actual.png')
    
    # Cumulative returns over time by instrument.
    # Only aggregate numeric columns: strategy returns.
    grouped = strategy_df.groupby('date')[['dir_strategy_return', 'conf_strategy_return', 'size_strategy_return']].mean().reset_index()
    grouped['dir_cum'] = (1 + grouped['dir_strategy_return']).cumprod() - 1
    grouped['conf_cum'] = (1 + grouped['conf_strategy_return']).cumprod() - 1
    grouped['size_cum'] = (1 + grouped['size_strategy_return']).cumprod() - 1
    
    plt.figure(figsize=(12, 8))
    plt.plot(grouped['date'], grouped['dir_cum'], label='Directional')
    plt.plot(grouped['date'], grouped['conf_cum'], label='Confidence-Weighted', linestyle='--')
    plt.plot(grouped['date'], grouped['size_cum'], label='Size-Proportional', linestyle='-.')
    plt.title('Cumulative Trading Strategy Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/evaluation/strategy_cumulative_returns.png')
    
def compute_prior_and_covariance_for_date(actual, etf_tickers, current_date, lookback=LOOKBACK_WEEKS):
    """
    For a given current_date, compute the prior weekly return vector (pi) and covariance matrix (Sigma)
    using the previous 'lookback' weeks of data for the specified ETF tickers.
    
    Note: The data is daily returns, not prices. We use these returns directly.
    """
    actual_sorted = actual.sort_values('date')
    actual_sorted.set_index('date', inplace=True)
    
    historical = actual_sorted[actual_sorted.index < current_date].copy()
    if len(historical) < lookback:
        print(f"Warning: Not enough historical data before {current_date}. Using available data: {len(historical)} weeks.")
        if len(historical) < 4:  # Minimum data requirement
            raise ValueError(f"Insufficient historical data before {current_date}. Need at least 4 weeks.")
    
    historical = historical.tail(min(lookback, len(historical)))
    
    # The data already contains returns, so we can use it directly
    returns = pd.DataFrame()
    for etf in etf_tickers:
        if etf in historical.columns:
            returns[etf] = historical[etf]  # These are already returns, not prices
        else:
            print(f"Warning: {etf} not found in historical data.")
    
    # Clean up the returns data
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(returns) == 0:
        raise ValueError(f"No valid return data available before {current_date}.")
    
    pi = returns.mean().values.reshape(-1, 1)
    Sigma = returns.cov().values
    
    # Ensure covariance matrix is positive definite
    min_eig = np.min(np.linalg.eigvals(Sigma))
    if min_eig < 0:
        Sigma -= 10*min_eig * np.eye(*Sigma.shape)
    
    return pi, Sigma

def apply_black_litterman(pi, Sigma, view_df_week):
    """
    Apply the Black-Litterman model using:
        r_BL = [Σ⁻¹ + Ω⁻¹]⁻¹ [Σ⁻¹*pi + Ω⁻¹*q]
    For the given week, the view vector q and confidence are extracted from view_df_week.
    Omega is built as a diagonal matrix with entries: UNCERTAINTY_SCALE*(1 - confidence) + EPSILON.
    """
    etf_list = list(instrument_to_etf.values())
    q = []
    omega_diags = []
    for etf in etf_list:
        instrument = [inst for inst, ticker in instrument_to_etf.items() if ticker == etf]
        if instrument and (view_df_week['instrument'] == instrument[0]).any():
            row = view_df_week[view_df_week['instrument'] == instrument[0]].iloc[0]
            # Cap extremely large values to prevent numerical issues
            pred_return = np.clip(row['predicted_return'], -0.2, 0.2)
            q.append(pred_return)
            conf = np.clip(row['confidence'], 0.01, 0.99)  # Ensure confidence is in reasonable range
            omega_value = UNCERTAINTY_SCALE * (1 - conf) + EPSILON
            omega_diags.append(omega_value)
        else:
            q.append(0.0)
            omega_diags.append(UNCERTAINTY_SCALE + EPSILON)
    
    q = np.array(q).reshape(-1, 1)
    Omega = np.diag(omega_diags)
    
    try:
        inv_Sigma = np.linalg.inv(Sigma)
        inv_Omega = np.linalg.inv(Omega)
        
        r_bl = np.linalg.inv(inv_Sigma + inv_Omega) @ (inv_Sigma @ pi + inv_Omega @ q)
        return r_bl
    except np.linalg.LinAlgError as e:
        print(f"Linear algebra error in BL calculation: {e}")
        # Return the prior as fallback
        return pi

def simulate_bl_portfolio_returns(actual, predictions):
    """
    Simulate a BL portfolio on a rolling, weekly basis.
    For each unique prediction date, compute the prior and covariance (using the past LOOKBACK_WEEKS),
    then use that week's LLM view (predicted_return and confidence) to compute the BL posterior returns.
    The weekly BL portfolio return is taken as the weighted sum of ETF returns using BL weights.
    Returns a DataFrame with columns: date and bl_portfolio_return.
    
    Note: The ETF data contains daily returns, not prices. We use these returns directly.
    """
    unique_dates = sorted(list(set(predictions['date'].unique())))
    portfolio_returns = []
    dates_list = []
    etf_tickers = list(instrument_to_etf.values())
    
    # Also calculate equal-weighted portfolio for comparison
    equal_weights_returns = []
    
    for current_date in unique_dates:
        try:
            view_df_week = predictions[predictions['date'] == current_date]
            pi, Sigma = compute_prior_and_covariance_for_date(actual, etf_tickers, current_date, lookback=LOOKBACK_WEEKS)
            r_bl = apply_black_litterman(pi, Sigma, view_df_week)
            
            # Weight optimization (equal risk contribution)
            weights = calculate_bl_weights(r_bl, Sigma)
            
            # Get next week's actual returns for the ETFs
            next_date = actual[actual['date'] > current_date]['date'].min()
            if pd.isna(next_date):
                continue
                
            next_week_returns = {}
            for etf in etf_tickers:
                if etf in actual.columns:
                    # For ETFs, use the return value directly since our data is returns, not prices
                    next_week_return = actual[actual['date'] == next_date][etf].values[0]
                    # Handle missing or invalid values
                    if pd.isna(next_week_return) or np.isinf(next_week_return):
                        next_week_return = 0
                    next_week_returns[etf] = next_week_return
            
            # Calculate weighted portfolio return
            if len(next_week_returns) == len(etf_tickers):
                next_returns_array = np.array([next_week_returns[etf] for etf in etf_tickers])
                
                # Cap extreme returns to prevent inf/nan issues
                next_returns_array = np.clip(next_returns_array, -0.5, 0.5)
                
                # Black-Litterman portfolio return
                bl_port_return = np.sum(weights * next_returns_array)
                portfolio_returns.append(bl_port_return)
                
                # Equal-weighted portfolio return
                equal_weights = np.ones(len(etf_tickers)) / len(etf_tickers)
                eq_port_return = np.sum(equal_weights * next_returns_array)
                equal_weights_returns.append(eq_port_return)
                
                dates_list.append(current_date)
        except Exception as e:
            print(f"Skipping date {current_date} due to error: {e}")
            continue
    
    port_df = pd.DataFrame({
        'date': dates_list, 
        'bl_portfolio_return': portfolio_returns,
        'equal_weighted_return': equal_weights_returns
    })
    port_df.sort_values('date', inplace=True)
    port_df.reset_index(drop=True, inplace=True)
    return port_df

def calculate_bl_weights(r_bl, Sigma):
    """
    Calculate portfolio weights based on Black-Litterman expected returns.
    This implements a simple approach where weights are proportional to return/risk ratio.
    """
    n_assets = len(r_bl)
    
    # Calculate risk (standard deviation) for each asset
    risk = np.sqrt(np.diag(Sigma))
    
    # Avoid division by zero
    risk = np.where(risk < EPSILON, EPSILON, risk)
    
    # Calculate return-to-risk ratio
    return_to_risk = r_bl.flatten() / risk
    
    # Handle possible NaNs or infinities
    return_to_risk = np.nan_to_num(return_to_risk, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Normalize to get weights (sum to 1)
    weights_sum = np.sum(np.abs(return_to_risk))
    if weights_sum > EPSILON:
        weights = return_to_risk / weights_sum
    else:
        # Equal weights if all return/risk ratios are near zero
        weights = np.ones(n_assets) / n_assets
    
    return weights

def calculate_portfolio_metrics(returns_df):
    """
    Calculate portfolio performance metrics:
    - Annual Return
    - Annual Standard Deviation
    - Sharpe Ratio
    - Maximum Drawdown
    """
    metrics = {}
    
    for col in ['bl_portfolio_return', 'equal_weighted_return']:
        if col not in returns_df.columns:
            continue
            
        returns = returns_df[col]
        
        # Clean returns data - replace inf/extreme values
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.dropna()
        
        if len(returns) == 0:
            print(f"Warning: No valid returns for {col} after cleaning.")
            metrics[col] = {
                'Annual Return': np.nan,
                'Annual Std Dev': np.nan,
                'Sharpe Ratio': np.nan,
                'Max Drawdown': np.nan,
                'Total Return': np.nan
            }
            continue
        
        # Weekly to Annual conversion (52 weeks in a year)
        # Use compounding for annual return
        try:
            annual_return = ((1 + returns).prod()) ** (52/len(returns)) - 1
        except (OverflowError, RuntimeWarning):
            print(f"Warning: Could not compute annual return for {col} due to extreme values.")
            annual_return = np.nan
            
        annual_std = returns.std() * np.sqrt(52)
        
        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        sharpe_ratio = annual_return / annual_std if annual_std > 0 and not np.isnan(annual_return) else 0
        
        # Maximum drawdown calculation
        try:
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max) - 1
            max_drawdown = drawdown.min()
        except Exception:
            print(f"Warning: Could not compute drawdown for {col}.")
            max_drawdown = np.nan
            
        # Total return calculation
        try:
            total_return = (1 + returns).prod() - 1
            if np.isinf(total_return) or np.isnan(total_return):
                total_return = np.nan
        except Exception:
            print(f"Warning: Could not compute total return for {col}.")
            total_return = np.nan
        
        metrics[col] = {
            'Annual Return': annual_return,
            'Annual Std Dev': annual_std,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Total Return': total_return
        }
    
    return metrics

def plot_bl_vs_equal_weighted(bl_df):
    """
    Plot the cumulative return of the BL portfolio against the equal-weighted portfolio.
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate cumulative returns
    bl_df['bl_cum_return'] = (1 + bl_df['bl_portfolio_return']).cumprod() - 1
    bl_df['eq_cum_return'] = (1 + bl_df['equal_weighted_return']).cumprod() - 1
    
    plt.plot(bl_df['date'], bl_df['bl_cum_return'], label='Black-Litterman Portfolio', marker='o')
    plt.plot(bl_df['date'], bl_df['eq_cum_return'], label='Equal-Weighted Portfolio', linestyle='--')
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Return: Black-Litterman vs. Equal-Weighted Portfolio')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('data/evaluation', exist_ok=True)
    plt.savefig('data/evaluation/BL_vs_Equal_Weighted_cumulative_return.png')
    plt.show()

def get_strategy_cumulative_returns(strategy_df):
    """
    From strategy_df, compute the mean trading strategy return for each date and compound them
    to obtain cumulative returns for each strategy.
    """
    # Only use numeric strategy return columns
    grouped = strategy_df.groupby('date')[['dir_strategy_return', 'conf_strategy_return', 'size_strategy_return']].mean().reset_index()
    grouped['dir_cum'] = (1 + grouped['dir_strategy_return']).cumprod() - 1
    grouped['conf_cum'] = (1 + grouped['conf_strategy_return']).cumprod() - 1
    grouped['size_cum'] = (1 + grouped['size_strategy_return']).cumprod() - 1
    return grouped[['date', 'dir_cum', 'conf_cum', 'size_cum']]

def plot_bl_vs_strategies(bl_df, strategy_cum_df):
    """
    Plot the cumulative return of the BL portfolio against the trading strategies and equal-weighted portfolio.
    """
    bl_df = bl_df.sort_values('date').reset_index(drop=True)
    bl_df['bl_cum_return'] = (1 + bl_df['bl_portfolio_return']).cumprod() - 1
    bl_df['eq_cum_return'] = (1 + bl_df['equal_weighted_return']).cumprod() - 1
    
    # Merge dataframes on date for plotting
    plot_df = pd.merge(bl_df[['date', 'bl_cum_return', 'eq_cum_return']], 
                       strategy_cum_df, on='date', how='inner')
    
    plt.figure(figsize=(12, 6))
    plt.plot(plot_df['date'], plot_df['bl_cum_return'], label='BL Portfolio', marker='o')
    plt.plot(plot_df['date'], plot_df['eq_cum_return'], label='Equal-Weighted Portfolio', marker='x')
    plt.plot(plot_df['date'], plot_df['dir_cum'], label='Directional Strategy', linestyle='--')
    plt.plot(plot_df['date'], plot_df['conf_cum'], label='Confidence-Weighted Strategy', linestyle='-.')
    plt.plot(plot_df['date'], plot_df['size_cum'], label='Size-Proportional Strategy', linestyle=':')
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Return: All Strategies Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('data/evaluation', exist_ok=True)
    plt.savefig('data/evaluation/All_Strategies_comparison.png')
    plt.show()

def main():
    predictions, actual_data = load_data()
    print(f"Predictions columns: {predictions.columns.tolist()}")
    print(f"Actual data columns: {actual_data.columns.tolist()}")
    print(f"Predictions date range: {predictions['date'].min()} to {predictions['date'].max()}")
    print(f"Actual data date range: {actual_data['date'].min()} to {actual_data['date'].max()}")
    
    print("Preparing evaluation data...")
    eval_df = prepare_evaluation_data(predictions, actual_data)
    if eval_df.empty:
        print("No matching evaluation data found. Exiting.")
        return
    
    print("Calculating performance metrics...")
    overall_metrics, instrument_metrics = calculate_metrics(eval_df)
    
    print("Calculating trading strategy performance...")
    strategy_perf, instrument_strategy_perf, strategy_df = calculate_trading_performance(eval_df)
    
    print("Creating visualizations for trading strategies...")
    create_visualizations(eval_df, strategy_df)
    
    print("Simulating BL Portfolio returns on a rolling basis...")
    bl_port_df = simulate_bl_portfolio_returns(actual_data, predictions)
    
    # Save portfolio returns before any further processing
    os.makedirs('data/evaluation', exist_ok=True)
    bl_port_df.to_csv('data/evaluation/portfolio_returns.csv', index=False)
    
    # Handle NaN/Inf values in portfolio returns
    bl_port_df = bl_port_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(bl_port_df) == 0:
        print("Warning: No valid portfolio returns after cleaning.")
        return
    
    print("Computing cumulative returns for trading strategies...")
    strategy_cum_df = get_strategy_cumulative_returns(strategy_df)
    
    print("Calculating portfolio performance metrics...")
    portfolio_metrics = calculate_portfolio_metrics(bl_port_df)
    
    print("Plotting BL vs Equal-Weighted Portfolio comparison...")
    plot_bl_vs_equal_weighted(bl_port_df)
    
    print("Plotting cumulative returns for all strategies...")
    plot_bl_vs_strategies(bl_port_df, strategy_cum_df)
    
    print("\n==== OVERALL PREDICTION METRICS ====")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n==== METRICS BY INSTRUMENT ====")
    for instrument, metrics in instrument_metrics.items():
        print(f"\n{instrument}:")
        for metric, value in metrics.items():
            if metric != 'Count':
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    print("\n==== TRADING STRATEGY PERFORMANCE ====")
    for strategy, perf in strategy_perf.items():
        strat_name = strategy.replace('_strategy_return', '')
        print(f"\n{strat_name.capitalize()} Strategy:")
        for metric, value in perf.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\n==== PORTFOLIO PERFORMANCE METRICS ====")
    for portfolio, metrics in portfolio_metrics.items():
        port_name = "Black-Litterman Portfolio" if portfolio == "bl_portfolio_return" else "Equal-Weighted Portfolio"
        print(f"\n{port_name}:")
        for metric, value in metrics.items():
            if not np.isnan(value) and not np.isinf(value):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: N/A (could not compute)")
    
    # Save results
    eval_df.to_csv('data/evaluation/prediction_evaluation.csv', index=False)
    
    print("\nEvaluation data saved to 'data/evaluation/prediction_evaluation.csv'")
    print("Portfolio returns saved to 'data/evaluation/portfolio_returns.csv'")
    print("Visualizations saved to 'data/evaluation/' directory")

if __name__ == "__main__":
    main()
