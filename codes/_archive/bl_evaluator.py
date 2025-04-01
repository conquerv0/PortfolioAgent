import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----- Global Parameters -----
LOOKBACK_WEEKS = 52        # number of weeks to use for computing the prior and covariance
UNCERTAINTY_SCALE = 0.01     # scaling factor for view uncertainty
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
    Predictions come from 'data/fi_weekly_predictions.csv' and actual Treasury ETF price data from
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
    
    return predictions, actual_data

def prepare_evaluation_data(predictions, actual_data):
    """
    Prepare evaluation data by merging predictions with next week's actual ETF returns.
    We create a "date_only" column (without time) for consistent merging.
    Only the ETF price columns for IEF, IEI, SHV, SHY, TLH, and TLT are used.
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
    
    actual_returns = pd.DataFrame({'date': actual_data['date']})
    for etf in etf_tickers:
        if etf in actual_data.columns:
            actual_data[f'{etf}_return'] = actual_data[etf].pct_change()
            actual_returns[f'{etf}_return'] = actual_data[f'{etf}_return'].shift(-1)
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

def calculate_metrics(eval_df):
    """
    Calculate performance metrics for the predictions.
    """
    required_cols = ['predicted_return', 'actual_return']
    missing_cols = [col for col in required_cols if col not in eval_df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in evaluation data: {missing_cols}")
    
    eval_clean = eval_df.dropna(subset=required_cols)
    
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
    strategy_df = eval_df.copy().dropna(subset=['predicted_return', 'actual_return'])
    
    strategy_df['dir_strategy_return'] = np.sign(strategy_df['predicted_return']) * strategy_df['actual_return']
    strategy_df['conf_strategy_return'] = np.sign(strategy_df['predicted_return']) * strategy_df['confidence'] * strategy_df['actual_return']
    norm_factor = strategy_df['predicted_return'].abs().mean()
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
    sns.scatterplot(x='predicted_return', y='actual_return',
                    hue='instrument', size='confidence',
                    data=eval_df.dropna(subset=['predicted_return', 'actual_return']))
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
    plt.tight_layout()
    plt.savefig('data/evaluation/strategy_cumulative_returns.png')
    
def compute_prior_and_covariance_for_date(actual, etf_tickers, current_date, lookback=LOOKBACK_WEEKS):
    """
    For a given current_date, compute the prior weekly return vector (pi) and covariance matrix (Sigma)
    using the previous 'lookback' weeks of data for the specified ETF tickers.
    """
    historical = actual[actual.index < current_date].copy()
    if len(historical) < lookback:
        raise ValueError(f"Not enough historical data before {current_date}. Needed {lookback} weeks.")
    historical = historical.tail(lookback)
    
    returns = pd.DataFrame()
    for etf in etf_tickers:
        if etf in historical.columns:
            returns[etf] = historical[etf].pct_change()
        else:
            print(f"Warning: {etf} not found in historical data.")
    returns = returns.dropna()
    pi = returns.mean().values.reshape(-1, 1)
    Sigma = returns.cov().values
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
            q.append(row['predicted_return'])
            conf = row['confidence']
            omega_value = UNCERTAINTY_SCALE * (1 - conf) + EPSILON
            omega_diags.append(omega_value)
        else:
            q.append(0.0)
            omega_diags.append(UNCERTAINTY_SCALE + EPSILON)
    q = np.array(q).reshape(-1, 1)
    Omega = np.diag(omega_diags)
    
    inv_Sigma = np.linalg.inv(Sigma)
    inv_Omega = np.linalg.inv(Omega)
    
    r_bl = np.linalg.inv(inv_Sigma + inv_Omega) @ (inv_Sigma @ pi + inv_Omega @ q)
    return r_bl

def simulate_bl_portfolio_returns(actual, predictions):
    """
    Simulate a BL portfolio on a rolling, weekly basis.
    For each unique prediction date, compute the prior and covariance (using the past LOOKBACK_WEEKS),
    then use that week's LLM view (predicted_return and confidence) to compute the BL posterior returns.
    The weekly BL portfolio return is taken as the average of the BL posterior return vector.
    Returns a DataFrame with columns: date and bl_portfolio_return.
    """
    unique_dates = sorted(list(set(predictions['date'].unique())))
    portfolio_returns = []
    dates_list = []
    etf_tickers = list(instrument_to_etf.values())
    
    for current_date in unique_dates:
        try:
            view_df_week = predictions[predictions['date'] == current_date]
            pi, Sigma = compute_prior_and_covariance_for_date(actual, etf_tickers, current_date, lookback=LOOKBACK_WEEKS)
            r_bl = apply_black_litterman(pi, Sigma, view_df_week)
            weekly_port_return = np.mean(r_bl)
            portfolio_returns.append(weekly_port_return)
            dates_list.append(current_date)
        except Exception as e:
            print(f"Skipping date {current_date} due to error: {e}")
            continue
    
    port_df = pd.DataFrame({'date': dates_list, 'bl_portfolio_return': portfolio_returns})
    port_df.sort_values('date', inplace=True)
    port_df.reset_index(drop=True, inplace=True)
    return port_df

def get_strategy_cumulative_returns(strategy_df):
    """
    From strategy_df, compute the mean trading strategy return for each date and compound them
    to obtain cumulative returns for each strategy.
    """
    # Only use numeric strategy return columns.
    grouped = strategy_df.groupby('date')[['dir_strategy_return', 'conf_strategy_return', 'size_strategy_return']].mean().reset_index()
    grouped['dir_cum'] = (1 + grouped['dir_strategy_return']).cumprod() - 1
    grouped['conf_cum'] = (1 + grouped['conf_strategy_return']).cumprod() - 1
    grouped['size_cum'] = (1 + grouped['size_strategy_return']).cumprod() - 1
    return grouped[['date', 'dir_cum', 'conf_cum', 'size_cum']]

def plot_bl_vs_strategies(bl_df, strategy_cum_df):
    """
    Plot the cumulative return of the BL portfolio against the three trading strategies.
    """
    bl_df = bl_df.sort_values('date').reset_index(drop=True)
    bl_df['bl_cum_return'] = (1 + bl_df['bl_portfolio_return']).cumprod() - 1
    
    plt.figure(figsize=(12, 6))
    plt.plot(bl_df['date'], bl_df['bl_cum_return'], label='BL Portfolio', marker='o')
    plt.plot(strategy_cum_df['date'], strategy_cum_df['dir_cum'], label='Directional Strategy', linestyle='--')
    plt.plot(strategy_cum_df['date'], strategy_cum_df['conf_cum'], label='Confidence-Weighted Strategy', linestyle='-.')
    plt.plot(strategy_cum_df['date'], strategy_cum_df['size_cum'], label='Size-Proportional Strategy', linestyle=':')
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Return: BL Portfolio vs. Trading Strategies')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('data/evaluation', exist_ok=True)
    plt.savefig('data/evaluation/BL_vs_Strategies_cumulative_return.png')
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
    
    print("Computing cumulative returns for trading strategies...")
    strategy_cum_df = get_strategy_cumulative_returns(strategy_df)
    
    print("Plotting cumulative returns: BL Portfolio vs. Trading Strategies...")
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
    
    eval_df.to_csv('data/evaluation/prediction_evaluation.csv', index=False)
    print("\nEvaluation data saved to 'data/evaluation/prediction_evaluation.csv'")
    print("Visualizations saved to 'data/evaluation/' directory")

if __name__ == "__main__":
    main()
