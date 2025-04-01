import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    Prepare evaluation data by matching predictions to next week's actual ETF returns.
    For each treasury ETF instrument, compute the weekly return (price change) and merge with predictions.
    """
    print(f"Actual data columns: {actual_data.columns.tolist()}")
    
    # Mapping from instrument names to their ETF tickers
    treasury_to_etf = {
        "Short-Term Treasury": "SHV",
        "1-3 Year Treasury": "SHY",
        "3-7 Year Treasury": "IEI",
        "7-10 Year Treasury": "IEF",
        "10-20 Year Treasury": "TLH",
        "20+ Year Treasury": "TLT"
    }
    
    etf_tickers = list(treasury_to_etf.values())
    # Create a new DataFrame to hold next week's returns for each ETF.
    actual_returns = pd.DataFrame({'date': actual_data['date']})
    
    for etf in etf_tickers:
        if etf in actual_data.columns:
            # Compute return as percentage change (current price / previous price - 1)
            actual_data[f'{etf}_return'] = actual_data[etf].pct_change()
            # Shift returns so that the return corresponding to a date is the return for the following week.
            actual_returns[f'{etf}_return'] = actual_data[f'{etf}_return'].shift(-1)
        else:
            print(f"Warning: {etf} column not found in actual data.")
    
    # Drop the last row since next week's return is not available.
    actual_returns = actual_returns.dropna()
    
    # Merge predictions with actual returns based on date.
    evaluation_data = []
    for _, row in predictions.iterrows():
        date = row['date']
        instrument = row['instrument']  # e.g. "Short-Term Treasury"
        etf_ticker = treasury_to_etf.get(instrument, None)
        if etf_ticker is None:
            continue
        
        matching_returns = actual_returns[actual_returns['date'] == date]
        if not matching_returns.empty:
            actual_return = matching_returns[f'{etf_ticker}_return'].values[0]
            eval_row = row.copy()
            eval_row['actual_return'] = actual_return
            evaluation_data.append(eval_row)
    
    eval_df = pd.DataFrame(evaluation_data)
    return eval_df

def calculate_metrics(eval_df):
    """
    Calculate performance metrics for the predictions.
    """
    # Check if the required columns exist
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
    
    # Directional accuracy: fraction where the sign of predicted_return matches that of actual_return.
    correct_direction = (np.sign(eval_clean['predicted_return']) == np.sign(eval_clean['actual_return'])).mean()
    metrics['Directional Accuracy'] = correct_direction
    
    # Confidence-weighted correlation.
    eval_clean['weighted_prediction'] = eval_clean['predicted_return'] * eval_clean['confidence']
    weighted_correlation = eval_clean[['actual_return', 'weighted_prediction']].corr().iloc[0, 1]
    metrics['Weighted Correlation'] = weighted_correlation
    
    # Calculate metrics by instrument.
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
    Calculate performance metrics for a simple trading strategy based on predictions.
    """
    strategy_df = eval_df.copy().dropna(subset=['predicted_return', 'actual_return'])
    
    # Simple directional strategy: long/short based on predicted return sign.
    strategy_df['dir_strategy_return'] = np.sign(strategy_df['predicted_return']) * strategy_df['actual_return']
    
    # Confidence-weighted strategy.
    strategy_df['conf_strategy_return'] = np.sign(strategy_df['predicted_return']) * strategy_df['confidence'] * strategy_df['actual_return']
    
    # Size-proportional strategy: scale position size based on predicted return magnitude.
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
    
    # Performance by instrument.
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
    
    # 1. Scatter plot of predicted vs actual returns by instrument.
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
    
    # 2. Cumulative returns over time by instrument.
    time_perf = strategy_df.groupby(['date', 'instrument'])['dir_strategy_return'].mean().reset_index()
    time_perf_pivot = time_perf.pivot(index='date', columns='instrument', values='dir_strategy_return')
    
    plt.figure(figsize=(12, 8))
    time_perf_pivot.cumsum().plot()
    plt.title('Cumulative Strategy Returns by Instrument')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.tight_layout()
    plt.savefig('data/evaluation/cumulative_returns_by_instrument.png')
    
    # 3. Confidence vs directional accuracy.
    unique_conf = strategy_df['confidence'].nunique()
    try:
        if unique_conf >= 5:
            strategy_df['confidence_bin'] = pd.qcut(strategy_df['confidence'], 5, labels=False, duplicates='drop')
        else:
            if unique_conf <= 1:
                raise ValueError("Not enough unique confidence values for binning")
            else:
                strategy_df['confidence_bin'] = pd.cut(strategy_df['confidence'], bins=min(unique_conf, 5), labels=False)
        
        conf_acc = strategy_df.groupby('confidence_bin').apply(
            lambda x: (np.sign(x['predicted_return']) == np.sign(x['actual_return'])).mean()
        ).reset_index()
        conf_acc.columns = ['confidence_bin', 'directional_accuracy']
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='confidence_bin', y='directional_accuracy', data=conf_acc)
        plt.title('Directional Accuracy by Confidence Bin')
        plt.xlabel('Confidence Bin (Low to High)')
        plt.ylabel('Directional Accuracy')
        plt.tight_layout()
        plt.savefig('data/evaluation/accuracy_by_confidence.png')
    except Exception as e:
        print(f"Error creating confidence bin visualization: {e}")
        plt.figure(figsize=(10, 6))
        correct_dir = np.sign(strategy_df['predicted_return']) == np.sign(strategy_df['actual_return'])
        plt.scatter(strategy_df['confidence'], correct_dir, alpha=0.5)
        plt.title('Confidence vs Prediction Accuracy')
        plt.xlabel('Confidence Score')
        plt.ylabel('Correct Direction (1=Yes, 0=No)')
        plt.tight_layout()
        plt.savefig('data/evaluation/confidence_vs_accuracy.png')
    
    # 4. Strategy performance comparison.
    strategy_cols = ['dir_strategy_return', 'conf_strategy_return', 'size_strategy_return']
    cumulative_returns = (1 + strategy_df[strategy_cols]).cumprod() - 1
    plt.figure(figsize=(12, 8))
    cumulative_returns.plot()
    plt.title('Cumulative Returns by Trading Strategy')
    plt.xlabel('Observation')
    plt.ylabel('Cumulative Return')
    plt.legend(['Directional', 'Confidence-Weighted', 'Size-Proportional'])
    plt.tight_layout()
    plt.savefig('data/evaluation/strategy_comparison.png')

def main():
    predictions, actual_data = load_data()
    print(f"Predictions columns: {predictions.columns.tolist()}")
    print(f"Actual data columns: {actual_data.columns.tolist()}")
    print(f"Predictions date range: {predictions['date'].min()} to {predictions['date'].max()}")
    print(f"Actual data date range: {actual_data['date'].min()} to {actual_data['date'].max()}")
    
    print("Preparing evaluation data...")
    eval_df = prepare_evaluation_data(predictions, actual_data)
    
    # If no evaluation data, exit.
    if eval_df.empty:
        print("No matching evaluation data found. Exiting.")
        return
    
    print("Calculating performance metrics...")
    overall_metrics, instrument_metrics = calculate_metrics(eval_df)
    
    print("Calculating trading strategy performance...")
    strategy_perf, instrument_strategy_perf, strategy_df = calculate_trading_performance(eval_df)
    
    print("Creating visualizations...")
    create_visualizations(eval_df, strategy_df)
    
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
