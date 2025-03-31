import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """
    Load prediction and actual data files
    """
    print("Loading prediction data...")
    predictions = pd.read_csv('data/fx_weekly_predictions.csv')
    predictions['date'] = pd.to_datetime(predictions['date'])
    
    print("Loading actual ETF data...")
    actual_data = pd.read_csv('data/fx_combined_features_weekly.csv')
    # Make sure the date column is properly set
    if 'Unnamed: 0' in actual_data.columns:
        actual_data['date'] = pd.to_datetime(actual_data['Unnamed: 0'])
        actual_data = actual_data.drop('Unnamed: 0', axis=1)
    else:
        # If there's no unnamed column, the first column might be the date but with a different name
        first_col = actual_data.columns[0]
        actual_data['date'] = pd.to_datetime(actual_data[first_col])
        actual_data = actual_data.drop(first_col, axis=1)
    
    return predictions, actual_data

def prepare_evaluation_data(predictions, actual_data):
    """
    Prepare the data for evaluation by matching predictions to next week's actual returns
    """
    # Make sure actual_data has a proper date column
    print(f"Actual data columns: {actual_data.columns}")
    
    # Calculate returns for each ETF
    etf_cols = ['FXE', 'FXB', 'FXY', 'FXF', 'FXC']
    
    # Create a new dataframe with shifted values (next week's data)
    actual_returns = pd.DataFrame({'date': actual_data['date']})
    
    for etf in etf_cols:
        # Calculate return as current price / previous price - 1
        actual_data[f'{etf}_return'] = actual_data[etf].pct_change()
        
        # Shift returns back by 1 week, so that for each date, we have the next week's return
        actual_returns[f'{etf}_return'] = actual_data[f'{etf}_return'].shift(-1)
    
    # Drop the last row since we don't have next week's returns for it
    actual_returns = actual_returns.dropna()
    
    # Merge predictions with actual returns
    # First create a mapping for ETF to currency pair
    etf_to_pair = {
        'FXE': 'EUR/USD',
        'FXB': 'GBP/USD',
        'FXY': 'USD/JPY',
        'FXF': 'USD/CHF',
        'FXC': 'USD/CAD'
    }
    
    # Now prepare evaluation data
    evaluation_data = []
    
    for _, row in predictions.iterrows():
        date = row['date']
        etf = row['etf']
        
        # Find matching date in actual returns
        matching_returns = actual_returns[actual_returns['date'] == date]
        
        if not matching_returns.empty:
            actual_return = matching_returns[f'{etf}_return'].values[0]
            
            # Create evaluation row
            eval_row = row.copy()
            eval_row['actual_return'] = actual_return
            evaluation_data.append(eval_row)
    
    # Create dataframe from evaluation data
    eval_df = pd.DataFrame(evaluation_data)
    
    return eval_df

def calculate_metrics(eval_df):
    """
    Calculate performance metrics for the predictions
    """
    # Filter out any rows with missing data
    eval_clean = eval_df.dropna(subset=['predicted_return', 'actual_return'])
    
    # Calculate metrics
    metrics = {
        'Mean Squared Error': mean_squared_error(eval_clean['actual_return'], eval_clean['predicted_return']),
        'Root Mean Squared Error': np.sqrt(mean_squared_error(eval_clean['actual_return'], eval_clean['predicted_return'])),
        'Mean Absolute Error': mean_absolute_error(eval_clean['actual_return'], eval_clean['predicted_return']),
        'R-squared': r2_score(eval_clean['actual_return'], eval_clean['predicted_return']),
    }
    
    # Calculate correlation
    correlation = eval_clean[['actual_return', 'predicted_return']].corr().iloc[0, 1]
    metrics['Correlation'] = correlation
    
    # Calculate directional accuracy (if the sign of prediction matches actual)
    correct_direction = (np.sign(eval_clean['predicted_return']) == np.sign(eval_clean['actual_return'])).mean()
    metrics['Directional Accuracy'] = correct_direction
    
    # Calculate weighted returns (prediction confidence * predicted return)
    eval_clean['weighted_prediction'] = eval_clean['predicted_return'] * eval_clean['confidence']
    weighted_correlation = eval_clean[['actual_return', 'weighted_prediction']].corr().iloc[0, 1]
    metrics['Weighted Correlation'] = weighted_correlation
    
    # Calculate metrics by currency pair
    pair_metrics = {}
    for pair in eval_clean['currency_pair'].unique():
        pair_data = eval_clean[eval_clean['currency_pair'] == pair]
        pair_metrics[pair] = {
            'MSE': mean_squared_error(pair_data['actual_return'], pair_data['predicted_return']),
            'Directional Accuracy': (np.sign(pair_data['predicted_return']) == np.sign(pair_data['actual_return'])).mean(),
            'Correlation': pair_data[['actual_return', 'predicted_return']].corr().iloc[0, 1],
            'Count': len(pair_data)
        }
    
    return metrics, pair_metrics

def calculate_trading_performance(eval_df):
    """
    Calculate performance metrics for a simple trading strategy based on predictions
    """
    # Create a copy of the evaluation dataframe for strategy analysis
    strategy_df = eval_df.copy()
    
    # Filter out rows with missing data
    strategy_df = strategy_df.dropna(subset=['predicted_return', 'actual_return'])
    
    # Calculate returns for different strategies
    # 1. Simple directional strategy (long/short based on prediction sign)
    strategy_df['dir_strategy_return'] = np.sign(strategy_df['predicted_return']) * strategy_df['actual_return']
    
    # 2. Confidence-weighted strategy
    strategy_df['conf_strategy_return'] = np.sign(strategy_df['predicted_return']) * strategy_df['confidence'] * strategy_df['actual_return']
    
    # 3. Size-proportional strategy (size position based on predicted magnitude)
    norm_factor = strategy_df['predicted_return'].abs().mean()  # Normalization factor
    strategy_df['size_strategy_return'] = (strategy_df['predicted_return'] / norm_factor) * strategy_df['actual_return']
    
    # Calculate strategy performance metrics
    strategies = ['dir_strategy_return', 'conf_strategy_return', 'size_strategy_return']
    strategy_performance = {}
    baseline_return = strategy_df['actual_return'].mean()
    
    for strategy in strategies:
        strategy_returns = strategy_df[strategy]
        strategy_performance[strategy] = {
            'Mean Return': strategy_returns.mean(),
            'Cumulative Return': (1 + strategy_returns).prod() - 1,
            'Sharpe Ratio': strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0,
            'Win Rate': (strategy_returns > 0).mean(),
            'vs Baseline': strategy_returns.mean() - baseline_return
        }
    
    # Calculate performance by currency pair and strategy
    pair_strategy_performance = {}
    for pair in strategy_df['currency_pair'].unique():
        pair_data = strategy_df[strategy_df['currency_pair'] == pair]
        pair_strategy_performance[pair] = {}
        
        pair_baseline = pair_data['actual_return'].mean()
        
        for strategy in strategies:
            strategy_returns = pair_data[strategy]
            pair_strategy_performance[pair][strategy] = {
                'Mean Return': strategy_returns.mean(),
                'Cumulative Return': (1 + strategy_returns).prod() - 1,
                'vs Baseline': strategy_returns.mean() - pair_baseline
            }
    
    return strategy_performance, pair_strategy_performance, strategy_df

def create_visualizations(eval_df, strategy_df):
    """
    Create visualizations to analyze prediction performance
    """
    # Create output directory if it doesn't exist
    os.makedirs('data/evaluation', exist_ok=True)
    
    # 1. Scatter plot of predicted vs actual returns
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='predicted_return', y='actual_return', 
                    hue='currency_pair', size='confidence', 
                    data=eval_df.dropna(subset=['predicted_return', 'actual_return']))
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.title('Predicted vs Actual Returns')
    plt.xlabel('Predicted Return')
    plt.ylabel('Actual Return')
    plt.tight_layout()
    plt.savefig('data/evaluation/predicted_vs_actual.png')
    
    # 2. Performance over time by currency pair
    # Group by date and currency pair to calculate average performance
    time_perf = strategy_df.groupby(['date', 'currency_pair'])['dir_strategy_return'].mean().reset_index()
    time_perf_pivot = time_perf.pivot(index='date', columns='currency_pair', values='dir_strategy_return')
    
    plt.figure(figsize=(12, 8))
    time_perf_pivot.cumsum().plot()
    plt.title('Cumulative Strategy Returns by Currency Pair')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.tight_layout()
    plt.savefig('data/evaluation/cumulative_returns_by_pair.png')
    
    # 3. Confidence vs accuracy
    # Check the number of unique confidence values
    unique_confidence = strategy_df['confidence'].nunique()
    print(f"Number of unique confidence values: {unique_confidence}")
    
    try:
        # Try to create confidence bins, but handle duplicate bin edges
        if unique_confidence >= 5:
            # If we have enough unique values, use qcut with duplicates='drop'
            strategy_df['confidence_bin'] = pd.qcut(strategy_df['confidence'], 5, labels=False, duplicates='drop')
        else:
            # If we have fewer unique values, use fewer bins or just use the values directly
            if unique_confidence <= 1:
                # If only one confidence value, we can't bin
                print("Only one unique confidence value - skipping confidence binning analysis")
                raise ValueError("Not enough unique confidence values for binning")
            else:
                # Use cut instead of qcut with the number of unique values
                strategy_df['confidence_bin'] = pd.cut(strategy_df['confidence'], 
                                                       bins=min(unique_confidence, 5), 
                                                       labels=False)
        
        # Calculate directional accuracy by confidence bin
        conf_accuracy = strategy_df.groupby('confidence_bin').apply(
            lambda x: (np.sign(x['predicted_return']) == np.sign(x['actual_return'])).mean()
        ).reset_index()
        conf_accuracy.columns = ['confidence_bin', 'directional_accuracy']
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='confidence_bin', y='directional_accuracy', data=conf_accuracy)
        plt.title('Directional Accuracy by Confidence Bin')
        plt.xlabel('Confidence Bin (Low to High)')
        plt.ylabel('Directional Accuracy')
        plt.tight_layout()
        plt.savefig('data/evaluation/accuracy_by_confidence.png')
        
    except Exception as e:
        print(f"Error creating confidence bin visualization: {e}")
        print("Creating alternate confidence visualization...")
        
        # Alternative: Plot scatter of confidence vs accuracy
        plt.figure(figsize=(10, 6))
        correct_direction = np.sign(strategy_df['predicted_return']) == np.sign(strategy_df['actual_return'])
        plt.scatter(strategy_df['confidence'], correct_direction, alpha=0.5)
        plt.title('Confidence vs Prediction Accuracy')
        plt.xlabel('Confidence Score')
        plt.ylabel('Correct Direction (1=Yes, 0=No)')
        plt.tight_layout()
        plt.savefig('data/evaluation/confidence_vs_accuracy.png')
    
    # 4. Strategy performance comparison
    strategy_cols = ['dir_strategy_return', 'conf_strategy_return', 'size_strategy_return']
    cumulative_returns = (1 + strategy_df[strategy_cols]).cumprod() - 1
    
    plt.figure(figsize=(12, 8))
    cumulative_returns.plot()
    plt.title('Cumulative Returns by Strategy')
    plt.xlabel('Observation')
    plt.ylabel('Cumulative Return')
    plt.legend(['Directional', 'Confidence-Weighted', 'Size-Proportional'])
    plt.tight_layout()
    plt.savefig('data/evaluation/strategy_comparison.png')

def main():
    # Load data
    predictions, actual_data = load_data()
    
    # Debug output
    print(f"Predictions columns: {predictions.columns}")
    print(f"Actual data columns: {actual_data.columns}")
    print(f"Predictions date range: {predictions['date'].min()} to {predictions['date'].max()}")
    print(f"Actual data date range: {actual_data['date'].min()} to {actual_data['date'].max()}")
    
    # Prepare evaluation data
    print("Preparing evaluation data...")
    eval_df = prepare_evaluation_data(predictions, actual_data)
    
    # Calculate performance metrics
    print("Calculating performance metrics...")
    overall_metrics, pair_metrics = calculate_metrics(eval_df)
    
    # Calculate trading strategy performance
    print("Calculating trading strategy performance...")
    strategy_perf, pair_strategy_perf, strategy_df = calculate_trading_performance(eval_df)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(eval_df, strategy_df)
    
    # Display the results
    print("\n==== OVERALL PREDICTION METRICS ====")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n==== METRICS BY CURRENCY PAIR ====")
    for pair, metrics in pair_metrics.items():
        print(f"\n{pair}:")
        for metric, value in metrics.items():
            if metric != 'Count':
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    print("\n==== TRADING STRATEGY PERFORMANCE ====")
    for strategy, perf in strategy_perf.items():
        strategy_name = strategy.replace('_strategy_return', '')
        print(f"\n{strategy_name.capitalize()} Strategy:")
        for metric, value in perf.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save evaluation results to CSV
    eval_df.to_csv('data/evaluation/prediction_evaluation.csv', index=False)
    
    print("\nEvaluation data saved to 'data/evaluation/prediction_evaluation.csv'")
    print("Visualizations saved to 'data/evaluation/' directory")

if __name__ == "__main__":
    main()
