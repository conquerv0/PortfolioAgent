import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

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

def sanitize_filename(name):
    """
    Sanitize a string to be used as a filename
    Replaces slashes and other invalid characters with underscores
    """
    return re.sub(r'[\\/*?:"<>|]', '_', name)

def prepare_evaluation_data(predictions, actual_data):
    """
    Prepare the data for evaluation by matching predictions to next week's actual returns and volatility
    """
    # Make sure actual_data has a proper date column
    print(f"Actual data columns: {actual_data.columns}")
    
    # Calculate returns for each ETF
    etf_cols = ['FXE', 'FXB', 'FXY', 'FXF', 'FXC']
    
    # Create a new dataframe with shifted values (next week's data)
    actual_returns = pd.DataFrame({'date': actual_data['date']})
    
    # Create realized returns and volatility
    for etf in etf_cols:
        # Calculate return as current price / previous price - 1
        actual_data[f'{etf}_return'] = actual_data[etf].pct_change()
        
        # Shift returns back by 1 week, so that for each date, we have the next week's return
        actual_returns[f'{etf}_return'] = actual_data[f'{etf}_return'].shift(-1)
        
        # Calculate realized volatility - use the 1-month volatility as a proxy
        # for the actual realized volatility in the next period
        if f'{etf}_vol_1m' in actual_data.columns:
            actual_returns[f'{etf}_realized_vol'] = actual_data[f'{etf}_vol_1m'].shift(-1)

    # Drop the last row since we don't have next week's returns for it
    actual_returns = actual_returns.dropna(subset=[f'{etf}_return' for etf in etf_cols])
    
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
            
            # Add realized volatility if available
            if f'{etf}_realized_vol' in matching_returns.columns:
                eval_row['actual_volatility'] = matching_returns[f'{etf}_realized_vol'].values[0]
            
            evaluation_data.append(eval_row)
    
    # Create dataframe from evaluation data
    eval_df = pd.DataFrame(evaluation_data)
    
    # Print summary statistics
    print(f"Total prediction-actual pairs: {len(eval_df)}")
    print(f"Date range: {eval_df['date'].min()} to {eval_df['date'].max()}")
    print(f"Number of instruments: {eval_df['instrument'].nunique()}")
    
    return eval_df

def calculate_metrics(eval_df, target_col='return'):
    """
    Calculate performance metrics for the predictions
    
    Parameters:
    -----------
    eval_df : pandas.DataFrame
        DataFrame containing prediction and actual data
    target_col : str
        Target column to evaluate ('return' or 'volatility')
    """
    # Determine the column names based on target_col
    predicted_col = f'predicted_{target_col}'
    actual_col = f'actual_{target_col}'
    
    # Filter out any rows with missing data
    eval_clean = eval_df.dropna(subset=[predicted_col, actual_col])
    
    if len(eval_clean) == 0:
        print(f"Warning: No data available for {target_col} evaluation")
        return {}, {}
    
    # Calculate metrics
    metrics = {
        'Mean Squared Error': mean_squared_error(eval_clean[actual_col], eval_clean[predicted_col]),
        'Root Mean Squared Error': np.sqrt(mean_squared_error(eval_clean[actual_col], eval_clean[predicted_col])),
        'Mean Absolute Error': mean_absolute_error(eval_clean[actual_col], eval_clean[predicted_col]),
        'R-squared': r2_score(eval_clean[actual_col], eval_clean[predicted_col]),
    }
    
    # Calculate correlation if there's variation in both series
    if eval_clean[predicted_col].std() > 0 and eval_clean[actual_col].std() > 0:
        correlation = eval_clean[[actual_col, predicted_col]].corr().iloc[0, 1]
        metrics['Correlation'] = correlation
    else:
        metrics['Correlation'] = np.nan
    
    # Calculate directional accuracy (if the sign of prediction matches actual)
    # For volatility, we're checking if both predict an increase or decrease in volatility
    # For returns, we check if both predict positive or negative returns
    correct_direction = (np.sign(eval_clean[predicted_col]) == np.sign(eval_clean[actual_col])).mean()
    metrics['Directional Accuracy'] = correct_direction
    
    # Calculate weighted predictions (prediction confidence * predicted value)
    eval_clean['weighted_prediction'] = eval_clean[predicted_col] * eval_clean['confidence']
    # Check if there's sufficient variation to calculate correlation
    if eval_clean['weighted_prediction'].std() > 0 and eval_clean[actual_col].std() > 0:
        weighted_correlation = eval_clean[[actual_col, 'weighted_prediction']].corr().iloc[0, 1]
        metrics['Weighted Correlation'] = weighted_correlation
    
    # Calculate metrics by currency pair/instrument
    pair_metrics = {}
    for instrument in eval_clean['instrument'].unique():
        pair_data = eval_clean[eval_clean['instrument'] == instrument]
        
        # Skip if too few data points
        if len(pair_data) < 3:
            continue
        
        pair_metrics_dict = {
            'MSE': mean_squared_error(pair_data[actual_col], pair_data[predicted_col]),
            'MAE': mean_absolute_error(pair_data[actual_col], pair_data[predicted_col]),
            'Count': len(pair_data)
        }
        
        # Calculate correlation if there's variation in both series
        if pair_data[predicted_col].std() > 0 and pair_data[actual_col].std() > 0:
            correlation = pair_data[[actual_col, predicted_col]].corr().iloc[0, 1]
            pair_metrics_dict['Correlation'] = correlation
        else:
            pair_metrics_dict['Correlation'] = np.nan
        
        # Add directional accuracy for both returns and volatility
        pair_metrics_dict['Directional Accuracy'] = (np.sign(pair_data[predicted_col]) == np.sign(pair_data[actual_col])).mean()
        
        pair_metrics[instrument] = pair_metrics_dict
    
    return metrics, pair_metrics

def create_visualizations(eval_df, target_col='return'):
    """
    Create visualizations to analyze prediction performance
    
    Parameters:
    -----------
    eval_df : pandas.DataFrame
        DataFrame containing prediction and actual data
    target_col : str
        Target column to visualize ('return' or 'volatility')
    """
    # Determine the column names based on target_col
    predicted_col = f'predicted_{target_col}'
    actual_col = f'actual_{target_col}'
    
    # Create output directory if it doesn't exist
    os.makedirs('data/evaluation', exist_ok=True)
    
    # Filter out any rows with missing data
    eval_clean = eval_df.dropna(subset=[predicted_col, actual_col])
    
    if len(eval_clean) == 0:
        print(f"Warning: No data available for {target_col} visualization")
        return
    
    # Generate only the confidence vs accuracy chart
    # Check the number of unique confidence values
    unique_confidence = eval_clean['confidence'].nunique()
    print(f"Number of unique confidence values: {unique_confidence}")
    
    try:
        # Try to create confidence bins, but handle duplicate bin edges
        if unique_confidence >= 5:
            # If we have enough unique values, use qcut with duplicates='drop'
            bins = pd.qcut(eval_clean['confidence'], 5, duplicates='drop')
            eval_clean['confidence_bin'] = bins
            
            # Get the bin ranges for labeling
            bin_labels = [f"{b.left:.2f}-{b.right:.2f}" for b in bins.cat.categories]
        else:
            # If we have fewer unique values, use fewer bins
            if unique_confidence <= 1:
                # If only one confidence value, we can't bin
                print("Only one unique confidence value - skipping confidence binning analysis")
                raise ValueError("Not enough unique confidence values for binning")
            else:
                # Use cut instead of qcut with the number of unique values
                bins = pd.cut(eval_clean['confidence'], bins=min(unique_confidence, 5))
                eval_clean['confidence_bin'] = bins
                
                # Get the bin ranges for labeling
                bin_labels = [f"{b.left:.2f}-{b.right:.2f}" for b in bins.categories]
        
        # Calculate directional accuracy by confidence bin
        conf_accuracy = eval_clean.groupby('confidence_bin').apply(
            lambda x: (np.sign(x[predicted_col]) == np.sign(x[actual_col])).mean()
        ).reset_index()
        
        # Create a simpler version of the bin labels for plotting
        conf_accuracy['bin_label'] = [f"{b.left:.2f}-{b.right:.2f}" for b in conf_accuracy['confidence_bin']]
        conf_accuracy = conf_accuracy.sort_values(by='bin_label')
        
        # Plot with actual confidence ranges
        plt.figure(figsize=(12, 6))
        bar_plot = sns.barplot(x='bin_label', y=0, data=conf_accuracy)
        plt.title(f'Directional Accuracy by Confidence Level - {target_col.capitalize()}')
        plt.xlabel('Confidence Range')
        plt.ylabel('Directional Accuracy')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'data/evaluation/accuracy_by_confidence_{target_col}.png')
        plt.close()
        
    except Exception as e:
        print(f"Error creating confidence bin visualization: {e}")
        print("Creating alternate confidence visualization...")
        
        # Alternative: Create a binned scatter plot
        plt.figure(figsize=(10, 6))
        correct_direction = np.sign(eval_clean[predicted_col]) == np.sign(eval_clean[actual_col])
        
        # Get unique confidence values and sort them
        confidence_values = sorted(eval_clean['confidence'].unique())
        accuracy_by_conf = []
        
        # Calculate accuracy for each unique confidence value
        for conf in confidence_values:
            mask = eval_clean['confidence'] == conf
            accuracy = correct_direction[mask].mean()
            accuracy_by_conf.append(accuracy)
        
        # Create a bar plot with actual confidence values
        plt.bar(
            [str(round(c, 2)) for c in confidence_values],
            accuracy_by_conf, 
            alpha=0.7
        )
        plt.title(f'Confidence vs Prediction Accuracy - {target_col.capitalize()}')
        plt.xlabel('Confidence Value')
        plt.ylabel('Directional Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'data/evaluation/accuracy_by_confidence_{target_col}.png')
        plt.close()

def analyze_pair_metrics(eval_df, target_col='return'):
    """
    Analyze prediction performance metrics by currency pair/instrument
    
    Parameters:
    -----------
    eval_df : pandas.DataFrame
        DataFrame containing prediction and actual data
    target_col : str
        Target column to analyze ('return' or 'volatility')
    """
    # Determine the column names based on target_col
    predicted_col = f'predicted_{target_col}'
    actual_col = f'actual_{target_col}'
    
    # Create output directory if it doesn't exist
    os.makedirs('data/evaluation', exist_ok=True)
    
    # Filter out rows with missing data
    eval_clean = eval_df.dropna(subset=[predicted_col, actual_col])
    
    if len(eval_clean) == 0:
        print(f"Warning: No data available for {target_col} analysis")
        return None
    
    # Calculate metrics for each instrument
    instruments = eval_clean['instrument'].unique()
    
    # Create a dataframe to store the results
    metrics_df = pd.DataFrame(columns=['Instrument', 'Avg Confidence', 'MSE', 'MAE', 'Correlation', 'Directional Accuracy', 'Sample Size'])
    
    # Calculate metrics for each instrument
    for instrument in instruments:
        instrument_data = eval_clean[eval_clean['instrument'] == instrument]
        
        # Skip if too few data points
        if len(instrument_data) < 3:
            continue
        
        # Calculate metrics
        avg_confidence = instrument_data['confidence'].mean()
        mse = mean_squared_error(instrument_data[actual_col], instrument_data[predicted_col])
        mae = mean_absolute_error(instrument_data[actual_col], instrument_data[predicted_col])
        
        # Calculate correlation if there's variation in both series
        if instrument_data[predicted_col].std() > 0 and instrument_data[actual_col].std() > 0:
            correlation = instrument_data[[actual_col, predicted_col]].corr().iloc[0, 1]
        else:
            correlation = np.nan
        
        # Calculate directional accuracy
        dir_accuracy = (np.sign(instrument_data[predicted_col]) == np.sign(instrument_data[actual_col])).mean()
        
        # Create result row
        result_row = {
            'Instrument': instrument,
            'Avg Confidence': avg_confidence,
            'MSE': mse,
            'MAE': mae,
            'Correlation': correlation,
            'Directional Accuracy': dir_accuracy,
            'Sample Size': len(instrument_data)
        }
        
        # Add to dataframe
        metrics_df = pd.concat([metrics_df, pd.DataFrame([result_row])], ignore_index=True)
    
    # Sort by directional accuracy
    metrics_df = metrics_df.sort_values('Directional Accuracy', ascending=False)
    
    # Create a comparative bar plot
    plt.figure(figsize=(12, 8))
    
    # Set up positions for bars
    bar_positions = np.arange(len(metrics_df))
    bar_width = 0.35
    
    # Create confidence bars
    plt.bar(bar_positions - bar_width/2, metrics_df['Avg Confidence'], 
            width=bar_width, color='skyblue', label='Avg Confidence')
    
    # Create directional accuracy bars
    plt.bar(bar_positions + bar_width/2, metrics_df['Directional Accuracy'], 
            width=bar_width, color='orange', label='Directional Accuracy')
    
    # Add instrument names on x-axis
    plt.xticks(bar_positions, metrics_df['Instrument'])
    
    # Add labels and title
    plt.xlabel('Instrument')
    plt.ylabel('Value')
    plt.title(f'Prediction Performance Metrics by Instrument - {target_col.capitalize()}')
    plt.legend()
    
    # Add sample size as text above each instrument
    max_val = max(
        metrics_df['Avg Confidence'].max(), 
        metrics_df['Directional Accuracy'].max()
    )
    
    for i, pos in enumerate(bar_positions):
        plt.text(pos, max_val + 0.05, 
                 f"n={metrics_df['Sample Size'].iloc[i]}", 
                 ha='center')
    
    plt.ylim(0, 1.1)  # Set y-axis limit to accommodate the sample size text
    plt.tight_layout()
    plt.savefig(f'data/evaluation/instrument_metrics_{target_col}.png')
    plt.close()
    
    # Save the data to CSV
    metrics_df.to_csv(f'data/evaluation/instrument_metrics_{target_col}.csv', index=False)
    
    # Print the results
    print(f"\n==== METRICS BY INSTRUMENT - {target_col.upper()} ====")
    for _, row in metrics_df.iterrows():
        print(f"{row['Instrument']}:")
        print(f"  Avg Confidence: {row['Avg Confidence']:.4f}")
        print(f"  Directional Accuracy: {row['Directional Accuracy']:.4f}")
        print(f"  Sample Size: {row['Sample Size']}")
    
    return metrics_df

def main():
    # Load data
    predictions, actual_data = load_data()
    
    # Debug output
    print(f"Predictions columns: {predictions.columns}")
    print(f"Actual data columns: {actual_data.columns}")
    print(f"Predictions date range: {predictions['date'].min()} to {predictions['date'].max()}")
    print(f"Actual data date range: {actual_data['date'].min()} to {actual_data['date'].max()}")
    
    # Create evaluation directory if it doesn't exist
    os.makedirs('data/evaluation', exist_ok=True)
    
    # Prepare evaluation data
    print("Preparing evaluation data...")
    eval_df = prepare_evaluation_data(predictions, actual_data)
    
    # Save full evaluation data for reference
    eval_df.to_csv('data/evaluation/full_evaluation_data.csv', index=False)
    
    # Analyze returns predictions
    print("\n==== ANALYZING RETURN PREDICTIONS ====")
    # Calculate performance metrics for returns
    return_metrics, return_pair_metrics = calculate_metrics(eval_df, target_col='return')
    
    # Create return accuracy by confidence chart only
    create_visualizations(eval_df, target_col='return')
    
    # Create instrument metrics chart for returns
    return_metrics_by_pair = analyze_pair_metrics(eval_df, target_col='return')
    
    # Display key return results
    print("\n==== OVERALL RETURN PREDICTION METRICS ====")
    print(f"Directional Accuracy: {return_metrics.get('Directional Accuracy', 'N/A'):.4f}")
    print(f"Average Confidence: {eval_df['confidence'].mean():.4f}")
    
    # Analyze volatility predictions (if available)
    if 'predicted_volatility' in eval_df.columns and 'actual_volatility' in eval_df.columns:
        print("\n==== ANALYZING VOLATILITY PREDICTIONS ====")
        # Calculate performance metrics for volatility
        vol_metrics, vol_pair_metrics = calculate_metrics(eval_df, target_col='volatility')
        
        # Create volatility accuracy by confidence chart only
        create_visualizations(eval_df, target_col='volatility')
        
        # Create instrument metrics chart for volatility
        vol_metrics_by_pair = analyze_pair_metrics(eval_df, target_col='volatility')
        
        # Display key volatility results
        print("\n==== OVERALL VOLATILITY PREDICTION METRICS ====")
        print(f"Directional Accuracy: {vol_metrics.get('Directional Accuracy', 'N/A'):.4f}")
        print(f"Average Confidence: {eval_df.dropna(subset=['predicted_volatility', 'actual_volatility'])['confidence'].mean():.4f}")
    else:
        print("\nNo volatility prediction data available for analysis")
    
    print("\nEvaluation complete. Results saved to 'data/evaluation/' directory")

if __name__ == "__main__":
    main()
