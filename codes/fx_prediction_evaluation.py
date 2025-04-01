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

def create_visualizations(eval_df):
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
    
    # 2. Confidence vs accuracy
    # Check the number of unique confidence values
    eval_clean = eval_df.dropna(subset=['predicted_return', 'actual_return'])
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
            lambda x: (np.sign(x['predicted_return']) == np.sign(x['actual_return'])).mean()
        ).reset_index()
        
        # Create a simpler version of the bin labels for plotting
        conf_accuracy['bin_label'] = [f"{b.left:.2f}-{b.right:.2f}" for b in conf_accuracy['confidence_bin']]
        conf_accuracy = conf_accuracy.sort_values(by='bin_label')
        
        # Plot with actual confidence ranges
        plt.figure(figsize=(12, 6))
        bar_plot = sns.barplot(x='bin_label', y=0, data=conf_accuracy)
        plt.title('Directional Accuracy by Confidence Level')
        plt.xlabel('Confidence Range')
        plt.ylabel('Directional Accuracy')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data/evaluation/accuracy_by_confidence.png')
        
    except Exception as e:
        print(f"Error creating confidence bin visualization: {e}")
        print("Creating alternate confidence visualization...")
        
        # Alternative: Create a binned scatter plot
        plt.figure(figsize=(10, 6))
        correct_direction = np.sign(eval_clean['predicted_return']) == np.sign(eval_clean['actual_return'])
        
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
        plt.title('Confidence vs Prediction Accuracy')
        plt.xlabel('Confidence Value')
        plt.ylabel('Directional Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data/evaluation/confidence_vs_accuracy.png')

def analyze_pair_confidence(eval_df):
    """
    Analyze the relationship between confidence levels and accuracy for each currency pair
    """
    # Create output directory if it doesn't exist
    os.makedirs('data/evaluation', exist_ok=True)
    
    # Filter out rows with missing data
    eval_clean = eval_df.dropna(subset=['predicted_return', 'actual_return'])
    
    # Calculate directional accuracy for each pair
    pairs = eval_clean['currency_pair'].unique()
    
    # Create a dataframe to store the results
    pair_confidence_df = pd.DataFrame(columns=['Currency Pair', 'Avg Confidence', 'Directional Accuracy', 'Sample Size'])
    
    # Calculate metrics for each pair
    for pair in pairs:
        pair_data = eval_clean[eval_clean['currency_pair'] == pair]
        
        # Calculate average confidence
        avg_confidence = pair_data['confidence'].mean()
        
        # Calculate directional accuracy
        dir_accuracy = (np.sign(pair_data['predicted_return']) == np.sign(pair_data['actual_return'])).mean()
        
        # Add to dataframe
        pair_confidence_df = pair_confidence_df._append({
            'Currency Pair': pair,
            'Avg Confidence': avg_confidence,
            'Directional Accuracy': dir_accuracy,
            'Sample Size': len(pair_data)
        }, ignore_index=True)
    
    # Sort by directional accuracy
    pair_confidence_df = pair_confidence_df.sort_values('Directional Accuracy', ascending=False)
    
    # Create a bar plot comparing confidence and accuracy by pair
    plt.figure(figsize=(12, 8))
    
    # Set up positions for bars
    bar_positions = np.arange(len(pairs))
    bar_width = 0.35
    
    # Create bars
    plt.bar(bar_positions - bar_width/2, pair_confidence_df['Avg Confidence'], 
            width=bar_width, color='skyblue', label='Avg Confidence')
    plt.bar(bar_positions + bar_width/2, pair_confidence_df['Directional Accuracy'], 
            width=bar_width, color='orange', label='Directional Accuracy')
    
    # Add pair names on x-axis
    plt.xticks(bar_positions, pair_confidence_df['Currency Pair'])
    
    # Add labels and title
    plt.xlabel('Currency Pair')
    plt.ylabel('Value')
    plt.title('Average Confidence vs Directional Accuracy by Currency Pair')
    plt.legend()
    
    # Add sample size as text above each pair
    for i, pos in enumerate(bar_positions):
        plt.text(pos, max(pair_confidence_df['Avg Confidence'].max(), 
                          pair_confidence_df['Directional Accuracy'].max()) + 0.05, 
                 f"n={pair_confidence_df['Sample Size'].iloc[i]}", 
                 ha='center')
    
    plt.ylim(0, 1.1)  # Set y-axis limit to accommodate the sample size text
    plt.tight_layout()
    plt.savefig('data/evaluation/pair_confidence_vs_accuracy.png')
    
    # Save the data to CSV
    pair_confidence_df.to_csv('data/evaluation/pair_confidence_accuracy.csv', index=False)
    
    # Print the results
    print("\n==== CONFIDENCE & ACCURACY BY CURRENCY PAIR ====")
    for _, row in pair_confidence_df.iterrows():
        print(f"{row['Currency Pair']}:")
        print(f"  Avg Confidence: {row['Avg Confidence']:.4f}")
        print(f"  Directional Accuracy: {row['Directional Accuracy']:.4f}")
        print(f"  Sample Size: {row['Sample Size']}")
    
    return pair_confidence_df

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
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(eval_df)
    
    # Analyze confidence and accuracy by currency pair
    print("Analyzing confidence and accuracy by currency pair...")
    pair_confidence_df = analyze_pair_confidence(eval_df)
    
    # Display the results
    print("\n==== OVERALL PREDICTION METRICS ====")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    
    # Save evaluation results to CSV
    eval_df.to_csv('data/evaluation/prediction_evaluation.csv', index=False)
    
    print("\nEvaluation data saved to 'data/evaluation/prediction_evaluation.csv'")
    print("Visualizations saved to 'data/evaluation/' directory")

if __name__ == "__main__":
    main()
