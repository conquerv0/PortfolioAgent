from Evaluator import Evaluator
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


class ForexEvaluator(Evaluator):
    """
    Evaluator for Forex predictions.
    Inherits base functionality from Evaluator class.
    """
    
    def __init__(self, predictions_file, actual_data_file, output_dir):
        """
        Initialize the Forex evaluator.
        """
        super().__init__(predictions_file, actual_data_file, output_dir)
    
    def prepare_evaluation_data(self):
        """
        Prepare evaluation data by matching predictions to next week's actual Forex ETF returns.
        """
        print(f"Preparing evaluation data for Forex...")
        
        # Calculate returns for each Forex ETF
        etf_cols = ['FXE', 'FXB', 'FXY', 'FXF', 'FXC']
        
        # Create a new dataframe with shifted values (next week's data)
        actual_returns = pd.DataFrame({'date': self.actual_data['date']})
        
        for etf in etf_cols:
            # Check if the ETF column exists
            if etf in self.actual_data.columns:
                # Calculate return as current price / previous price - 1
                self.actual_data[f'{etf}_return'] = self.actual_data[etf].pct_change()
                
                # Shift returns back by 1 week, so that for each date, we have the next week's return
                actual_returns[f'{etf}_return'] = self.actual_data[f'{etf}_return'].shift(-1)
            else:
                print(f"Warning: {etf} column not found in actual data.")
        
        # Drop the last row since we don't have next week's returns for it
        actual_returns = actual_returns.dropna()
        
        # Mapping for ETF to currency pair (if needed for display)
        etf_to_pair = {
            'FXE': 'EUR/USD',
            'FXB': 'GBP/USD',
            'FXY': 'USD/JPY',
            'FXF': 'USD/CHF',
            'FXC': 'USD/CAD'
        }
        
        # Now prepare evaluation data
        evaluation_data = []
        
        for _, row in self.predictions.iterrows():
            date = row['date']
            etf = row['etf']
            
            # Find matching date in actual returns
            matching_returns = actual_returns[actual_returns['date'] == date]
            
            if not matching_returns.empty:
                # Check if this ETF's return is available
                etf_return_col = f'{etf}_return'
                if etf_return_col in matching_returns.columns:
                    actual_return = matching_returns[etf_return_col].values[0]
                    
                    # Create evaluation row
                    eval_row = row.copy()
                    eval_row['actual_return'] = actual_return
                    evaluation_data.append(eval_row)
        
        # Create dataframe from evaluation data
        eval_df = pd.DataFrame(evaluation_data)
        
        return eval_df
    
    def create_visualizations(self):
        """
        Create visualizations to analyze prediction performance.
        """
        # Call the base class visualizations first
        super().create_visualizations()
        
        # Add additional Forex-specific visualization:
        # Currency pair confidence vs accuracy
        eval_clean = self.eval_df.dropna(subset=['predicted_return', 'actual_return'])
        pair_confidence_df = pd.DataFrame(columns=['Currency Pair', 'Avg Confidence', 'Directional Accuracy', 'Sample Size'])
        
        for pair in eval_clean['currency_pair'].unique():
            pair_data = eval_clean[eval_clean['currency_pair'] == pair]
            avg_confidence = pair_data['confidence'].mean()
            dir_accuracy = (np.sign(pair_data['predicted_return']) == np.sign(pair_data['actual_return'])).mean()
            
            pair_confidence_df = pd.concat([pair_confidence_df, pd.DataFrame({
                'Currency Pair': [pair],
                'Avg Confidence': [avg_confidence],
                'Directional Accuracy': [dir_accuracy],
                'Sample Size': [len(pair_data)]
            })], ignore_index=True)
        
        # Sort by directional accuracy
        pair_confidence_df = pair_confidence_df.sort_values('Directional Accuracy', ascending=False)
        
        plt.figure(figsize=(12, 8))
        bar_positions = np.arange(len(pair_confidence_df))
        bar_width = 0.35
        
        plt.bar(bar_positions - bar_width/2, pair_confidence_df['Avg Confidence'], 
                width=bar_width, color='skyblue', label='Avg Confidence')
        plt.bar(bar_positions + bar_width/2, pair_confidence_df['Directional Accuracy'], 
                width=bar_width, color='orange', label='Directional Accuracy')
        
        plt.xticks(bar_positions, pair_confidence_df['Currency Pair'])
        plt.xlabel('Currency Pair')
        plt.ylabel('Value')
        plt.title('Average Confidence vs Directional Accuracy by Currency Pair')
        plt.legend()
        
        for i, pos in enumerate(bar_positions):
            plt.text(pos, max(pair_confidence_df['Avg Confidence'].max(), 
                              pair_confidence_df['Directional Accuracy'].max()) + 0.05, 
                     f"n={pair_confidence_df['Sample Size'].iloc[i]}", 
                     ha='center')
        
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pair_confidence_vs_accuracy.png'))
        
        # Save pair confidence analysis to CSV
        pair_confidence_df.to_csv(os.path.join(self.output_dir, 'pair_confidence_accuracy.csv'), index=False) 