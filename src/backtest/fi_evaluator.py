from Evaluator import Evaluator
import pandas as pd
import os
import matplotlib.pyplot as plt


class FixedIncomeEvaluator(Evaluator):
    """
    Evaluator for Fixed Income predictions.
    Inherits base functionality from Evaluator class.
    """
    
    def __init__(self, predictions_file, actual_data_file, output_dir):
        """
        Initialize the Fixed Income evaluator.
        """
        super().__init__(predictions_file, actual_data_file, output_dir)
    
    def prepare_evaluation_data(self):
        """
        Prepare evaluation data by matching predictions to next week's actual ETF returns.
        For each treasury ETF instrument, compute the weekly return (price change) and merge with predictions.
        """
        print(f"Preparing evaluation data for Fixed Income...")
        
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
        # Create a new DataFrame to hold next week's returns for each ETF
        actual_returns = pd.DataFrame({'date': self.actual_data['date']})
        
        for etf in etf_tickers:
            if etf in self.actual_data.columns:
                # Data already contains returns, so just shift to get next week's returns
                actual_returns[f'{etf}_return'] = self.actual_data[etf].shift(-1)
            else:
                print(f"Warning: {etf} column not found in actual data.")
        
        # Drop the last row since next week's return is not available
        actual_returns = actual_returns.dropna()
        
        # Merge predictions with actual returns based on date
        evaluation_data = []
        for _, row in self.predictions.iterrows():
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
    
    def create_visualizations(self):
        """
        Create visualizations to analyze prediction performance.
        """
        # Call the base class visualizations first
        super().create_visualizations()
        
        # Add additional fixed income specific visualization:
        # Cumulative returns over time by instrument
        time_perf = self.strategy_df.groupby(['date', 'instrument'])['dir_strategy_return'].mean().reset_index()
        time_perf_pivot = time_perf.pivot(index='date', columns='instrument', values='dir_strategy_return')
        
        plt.figure(figsize=(12, 8))
        time_perf_pivot.cumsum().plot()
        plt.title('Cumulative Strategy Returns by Instrument')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cumulative_returns_by_instrument.png')) 