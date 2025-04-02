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

        # Ensure the 'date' column is a datetime and set it as the index
        self.actual_data['date'] = pd.to_datetime(self.actual_data['date'])
        actual_data = self.actual_data.set_index('date')

        # Aggregate daily returns into weekly returns using Friday as the week-end.
        # Here, we compound the returns: (product of (1+daily_return)) - 1
        weekly_returns = actual_data[etf_tickers].resample('W-FRI').apply(lambda x: (x + 1).prod() - 1)

        # Shift weekly returns to get next week's return
        weekly_returns_shifted = weekly_returns.shift(-1).dropna().reset_index()

        # Merge predictions (assumed to be weekly) with actual weekly returns based on the date.
        evaluation_data = []
        for _, row in self.predictions.iterrows():
            pred_date = pd.to_datetime(row['date'])
            instrument = row['instrument']  # e.g., "Short-Term Treasury"
            etf_ticker = treasury_to_etf.get(instrument)
            if etf_ticker is None:
                continue
            matching_week = weekly_returns_shifted[weekly_returns_shifted['date'] == pred_date]
            if not matching_week.empty:
                actual_return = matching_week[etf_ticker].values[0]
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