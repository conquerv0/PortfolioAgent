import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from abc import ABC, abstractmethod


class Evaluator(ABC):
    """
    Base class for evaluating agent predictions performance.
    Provides common functionality for loading data, calculating metrics, and visualizing results.
    """
    
    def __init__(self, predictions_file, actual_data_file, output_dir='data/evaluation'):
        """
        Initialize the evaluator with file paths for predictions and actual data.
        
        Args:
            predictions_file (str): Path to the predictions CSV file
            actual_data_file (str): Path to the actual data CSV file
            output_dir (str): Directory to save evaluation results and visualizations
        """
        self.predictions_file = predictions_file
        self.actual_data_file = actual_data_file
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data attributes
        self.predictions = None
        self.actual_data = None
        self.eval_df = None
        self.metrics = None
        self.instrument_metrics = None
        self.strategy_df = None
        self.strategy_performance = None
        self.instrument_strategy_perf = None
    
    def load_data(self):
        """
        Load prediction and actual data from files.
        """
        print(f"Loading prediction data from {self.predictions_file}...")
        self.predictions = pd.read_csv(self.predictions_file)
        self.predictions['date'] = pd.to_datetime(self.predictions['date'])
        
        print(f"Loading actual data from {self.actual_data_file}...")
        self.actual_data = pd.read_csv(self.actual_data_file)
        
        # Handle date column which might be in different formats
        if 'Unnamed: 0' in self.actual_data.columns:
            self.actual_data['date'] = pd.to_datetime(self.actual_data['Unnamed: 0'])
            self.actual_data = self.actual_data.drop('Unnamed: 0', axis=1)
        else:
            first_col = self.actual_data.columns[0]
            self.actual_data['date'] = pd.to_datetime(self.actual_data[first_col])
            self.actual_data = self.actual_data.drop(first_col, axis=1)
        
        # Debug information
        print(f"Predictions columns: {self.predictions.columns.tolist()}")
        print(f"Actual data columns: {self.actual_data.columns.tolist()}")
        print(f"Predictions date range: {self.predictions['date'].min()} to {self.predictions['date'].max()}")
        print(f"Actual data date range: {self.actual_data['date'].min()} to {self.actual_data['date'].max()}")
        
        return self.predictions, self.actual_data
    
    @abstractmethod
    def prepare_evaluation_data(self):
        """
        Prepare evaluation data by matching predictions to actual returns.
        This method should be implemented by subclasses.
        
        Returns:
            DataFrame: Evaluation data with matched predictions and actual returns
        """
        pass
    
    def calculate_metrics(self, eval_df=None):
        """
        Calculate performance metrics for the predictions.
        
        Args:
            eval_df (DataFrame, optional): Evaluation data. If None, uses self.eval_df.
            
        Returns:
            tuple: (overall_metrics, instrument_metrics)
        """
        if eval_df is None:
            eval_df = self.eval_df
        
        # Check if the required columns exist
        required_cols = ['predicted_return', 'actual_return']
        missing_cols = [col for col in required_cols if col not in eval_df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns in evaluation data: {missing_cols}")
        
        eval_clean = eval_df.dropna(subset=required_cols)
        
        # Calculate overall metrics
        metrics = {
            'Mean Squared Error': mean_squared_error(eval_clean['actual_return'], eval_clean['predicted_return']),
            'Root Mean Squared Error': np.sqrt(mean_squared_error(eval_clean['actual_return'], eval_clean['predicted_return'])),
            'Mean Absolute Error': mean_absolute_error(eval_clean['actual_return'], eval_clean['predicted_return']),
            'R-squared': r2_score(eval_clean['actual_return'], eval_clean['predicted_return'])
        }
        
        # Calculate correlation
        correlation = eval_clean[['actual_return', 'predicted_return']].corr().iloc[0, 1]
        metrics['Correlation'] = correlation
        
        # Directional accuracy
        correct_direction = (np.sign(eval_clean['predicted_return']) == np.sign(eval_clean['actual_return'])).mean()
        metrics['Directional Accuracy'] = correct_direction
        
        # Confidence-weighted correlation
        if 'confidence' in eval_clean.columns:
            eval_clean['weighted_prediction'] = eval_clean['predicted_return'] * eval_clean['confidence']
            weighted_correlation = eval_clean[['actual_return', 'weighted_prediction']].corr().iloc[0, 1]
            metrics['Weighted Correlation'] = weighted_correlation
        
        # Get instrument column name (could be 'instrument', 'currency_pair', etc.)
        instrument_col = self.get_instrument_column()
        
        # Calculate metrics by instrument
        instrument_metrics = {}
        for instrument in eval_clean[instrument_col].unique():
            inst_data = eval_clean[eval_clean[instrument_col] == instrument]
            instrument_metrics[instrument] = {
                'MSE': mean_squared_error(inst_data['actual_return'], inst_data['predicted_return']),
                'Directional Accuracy': (np.sign(inst_data['predicted_return']) == np.sign(inst_data['actual_return'])).mean(),
                'Correlation': inst_data[['actual_return', 'predicted_return']].corr().iloc[0, 1],
                'Count': len(inst_data)
            }
        
        self.metrics = metrics
        self.instrument_metrics = instrument_metrics
        
        return metrics, instrument_metrics
    
    def get_instrument_column(self):
        """
        Determine which column contains the instrument/asset information.
        
        Returns:
            str: Name of the instrument column
        """
        instrument_columns = ['instrument', 'currency_pair', 'etf', 'asset']
        for col in instrument_columns:
            if col in self.eval_df.columns:
                return col
        
        # Default to first string column that's not 'date'
        for col in self.eval_df.columns:
            if col != 'date' and self.eval_df[col].dtype == 'object':
                return col
        
        raise ValueError("Could not determine instrument column in evaluation data")
    
    def calculate_trading_performance(self):
        """
        Calculate performance metrics for trading strategies.
        
        Returns:
            tuple: (strategy_performance, instrument_strategy_perf, strategy_df)
        """
        # Filter out any rows with missing data
        strategy_df = self.eval_df.copy().dropna(subset=['predicted_return', 'actual_return'])
        
        # Simple directional strategy: long/short based on predicted direction
        strategy_df['dir_strategy_return'] = np.sign(strategy_df['predicted_return']) * strategy_df['actual_return']
        
        # Confidence-weighted strategy
        strategy_df['conf_strategy_return'] = np.sign(strategy_df['predicted_return']) * strategy_df['confidence'] * strategy_df['actual_return']
        
        # Size-proportional strategy (scale position by prediction magnitude)
        norm_factor = strategy_df['predicted_return'].abs().mean()
        if norm_factor > 0:
            strategy_df['size_strategy_return'] = (strategy_df['predicted_return'] / norm_factor) * strategy_df['actual_return']
        else:
            strategy_df['size_strategy_return'] = strategy_df['dir_strategy_return']  # Fallback if normalization fails
        
        # Calculate strategy performance metrics
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
        
        # Calculate performance by instrument
        instrument_col = self.get_instrument_column()
        instrument_strategy_perf = {}
        
        for instrument in strategy_df[instrument_col].unique():
            inst_data = strategy_df[strategy_df[instrument_col] == instrument]
            instrument_strategy_perf[instrument] = {}
            inst_baseline = inst_data['actual_return'].mean()
            
            for strat in strategies:
                strat_returns = inst_data[strat]
                instrument_strategy_perf[instrument][strat] = {
                    'Mean Return': strat_returns.mean(),
                    'Cumulative Return': (1 + strat_returns).prod() - 1,
                    'vs Baseline': strat_returns.mean() - inst_baseline
                }
        
        # Print trading strategy results
        print("\n==== TRADING STRATEGY PERFORMANCE ====")
        for strategy, perf in strategy_performance.items():
            strat_name = strategy.replace('_strategy_return', '')
            print(f"\n{strat_name.capitalize()} Strategy:")
            for metric, value in perf.items():
                print(f"  {metric}: {value:.4f}")
        
        self.strategy_df = strategy_df
        self.strategy_performance = strategy_performance
        self.instrument_strategy_perf = instrument_strategy_perf
        
        return strategy_performance, instrument_strategy_perf, strategy_df
    
    def create_scatter_plot(self):
        """
        Create a scatter plot of predicted vs actual returns.
        """
        instrument_col = self.get_instrument_column()
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='predicted_return', y='actual_return',
                        hue=instrument_col, size='confidence',
                        data=self.eval_df.dropna(subset=['predicted_return', 'actual_return']))
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
        plt.title('Predicted vs Actual Returns')
        plt.xlabel('Predicted Return')
        plt.ylabel('Actual Return')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'predicted_vs_actual.png'))
    
    def create_confidence_accuracy_plot(self):
        """
        Create a visualization of confidence vs. directional accuracy.
        """
        eval_clean = self.eval_df.dropna(subset=['predicted_return', 'actual_return'])
        unique_confidence = eval_clean['confidence'].nunique()
        
        try:
            # Try to create confidence bins, handling case of few unique values
            if unique_confidence >= 5:
                bins = pd.qcut(eval_clean['confidence'], 5, duplicates='drop')
                eval_clean['confidence_bin'] = bins
            else:
                if unique_confidence <= 1:
                    raise ValueError("Not enough unique confidence values for binning")
                else:
                    bins = pd.cut(eval_clean['confidence'], bins=min(unique_confidence, 5))
                    eval_clean['confidence_bin'] = bins
            
            # Calculate directional accuracy by confidence bin
            conf_accuracy = eval_clean.groupby('confidence_bin').apply(
                lambda x: (np.sign(x['predicted_return']) == np.sign(x['actual_return'])).mean()
            ).reset_index()
            
            # Create labels for bins
            conf_accuracy['bin_label'] = [str(b) for b in conf_accuracy['confidence_bin']]
            
            # Plot
            plt.figure(figsize=(12, 6))
            plt.bar(conf_accuracy['bin_label'], conf_accuracy[0])
            plt.title('Directional Accuracy by Confidence Level')
            plt.xlabel('Confidence Range')
            plt.ylabel('Directional Accuracy')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'accuracy_by_confidence.png'))
            
        except Exception as e:
            print(f"Error creating confidence bin visualization: {e}")
            # Alternative visualization
            plt.figure(figsize=(10, 6))
            correct_direction = np.sign(eval_clean['predicted_return']) == np.sign(eval_clean['actual_return'])
            plt.scatter(eval_clean['confidence'], correct_direction, alpha=0.5)
            plt.title('Confidence vs Prediction Accuracy')
            plt.xlabel('Confidence Value')
            plt.ylabel('Correct Direction (1=Yes, 0=No)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'confidence_vs_accuracy.png'))
    
    def create_strategy_comparison_plot(self):
        """
        Create a plot comparing the performance of different trading strategies.
        """
        if self.strategy_df is None:
            print("Strategy dataframe not found. Run calculate_trading_performance() first.")
            return
            
        strategy_cols = ['dir_strategy_return', 'conf_strategy_return', 'size_strategy_return']
        cumulative_returns = (1 + self.strategy_df[strategy_cols]).cumprod() - 1
        
        plt.figure(figsize=(12, 8))
        cumulative_returns.plot()
        plt.title('Cumulative Returns by Trading Strategy')
        plt.xlabel('Observation')
        plt.ylabel('Cumulative Return')
        plt.legend(['Directional', 'Confidence-Weighted', 'Size-Proportional'])
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'strategy_comparison.png'))
    
    def create_visualizations(self):
        """
        Create visualizations to analyze prediction performance.
        This method can be overridden by subclasses to add more specific visualizations.
        """
        if self.eval_df is None or self.strategy_df is None:
            print("Evaluation data or strategy dataframe not available. Run prepare_evaluation_data() and calculate_trading_performance() first.")
            return
        
        # 1. Scatter plot of predicted vs actual returns
        self.create_scatter_plot()
        
        # 2. Confidence vs accuracy plot
        self.create_confidence_accuracy_plot()
        
        # 3. Strategy comparison
        self.create_strategy_comparison_plot()
    
    def print_metrics(self):
        """
        Print evaluation metrics to console.
        """
        if not self.metrics or not self.instrument_metrics:
            print("Metrics not calculated yet. Run calculate_metrics() first.")
            return
        
        print("\n==== OVERALL PREDICTION METRICS ====")
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\n==== METRICS BY INSTRUMENT ====")
        for instrument, metrics in self.instrument_metrics.items():
            print(f"\n{instrument}:")
            for metric, value in metrics.items():
                if metric != 'Count':
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
    
    def save_results(self):
        """
        Save evaluation results to CSV.
        """
        if self.eval_df is None:
            print("No evaluation data to save.")
            return
        
        output_file = os.path.join(self.output_dir, "prediction_evaluation.csv")
        self.eval_df.to_csv(output_file, index=False)
        print(f"\nEvaluation data saved to '{output_file}'")
        print(f"Visualizations saved to '{self.output_dir}' directory")
    
    def evaluate(self):
        """
        Run the complete evaluation process.
        """
        # Load data
        self.load_data()
        
        # Prepare evaluation data
        print("Preparing evaluation data...")
        self.eval_df = self.prepare_evaluation_data()
        
        # If no evaluation data, exit
        if self.eval_df.empty:
            print("No matching evaluation data found. Exiting.")
            return
        
        # Calculate metrics
        print("Calculating performance metrics...")
        self.calculate_metrics()
        
        # Calculate trading performance
        print("Calculating trading strategy performance...")
        self.calculate_trading_performance()
        
        # Create visualizations
        print("Creating visualizations...")
        self.create_visualizations()
        
        # Print metrics
        self.print_metrics()
        
        # Save results
        self.save_results()
