import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import os

class ManagerAgent:
    """
    An agent that dynamically assigns portfolio weights based on monthly returns from predictions.
    This agent implements various portfolio optimization strategies and can adapt weights based on
    predicted returns and risk metrics.
    """
    
    def __init__(self, risk_aversion: float = 1.0, min_weight: float = 0.0, max_weight: float = 1.0):
        """
        Initialize the ManagerAgent.
        
        Args:
            risk_aversion: Risk aversion parameter for mean-variance optimization
            min_weight: Minimum allowed weight for any asset
            max_weight: Maximum allowed weight for any asset
        """
        self.risk_aversion = risk_aversion
        self.min_weight = min_weight
        self.max_weight = max_weight
        
    def load_predictions(self, predictions_file: str) -> pd.DataFrame:
        """
        Load and process predictions from file.
        
        Args:
            predictions_file: Path to the predictions CSV file
            
        Returns:
            DataFrame containing processed predictions
        """
        if not os.path.exists(predictions_file):
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
            
        predictions = pd.read_csv(predictions_file)
        predictions['date'] = pd.to_datetime(predictions['date'])
        return predictions
        
    def compute_monthly_returns(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Compute monthly returns from predictions.
        
        Args:
            predictions: DataFrame containing predictions
            
        Returns:
            DataFrame with monthly returns
        """
        # Group by month and compute mean returns
        monthly_returns = predictions.groupby([
            predictions['date'].dt.year,
            predictions['date'].dt.month,
            'etf'
        ])['predicted_return'].mean().reset_index()
        
        # Create proper date column
        monthly_returns['date'] = pd.to_datetime(
            monthly_returns[['year', 'month']].assign(day=1)
        )
        
        return monthly_returns
        
    def compute_covariance_matrix(self, monthly_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute covariance matrix from monthly returns.
        
        Args:
            monthly_returns: DataFrame containing monthly returns
            
        Returns:
            Covariance matrix
        """
        # Pivot to get returns matrix
        returns_matrix = monthly_returns.pivot(
            index='date',
            columns='etf',
            values='predicted_return'
        )
        
        # Compute covariance matrix
        return returns_matrix.cov()
        
    def mean_variance_optimization(self, 
                                 expected_returns: pd.Series,
                                 covariance_matrix: pd.DataFrame) -> pd.Series:
        """
        Perform mean-variance optimization to get optimal weights.
        
        Args:
            expected_returns: Series of expected returns
            covariance_matrix: Covariance matrix of returns
            
        Returns:
            Series of optimal weights
        """
        n_assets = len(expected_returns)
        
        # Convert to numpy arrays
        mu = expected_returns.values.reshape(-1, 1)
        Sigma = covariance_matrix.values
        
        # Add small regularization to ensure positive definiteness
        Sigma = Sigma + 1e-6 * np.eye(n_assets)
        
        # Compute optimal weights using mean-variance optimization
        Sigma_inv = np.linalg.inv(Sigma)
        ones = np.ones((n_assets, 1))
        
        # Compute optimal weights
        w = (1 / self.risk_aversion) * Sigma_inv @ mu
        
        # Normalize weights to sum to 1
        w = w / np.sum(w)
        
        # Apply weight constraints
        w = np.clip(w, self.min_weight, self.max_weight)
        w = w / np.sum(w)  # Renormalize after clipping
        
        # Convert to Series
        weights = pd.Series(w.flatten(), index=expected_returns.index)
        
        return weights
        
    def assign_weights(self, predictions_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main method to assign weights based on predictions.
        
        Args:
            predictions_file: Path to the predictions CSV file
            
        Returns:
            Tuple of (monthly_weights, monthly_returns)
        """
        # Load and process predictions
        predictions = self.load_predictions(predictions_file)
        
        # Compute monthly returns
        monthly_returns = self.compute_monthly_returns(predictions)
        
        # Compute covariance matrix
        covariance_matrix = self.compute_covariance_matrix(monthly_returns)
        
        # Get unique dates and assets
        dates = monthly_returns['date'].unique()
        assets = monthly_returns['etf'].unique()
        
        # Initialize weights DataFrame
        weights = pd.DataFrame(index=dates, columns=assets)
        
        # Assign weights for each month
        for date in dates:
            # Get expected returns for this month
            month_returns = monthly_returns[month_returns['date'] == date]
            expected_returns = month_returns.set_index('etf')['predicted_return']
            
            # Perform optimization
            month_weights = self.mean_variance_optimization(
                expected_returns,
                covariance_matrix
            )
            
            # Store weights
            weights.loc[date] = month_weights
            
        return weights, monthly_returns.pivot(
            index='date',
            columns='etf',
            values='predicted_return'
        )
        
    def save_weights(self, weights: pd.DataFrame, output_file: str):
        """
        Save weights to CSV file.
        
        Args:
            weights: DataFrame containing weights
            output_file: Path to save weights
        """
        weights.to_csv(output_file)
        print(f"Weights saved to {output_file}")
