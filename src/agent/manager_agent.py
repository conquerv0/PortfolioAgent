import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

class PortfolioManager:
    def __init__(self, initial_weights: Dict[str, float] = None):
        """
        Initialize the PortfolioManager with initial weights for each asset class.
        
        Args:
            initial_weights: Dictionary mapping asset class names to their initial weights.
                           If None, equal weights (0.25) will be used for all asset classes.
        """
        self.asset_classes = ['commodity', 'equity', 'fx', 'fi']
        self.initial_weights = initial_weights or {asset: 0.25 for asset in self.asset_classes}
        self.current_weights = self.initial_weights.copy()
        self.performance_history = {asset: [] for asset in self.asset_classes}
        self.weight_history = {asset: [self.initial_weights[asset]] for asset in self.asset_classes}
        self.dates = []
        
    def update_weights(self, returns_data: Dict[str, pd.DataFrame], lookback_period: int = 1):
        """
        Update portfolio weights based on performance comparison with equal-weighted strategy.
        Increase weight if BL return > equal-weighted return, decrease otherwise.
        
        Args:
            returns_data: Dictionary mapping asset class names to their returns DataFrames
            lookback_period: Number of months to look back for performance calculation
        """
        # Calculate relative performance for each asset class
        performances = {}
        for asset, df in returns_data.items():
            # Get the last 'lookback_period' months of returns
            recent_bl_returns = df['bl_portfolio_return'].tail(lookback_period)
            recent_eq_returns = df['equal_weighted_return'].tail(lookback_period)
            
            # Calculate cumulative performance difference
            bl_perf = (1 + recent_bl_returns).prod() - 1
            eq_perf = (1 + recent_eq_returns).prod() - 1
            relative_perf = bl_perf - eq_perf
            performances[asset] = relative_perf
            
        # Adjust weights based on performance vs equal-weighted
        new_weights = {}
        for asset in self.asset_classes:
            if asset in performances:
                # If BL outperforms equal-weighted, increase weight by 20%
                if performances[asset] > 0:
                    new_weights[asset] = min(self.current_weights[asset] * 1.2, 0.4)
                # If BL underperforms equal-weighted, decrease weight by 20%
                else:
                    new_weights[asset] = max(self.current_weights[asset] * 0.8, 0.1)
            else:
                new_weights[asset] = self.current_weights[asset]
        
        # Normalize weights to sum to 1
        total_weight = sum(new_weights.values())
        self.current_weights = {asset: weight/total_weight for asset, weight in new_weights.items()}
        
        # Store performance history and weights
        for asset in self.asset_classes:
            if asset in performances:
                self.performance_history[asset].append(performances[asset])
                self.weight_history[asset].append(self.current_weights[asset])
            else:
                self.performance_history[asset].append(0)
                self.weight_history[asset].append(self.current_weights[asset])
                
    def get_current_weights(self) -> Dict[str, float]:
        """Return the current portfolio weights."""
        return self.current_weights
    
    def get_performance_history(self) -> Dict[str, List[float]]:
        """Return the performance history for each asset class."""
        return self.performance_history
    
    def get_weight_history(self) -> Dict[str, List[float]]:
        """Return the weight history for each asset class."""
        return self.weight_history

def load_portfolio_returns(asset_classes: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load portfolio returns data for each asset class.
    
    Args:
        asset_classes: List of asset class names to load data for
        
    Returns:
        Dictionary mapping asset class names to their returns DataFrames
    """
    returns_data = {}
    for asset in asset_classes:
        file_path = f'data/evaluation/{asset}/{asset}_portfolio_returns_backtest.csv'
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            returns_data[asset] = df
        except FileNotFoundError:
            print(f"Warning: Could not find returns data for {asset}")
    return returns_data

def calculate_performance_metrics(returns_series, weights_history=None, risk_free_rate=0.03):
    """
    Calculate performance metrics from a return series.
    
    Args:
        returns_series: Series or array of returns
        weights_history: List of portfolio weights over time (for turnover calculation)
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary of metrics including:
        - Annualized Return
        - Annualized Standard Deviation
        - Sharpe Ratio
        - Maximum Drawdown
        - Average Turnover (if weights_history is provided)
    """
    total_return = (1 + returns_series).prod() - 1
    n_periods = len(returns_series)
    annual_return = (1 + total_return) ** (52 / n_periods) - 1  # Assuming weekly returns
    annual_std = returns_series.std() * np.sqrt(52)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_std if annual_std > 0 else np.nan
    
    cum_returns = (1 + returns_series).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    metrics = {
        'Annualized Return': annual_return,
        'Annualized Std Dev': annual_std,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }
    
    if weights_history is not None and len(weights_history) > 1:
        turnovers = []
        for i in range(1, len(weights_history)):
            prev = weights_history[i-1]
            current = weights_history[i]
            turnover = np.sum(np.abs(current - prev))
            turnovers.append(turnover)
        avg_turnover = np.mean(turnovers)
        metrics['Average Turnover'] = avg_turnover
    else:
        metrics['Average Turnover'] = np.nan
    
    return metrics

def plot_portfolio_comparison(manager: PortfolioManager, returns_data: Dict[str, pd.DataFrame], dates_list):
    """
    Plot comparison between managed portfolio and equal-weighted portfolio.
    """
    plt.figure(figsize=(12, 6))
    
    # Find common date range
    all_dates = []
    for df in returns_data.values():
        all_dates.extend(df['date'].tolist())
    common_dates = sorted(list(set(all_dates)))
    
    # Create a DataFrame to store aligned returns
    aligned_returns = pd.DataFrame(index=common_dates)
    aligned_returns.index = pd.DatetimeIndex(aligned_returns.index)
    
    # Calculate managed portfolio returns
    for asset, df in returns_data.items():
        df = df.set_index('date')
        # Store BL returns
        aligned_returns[f'{asset}_bl_return'] = df['bl_portfolio_return']
    
    # Calculate managed portfolio returns with time-varying weights
    managed_returns = pd.Series(0.0, index=pd.DatetimeIndex(dates_list))
    equal_weighted_returns = pd.Series(0.0, index=pd.DatetimeIndex(dates_list))
    
    # Convert weight history to DataFrame for easier indexing
    weight_history = {}
    for asset in manager.asset_classes:
        weight_history[asset] = manager.weight_history[asset][1:]  # Skip initial weight
    
    # Calculate returns for each date using the corresponding weights
    for i, date in enumerate(dates_list):
        if date in aligned_returns.index:
            for asset in manager.asset_classes:
                if f'{asset}_bl_return' in aligned_returns.columns and i < len(weight_history[asset]):
                    # Use the weight for this specific period
                    weight = weight_history[asset][i]
                    bl_return = aligned_returns.loc[date, f'{asset}_bl_return']
                    
                    if not pd.isna(bl_return):
                        managed_returns[date] += bl_return * weight
                        # Use the same BL returns but with equal weights (1/4 each)
                        equal_weighted_returns[date] += bl_return * 0.25
    
    # Calculate cumulative returns
    cumulative_managed = (1 + managed_returns).cumprod() - 1
    cumulative_equal = (1 + equal_weighted_returns).cumprod() - 1
    
    # Plot both portfolios
    plt.plot(dates_list, cumulative_managed, label='Dynamic Weight Portfolio', linewidth=2, color='blue')
    plt.plot(dates_list, cumulative_equal, label='Equal-Weighted Portfolio', linewidth=2, color='red', linestyle='--')
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Portfolio Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('data/evaluation/manager', exist_ok=True)
    plt.savefig('data/evaluation/manager/manager_agent_portfolio_comparison.png')
    
    # Calculate final performance metrics
    final_managed_return = cumulative_managed.iloc[-1]
    final_equal_return = cumulative_equal.iloc[-1]
    print(f"\nFinal Cumulative Returns:")
    print(f"Dynamic Weight Portfolio: {final_managed_return:.2%}")
    print(f"Equal-Weighted Portfolio: {final_equal_return:.2%}")
    print(f"Outperformance: {(final_managed_return - final_equal_return):.2%}")
    
    # Calculate and display performance metrics
    print("\nPerformance Metrics:")
    # Convert weight_history to the format needed for turnover calculation
    flattened_weights = []
    for i in range(len(dates_list)):
        weights = []
        for asset in manager.asset_classes:
            if i < len(weight_history[asset]):
                weights.append(weight_history[asset][i])
        if weights:
            flattened_weights.append(np.array(weights))
    
    # Create equal weights history - constant 0.25 for each asset
    equal_flattened_weights = []
    for i in range(len(dates_list)):
        equal_flattened_weights.append(np.array([0.25] * len(manager.asset_classes)))
    
    managed_metrics = calculate_performance_metrics(managed_returns, flattened_weights)
    equal_metrics = calculate_performance_metrics(equal_weighted_returns, equal_flattened_weights)
    
    print("\nDynamic Weight Portfolio:")
    for key, value in managed_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nEqual-Weighted Portfolio:")
    for key, value in equal_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return managed_returns, equal_weighted_returns, managed_metrics, equal_metrics

def plot_weight_evolution(manager: PortfolioManager, dates_list):
    """
    Plot the evolution of portfolio weights over time.
    """
    plt.figure(figsize=(12, 6))
    
    weight_history = manager.get_weight_history()
    
    # Skip the initial weights (which are just the starting weights)
    for asset in manager.asset_classes:
        weights = weight_history[asset][1:]  # Skip first entry (initial weight)
        plt.plot(dates_list, weights, label=asset, linewidth=2)
    
    plt.xlabel('Date')
    plt.ylabel('Portfolio Weight')
    plt.title('Portfolio Weight Evolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('data/evaluation/manager', exist_ok=True)
    plt.savefig('data/evaluation/manager/manager_agent_weight_evolution.png')

def simulate_rolling_portfolio_management(asset_classes: List[str], lookback_period: int = 1) -> Tuple[PortfolioManager, List[datetime]]:
    """
    Simulate rolling portfolio management by updating weights at each date point.
    
    Args:
        asset_classes: List of asset classes to include
        lookback_period: Number of periods to look back for performance evaluation
        
    Returns:
        manager: Trained PortfolioManager
        common_dates: List of dates used in simulation
    """
    # Initialize portfolio manager
    manager = PortfolioManager()
    
    # Load portfolio returns data
    returns_data = load_portfolio_returns(asset_classes)
    
    # Find common dates across all asset classes
    common_dates = []
    for asset, df in returns_data.items():
        if len(common_dates) == 0:
            common_dates = df['date'].tolist()
        else:
            common_dates = [date for date in common_dates if date in df['date'].tolist()]
    
    common_dates = sorted(common_dates)
    
    # Start with lookback_period to have enough data
    for i in range(lookback_period, len(common_dates)):
        current_date = common_dates[i]
        
        # Create a subset of data up to the current date
        current_data = {}
        for asset, df in returns_data.items():
            current_data[asset] = df[df['date'] <= current_date].copy()
        
        # Update weights based on data up to current date
        manager.update_weights(current_data, lookback_period)
        
    return manager, common_dates[lookback_period:]

def save_weight_history_to_df(manager: PortfolioManager, dates_list):
    """
    Convert the weight history to a DataFrame and save it to a CSV file.
    
    Args:
        manager: PortfolioManager instance with weight history
        dates_list: List of dates corresponding to weight changes
    
    Returns:
        DataFrame containing the weight history
    """
    # Create a DataFrame with dates as index
    weight_df = pd.DataFrame(index=dates_list)
    
    # Skip the initial weights (which are just the starting weights)
    weight_history = manager.get_weight_history()
    for asset in manager.asset_classes:
        weights = weight_history[asset][1:]  # Skip first entry (initial weight)
        if len(weights) == len(dates_list):
            weight_df[asset] = weights
        else:
            # Handle cases where lengths don't match
            print(f"Warning: Number of weights for {asset} ({len(weights)}) doesn't match number of dates ({len(dates_list)})")
            # Pad with NaN if needed
            padded_weights = weights + [np.nan] * (len(dates_list) - len(weights))
            weight_df[asset] = padded_weights[:len(dates_list)]
    
    # Save to CSV
    os.makedirs('data/evaluation/manager', exist_ok=True)
    weight_df.to_csv('data/evaluation/manager/manager_agent_weight_history_monthly.csv')
    
    # Also create a monthly resampled version
    weight_df.index = pd.DatetimeIndex(weight_df.index)
    monthly_weights = weight_df.resample('MS').first()  # MS = Month Start
    monthly_weights.to_csv('data/evaluation/manager/manager_agent_weight_history_monthly_resampled.csv')
    
    return weight_df, monthly_weights

def main():
    # Simulate rolling portfolio management
    asset_classes = ['commodity', 'equity', 'fx', 'fi']
    manager, dates_list = simulate_rolling_portfolio_management(asset_classes, lookback_period=1)
    
    # Load complete portfolio data for visualization
    returns_data = load_portfolio_returns(asset_classes)
    
    # Print current weights
    print("\nFinal Portfolio Weights:")
    for asset, weight in manager.get_current_weights().items():
        print(f"{asset}: {weight:.2%}")
    
    # Plot performance charts and calculate metrics
    managed_returns, equal_returns, managed_metrics, equal_metrics = plot_portfolio_comparison(manager, returns_data, dates_list)
    plot_weight_evolution(manager, dates_list)
    
    # Save weight history to DataFrame
    weight_df, monthly_weights = save_weight_history_to_df(manager, dates_list)
    
    # Print the monthly weights
    print("\nMonthly Portfolio Weights:")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.precision', 4)
    print(monthly_weights)
    
    # Create a DataFrame to store and save metrics
    metrics_df = pd.DataFrame({
        'Metric': list(managed_metrics.keys()),
        'Dynamic Portfolio': list(managed_metrics.values()),
        'Equal-Weighted Portfolio': list(equal_metrics.values())
    })
    
    os.makedirs('data/evaluation/manager', exist_ok=True)
    metrics_df.to_csv('data/evaluation/manager/manager_agent_performance_metrics.csv', index=False)

if __name__ == "__main__":
    main()
