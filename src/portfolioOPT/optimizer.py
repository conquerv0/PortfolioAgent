import numpy as np
import pandas as pd
import cvxpy as cp

class PortfolioOptimizerModule:
    def __init__(self, portfolio: dict, expected_returns: pd.Series, cov_matrix: pd.DataFrame):
        """
        Initializes the optimizer with a portfolio and the corresponding market data.

        Parameters:
            portfolio (dict): A dictionary representing the ETF portfolio.
                Example structure:
                    {
                        "treasuries": [
                            {"name": "Short-Term Treasury", "etf": "SHV", "maturity": "0-1yr", "weight": 0.0},
                            {"name": "1-3 Year Treasury", "etf": "SHY", "maturity": "1-3yr", "weight": 0.0},
                            ... (more entries)
                        ]
                    }
            expected_returns (pd.Series): Prior estimate of annual expected returns for each asset (index must match ETF tickers).
            cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
        """
        self.portfolio = portfolio
        self.mu = expected_returns
        self.Sigma = cov_matrix

    def mean_variance_optimization(self, target_return: float = None, allow_short: bool = False) -> pd.Series:
        """
        Solves a mean-variance optimization problem.
        
        If target_return is specified, the optimizer minimizes portfolio variance subject to achieving at least that return.
        Otherwise, it finds the minimum variance portfolio.
        
        Parameters:
            target_return (float, optional): The minimum required portfolio return.
            allow_short (bool): If False, weights are constrained to be non-negative.
        
        Returns:
            pd.Series: Optimal weights indexed by asset ticker.
        """
        n = len(self.mu)
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, self.Sigma))
        constraints = [cp.sum(w) == 1]
        if not allow_short:
            constraints.append(w >= 0)
        if target_return is not None:
            constraints.append(self.mu.values @ w >= target_return)
        prob = cp.Problem(objective, constraints)
        prob.solve()
        if w.value is None:
            raise ValueError("Mean-variance optimization failed to find a solution.")
        weights = pd.Series(w.value, index=self.mu.index)
        return weights

    def black_litterman_allocation(self, view_dict: dict, tau: float = 0.05, delta: float = 2.5) -> pd.Series:
        """
        Implements a basic Black-Litterman allocation.
        
        Given a dictionary of absolute views (e.g. {'AAPL': 0.10, 'MSFT': 0.08}),
        constructs the picking matrix P and views vector Q (assuming one view per asset).
        It then computes the posterior expected returns using the formula:
        
            E(R) = [ (tau*Sigma)^(-1) + P^T Omega^(-1) P ]^(-1) [ (tau*Sigma)^(-1) * pi + P^T Omega^(-1) * Q ]
        
        Finally, optimal weights are calculated as proportional to:
        
            w ∝ (delta * Sigma)^(-1) E(R)
        
        Parameters:
            view_dict (dict): Dictionary of absolute views for some assets.
            tau (float): Scalar tuning parameter.
            delta (float): Risk-aversion parameter.
        
        Returns:
            pd.Series: Optimal weights indexed by asset ticker.
        """
        # Prior returns vector (pi)
        pi = self.mu.values.reshape(-1, 1)
        # Build picking matrix P and views vector Q from view_dict:
        assets_view = list(view_dict.keys())
        Q = np.array([view_dict[a] for a in assets_view]).reshape(-1, 1)
        n = len(self.mu)
        P = np.zeros((len(assets_view), n))
        asset_idx = {asset: i for i, asset in enumerate(self.mu.index)}
        for i, asset in enumerate(assets_view):
            if asset not in asset_idx:
                raise ValueError(f"Asset '{asset}' in view_dict not found in expected_returns index.")
            P[i, asset_idx[asset]] = 1

        # Uncertainty matrix Omega = tau * P * Sigma * P^T
        Sigma_vals = self.Sigma.values
        Omega = tau * (P @ Sigma_vals @ P.T)
        Sigma_inv = np.linalg.inv(tau * Sigma_vals)
        Omega_inv = np.linalg.inv(Omega)
        M = np.linalg.inv(Sigma_inv + P.T @ Omega_inv @ P)
        E_R = M @ (Sigma_inv @ pi + P.T @ Omega_inv @ Q)
        # Compute weights: w ∝ (delta * Sigma)^(-1) E_R
        weights_unscaled = np.linalg.inv(delta * Sigma_vals) @ E_R
        weights = weights_unscaled / np.sum(weights_unscaled)
        weights = pd.Series(weights.flatten(), index=self.mu.index)
        return weights

    def update_portfolio_weights(self, new_weights: pd.Series) -> dict:
        """
        Modifies the current ETF portfolio weights using the provided new weights.
        
        The portfolio (e.g. under the key "treasuries") is assumed to be a list of dicts,
        each with an "etf" key and a "weight" key. This function updates the "weight"
        for each asset found in new_weights.
        
        Parameters:
            new_weights (pd.Series): New weights indexed by asset ticker.
        
        Returns:
            dict: The updated portfolio dictionary.
        """
        # Assume the portfolio structure has a key "treasuries" containing asset dicts.
        for asset in self.portfolio.get("treasuries", []):
            ticker = asset.get("etf")
            if ticker in new_weights:
                asset["weight"] = float(new_weights[ticker])
        return self.portfolio

# Example usage:
if __name__ == "__main__":
    # Example ETF portfolio (Fixed Income, "treasuries")
    fixed_income_portfolio = PORTFOLIO['treasuries']
    # Dummy expected returns and covariance for demonstration.
    tickers = ["SHV", "SHY", "IEI", "IEF", "TLH", "TLT"]
    mu_example = pd.Series([0.04, 0.05, 0.06, 0.055, 0.07, 0.065], index=tickers)
    cov_example = pd.DataFrame(
        np.array([
            [0.001, 0.0002, 0.0003, 0.0002, 0.0001, 0.0002],
            [0.0002, 0.0015, 0.00025, 0.0003, 0.0002, 0.0003],
            [0.0003, 0.00025, 0.002, 0.00035, 0.0003, 0.00025],
            [0.0002, 0.0003, 0.00035, 0.0018, 0.0002, 0.0003],
            [0.0001, 0.0002, 0.0003, 0.0002, 0.0025, 0.0004],
            [0.0002, 0.0003, 0.00025, 0.0003, 0.0004, 0.0022]
        ]),
        index=tickers, columns=tickers
    )

    optimizer_module = PortfolioOptimizerModule(fixed_income_portfolio, mu_example, cov_example)

    # Run mean-variance optimization to generate new weights.
    mv_weights = optimizer_module.mean_variance_optimization(allow_short=False)
    print("Optimized Mean-Variance Weights:")
    print(mv_weights)

    # Alternatively, run Black-Litterman allocation (with a view dict).
    view_dict = {"TLT": 0.08, "TLH": 0.09}  # Example views on two ETFs.
    bl_weights = optimizer_module.black_litterman_allocation(view_dict=view_dict, tau=0.05, delta=2.5)
    print("\nOptimized Black-Litterman Weights:")
    print(bl_weights)

    # Update the ETF portfolio with new weights (choose one set of weights)
    updated_portfolio = optimizer_module.update_portfolio_weights(new_weights=mv_weights)
    print("\nUpdated ETF Portfolio:")
    print(updated_portfolio)
