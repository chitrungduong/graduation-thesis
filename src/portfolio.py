"""
portfolio.py - Portfolio Optimization

Mean-variance optimization with Ledoit-Wolf covariance shrinkage.
Implements tangency portfolio and equal-weight portfolio construction.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import MAX_WEIGHT


def ledoit_wolf_shrinkage(returns):
    """Estimate covariance matrix using Ledoit-Wolf shrinkage.

    Args:
        returns: Stock returns DataFrame or array (T x N)

    Returns:
        shrunk_cov: Shrinkage covariance matrix (N x N)
        delta: Shrinkage intensity [0, 1]
    """
    returns_array = returns.values if hasattr(returns, 'values') else returns
    T, N = returns_array.shape

    # Sample covariance matrix
    sample_cov = np.cov(returns_array, rowvar=False, ddof=1)

    # Shrinkage target: constant correlation matrix
    prior = np.trace(sample_cov) / N * np.eye(N)

    # Center the data
    mu = returns_array.mean(axis=0, keepdims=True)
    X_centered = returns_array - mu

    # Calculate optimal shrinkage intensity
    X2 = X_centered ** 2
    sample_var = (X_centered.T @ X_centered) / T

    phi_mat = (X2.T @ X2) / T - sample_var ** 2
    phi = np.sum(phi_mat)

    gamma = np.linalg.norm(sample_cov - prior, 'fro') ** 2

    kappa = phi / gamma
    delta = max(0, min(1, kappa / T))

    # Shrinkage combination
    shrunk_cov = delta * prior + (1 - delta) * sample_cov

    return shrunk_cov, delta


def calculate_tangency_portfolio(returns, rf, use_shrinkage=True, max_weight=MAX_WEIGHT):
    """Calculate tangency portfolio (maximum Sharpe ratio).

    Args:
        returns: Training returns DataFrame (T x N)
        rf: Annual risk-free rate
        use_shrinkage: Whether to use Ledoit-Wolf shrinkage
        max_weight: Maximum weight per asset

    Returns:
        weights: Optimal portfolio weights (N,)
        shrinkage: Shrinkage intensity (0 if use_shrinkage=False)
    """
    # Expected returns (annualized)
    mu = returns.mean() * 12

    # Covariance matrix (annualized)
    if use_shrinkage:
        cov_monthly, shrinkage = ledoit_wolf_shrinkage(returns)
        cov = cov_monthly * 12
    else:
        cov = returns.cov() * 12
        shrinkage = 0.0

    n_assets = len(mu)

    # Objective: negative Sharpe ratio (for minimization)
    def neg_sharpe(w):
        port_return = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        return -(port_return - rf) / (port_vol + 1e-10)

    # Constraints and bounds
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, max_weight) for _ in range(n_assets)]

    # Initial guess: equal weights
    w0 = np.ones(n_assets) / n_assets

    # Optimize
    result = minimize(
        neg_sharpe,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )

    # Process results
    if result.success:
        weights = result.x
        weights[weights < 1e-4] = 0
        weights = weights / weights.sum()
    else:
        # Fallback to equal weights
        weights = np.ones(n_assets) / n_assets
        weights = np.minimum(weights, max_weight)
        weights = weights / weights.sum()

    return weights, shrinkage


def create_equal_weight_portfolio(returns):
    """Create equal-weight portfolio.

    Args:
        returns: Returns DataFrame (T x N)

    Returns:
        weights: Equal weights (N,)
    """
    n_assets = returns.shape[1]
    weights = np.ones(n_assets) / n_assets
    return weights


def calculate_portfolio_returns(returns, weights):
    """Calculate portfolio returns given weights.

    Args:
        returns: Returns DataFrame (T x N)
        weights: Portfolio weights (N,)

    Returns:
        portfolio_returns: Portfolio returns series (T,)
    """
    portfolio_returns = (returns * weights).sum(axis=1)
    return portfolio_returns
