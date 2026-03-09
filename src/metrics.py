"""
metrics.py - Performance Metrics

Calculate portfolio performance statistics including annualized return,
volatility, Sharpe ratio, and maximum drawdown.
"""

import numpy as np
import pandas as pd


def calculate_performance_metrics(returns, rf, annualization_factor=12):
    """Calculate portfolio performance metrics.

    Args:
        returns: Portfolio returns series (T,)
        rf: Annual risk-free rate
        annualization_factor: 12 for monthly data

    Returns:
        metrics: Dictionary with performance statistics
    """
    returns_array = returns.values if hasattr(returns, 'values') else returns
    n_periods = len(returns_array)

    # Annualized return
    total_return = (1 + returns_array).prod() - 1
    annual_return = (1 + total_return) ** (annualization_factor / n_periods) - 1

    # Annualized volatility
    volatility = returns_array.std() * np.sqrt(annualization_factor)

    # Sharpe ratio
    if volatility > 0:
        sharpe_ratio = (annual_return - rf) / volatility
    else:
        sharpe_ratio = 0

    # Maximum drawdown
    cumulative = (1 + returns_array).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }
