"""
data_loader.py - Data Loading and Train-Test Split

Functions for loading monthly returns, daily OHLCV data, and performing
temporal train-test splits for walk-forward backtesting.
"""

import os

import pandas as pd


# Get the absolute path to the project root (Lab Project/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_monthly_returns(filepath=None):
    """Load monthly stock returns.

    Args:
        filepath: Path to monthly returns CSV file

    Returns:
        returns: DataFrame with monthly returns (T x N)
    """
    if filepath is None:
        filepath = os.path.join(PROJECT_ROOT, 'data', 'vn30_monthly_returns_2020_2025.csv')
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.index.name = 'date'
    return df


def load_daily_ohlcv(filepath=None):
    """Load daily OHLCV data.

    Args:
        filepath: Path to daily OHLCV CSV file

    Returns:
        df: DataFrame with daily OHLCV data
    """
    if filepath is None:
        filepath = os.path.join(PROJECT_ROOT, 'data', 'vn30_daily_ohlcv_2020_2025_RAW.csv')
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])
    return df


def split_train_test(returns, test_year):
    """Split data by year for walk-forward testing.

    Args:
        returns: Monthly returns DataFrame (T x N)
        test_year: Test year (e.g., 2023, 2024, 2025)

    Returns:
        train_returns: Training data (all years before test_year)
        test_returns: Test data (only test_year)
        n_train: Number of training observations
        n_test: Number of test observations
    """
    train = returns[returns.index.year < test_year]
    test = returns[returns.index.year == test_year]

    return train, test, len(train), len(test)
