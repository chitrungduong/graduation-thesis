"""
features.py - Feature Engineering

Calculate technical indicators from monthly stock returns.
All features are lagged by 1 month to prevent look-ahead bias.
"""

import numpy as np
import pandas as pd

from config import (
    MOMENTUM_3M,
    MOMENTUM_6M,
    VOLATILITY_WINDOW,
    MARKET_CORR_WINDOW,
    DOWNSIDE_WINDOW,
    BETA_WINDOW,
    FEATURE_LAG
)


def calculate_features(monthly_returns, stock):
    """Calculate technical features for one stock.

    Args:
        monthly_returns: Monthly returns DataFrame (T x N)
        stock: Stock ticker symbol

    Returns:
        features: DataFrame with 6 features (T x 6)
    """
    stock_returns = monthly_returns[stock]
    market_returns = monthly_returns['VN30']

    # Initialize features DataFrame
    features = pd.DataFrame(index=stock_returns.index)

    # F1: 3-month momentum
    features['momentum_3m'] = stock_returns.rolling(MOMENTUM_3M).mean()

    # F2: 6-month momentum
    features['momentum_6m'] = stock_returns.rolling(MOMENTUM_6M).mean()

    # F3: 6-month volatility
    features['volatility_6m'] = stock_returns.rolling(VOLATILITY_WINDOW).std()

    # F4: Market correlation (rolling 12-month)
    features['market_corr'] = stock_returns.rolling(MARKET_CORR_WINDOW).corr(market_returns)

    # F5: Downside deviation (6-month)
    negative_returns = stock_returns[stock_returns < 0]
    features['downside_dev'] = negative_returns.rolling(
        DOWNSIDE_WINDOW,
        min_periods=1
    ).std()
    features['downside_dev'] = features['downside_dev'].fillna(
        stock_returns.rolling(DOWNSIDE_WINDOW).std()
    )

    # F6: Beta (CAPM systematic risk)
    cov = stock_returns.rolling(BETA_WINDOW).cov(market_returns)
    var = market_returns.rolling(BETA_WINDOW).var()
    features['beta'] = cov / (var + 1e-10)

    # Lag all features by 1 month to prevent look-ahead bias
    features = features.shift(FEATURE_LAG)

    return features


def create_feature_matrix(monthly_returns, train_stocks):
    """Create feature matrix for all stocks.

    Args:
        monthly_returns: Monthly returns DataFrame (T x N)
        train_stocks: List of stock tickers

    Returns:
        features_dict: Dictionary mapping stock -> features DataFrame
    """
    features_dict = {}

    for stock in train_stocks:
        features = calculate_features(monthly_returns, stock)
        features_dict[stock] = features

    return features_dict
