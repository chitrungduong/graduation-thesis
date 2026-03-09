"""
ann_model.py - Neural Network Training and Stock Screening

Train MLP neural networks with GridSearchCV to predict stock returns.
Select top-K stocks based on cross-validated prediction accuracy.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from config import (
    RANDOM_SEED,
    ANN_MAX_ITER,
    ANN_EARLY_STOPPING,
    ANN_N_ITER_NO_CHANGE,
    ANN_VALIDATION_FRACTION,
    ANN_CV_SPLITS,
    ANN_N_JOBS,
    ANN_HIDDEN_LAYERS,
    ANN_ACTIVATION,
    ANN_ALPHA,
    ANN_LEARNING_RATE
)


def train_ann(X_train, y_train):
    """Train ANN with GridSearchCV for hyperparameter tuning.

    Args:
        X_train: Feature matrix (N x 6)
        y_train: Target returns (N,)

    Returns:
        best_model: Trained MLPRegressor with best hyperparameters
        cv_mse: Cross-validated mean squared error
        scaler: Fitted StandardScaler for features
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Define hyperparameter grid
    param_grid = {
        'hidden_layer_sizes': ANN_HIDDEN_LAYERS,
        'activation': ANN_ACTIVATION,
        'alpha': ANN_ALPHA,
        'learning_rate_init': ANN_LEARNING_RATE
    }

    # Base MLP model
    mlp = MLPRegressor(
        max_iter=ANN_MAX_ITER,
        random_state=RANDOM_SEED,
        early_stopping=ANN_EARLY_STOPPING,
        n_iter_no_change=ANN_N_ITER_NO_CHANGE,
        validation_fraction=ANN_VALIDATION_FRACTION
    )

    # Grid search with time series cross-validation
    grid_search = GridSearchCV(
        mlp,
        param_grid,
        cv=TimeSeriesSplit(n_splits=ANN_CV_SPLITS),
        scoring='neg_mean_squared_error',
        n_jobs=ANN_N_JOBS
    )

    grid_search.fit(X_scaled, y_train)

    # Extract results
    best_model = grid_search.best_estimator_
    cv_mse = -grid_search.best_score_

    return best_model, cv_mse, scaler


def screen_stocks(train_returns, features_dict, top_k):
    """Screen stocks using ANN predictions.

    Trains ANN for each stock and selects top-K stocks based on
    cross-validated mean squared error (lower MSE = better).

    Args:
        train_returns: Training returns DataFrame (T x N)
        features_dict: Dictionary mapping stock -> features DataFrame
        top_k: Number of stocks to select

    Returns:
        selected_stocks: List of top-K stock tickers
        screening_results: Dictionary mapping stock -> CV-MSE score
        trained_models: Dictionary mapping stock -> model info
    """
    results = []
    trained_models = {}

    for stock in train_returns.columns:
        # Skip market index
        if stock == 'VN30':
            continue

        # Check if features exist
        if stock not in features_dict:
            continue

        # Get features and returns
        features = features_dict[stock]
        returns = train_returns[stock]

        # Align data and remove NaN
        valid_idx = features.dropna().index
        X = features.loc[valid_idx].values
        y = returns.loc[valid_idx].values

        # Train ANN
        try:
            model, cv_mse, scaler = train_ann(X, y)

            # Store results (negate MSE so higher is better for sorting)
            results.append({
                'stock': stock,
                'score': -cv_mse,
                'cv_mse': cv_mse
            })

            # Store trained model
            trained_models[stock] = {
                'model': model,
                'scaler': scaler,
                'X': X,
                'y': y
            }

        except Exception as e:
            print(f"  {stock}: Training failed - {e}")
            continue

    # Sort by score (higher is better, i.e., lower MSE)
    results.sort(key=lambda x: x['score'], reverse=True)

    # Select top-K stocks
    selected_stocks = [r['stock'] for r in results[:top_k]]

    # Create screening results dictionary
    screening_results = {r['stock']: r['score'] for r in results}

    return selected_stocks, screening_results, trained_models
