"""
explainability.py - Model Interpretability

SHAP (SHapley Additive exPlanations) analysis for neural network models.
Calculates and aggregates feature importance across stocks.
"""

import warnings

import numpy as np
import shap


def calculate_shap_values(model, scaler, X, feature_names):
    """
    Calculate SHAP values for neural network model.

    Args:
        model: Trained MLPRegressor
        scaler: Fitted StandardScaler
        X: Feature matrix
        feature_names: List of feature names

    Returns:
        Dictionary with shap_values, expected_value, feature_importance
    """
    X_scaled = scaler.transform(X)

    explainer = shap.KernelExplainer(model.predict, X_scaled[:100])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(X_scaled, silent=True)

    feature_importance = {}
    for i, name in enumerate(feature_names):
        feature_importance[name] = np.mean(np.abs(shap_values[:, i]))

    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    return {
        'shap_values': shap_values,
        'expected_value': explainer.expected_value,
        'feature_importance': feature_importance
    }


def aggregate_stock_importance(trained_models, top_k_stocks, feature_names):
    """
    Aggregate feature importance across selected stocks.

    Args:
        trained_models: Dictionary of trained models for each stock
        top_k_stocks: List of selected stock tickers
        feature_names: List of feature names

    Returns:
        Dictionary with shap_aggregated and stock_level_shap
    """
    all_shap = {feat: [] for feat in feature_names}
    stock_level_shap = {}

    for stock in top_k_stocks:
        if stock not in trained_models:
            continue

        model_data = trained_models[stock]
        model = model_data['model']
        scaler = model_data['scaler']
        X = model_data['X']

        try:
            result = calculate_shap_values(model, scaler, X, feature_names)
            shap_imp = result['feature_importance']
            stock_level_shap[stock] = shap_imp

            for feat in feature_names:
                all_shap[feat].append(shap_imp[feat])

        except Exception as e:
            print(f"Warning: Could not calculate importance for {stock}: {e}")
            continue

    shap_aggregated = {feat: np.mean(vals) for feat, vals in all_shap.items() if len(vals) > 0}
    shap_aggregated = dict(sorted(shap_aggregated.items(), key=lambda x: x[1], reverse=True))

    return {
        'shap_aggregated': shap_aggregated,
        'stock_level_shap': stock_level_shap
    }
