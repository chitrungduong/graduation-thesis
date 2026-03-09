import numpy as np

from config import RANDOM_SEED, RISK_FREE_RATE, PORTFOLIO_SIZE, MAX_WEIGHT
from src.data_loader import load_monthly_returns, split_train_test
from src.features import create_feature_matrix
from src.ann_model import screen_stocks
from src.portfolio import calculate_tangency_portfolio, create_equal_weight_portfolio, calculate_portfolio_returns
from src.metrics import calculate_performance_metrics
from src.statistical_tests import block_bootstrap_sharpe_ci, jobson_korkie_test, spa_test
from src.explainability import aggregate_stock_importance

np.random.seed(RANDOM_SEED)


def run_backtest(test_year):
    """
    Run backtest for one year.

    Args:
        test_year: Year to test on

    Returns:
        Dictionary with returns and metrics for all strategies
    """
    print(f"\nBACKTEST YEAR: {test_year}")
    print()

    returns = load_monthly_returns()
    train_returns, test_returns, n_train, n_test = split_train_test(returns, test_year)
    train_stocks = [col for col in train_returns.columns if col != 'VN30']

    print("STEP 1: Stock Screening with ANN")
    features_dict = create_feature_matrix(train_returns, train_stocks)
    selected_stocks, screening_results, trained_models = screen_stocks(
        train_returns[train_stocks], features_dict, PORTFOLIO_SIZE
    )

    all_results = sorted(screening_results.items(), key=lambda x: -x[1])
    top_5 = [f"{stock} ({-score:.4f})" for stock, score in all_results[:5]]
    print(f"  Selected Top {PORTFOLIO_SIZE}/{len(all_results)}: {', '.join(top_5)}...")
    print()

    print("STEP 2: Feature Importance (SHAP)")
    feature_names = ['momentum_3m', 'momentum_6m', 'volatility_6m', 'market_corr', 'downside_dev', 'beta']
    importance_results = aggregate_stock_importance(trained_models, selected_stocks, feature_names)
    shap_imp = importance_results['shap_aggregated']

    top_3_features = list(shap_imp.items())[:3]
    print(f"  Top 3 Features: {', '.join([f'{name} ({val:.4f})' for name, val in top_3_features])}")
    print()

    print("STEP 3: Ledoit-Wolf Covariance Shrinkage")
    rf = RISK_FREE_RATE[test_year]
    selected_train_returns = train_returns[selected_stocks]
    selected_test_returns = test_returns[selected_stocks]

    _, shrinkage = calculate_tangency_portfolio(selected_train_returns, rf, use_shrinkage=True, max_weight=MAX_WEIGHT)
    print(f"  Shrinkage Intensity = {shrinkage:.2f} ({shrinkage*100:.0f}%)")
    print()

    print("STEP 4: Portfolio Construction")

    ew_weights = create_equal_weight_portfolio(selected_test_returns)
    ew_returns = calculate_portfolio_returns(selected_test_returns, ew_weights)
    ew_metrics = calculate_performance_metrics(ew_returns, rf)

    tangency_weights, _ = calculate_tangency_portfolio(selected_train_returns, rf, use_shrinkage=True, max_weight=MAX_WEIGHT)
    tangency_returns = calculate_portfolio_returns(selected_test_returns, tangency_weights)
    tangency_metrics = calculate_performance_metrics(tangency_returns, rf)

    all_train_returns = train_returns[train_stocks]
    all_test_returns = test_returns[train_stocks]
    tangency_all_weights, _ = calculate_tangency_portfolio(all_train_returns, rf, use_shrinkage=False, max_weight=1.0)
    tangency_all_returns = calculate_portfolio_returns(all_test_returns, tangency_all_weights)
    tangency_all_metrics = calculate_performance_metrics(tangency_all_returns, rf)

    active_positions = [(s, w) for s, w in zip(selected_stocks, tangency_weights) if w > 0.001]
    print(f"  Allocation: {', '.join([f'{s} {w*100:.0f}%' for s, w in active_positions])}")
    print(f"  Active Stocks: {len(active_positions)}/{PORTFOLIO_SIZE}  |  Sharpe Ratio: {tangency_metrics['sharpe_ratio']:.2f}")

    return {
        'ew_returns': ew_returns,
        'tangency_returns': tangency_returns,
        'tangency_all_returns': tangency_all_returns,
        'rf': rf,
        'ew_metrics': ew_metrics,
        'tangency_metrics': tangency_metrics,
        'tangency_all_metrics': tangency_all_metrics
    }


def print_summary_table(backtest_results):
    """
    Print year-by-year performance summary.

    Args:
        backtest_results: Dictionary of results by year
    """
    print("\n")
    print("YEAR-BY-YEAR PERFORMANCE SUMMARY (2023-2025)")
    print()

    print(f"{'Year':<6} {'Portfolio Strategy':<35} {'Return':>10} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print()

    for year in [2023, 2024, 2025]:
        r = backtest_results[year]

        print(f"{year}   {'ANN + LW + Equal-Weight':<35} "
              f"{r['ew_metrics']['annual_return']*100:>9.2f}% "
              f"{r['ew_metrics']['volatility']*100:>7.2f}% "
              f"{r['ew_metrics']['sharpe_ratio']:>8.2f} "
              f"{r['ew_metrics']['max_drawdown']*100:>7.2f}%")

        print(f"{'':6} {'ANN + LW + Tangency':<35} "
              f"{r['tangency_metrics']['annual_return']*100:>9.2f}% "
              f"{r['tangency_metrics']['volatility']*100:>7.2f}% "
              f"{r['tangency_metrics']['sharpe_ratio']:>8.2f} "
              f"{r['tangency_metrics']['max_drawdown']*100:>7.2f}%")

        print(f"{'':6} {'All-Stocks Tangency':<35} "
              f"{r['tangency_all_metrics']['annual_return']*100:>9.2f}% "
              f"{r['tangency_all_metrics']['volatility']*100:>7.2f}% "
              f"{r['tangency_all_metrics']['sharpe_ratio']:>8.2f} "
              f"{r['tangency_all_metrics']['max_drawdown']*100:>7.2f}%")

        if year != 2025:
            print()

    print("\n")


def run_statistical_tests(backtest_results):
    """
    Run statistical validation tests.

    Args:
        backtest_results: Dictionary of results by year
    """
    print("STATISTICAL VALIDATION (Combined 2023-2025)")
    print()

    all_ew = np.concatenate([backtest_results[y]['ew_returns'].values for y in [2023, 2024, 2025]])
    all_tangency = np.concatenate([backtest_results[y]['tangency_returns'].values for y in [2023, 2024, 2025]])
    all_tangency_all = np.concatenate([backtest_results[y]['tangency_all_returns'].values for y in [2023, 2024, 2025]])
    avg_rf = np.mean([backtest_results[y]['rf'] for y in [2023, 2024, 2025]])

    print("TEST 1: Block Bootstrap Confidence Intervals (95% CI)")
    print()

    np.random.seed(RANDOM_SEED)
    boot_ew = block_bootstrap_sharpe_ci(all_ew, avg_rf, block_size=3)
    boot_tangency = block_bootstrap_sharpe_ci(all_tangency, avg_rf, block_size=3)
    boot_tangency_all = block_bootstrap_sharpe_ci(all_tangency_all, avg_rf, block_size=3)

    print(f"{'Portfolio Strategy':<35} {'Sharpe':>10} {'95% CI':>20}")
    print()
    print(f"{'ANN + LW + Equal-Weight':<35} {boot_ew['sharpe']:>10.2f} [{boot_ew['ci_lower']:>6.2f}, {boot_ew['ci_upper']:>6.2f}]")
    print(f"{'ANN + LW + Tangency':<35} {boot_tangency['sharpe']:>10.2f} [{boot_tangency['ci_lower']:>6.2f}, {boot_tangency['ci_upper']:>6.2f}]")
    print(f"{'All-Stocks Tangency':<35} {boot_tangency_all['sharpe']:>10.2f} [{boot_tangency_all['ci_lower']:>6.2f}, {boot_tangency_all['ci_upper']:>6.2f}]")

    print()
    print("TEST 2: Jobson-Korkie Pairwise Comparison")
    print()

    jk_tang_ew = jobson_korkie_test(all_tangency, all_ew, avg_rf)
    print("ANN + LW + Tangency  vs  ANN + LW + Equal-Weight:")
    print(f"  Sharpe Difference: {jk_tang_ew['sharpe_a'] - jk_tang_ew['sharpe_b']:+.2f}")
    print(f"  z-statistic: {jk_tang_ew['z_statistic']:.2f}")
    print(f"  p-value: {'<0.001 ***' if jk_tang_ew['p_value'] < 0.001 else f'{jk_tang_ew['p_value']:.4f}'}")

    print()
    jk_tang_all = jobson_korkie_test(all_tangency, all_tangency_all, avg_rf)
    print("ANN + LW + Tangency  vs  All-Stocks Tangency:")
    print(f"  Sharpe Difference: {jk_tang_all['sharpe_a'] - jk_tang_all['sharpe_b']:+.2f}")
    print(f"  z-statistic: {jk_tang_all['z_statistic']:.2f}")
    print(f"  p-value: {jk_tang_all['p_value']:.4f}")

    print()
    print("TEST 3: Superior Predictive Ability (SPA) Test")
    print()

    np.random.seed(RANDOM_SEED)
    spa_result = spa_test(all_tangency, [all_ew, all_tangency_all], n_bootstrap=10000)

    print("H0: ANN + LW + Tangency does NOT outperform all benchmarks")
    print(f"Test Statistic: {spa_result['max_diff']:.4f}")
    print(f"p-value: {spa_result['p_value']:.4f}")
    print(f"Decision: {'Reject H0 at alpha=0.05' if spa_result['significant'] else 'Cannot reject H0'}")

    print("\n")


def main():
    """
    Main execution function.
    """
    print("\n")
    print("APPLICATION OF MACHINE LEARNING IN PORTFOLIO MANAGEMENT")
    print("ON VIETNAM STOCK MARKET - VN30 Index (2023-2025)")
    print()

    backtest_results = {}
    for year in [2023, 2024, 2025]:
        backtest_results[year] = run_backtest(year)

    print_summary_table(backtest_results)
    run_statistical_tests(backtest_results)


if __name__ == "__main__":
    main()
