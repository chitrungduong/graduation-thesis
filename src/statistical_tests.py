"""
statistical_tests.py - Statistical Validation

Implements block bootstrap, Jobson-Korkie test, and Superior Predictive
Ability (SPA) test for portfolio performance comparison.
"""

import numpy as np
from scipy import stats


def block_bootstrap_sharpe_ci(returns, rf, n_bootstrap=10000, block_size=3, confidence=0.95, annualization_factor=12):
    """
    Block bootstrap confidence interval for Sharpe ratio.

    Args:
        returns: Portfolio returns
        rf: Annual risk-free rate
        n_bootstrap: Number of bootstrap samples
        block_size: Size of blocks to preserve autocorrelation
        confidence: Confidence level
        annualization_factor: 12 for monthly data

    Returns:
        Dictionary with sharpe, ci_lower, ci_upper, std_error
    """
    returns_array = returns.values if hasattr(returns, 'values') else returns
    n = len(returns_array)
    n_blocks = int(np.ceil(n / block_size))
    sharpe_samples = []

    for _ in range(n_bootstrap):
        boot_returns = []
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, max(1, n - block_size + 1))
            block = returns_array[start_idx:min(start_idx + block_size, n)]
            boot_returns.extend(block)

        boot_returns = np.array(boot_returns[:n])

        total_return = (1 + boot_returns).prod() - 1
        annual_return = (1 + total_return) ** (annualization_factor / n) - 1
        volatility = boot_returns.std() * np.sqrt(annualization_factor)

        if volatility > 0:
            sharpe = (annual_return - rf) / volatility
            sharpe_samples.append(sharpe)

    sharpe_samples = np.array(sharpe_samples)
    alpha = (1 - confidence) / 2

    total_return = (1 + returns_array).prod() - 1
    annual_return = (1 + total_return) ** (annualization_factor / n) - 1
    volatility = returns_array.std() * np.sqrt(annualization_factor)
    observed_sharpe = (annual_return - rf) / volatility if volatility > 0 else 0

    return {
        'sharpe': observed_sharpe,
        'ci_lower': np.percentile(sharpe_samples, alpha * 100),
        'ci_upper': np.percentile(sharpe_samples, (1 - alpha) * 100),
        'std_error': sharpe_samples.std()
    }


def jobson_korkie_test(returns_a, returns_b, rf, annualization_factor=12):
    """
    Jobson-Korkie test for comparing Sharpe ratios.

    Args:
        returns_a: Returns of strategy A
        returns_b: Returns of strategy B
        rf: Annual risk-free rate
        annualization_factor: 12 for monthly data

    Returns:
        Dictionary with sharpe_a, sharpe_b, z_statistic, p_value, significant
    """
    returns_a = returns_a.values if hasattr(returns_a, 'values') else returns_a
    returns_b = returns_b.values if hasattr(returns_b, 'values') else returns_b

    n = len(returns_a)

    total_return_a = (1 + returns_a).prod() - 1
    annual_return_a = (1 + total_return_a) ** (annualization_factor / n) - 1
    vol_a = returns_a.std() * np.sqrt(annualization_factor)
    sharpe_a = (annual_return_a - rf) / vol_a if vol_a > 0 else 0

    total_return_b = (1 + returns_b).prod() - 1
    annual_return_b = (1 + total_return_b) ** (annualization_factor / n) - 1
    vol_b = returns_b.std() * np.sqrt(annualization_factor)
    sharpe_b = (annual_return_b - rf) / vol_b if vol_b > 0 else 0

    excess_a = returns_a - rf / annualization_factor
    excess_b = returns_b - rf / annualization_factor

    var_a = np.var(excess_a, ddof=1) * annualization_factor
    var_b = np.var(excess_b, ddof=1) * annualization_factor
    cov_ab = np.cov(excess_a, excess_b)[0, 1] * annualization_factor

    if vol_a > 0 and vol_b > 0:
        theta = (var_a * sharpe_b**2 + var_b * sharpe_a**2 - 2 * sharpe_a * sharpe_b * cov_ab) / (2 * n)

        if theta > 0:
            z_stat = (sharpe_a - sharpe_b) / np.sqrt(theta)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0
            p_value = 1.0
    else:
        z_stat = 0
        p_value = 1.0

    return {
        'sharpe_a': sharpe_a,
        'sharpe_b': sharpe_b,
        'z_statistic': z_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def spa_test(strategy_returns, benchmark_returns, n_bootstrap=1000):
    """
    Superior Predictive Ability test for data snooping.

    Args:
        strategy_returns: Returns of strategy
        benchmark_returns: List of benchmark returns
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary with max_diff, p_value, significant
    """
    strategy_returns = strategy_returns.values if hasattr(strategy_returns, 'values') else strategy_returns

    if not isinstance(benchmark_returns, list):
        benchmark_returns = [benchmark_returns]

    benchmark_returns = [b.values if hasattr(b, 'values') else b for b in benchmark_returns]

    n = len(strategy_returns)
    n_benchmarks = len(benchmark_returns)

    diffs = np.zeros(n_benchmarks)
    for i, bench in enumerate(benchmark_returns):
        diffs[i] = np.mean(strategy_returns - bench)

    max_diff_observed = np.max(diffs)

    max_diffs_boot = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        boot_strategy = strategy_returns[indices]

        boot_diffs = np.zeros(n_benchmarks)
        for i, bench in enumerate(benchmark_returns):
            boot_bench = bench[indices]
            boot_mean_diff = np.mean(boot_strategy - boot_bench)
            centered_diff = boot_mean_diff - diffs[i]
            boot_diffs[i] = centered_diff

        max_diffs_boot.append(np.max(boot_diffs))

    max_diffs_boot = np.array(max_diffs_boot)
    p_value = np.mean(max_diffs_boot >= max_diff_observed)

    return {
        'max_diff': max_diff_observed,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
