import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# DATA DOWNLOAD
def download_data(tickers_dict, start, end):
    """Scarica prezzi e rinomina colonne coi nomi leggibili"""
    data = yf.download(list(tickers_dict.values()), start=start, end=end)['Close']

    # Inverti il dizionario: da {nome: ticker} a {ticker: nome}
    ticker_to_name = {v: k for k, v in tickers_dict.items()}

    # Rinomina usando il ticker effettivo di ogni colonna, non la posizione
    data.columns = [ticker_to_name[col] for col in data.columns]

    return data

def compute_monthly_returns(prices):
    """Da prezzi giornalieri a return mensili"""
    monthly_prices = prices.ffill().resample('ME').last()
    monthly_returns = monthly_prices.pct_change().dropna()
    return monthly_returns

# PORTFOLIO SIMULATION - MONTE CARLO METHOD
def monte_carlo_simulation(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    """Genera portafogli random e ne calcola return/vol/sharpe"""
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return = np.dot(mean_returns, weights) * 12
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 12, weights)))
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_volatility

    return results, weights_record

# EFFICIENT FRONTIER OPTIMIZATION

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(mean_returns, weights) * 12
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 12, weights)))
    return returns, std

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    portfolio_return, portfolio_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(portfolio_return - risk_free_rate) / portfolio_std

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

def build_bounds(asset_names, excluded_asset, default_bound):
    bounds = []
    for name in asset_names:
        if name == excluded_asset:
            bounds.append((0, 0))  # excluded
        else:
            bounds.append(default_bound)
    return tuple(bounds)

# MAXIMUM SHARPE RATIO PORTFOLIO
def optimize_max_sharpe(mean_returns, cov_matrix, bounds, risk_free_rate):
    num_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    init_guess = num_assets * [1. / num_assets]
    result = minimize(negative_sharpe, init_guess,
                               args=(mean_returns, cov_matrix, risk_free_rate),
                               method='SLSQP', bounds=bounds, constraints=constraints)
    weights = result.x
    ret, std = portfolio_performance(weights, mean_returns, cov_matrix)

    return weights, ret, std

# MINIMUM VARIANCE PORTFOLIO
def optimize_min_variance(mean_returns, cov_matrix, bounds):
    num_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    init_guess = num_assets * [1. / num_assets]
    result = minimize(portfolio_volatility, init_guess,
                            args=(mean_returns, cov_matrix),
                            method='SLSQP', bounds=bounds, constraints=constraints)
    weights = result.x
    ret, std = portfolio_performance(weights, mean_returns, cov_matrix)

    return weights, ret, std

# EFFICIENT FRONTIER LINE
def compute_efficient_frontier(mean_returns, cov_matrix, bounds, min_return, max_return, n_points):
    frontier_returns = []
    frontier_volatilities = []
# Range of target returns between min variance and max return
    target_returns = np.linspace(min_return, max_return * 1.2, n_points)
    num_assets = len(mean_returns)
    init_guess = num_assets * [1. / num_assets]
    for target in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x, t=target: portfolio_performance(x, mean_returns, cov_matrix)[0] - t}
        )
        result = minimize(portfolio_volatility, init_guess,
                          args=(mean_returns, cov_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            frontier_returns.append(target)
            frontier_volatilities.append(result.fun)

    return frontier_returns, frontier_volatilities

# FINAL PLOT - Monte Carlo + Efficient Frontier + Key Portfolios
def final_plot(results, frontier_returns, frontier_volatilities, max_sharpe_return, max_sharpe_std, min_var_return, min_var_std, save_path='efficient_frontier.png'):
    plt.figure(figsize=(12, 8))
# Monte Carlo cloud
    plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', alpha=0.4, s=5, label='Random Portfolios')
    plt.colorbar(label='Sharpe Ratio')
# Efficient frontier line
    plt.plot(frontier_volatilities, frontier_returns, 'r-', linewidth=2, label='Efficient Frontier')
# Maximum Sharpe Ratio portfolio
    plt.scatter(max_sharpe_std, max_sharpe_return, marker='*', color='red', s=500, label='Max Sharpe Ratio')
# Minimum Variance portfolio
    plt.scatter(min_var_std, min_var_return, marker='*', color='blue', s=500, label='Min Variance')
    plt.xlabel('Annual Volatility')
    plt.ylabel('Annual Return')
    plt.title('Efficient Frontier - Markowitz Portfolio Optimization')
    plt.legend(labelspacing=0.8)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# FINAL RESULT
if __name__ == "__main__":
    tickers = {
        'MSCI World': 'IWDA.L',
        'Emerging Markets IMI': 'EIMI.L',
        'MSCI Small Cap': 'WSML.L',
        'Bond Corporate Hedged': 'CRHG.L',
        'Physical Gold': 'EGLN.L'
    }
    prices = download_data(tickers, start='2020-07-01', end='2025-12-31')
    monthly_returns = compute_monthly_returns(prices)
    mean_returns = monthly_returns.mean()
    cov_matrix = monthly_returns.cov()

    results_montecarlo, weights_montecarlo = monte_carlo_simulation(mean_returns, cov_matrix, 10000, 0.02)

    bounds = build_bounds(mean_returns.index, excluded_asset=None, default_bound=(0, 1.0))

    max_sharpe_weights, max_sharpe_return, max_sharpe_std = optimize_max_sharpe(mean_returns, cov_matrix, bounds, 0.02)
    min_var_weights, min_var_return, min_var_std = optimize_min_variance(mean_returns, cov_matrix, bounds)

    eff_front_returns, eff_front_volatilities = compute_efficient_frontier(mean_returns, cov_matrix, bounds, min_var_return, max_sharpe_return, 100)
    final_plot(results_montecarlo, eff_front_returns, eff_front_volatilities, max_sharpe_return, max_sharpe_std, min_var_return, min_var_std)