import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from DAY_4_Try_Except import download_data_safe

tickers = {
    'MSCI World': 'IWDA.L',
    'Emerging Markets IMI': 'EIMI.L',
    'MSCI Small Cap': 'WSML.L',
    'Bond Corporate Hedged': 'CRHG.L',
    'Physical Gold': 'EGLN.L',
}

prices = download_data_safe(tickers, start='2020-01-01', end='2025-12-31')

returns = prices['MSCI World'].pct_change().dropna()

VaR_95 = returns.quantile(1 - 0.95)
print(f"VaR storico al 95%: {VaR_95:.2%}")

CVaR_95 = returns[returns <= VaR_95].mean()
print(f"CVaR storico al 95%: {CVaR_95:.2%}")

VaR_99 = returns.quantile(1 - 0.99)
print(f"VaR storico al 99%: {VaR_99:.2%}")

CVaR_99 = returns[returns <= VaR_99].mean()
print(f"CVaR storico al 99%: {CVaR_99:.2%}")

# Parametric VaR
mu = returns.mean()
sigma = returns.std()

VaR_95_parametric = stats.norm.ppf(0.05, mu, sigma)
VaR_99_parametric = stats.norm.ppf(0.01, mu, sigma)

print(f"VaR 95% storico: {VaR_95:.2%}   |   VaR 95% parametrico (normal): {VaR_95_parametric:.2%}")
print(f"VaR 99% storico: {VaR_99:.2%}   |   VaR 99% parametrico (normal): {VaR_99_parametric:.2%}")