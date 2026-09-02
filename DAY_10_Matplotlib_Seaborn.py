import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
from DAY_4_Try_Except import download_data_safe

tickers = {
    'MSCI World': 'IWDA.L',
    'Emerging Markets IMI': 'EIMI.L',
    'MSCI Small Cap': 'WSML.L',
    'Bond Corporate Hedged': 'CRHG.L',
    'Physical Gold': 'EGLN.L',
}

prices = download_data_safe(tickers, start='2020-01-01', end='2025-12-31')

# Equity Curve
returns = prices['MSCI World'].pct_change().dropna()
equity_curve = (1 + returns).cumprod()

plt.figure(figsize=(10, 5))
plt.plot(equity_curve)
plt.title('Equity Curve - MSCI World')
plt.xlabel('Date')
plt.ylabel('Cumulated value (base 1$)')
plt.show()

# Drawdown Curve
running_max = equity_curve.cummax() #Max value until the selected day
drawdown = (equity_curve - running_max) / running_max

plt.figure(figsize=(10, 5))
plt.plot(drawdown)
plt.title('Drawdown Curve - MSCI World')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
plt.show()

# Correlation Heatmap
returns_all = prices.pct_change().dropna()
corr_matrix = returns_all.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix - Daily Return')
plt.show()