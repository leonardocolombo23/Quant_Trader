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
skewness = returns.skew()
kurtosis = returns.kurtosis()
print(f"Skewness: {skewness:.3f}")
print(f"Kurtosis: {kurtosis:.3f}")


plt.figure(figsize=(10, 5))
plt.hist(returns, bins=100, density=True, alpha=0.6, label='Real Returns')

# Normal distribution with same mean and std
mu, sigma = returns.mean(), returns.std()
x = np.linspace(returns.min(), returns.max(), 200)
plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')

plt.legend()
plt.title('Real return distribution vs Normal distribution')
plt.show()