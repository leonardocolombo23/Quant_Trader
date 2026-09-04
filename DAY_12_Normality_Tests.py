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

# Jarque-Bera Test
jb_stat, jb_pvalue = stats.jarque_bera(returns)
print(f"JB_Stat: {jb_stat:.2f}")
print(f"JB_Pvalue: {jb_pvalue:.10f}")

# Box-Plot
plt.figure(figsize=(8, 5))
plt.boxplot(returns, vert=True)
plt.title('Boxplot - Daily Return MSCI World')
plt.ylabel('Return')
plt.show()

# QQ PLot
plt.figure(figsize=(7, 7))
stats.probplot(returns, dist="norm", plot=plt)
plt.title('QQ-Plot - Returns vs Normal')
plt.show()