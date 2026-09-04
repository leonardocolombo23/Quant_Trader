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

returns_all = prices.pct_change().dropna()

rolling_corr = returns_all['MSCI World'].rolling(90).corr(returns_all['Physical Gold'])

plt.figure(figsize=(12, 5))
plt.plot(rolling_corr)
plt.axhline(0, color='gray', linestyle='--')
plt.title('Rolling Correlation 90d: MSCI World vs Physical Gold')
plt.xlabel('Date')
plt.ylabel('Correlation')
plt.show()