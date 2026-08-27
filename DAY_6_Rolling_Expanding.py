import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from DAY_4_Try_Except import download_data_safe

tickers = {
    'MSCI World': 'IWDA.L',
    'Emerging Markets IMI': 'EIMI.L',
    'MSCI Small Cap': 'WSML.L',
    'Bond Corporate Hedged': 'CRHG.L',
    'Physical Gold': 'EGLN.L',
}

prices = download_data_safe(tickers, start='2020-07-01', end='2025-12-31')

moving_average_20 = prices['MSCI World'].rolling(20).mean()
moving_std_30 = prices['MSCI World'].rolling(30).std()
print(moving_average_20.head(25), moving_std_30.head(25))

expanding_mean = prices['MSCI World'].expanding().mean()
print(expanding_mean.head(10))

# Exercise 1
log_returns = np.log(prices['MSCI World'] / prices['MSCI World'].shift(1))

rolling_volatility_annualized = log_returns.rolling(30).std() * np.sqrt(252)

picco_data = rolling_volatility_annualized.idxmax()
print(picco_data)
print(prices['MSCI World'].loc[picco_data - pd.Timedelta(days=10): picco_data + pd.Timedelta(days=10)])

plt.plot(rolling_volatility_annualized)
plt.title('Annualized Volatility Rolling (30gg) - MSCI World')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.show()