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

prices_2020_2022 = download_data_safe(tickers, start='2020-01-01', end='2022-12-31')
prices_2023_2025 = download_data_safe(tickers, start='2023-01-01', end='2025-12-31')

prices_combined = pd.concat([prices_2020_2022, prices_2023_2025])

print(len(prices_2020_2022), len(prices_2023_2025), len(prices_combined))

# Duplicated
prices_a = download_data_safe(tickers, start='2020-01-01', end='2022-06-30')
prices_b = download_data_safe(tickers, start='2022-01-01', end='2022-12-31')  # si sovrappone!

prices_overlap = pd.concat([prices_a, prices_b])
print(len(prices_a), len(prices_b), len(prices_overlap))
print(prices_overlap.index.duplicated().sum())  # conta le date duplicate

prices_clean = prices_overlap[~prices_overlap.index.duplicated(keep='first')]

# Esercizio 1 - MSCI World

prices = download_data_safe(tickers, start='2020-01-01', end='2025-12-31')
returns = prices['MSCI World'].pct_change().dropna()
rolling_volatility = returns.rolling(30).std() * np.sqrt(252)

analysis_df = pd.concat([returns, rolling_volatility], axis=1)
analysis_df.columns = ['Daily Return', 'Rolling volatility 30d']
print(analysis_df.head(35))

# Esercizio 2 - merge

asset_info = pd.DataFrame({
    'name': ['MSCI World', 'Emerging Markets IMI', 'MSCI Small Cap', 'Bond Corporate Hedged', 'Physical Gold'],
    'asset_class': ['Equity', 'Equity', 'Equity', 'Bond', 'Commodity']
})

sharpe_df = pd.DataFrame({
    'name': ['MSCI World', 'Emerging Markets IMI', 'MSCI Small Cap', 'Bond Corporate Hedged'],
    'sharpe_ratio': [0.15, 0.42, 0.38, 0.61]  # Example values
})

total_df = pd.merge(asset_info, sharpe_df, on='name', how='left')
print(total_df)

