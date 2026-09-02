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

prices = download_data_safe(tickers, start='2010-07-01', end='2025-12-31')

tot_NaN_per_column = prices.isna().sum()
print(tot_NaN_per_column)

print(f"Righe totali: {len(prices)}")
print(f"Righe dopo dropna: {len(prices.dropna())}")

common_start = prices.dropna().index.min()
print(f"Il periodo comune a tutti gli asset inizia da: {common_start}")

# Outliers
returns = prices['MSCI World'].pct_change().dropna()
soglia = returns.std() * 4  # 4 deviazioni standard
outliers = returns[returns.abs() > soglia]
print(outliers)