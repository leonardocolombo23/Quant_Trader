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

prices['MSCI World'].resample('ME').last()   # ultimo prezzo di ogni mese
prices['MSCI World'].resample('ME').mean()   # media dei prezzi di ogni mese
prices['MSCI World'].resample('W').last()    # ultimo prezzo di ogni settimana
prices['MSCI World'].resample('YE').last()   # ultimo prezzo di ogni anno

prices['MSCI World'].groupby(prices.index.year).mean()   # media dei prezzi per ogni anno

# Esercizio 1
average_monthly_price = prices['MSCI World'].resample('ME').mean()
average_yearly_price = prices['MSCI World'].groupby(prices.index.year).mean()

# Esercizio 2
returns = prices['MSCI World'].pct_change().dropna()
day_of_the_week_returns = returns.groupby(returns.index.dayofweek).mean()
