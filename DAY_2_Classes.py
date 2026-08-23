import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from DAY_1_Markowitz_functions import download_data

tickers = {
    'MSCI World': 'IWDA.L',
    'Emerging Markets IMI': 'EIMI.L',
    'MSCI Small Cap': 'WSML.L',
    'Bond Corporate Hedged': 'CRHG.L',
    'Physical Gold': 'EGLN.L'
}

prices = download_data(tickers, start='2010-07-01', end='2025-12-31')

class Asset:
    def __init__(self, name, prices):
        self.name = name
        self.prices = prices

    def returns(self):
        """Calcola i return giornalieri dell'asset"""
        return self.prices.pct_change().dropna()

    def annualized_returns(self):
        """Calcola i return annuali dell'asset"""
        return self.returns().mean() * 252

    def annualized_volatility(self):
        """Calcola le devizioni standard annuali dell'asset"""
        return self.returns().std() * np.sqrt(252)

    def sharpe_ratio(self, risk_free_rate=0.02):
        """Calcola lo Sharpe Ratio annuale dell'asset"""
        return (self.annualized_returns() - risk_free_rate) / self.annualized_volatility()

    def summary(self):
        """Ritorna un dizionario con tutte le metriche principali dell'asset"""
        return {
            'name': self.name,
            'annualized_return': self.annualized_returns(),
            'annualized_volatility': self.annualized_volatility(),
            'sharpe_ratio': self.sharpe_ratio()
        }

assets = []
for name in prices.columns:
    asset = Asset(name, prices[name])
    assets.append(asset)

for asset in assets:
    s = asset.summary()
    print(f"{s['name']}: return={s['annualized_return']:.2%}, volatility={s['annualized_volatility']:.2%}, sharpe={s['sharpe_ratio']:.2f}")

r = pd.DataFrame([asset.summary() for asset in assets])


