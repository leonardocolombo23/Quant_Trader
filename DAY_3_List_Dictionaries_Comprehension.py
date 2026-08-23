import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from DAY_1_Markowitz_functions import download_data

# Solo i ticker che finiscono con '.L' (Londra)
#london_tickers = [t for t in tickers.values() if t.endswith('.L')]

#Si legge: "per ogni `t` in `tickers.values()`, se `t` finisce con '.L', includilo nella lista".

## Esercizio 1 per te

tickers = {
    'MSCI World': 'IWDA.L',
    'Emerging Markets IMI': 'EIMI.L',
    'MSCI Small Cap': 'WSML.L',
    'Bond Corporate Hedged': 'CRHG.L',
    'Physical Gold': 'EGLN.L'
}

tickers_list = [ticker for ticker in tickers if ticker != 'Physical Gold']

new_tickers_dict = {k: v.replace('.L', '') for k, v in tickers.items()}

print(tickers_list, new_tickers_dict)

sharpe_data = [
    ('MSCI World', 0.011),
    ('Emerging Markets IMI', 0.331),
    ('MSCI Small Cap', 0.263),
    ('Bond Corporate Hedged', 0.634),
    ('Physical Gold', 0.386)
]

asset_selected = [nome for nome, sharpe_ratio in sharpe_data if sharpe_ratio > 0.30]

print(asset_selected)