import yfinance as yf
import pandas as pd
from DAY_1_Markowitz_functions import download_data

def safe_divide(a, b):
    try:
        risultato = a / b
        return risultato

    except ZeroDivisionError:
        print(f"Attenzione: ZeroDivisionError")
        return None

Calcolo = safe_divide(10, 2)
Calcolo_1 = safe_divide(10, 0)

# DATA DOWNLOAD
def download_data_safe(tickers_dict, start, end):
    valid_data = pd.DataFrame()
    for name, ticker in tickers_dict.items():
        try:
            prices = yf.download(ticker, start=start, end=end)['Close']
            if prices.empty:
                print(f"Nessun dato per {name} ({ticker}), lo salto")
                continue
            valid_data[name] = prices
        except Exception as e:
            print(f"Errore su {name} ({ticker}): {e}")
            continue
    return valid_data

tickers = {
    'MSCI World': 'IWDA.L',
    'Emerging Markets IMI': 'EIMI.L',
    'MSCI Small Cap': 'WSML.L',
    'Bond Corporate Hedged': 'CRHG.L',
    'Physical Gold': 'EGLN.L',
    'Amazon': 'hjhjhbhv'
}

dati = download_data_safe(tickers, start='2020-07-01', end='2025-12-31')
