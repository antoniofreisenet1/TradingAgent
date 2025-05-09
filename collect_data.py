import yfinance as yf
import pandas as pd
import json
import os
from pathlib import Path

# ==========================
# LOCAL VARIABLES
# ==========================

path = "../TradingAgent/data/"
start_date = "2020-01-01"
end_date = "2025-05-01"

empty_tickers = []
total_tickers = 0
# ==========================
# FUNCTION TO FETCH DATA
# ==========================

def fetch_data(dir_name, tickers_dict):
    """Download historical data for given tickers and save as CSV."""

    global total_tickers
    for i, (name, symbol) in enumerate(tickers_dict.items()):
        print(f'{dir_name}: ({i + 1}/{len(tickers_dict)}) downloading symbol {symbol} . . .')

        # Ensure subdirectory exists
        os.makedirs(f'{path}/{dir_name}', exist_ok=True)

        try:
            # Download data from yfinance
            data = yf.download(symbol, start=start_date, end=end_date)

            total_tickers = total_tickers + 1
            # Check if any data is returned
            if data.empty:
                print(f"⚠️ WARNING: No data available for {symbol}")
                empty_tickers.append(symbol)

            # Ensure 'Close' column is properly filled
            if "Close" in data.columns:
                data["Close"] = data["Close"].ffill()

            # FIX: Flatten MultiIndex columns (if needed)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[1] if col[1] else col[0] for col in data.columns]  # Use second level if exists


            # FIX: Rename columns to remove duplicate ticker name
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            # Save data to CSV
            data.to_csv(f'{path}/{dir_name}/{symbol}.csv')

            print(f"✅ {symbol} saved successfully!")

        except Exception as e:
            print(f"❌ ERROR downloading {symbol}: {e}")


# ==========================
# FUNCTION TO DOWNLOAD ALL FILES
# ==========================

def download_files():
    """Load assets from JSON and download historical data."""

    # Ensure base directory exists
    os.makedirs(path, exist_ok=True)

    # Load JSON file
    with open("assets.json", "r") as file:
        metadata = json.load(file)

    for index in metadata.values():
        index_name = index["name"]
        components = index["components"]

        print(f'\n\n-------- DOWNLOADING COMPONENTS FOR {index_name} from {start_date} to {end_date} --------\n\n')
        fetch_data(index_name, components)

    print(f'\n FOUND {len(empty_tickers)} EMPTY TICKERS OUT OF {total_tickers} TICKERS \n')
    print(f'\n\n EMPTY TICKERS: {empty_tickers} \n\n')

# ==========================
# RUN THE SCRIPT
# ==========================
if __name__ == "__main__":
    download_files()
