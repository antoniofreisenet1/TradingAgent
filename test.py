import pandas as pd

filename="data/Stocks/AAPL.csv"
df = pd.read_csv(filename, parse_dates=["Date"])
print(df)