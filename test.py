import pandas as pd
import collect_data # Test 1
import dataTreatment

filename="data/Stocks/AAPL.csv"
df = pd.read_csv(filename, parse_dates=["Date"])
print(df)


df2 = dataTreatment.load_data()
print(df2)
print(type(df2))
print(df2[0])