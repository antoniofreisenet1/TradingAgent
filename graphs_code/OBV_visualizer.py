import matplotlib.pyplot as plt
import dataTreatment

filename="../data/Stocks/AAPL.csv"

df = dataTreatment.load_data(filename)
plt.plot(df.index, df["OBV"])
plt.title("On Balance Volume (normalizado)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
