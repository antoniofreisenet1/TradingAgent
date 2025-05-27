import matplotlib.pyplot as plt
from sympy import false

import dataTreatment

filename="../data/Stocks/AAPL.csv"

df = dataTreatment.load_data(filename, enable_debug=false)
plt.plot(df.index, df["OBV"])
plt.title("On Balance Volume (normalizado)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
