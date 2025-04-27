import pandas as pd
import collect_data # Test 1
import dataTreatment
import matplotlib.pyplot as plt
import tensorflow as tf

print("Version:", tf.__version__)
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))


filename="data/Stocks/AAPL.csv"
df = pd.read_csv(filename, parse_dates=["Date"])
print(df)


df2 = dataTreatment.load_data()
print(df2.values)
print(type(df2))

soloceros = df2[(df2 == 0).any(axis=1)]
print(soloceros)


#TODO : testear el modelo con los datos de test (2025-01 a 2025-04)