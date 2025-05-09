import pandas as pd
import collect_data # Test 1
import dataTreatment
import matplotlib.pyplot as plt
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Mostrar todos los logs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Forzar visibilidad de la GPU 0
print("Version:", tf.__version__)
print("Dispositivos disponibles:", tf.config.list_physical_devices())
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))


filename="data/Stocks/AAPL.csv"
df = pd.read_csv(filename, parse_dates=["Date"])
print(df)


df2 = dataTreatment.load_data()
print("Hello")
print(df2.shape)
print("ByeBYe")
print(df2.values)
print(type(df2))

soloceros = df2[(df2 == 0).any(axis=1)]
print(soloceros)


#TODO : testear el modelo con los datos de test (2025-01 a 2025-04)