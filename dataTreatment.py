import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(filename="data/Stocks/AAPL.csv"):
    # Cargar datos desde el archivo CSV
    df = pd.read_csv(filename, parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    # Calcular indicadores técnicos
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(window=14).mean()))
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    df["BB_Std"] = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (df["BB_Std"] * 2)
    df["BB_Lower"] = df["BB_Middle"] - (df["BB_Std"] * 2)
    df["ROC"] = df["Close"].pct_change(periods=10) * 100


    # Normalización de los datos
    df.dropna(inplace=True)  # Eliminar filas con valores NaN TODO: esto ahora mismo elimina todas las filas de indicadores tecnicos
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    print(" ============ FIRST 5 ROWS OF THE SCALED DATA ============ ")
    print(df_scaled.head())
    print(" ============ LAST 5 ROWS OF THE SCALED DATA ============ ")
    print(df_scaled.tail())

    # Convertir a NumPy array ??????
    data = df_scaled.values
    return data  # Devolvemos la matriz lista para el agente


