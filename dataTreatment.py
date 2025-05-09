import pandas as pd
import numpy as np
from    sklearn.preprocessing import MinMaxScaler

def load_data(filename="data/Stocks/AAPL.csv"):
    pd.options.display.float_format = '{:.10f}'.format
    # Cargar datos desde el archivo CSV
    df = pd.read_csv(filename, parse_dates=["Date"])
    df.set_index("Date", inplace=True) # CUIDADO: Columna "Date" es indice ahora!! NO SE INCLUYE (13 cols)
    soloceros = df[(df == 0).any(axis=1)]

    # Calcular indicadores técnicos
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean() # Las medias moviles no estaran disponibles hasta pasados 200 records.
    df["RSI"] = compute_rsi(df["Close"]) # Cambiado porque la formula anterior no tenia en cuenta la variacion media
    df["BB_Middle"] = df["Close"].rolling(window=20).mean() # No es necesario incluir la banda central
    df["BB_Std"] = df["Close"].rolling(window=20).std() # No es necesario incluir la desviacion estandar
    df["BB_Upper"] = df["BB_Middle"] + (df["BB_Std"] * 2)
    df["BB_Lower"] = df["BB_Middle"] - (df["BB_Std"] * 2)
    df["ROC"] = df["Close"].pct_change(periods=10) * 100
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum() # Direccion del cambio de precio * volumen


    # Normalización de los datos
    df.dropna(inplace=True)  # Eliminar filas con valores NaN !!! ATENCION: POR SMA_200 PERDEMOS 200 RECORDS !!!
    scaler = MinMaxScaler(feature_range=(0.001, 1))

    df_bycolumns = []
    # Normalizamos por columnas
    for column in df.columns:
        if(not column.endswith("Close")):
            values = df[[column]]
            normalized_values = scaler.fit_transform(values)
            normalized_column = pd.DataFrame(
                normalized_values,
                columns=[column],
                index=values.index
            )
            df_bycolumns.append(normalized_column)
        else:
            df_bycolumns.append(df[[column]]) # Mantenemos el precio de cierre
    df_scaled = pd.concat(df_bycolumns, axis=1)
    # df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)


    print(" ============ FIRST 5 ROWS OF THE SCALED DATA ============ ")
    print(df_scaled.head())
    print(" ============ LAST 5 ROWS OF THE SCALED DATA ============ ")
    print(df_scaled.tail())

    print(" ============ COLUMNS ============ ")
    print(df_scaled.columns)
    print(df_scaled.loc[:,"Close"])

    # Convertir a NumPy array: CAMBIADO: ahora lo convertimos dentro de trading_environment
    # data = df_scaled.values
    return df_scaled  # Devolvemos el dataframe sin convertirlo a matriz aún.


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi
