import datetime

import keras.src.saving
import yfinance as yf
import pandas as pd
import numpy as np
from trading_environment import TradingEnv
from trading_agent import TradingAgent
from dataTreatment import load_data_for_prediction


def calcula_fechas(ticker):
    # Funcion que calcula los 50 días anteriores a la fecha de hoy
    folder = "data/Stock Indices/"
    path = folder + ticker + ".csv"

    ayer = datetime.date.today() - datetime.timedelta(days=0)
    delta = ayer - datetime.timedelta(days=90) #cogemos 90 para contabilizar los dias que el mercado puede no estar abierto

    data = yf.download(ticker, start=delta, end=ayer)
    if data.empty:
        print(f"⚠️ WARNING: No data available for {ticker}")
        return 1

    # Ensure 'Close' column is properly filled
    if "Close" in data.columns:
        data["Close"] = data["Close"].ffill()

    # FIX: Flatten MultiIndex columns (if needed)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[1] if col[1] else col[0] for col in data.columns]  # Use second level if exists

    # FIX: Rename columns to remove duplicate ticker name
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    scaled_data = load_data_for_prediction(data, path, ticker)
    # hacer cosas para definir correctamente al agente
    print(scaled_data)
    window_size = 10 # window_size normalmente es la ventana temporal en la que miramos, pero quizas no sea útil ahora.
    feature_size = scaled_data.shape[1]
    state_size = window_size * feature_size

    last_window = scaled_data.tail(window_size)

    #Cargar info del agente
    agent = TradingAgent(state_size=state_size, action_size=3)
    agent.q_network=keras.src.saving.load_model("models/full_" + ticker + ".keras")
    agent.epsilon = agent.epsilon_min

    state = last_window.values.reshape(1, -1)
    q_values, quantity = agent.q_network.predict(state, verbose = 0)
    action = np.argmax(q_values)
    quantity = quantity.item() * 10

    actions_dict = {0:"Vender", 1: "Hold", 2: "Comprar"}
    print(f"\n La predicción del agente para el ticker {ticker} en el día de hoy, {datetime.date.today()}, es: \n")
    print("=" * 40)
    print(f"ACCION: {actions_dict[action]}")
    print(f"CANTIDAD: {quantity:.2f}")
    print("=" * 40)


calcula_fechas("IBEX")