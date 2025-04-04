import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#input_shape: numero de indicadores tecnicos a utilizar (Usaremos 8 dimensiones)
#output_shape: posibles movimientos. (3 - buy/hold/sell??)
def build_q_network(input_shape, output_shape):
    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(input_shape,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(output_shape, activation="linear")  # Q-values para cada acción: buy/hold/sell
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model

def build_regressor(input_shape):
    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(input_shape,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")  # Número de acciones
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model
