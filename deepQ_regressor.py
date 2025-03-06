import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_q_network(input_shape, output_shape):
    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(input_shape,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(output_shape, activation="linear")  # Q-values para cada acción
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model

def build_regressor(input_shape):
    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(input_shape,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")  # Número de acciones (0 a 1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model
