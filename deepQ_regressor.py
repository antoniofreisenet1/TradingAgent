import tensorflow as tf
import keras
from keras import layers


#input_shape: numero de indicadores tecnicos a utilizar (Usaremos 8 dimensiones)
#output_shape: posibles movimientos. (3 - buy/hold/sell??)
def build_combined_network(input_shape):
    inputs = keras.Input(shape=(input_shape,), name="network_input")

    # Capas compartidas
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)

    # Rama Q-values (3 acciones)
    q_branch = layers.Dense(32, activation="relu")(x)
    q_output = layers.Dense(3, activation="linear", name="q_output")(q_branch)

    # Rama cantidades(1 valor, entre 0 y 1, sera escalado luego)
    quantity_branch = layers.Dense(32, activation="relu")(x)
    quantity_output = layers.Dense(1, activation="sigmoid", name="quantity_output")(quantity_branch)

    model = keras.Model(inputs=inputs, outputs=[q_output, quantity_output])

    lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.001,
        staircase=False
    )
    # Define custom losses and compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss={
            "q_output": "huber",
            "quantity_output": "huber"  # ajustable: huber, mse son buenas
        },
        loss_weights={
            "q_output": 1.0,
            "quantity_output": 0.1
        }
    )

    model.summary()
    for layer in model.layers:
        print(f"{layer.name}: trainable = {layer.trainable}")
    return model

# def build_q_network(input_shape, output_shape):
#     model = keras.Sequential([
#         keras.Input(shape=(input_shape,)),
#         layers.Dense(128, activation="relu"),
#         layers.Dense(64, activation="relu"),
#         layers.Dense(output_shape, activation="linear")  # Q-values para cada acción: buy/hold/sell
#     ])
#
#     lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
#         initial_learning_rate=0.002,
#         decay_steps=1000,
#         decay_rate=0.001,
#         staircase=False
#     )
#     optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
#     # model.summary() # Solo para testing
#     model.compile(optimizer, loss="huber") # Considera la perdida de Huber para mercados confusos
#     return model


#Testing para comprobar si es mejor no usar el regresor.
# def build_regressor(input_shape):
#     model = keras.Sequential([
#         keras.Input(shape=(input_shape,)),
#         layers.Dense(128, activation="relu"),
#         layers.Dense(64, activation="relu"),
#         layers.Dense(1, activation="sigmoid")  # Número de acciones
#     ])
#     # model.summary() # Solo para testing
#     optimizer = keras.optimizers.Adam(learning_rate=0.001)
#     model.compile(optimizer, loss="huber")  # Considera la perdida de Huber para mercados confusos
#     return model
