from tensorflow.keras.models import load_model
from keras.utils import plot_model

q_net = load_model("models/q_network_aapl.keras", compile=False)
reg = load_model("models/regressor_aapl.keras", compile=False)

q_net.compile(optimizer="adam", loss="mse")
reg.compile(optimizer="adam", loss="mse")

q_net.summary()
reg.summary()


plot_model(q_net, show_shapes=True, show_layer_names=True, to_file='models/modelo_Qnet.png')
plot_model(reg, show_shapes=True, show_layer_names=True, to_file='models/modelo_regresor.png')

for layer in q_net.layers:
    print(f"Layer: {layer.name}")
    weights = layer.get_weights()
    print(weights)

for layer in reg.layers:
    print(f"Layer: {layer.name}")
    weights = layer.get_weights()
    print(weights)
#TODO: reformatear con interfaz de usuario: cargar modelo, entrenar y guardar nuevo modelo, eliminar modelo, hacer prediccion