import keras.src.saving.saving_api

from dataTreatment import load_data
from trading_environment import TradingEnv
from trading_agent import TradingAgent
import numpy as np
import matplotlib.pyplot as plt
import time


# IMPORTANTE: antes de ejecutar las funciones, se debe ejecutar al menos una vez el archivo collect_data.py
data = load_data()
split_index = int(0.8 * len(data))
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]


window_size = 10
feature_size = data.shape[1]
state_size = window_size * feature_size

env = TradingEnv(train_data, 10, "AAPL")
agent = TradingAgent(state_size=state_size, action_size=3) #State_size: tamaño vector de entrada: indicadores

episode_rewards = []
episode_net_worth = []
time_per_episode = []
episode_infos = []
mean_quantities = []
total_losses = []
q_losses = []
quantity_losses = []
actions = [] #buy, hold, sell
epsilon_history = []


# TODO: entrenar al agente por partes: train y test
for episode in range(100):
    start_time = time.time()
    currentState = env.reset()
    done = False
    total_reward = 0  # Acumulador de recompensa
    reward_by_episode = []
    step_counter = 0 # Acumulador de pasos
    actions_by_episode = [0,0,0]
    quantities_by_episode = []
    info = {}

    while not done:
        action, quantity = agent.act(currentState)
        next_state, reward, done, info = env.step(action, quantity)
        agent.remember(currentState, action, reward, next_state, done, quantity)
        currentState = next_state

        total_reward += reward
        actions_by_episode[action] += 1 # actualiza numero de compras/ventas/aguantes
        quantities_by_episode.append(quantity)

        step_counter += 1
        if step_counter % 5 == 0:  #entrena cada 5 pasos: anteriormente al final de cada episodio.
            agent.train()
            q_losses.append(agent.last_q_values_loss)
            total_losses.append(agent.last_total_loss)

    epsilon_history.append(agent.epsilon)
    actions.append(actions_by_episode)
    episode_rewards.append(total_reward)  # Guardamos resultado
    time_per_episode.append(time.time() - start_time)
    episode_infos.append(info.copy())
    episode_net_worth.append(info["net_worth"])
    mean_quantities.append(np.mean(quantities_by_episode))

    # Reporte
    print(f"\n=== Episodio {episode + 1} ===")
    print(f"Recompensa total: {total_reward:.4f}")
    print(f"Net Worth final: {info['net_worth']:.2f}")
    print(f"Balance final: {info['balance']:.2f}")
    print(f"Acciones en cartera: {info['shares_held']}")
    print(f"Precio de cierre: {info['current_price']:.2f}")
    print(f"Tiempo de episodio: {time_per_episode[-1]:.2f} segundos")
    print("=" * 40)

# TODO : evaluar precision y guardar pesos de las redes neuronales

print(f"Tiempo total: {sum(time_per_episode):.2f}")

agent.q_network.save("models/q_network_aapl.keras")
# agent.regressor.save("models/regressor_aapl.keras")

def evaluate_agent(agent, env):
    #todo
    state = env.reset()
    done = False
    total_reward = 0
    net_worths = []
    info = {}

    operaciones_totales = 0
    operaciones_ganadoras = 0

    prev_net_worth = env.net_worth

    while not done:
        action, quantity = agent.act(state)
        state, reward, done, info = env.step(action, quantity)
        total_reward += reward
        net_worths.append(info["net_worth"])

        # Contamos operaciones activas
        if action in [0, 2]:  # Sell o Buy
            operaciones_totales += 1
            if info["net_worth"] > prev_net_worth:
                operaciones_ganadoras += 1

        prev_net_worth = info["net_worth"]

    # TODO Calcular métricas tras entrenamiento
    final_net_worths = np.array(episode_net_worth) #con la lista vamos bien para datos, pero no para operar con vectores

    # Media de ganancias netas
    mean_return = np.mean([nw - env.initial_balance for nw in final_net_worths])

    # Sharpe ratio (retorno medio / desviacion estandar)
    returns = np.diff(final_net_worths)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-6)  # evitamos division por cero

    # Maximo drawdown
    peak = np.maximum.accumulate(final_net_worths)
    drawdown = (final_net_worths - peak) / (peak + 1e-6)
    max_drawdown = np.min(drawdown)

    if operaciones_totales > 0:
        precision = operaciones_ganadoras / operaciones_totales
    else:
        precision = 0.0

    print(f"\nEvaluación del agente:")
    print(f"Media de retorno neto: {mean_return:.2f}")
    print(f"Ratio de Sharpe: {sharpe:.4f}")
    print(f"Máximo Drawdown: {max_drawdown:.4%}")

    print(f"Net worth final: {info['net_worth']:.2f}")
    print(f"Precisión del agente: {precision:.2%}")
    return net_worths

evaluate_agent(agent, env)



plt.plot(episode_rewards)
plt.title("Recompensa por episodio")
plt.xlabel("Episodio")
plt.ylabel("Reward total")
plt.show()


plt.plot(episode_net_worth)
plt.title("Net Worth por episodio")
plt.xlabel("Episodio")
plt.ylabel("Net Worth final")
plt.show()


plt.plot(q_losses, label="Q-output loss")
plt.plot(quantity_losses, label="Quantity-output loss")
plt.plot(total_losses, label="Total output loss")
plt.title("Evolución de las pérdidas por salida")
plt.xlabel("Episodio")
plt.ylabel("Pérdida")
plt.legend()
plt.grid(True)
plt.show()

# Convertimos la lista de listas en array para graficar facilmente
actions_array = np.array(actions)  # shape: (episodios, 3)

plt.plot(actions_array[:, 0], label="Buy") # con este slicing incluimos todas las filas, pero solo la columna 0
plt.plot(actions_array[:, 1], label="Hold")
plt.plot(actions_array[:, 2], label="Sell")
plt.title("Acciones tomadas por tipo (por episodio)")
plt.xlabel("Episodio")
plt.ylabel("Número de acciones")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(mean_quantities)
plt.title("Cantidad promedio comprada/vendida")
plt.xlabel("Episodio")
plt.ylabel("Cantidad media")
plt.grid(True)
plt.show()

plt.plot(epsilon_history)
plt.title("Epsilon por episodio")
plt.xlabel("Episodio")
plt.ylabel("Epsilon")
plt.grid(True)
plt.show()

print("\n=== PREDICCIÓN FINAL DEL AGENTE ===")

# Creamos un nuevo entorno sobre los datos de test
final_env = TradingEnv(test_data, window_size, "AAPL")
final_env.reset()

# Posicionamos el entorno en el último paso disponible
final_env.current_step = len(test_data) - 1  # o -2 para evitar out of bounds

# Obtenemos el último estado observable
final_state = final_env._next_observation()

# El agente toma una decisión
final_action, final_quantity = agent.act(final_state)

# Interpretamos la acción
acciones = ["SELL", "HOLD", "BUY"]
print(f"Acción sugerida: {acciones[final_action]}")
if final_action in [0, 2]:  # SELL o BUY
    print(f"Cantidad sugerida: {int(final_quantity)} acciones")
else:
    print("No se recomienda ninguna acción activa.")
