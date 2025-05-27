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



# TODO: entrenar al agente por partes: train y test
for episode in range(100):
    start_time = time.time()
    currentState = env.reset()
    done = False
    total_reward = 0  # Acumulador de recompensa
    reward_by_episode = []
    step_counter = 0 # Acumulador de pasos
    info = {}
    episode_infos = []

    while not done:
        action, quantity = agent.act(currentState)
        next_state, reward, done, info = env.step(action, quantity)
        agent.remember(currentState, action, reward, next_state, done, quantity)
        currentState = next_state

        total_reward += reward

        step_counter += 1
        if step_counter % 5 == 0:  #entrena cada 5 pasos: anteriormente al final de cada episodio.
            agent.train()

    agent.train()

    episode_rewards.append(total_reward)  # Guardamos resultado
    time_per_episode.append(time.time() - start_time)
    episode_infos.append(info.copy())
    episode_net_worth.append(info["net_worth"])

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
agent.regressor.save("models/regressor_aapl.keras")

def evaluate_agent(agent, env):
    #todo
    state = env.reset()
    done = False
    total_reward = 0
    net_worths = []

    while not done:
        action, quantity = agent.act(state)
        state, reward, done, info = env.step(action, quantity)
        total_reward += reward
        net_worths.append(info["net_worth"])

    # Calcular métricas tras entrenamiento
    final_net_worths = np.array([info['net_worth'] for info in episode_infos]) #con la lista vamos bien para datos, pero no para operar con vectores

    # Media de ganancias netas
    mean_return = np.mean([nw - env.initial_balance for nw in final_net_worths])

    # Sharpe ratio (retorno medio / desviación estándar)
    returns = np.diff(final_net_worths)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-6)  # evitamos división por cero

    # Máximo drawdown
    peak = np.maximum.accumulate(final_net_worths)
    drawdown = (final_net_worths - peak) / (peak + 1e-6)
    max_drawdown = np.min(drawdown)

    print(f"\nEvaluación del agente:")
    print(f"Media de retorno neto: {mean_return:.2f}")
    print(f"Ratio de Sharpe: {sharpe:.4f}")
    print(f"Máximo Drawdown: {max_drawdown:.4%}")

    print(f"Net worth final: {info['net_worth']:.2f}")
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