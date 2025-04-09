import keras.src.saving.saving_api

from dataTreatment import load_data
from trading_environment import TradingEnv
from trading_agent import TradingAgent
import time


# IMPORTANTE: antes de ejecutar las funciones, se debe ejecutar al menos una vez el archivo collect_data.py
data = load_data()
env = TradingEnv(data)
agent = TradingAgent(state_size=data.shape[1], action_size=3)

episode_rewards = []
time_per_episode = []
# TODO: entrenar al agente
for episode in range(100):
    start_time = time.time()
    currentState = env.reset()
    done = False
    total_reward = 0  # Acumulador de recompensa

    while not done:
        action, quantity = agent.act(currentState)
        next_state, reward, done, _ = env.step(action)
        agent.remember(currentState, action, reward, next_state, done)
        state = next_state
        total_reward += reward  # Sumamos recompensa

    agent.train()
    episode_rewards.append(total_reward)  # Guardamos resultado

    time_per_episode.append(time.time() - start_time)

    print(f"Episodio {episode + 1}: Recompensa total = {total_reward:.2f}")
    print(f"Tiempo tomado en este episodio de entrenamiento = {time_per_episode[-1]:.2f}")

# TODO : evaluar precision y guardar pesos de las redes neuronales

print(f"Tiempo total: {sum(time_per_episode):.2f}")

agent.q_network.save("models/q_network_aapl.keras")
agent.regressor.save("models/regressor_aapl.keras")
