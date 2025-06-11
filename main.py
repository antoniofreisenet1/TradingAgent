import keras.src.saving.saving_api

from dataTreatment import load_data
from trading_environment import TradingEnv
from trading_agent import TradingAgent
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import time
import os
import gc


# IMPORTANTE: antes de ejecutar las funciones, se debe ejecutar al menos una vez el archivo collect_data.py



#DONE: AAPL, JPM, CL=F, GC=F, BTC-USD

def train(agent, env, episodes=100):
    episode_rewards = []
    episode_net_worth = []
    time_per_episode = []
    episode_infos = []
    mean_quantities = []
    total_losses = []
    q_losses = []
    quantity_losses = []
    actions = []
    epsilon_history = []
    mean_returns = []

    for episode in range(episodes):
        start_time = time.time()
        currentState = env.reset()
        done = False
        total_reward = 0
        step_counter = 0
        actions_by_episode = [0, 0, 0]
        quantities_by_episode = []
        info = {}

        while not done:
            action, quantity = agent.act(currentState)
            next_state, reward, done, info = env.step(action, quantity)
            agent.remember(currentState, action, reward, next_state, done, quantity)
            currentState = next_state
            total_reward += reward
            actions_by_episode[action] += 1
            quantities_by_episode.append(quantity)

            step_counter += 1
            if step_counter % 5 == 0:
                agent.train()
                q_losses.append(agent.last_q_values_loss)
                total_losses.append(agent.last_total_loss)

        epsilon_history.append(agent.epsilon)
        actions.append(actions_by_episode)
        episode_rewards.append(total_reward)
        time_per_episode.append(time.time() - start_time)
        episode_infos.append(info.copy())
        episode_net_worth.append(info["net_worth"])
        mean_quantities.append(np.mean(quantities_by_episode))


        print(f"\n=== Episodio {episode + 1} ===")
        print(f"Recompensa total: {total_reward:.4f}")
        print(f"Net Worth final: {info['net_worth']:.2f}")
        print(f"Balance final: {info['balance']:.2f}")
        print(f"Acciones en cartera: {info['shares_held']}")
        print(f"Precio de cierre: {info['current_price']:.2f}")
        print(f"Tiempo de episodio: {time_per_episode[-1]:.2f} segundos")
        print("=" * 40)

    final_net_worths = np.array(episode_net_worth)
    mean_return_train = np.mean(final_net_worths - env.initial_balance)

    return {
        "rewards": episode_rewards,
        "net_worth": episode_net_worth,
        "times": time_per_episode,
        "infos": episode_infos,
        "mean_quantities": mean_quantities,
        "total_losses": total_losses,
        "q_losses": q_losses,
        "quantity_losses": quantity_losses,
        "actions": actions,
        "epsilons": epsilon_history,
        "mean_return_train": mean_return_train
    }



def evaluate_agent(agent, env, ticker):
    state = env.reset()
    done = False
    total_reward = 0
    net_worths = []
    rewards_test = []
    actions_test = [0,0,0]
    info = {}

    operaciones_totales = 0
    operaciones_ganadoras = 0

    prev_net_worth = env.net_worth

    while not done:
        action, quantity = agent.act(state)
        state, reward, done, info = env.step(action, quantity)
        total_reward += reward
        net_worths.append(info["net_worth"])
        rewards_test.append(total_reward)
        actions_test[action] += 1

        # Contamos operaciones activas
        if action in [0, 2]:  # Sell o Buy
            operaciones_totales += 1
            if info["net_worth"] > prev_net_worth:
                operaciones_ganadoras += 1

        prev_net_worth = info["net_worth"]

    final_net_worths = np.array(net_worths) #con la lista vamos bien para datos, pero no para operar con vectores

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

    os.makedirs("results", exist_ok=True)
    with open(f"results/evaluation_{ticker}.txt", "w") as f:
        f.write(f"Evaluación del agente ({ticker}):\n")
        f.write(f"Net worth final: {info['net_worth']:.2f}\n")
        f.write(f"Media de retorno neto: {mean_return:.2f}\n")
        f.write(f"Ratio de Sharpe: {sharpe:.4f}\n")
        f.write(f"Máximo Drawdown: {max_drawdown:.4%}\n")
        f.write(f"Precisión: {precision:.2%}\n")
        f.write(f"Recompensa total acumulada: {total_reward:.2f}\n")

    print(f"\nEvaluación del agente en evaluación {ticker}:")
    print(f"Media de retorno neto: {mean_return:.2f}")
    print(f"Ratio de Sharpe: {sharpe:.4f}")
    print(f"Máximo Drawdown: {max_drawdown:.4%}")

    print(f"Net worth final: {info['net_worth']:.2f}")
    print(f"Precisión del agente: {precision:.2%}")

    os.makedirs("results/", exist_ok=True)
    return net_worths, actions_test, rewards_test


def genera_graficas(ticker, stats, net_worths_test, rewards_test=None, actions_test=None):

    os.makedirs("plots", exist_ok=True)

    def save_plot(fig, name):
        fig.savefig(f"plots/{name}_{ticker}.png")
        plt.close(fig)

    # === Graficas de entrenamiento ===
    fig, ax = plt.subplots()
    ax.plot(stats["rewards"])
    ax.set_title(f"Recompensa por episodio: {ticker}")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Reward total")
    save_plot(fig, "reward_train")

    fig, ax = plt.subplots()
    ax.plot(stats["net_worth"])
    ax.set_title(f"Net Worth por episodio: {ticker}")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Net Worth final")
    save_plot(fig, "networth_train")

    fig, ax = plt.subplots()
    ax.plot(stats["q_losses"], label="Q Loss")
    ax.plot(stats["total_losses"], label="Total Loss")
    ax.set_title(f"Pérdidas por episodio: {ticker}")
    ax.set_xlabel("Paso de entrenamiento")
    ax.set_ylabel("Pérdida")
    ax.legend()
    ax.grid(True)
    save_plot(fig, "losses_train")

    fig, ax = plt.subplots()
    actions_array = np.array(stats["actions"])
    ax.plot(actions_array[:, 0], label="Sell")
    ax.plot(actions_array[:, 1], label="Hold")
    ax.plot(actions_array[:, 2], label="Buy")
    ax.set_title(f"Acciones por tipo (por episodio): {ticker}")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Número de acciones")
    ax.legend()
    ax.grid(True)
    save_plot(fig, "actions_train")

    fig, ax = plt.subplots()
    ax.plot(stats["mean_quantities"])
    ax.set_title(f"Cantidad promedio comprada/vendida: {ticker}")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Cantidad")
    ax.grid(True)
    save_plot(fig, "quantities_train")

    fig, ax = plt.subplots()
    ax.plot(stats["epsilons"])
    ax.set_title(f"Evolución de Epsilon: {ticker}")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Epsilon")
    ax.grid(True)
    save_plot(fig, "epsilon_train")

    # === Gráficas de evaluación (test) ===
    fig, ax = plt.subplots()
    ax.plot(net_worths_test)
    ax.set_title(f"Net Worth en datos de test: {ticker}")
    ax.set_xlabel("Paso")
    ax.set_ylabel("Net Worth")
    ax.grid(True)
    save_plot(fig, "networth_test")

    if rewards_test is not None:
        fig, ax = plt.subplots()
        ax.plot(rewards_test)
        ax.set_title(f"Recompensa por paso (test): {ticker}")
        ax.set_xlabel("Paso")
        ax.set_ylabel("Reward")
        ax.grid(True)
        save_plot(fig, "rewards_test")

    if actions_test is not None:
        fig, ax = plt.subplots()
        ax.bar(["Sell", "Hold", "Buy"], actions_test)
        ax.set_title(f"Acciones tomadas (test): {ticker}")
        ax.set_ylabel("Frecuencia")
        save_plot(fig, "actions_test")


file_list = [
    "data/Stocks/AAPL.csv",
    "data/Stocks/JPM.csv",
    "data/Commodity Futures/CL=F.csv",
    "data/Commodity Futures/GC=F.csv",
    "data/Stock Indices/DJI.csv",
    "data/Stock Indices/IBEX.csv",
    "data/Cryptocurrencies/BTC-USD.csv",
    "data/Cryptocurrencies/DOGE-USD.csv",
]

for filename in file_list:
    print("\n" + "=" * 60)
    print(f"Procesando ticker desde archivo: {filename}")
    print("=" * 60)

    # === Preparación de datos ===
    ticker = filename.split("/")[-1].split(".")[0]
    data = load_data(filename)
    split_index = int(0.8 * len(data))
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    window_size = 10
    feature_size = data.shape[1]
    state_size = window_size * feature_size

    env_train = TradingEnv(train_data, window_size, ticker)
    env_test = TradingEnv(test_data, window_size, ticker)
    agent = TradingAgent(state_size=state_size, action_size=3)

    # === Entrenamiento ===
    stats = train(agent, env_train, episodes=100)
    agent.q_network.save(f"models/full_{ticker}.keras")

    # === Evaluacion ===
    net_worths_test, actions_test, rewards_test = evaluate_agent(agent, env_test, ticker)

    # === Graficas ===
    genera_graficas(ticker, stats, net_worths_test, rewards_test, actions_test)

    # === Prediccion final del agente ===
    print("\n=== PREDICCION FINAL DEL AGENTE ===")
    final_env = TradingEnv(test_data, window_size, ticker)
    final_env.reset()
    final_env.current_step = len(test_data) - 1  # posiciona al final del test set
    final_state = final_env._next_observation()
    final_action, final_quantity = agent.act(final_state)

    acciones = ["SELL", "HOLD", "BUY"]
    print(f"Accion sugerida: {acciones[final_action]}")
    if final_action in [0, 2]:
        print(f"Cantidad sugerida: {int(final_quantity)} acciones")
    else:
        print("No se recomienda ninguna accion activa.")

    # === Limpieza de memoria ===
    print(f"\nLimpieza de memoria para {ticker}...")
    del agent, env_train, env_test, data, train_data, test_data, stats
    del net_worths_test, actions_test, rewards_test, final_env, final_state
    gc.collect()
    K.clear_session()

print("\n Proceso completado para todos los tickers.")
