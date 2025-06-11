# agent.py
import numpy as np
import random
from collections import deque
from deepQ_regressor import * #build_q_network, build_regressor


class TradingAgent:
    def __init__(self, state_size, action_size):
        self.last_total_loss = 0
        self.last_q_values_loss = 0
        self.last_quantity_loss = 0
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = build_combined_network(state_size) #regressor integrado en la red neuronal conjunta
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0 # parametro de exploracion
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99 # 0,99 es el mejor hasta ahora
        self.gamma = 0.97 # parametro de preferencia por recompensa cercana (greedy)
        self.batch_size = 64
        self.target_network = build_combined_network(state_size)
        self.target_update_counter = 0


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), np.random.rand()
        state = state.reshape(1, -1) #aplanamos para poder usar el stack de estados como entrada
        q_values, quantity = self.q_network.predict(state, verbose=0) #
        action = np.argmax(q_values[0])
        num_shares = quantity.item() * 10 # bien podria entrenar la red sobre 10 pero weno
        return action, num_shares

    def remember(self, state, action, reward, next_state, done, quantity):
        self.memory.append((state, action, reward, next_state, done, quantity*10)) #example: TradingEnv_2(day 2 of trading), 1 (Hold), 0.25 (New Percentual reward), 2,

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states = np.stack([transition[0].reshape(-1) for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.stack([transition[3].reshape(-1) for transition in batch])
        dones = np.array([transition[4] for transition in batch])
        quantity_target = np.array([transition[5] for transition in batch]) / 10.0 #entre 10 para mantener prediccion de la red
        # Predecir Q values para los proximos estados
        next_q_values, _ = self.target_network.predict(next_states, verbose=0) # Red objetivo para mayor estabilidad
        max_next_q_values = np.amax(next_q_values, axis=1)

        # Computa los Q-Values objetivo
        targets = rewards + (1 - dones.astype(int)) * self.gamma * max_next_q_values #convertir dones a int

        target_q, _ = self.q_network.predict(states, verbose=0) # Prediccion de q values para los estados actuales

        # Actualiza solo la accion escogida
        for i in range(self.batch_size):
            target_q[i][actions[i]] = targets[i]

        # Ajustar modelo
        history = self.q_network.fit(states, {"q_output": target_q, "quantity_output": quantity_target.reshape(-1, 1)}, epochs=1, verbose=0)

        self.last_quantity_loss = history.history["quantity_output_loss"][0]
        self.last_q_values_loss = history.history["q_output_loss"][0]
        self.last_total_loss = history.history["loss"][0]

        self.target_update_counter += 1
        if self.target_update_counter >= 50:
            self.target_network.set_weights(self.q_network.get_weights())
            self.target_update_counter = 0

        # Reducir epsilon (menos exploración con el tiempo)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay