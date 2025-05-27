# agent.py
import numpy as np
import random
from collections import deque
from deepQ_regressor import build_q_network, build_regressor


class TradingAgent:
    def __init__(self, state_size, action_size):
        #TODO: añadir documentacion explicando el agente
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = build_q_network(state_size, action_size)
        self.regressor = build_regressor(state_size)
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0 # parametro de exploracion
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.gamma = 0.95 # parametro de preferencia por recompensa cercana (greedy)
        self.batch_size = 64
        self.target_network = build_q_network(state_size, action_size)
        self.update_target_network()
        self.target_update_counter = 0

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), np.random.rand()
        state = state.reshape(1, -1) #aplanamos para poder usar el stack de estados como entrada
        q_values = self.q_network.predict(state, verbose=0) # TODO: preguntar por reescalado 3D con q_values = self.q_network.predict(state[np.newaxis, :, :], verbose=0)

        action = np.argmax(q_values[0])
        num_shares = self.regressor.predict(state, verbose=0)[0][0] * 100 # bien podria entrenar la red sobre 10 pero weno
        if num_shares < 1:
            num_shares = 1
        return action, int(num_shares)

    def remember(self, state, action, reward, next_state, done, quantity):
        self.memory.append((state, action, reward, next_state, done, quantity)) #example: TradingEnv_2(day 2 of trading), 1 (Hold), 0.25 (New Percentual reward), 2,

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        #TODO: Considerar cambiar stacks y arrays a tensores de TensorFlow.
        states = np.stack([transition[0].reshape(-1) for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.stack([transition[3].reshape(-1) for transition in batch])
        dones = np.array([transition[4] for transition in batch])

        # Predecir Q values para los proximos estados
        next_q_values = self.target_network.predict(next_states, verbose=0) # Red objetivo para mayor estabilidad
        max_next_q_values = np.amax(next_q_values, axis=1)

        # Computa los Q-Values objetivo
        targets = rewards + (1 - dones.astype(int)) * self.gamma * max_next_q_values #convertir dones a int

        target_f = self.q_network.predict(states, verbose=0) # Prediccion de q values para los estados actuales

        # Actualiza solo la accion escogida
        for i in range(self.batch_size):
            target_f[i][actions[i]] = targets[i]

        # Ajustar modelo
        self.q_network.fit(states, target_f, epochs=1, verbose=0)

        self.target_update_counter += 1
        if self.target_update_counter >= 50:
            self.update_target_network()
            self.target_update_counter = 0

        # Entrenar regresor
        quantity_target = np.array([transition[5] for transition in batch]) / 10.0
        self.regressor.fit(states, quantity_target.reshape(-1, 1), epochs=1, verbose=0)

        # Reducir epsilon (menos exploración con el tiempo)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())
