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
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0 # parametro de exploracion
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95 # parametro de preferencia por recompensa cercana (greedy)
        self.batch_size = 32

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), np.random.rand()
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        action = np.argmax(q_values[0])
        num_shares = self.regressor.predict(state.reshape(1, -1), verbose=0)[0][0] * 10
        return action, int(num_shares)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in batch:
            # === Q-Network ===
            target = reward
            if not done:
                target += self.gamma * np.amax(
                    self.q_network.predict(next_state.reshape(1, -1), verbose=0)[0]
                )

            target_f = self.q_network.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target

            self.q_network.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

            # === Regresor (cantidad de acciones) ===
            # Reentrenamos el regresor con el target como proporción de éxito
            # Normalizamos el target para que esté entre 0 y 1
            quantity_target = np.clip(reward / 1000.0, 0, 1)
            self.regressor.fit(state.reshape(1, -1), np.array([[quantity_target]]), epochs=1, verbose=0)

        # Reducir epsilon (menos exploración con el tiempo)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
