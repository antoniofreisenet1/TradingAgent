# agent.py
import numpy as np
import random
from collections import deque
from deepQ_regressor import build_q_network, build_regressor

class TradingAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = build_q_network(state_size, action_size)
        self.regressor = build_regressor(state_size)
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.batch_size = 32

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), np.random.rand()
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        action = np.argmax(q_values[0])
        num_shares = self.regressor.predict(state.reshape(1, -1), verbose=0)[0][0] * 10
        return action, int(num_shares)
