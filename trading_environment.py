import numpy as np
import gym
from gym import spaces


class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = 10000

        #  Declaramos el espacio de observación para entrenar al agente-
        self.observation_space = spaces.Box(low=0, high=1, shape=(data.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: Sell, 1: Hold, 2: Buy

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = 10000
        return self._next_observation()

    def _next_observation(self):
        return self.data[self.current_step]

    def step(self, action):
        current_price = self.data[self.current_step][3]  # Precio de cierre normalizado

        if action == 0 and self.shares_held > 0:
            self.balance += self.shares_held * current_price * 100
            self.shares_held = 0 #TODO Esto vende todas las acciones automaticamente, ignorando la logica del algoritmo.
        elif action == 2:
            num_shares = self.balance // (current_price * 100)
            self.shares_held += num_shares
            self.balance -= num_shares * current_price * 100

        self.net_worth = self.balance + self.shares_held * current_price * 100
        reward = self.net_worth - 10000
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        return self._next_observation(), reward, done, {}
