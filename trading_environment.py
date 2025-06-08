import numpy as np
import gym
from joblib import load
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, data, window_size=10, ticker_name = "AAPL"):
        super(TradingEnv, self).__init__()
        self.data = data.values # Los entornos de gym esperan espacios con numpy.arrays
        self.current_step = 0
        self.initial_balance = 10000 # Parametro modificable: indica el dinero inicial
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.percent_max_buy_sell = 0.2
        self.percent_penalize_hold = 0.005
        self.history = [] # Historial: sólo para propósitos de visualización
        self.close_scaler = load("scalers/scaler_Close_" + ticker_name + ".pkl")

        #  Declaramos el espacio de observación para entrenar al agente-
        self.action_space = spaces.Discrete(3)  # 0: Sell, 1: Hold, 2: Buy
        self.window_size = window_size
        self.observation_space = spaces.Box(low=0, high=1, shape=(window_size, data.shape[1]), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.history = []
        return self._next_observation()

    def _next_observation(self):
        #Añade contexto temporal
        start = max(0, self.current_step - self.window_size + 1)
        obs = self.data[start:self.current_step + 1]
        if len(obs) < self.window_size:
            first_row = obs[0]
            padding = np.repeat(first_row[np.newaxis, :], self.window_size - len(obs), axis = 0) #changed to a repeat of first value
            obs = np.vstack((padding, obs))
        return obs

    def step(self, action, quantity = 0):
        if self.current_step >= len(self.data):
            raise Exception("Episode already finished")
        normal_close = self.data[self.current_step][3] # Comprobar si no es mas facil pasar una matriz entera
        current_price = self.close_scaler.inverse_transform([[normal_close]])[0][0] # Precio de cierre sin normalizar
        if action == 0 and self.shares_held > 0:  # Vender
            if quantity < 0:
                quantity = 0 # Check por si acaso quantity es negativo
            max_sale = max(1.0, self.shares_held * self.percent_max_buy_sell)
            num_shares = min(1.0 * quantity, max_sale, self.shares_held) #venta conservadora
            self.balance += num_shares * current_price
            self.shares_held -= num_shares
        elif action == 2:  # Comprar
            if quantity < 0:
                quantity = 0 # Check por si acaso quantity es negativo
            max_investment = self.balance * self.percent_max_buy_sell
            max_shares = max_investment / current_price
            num_shares = min(max(quantity, 0), max_shares) # Siempre entero, siempre positivo, pero siempre menor posible todo compra arriesgada???
            self.shares_held += num_shares
            self.balance -= num_shares * current_price

        self.net_worth = self.balance + self.shares_held * current_price # Valor total de la cartera
        # print(f"action: {action},current_price:{current_price}, balance: {self.balance}, net_worth: {self.net_worth}, quantity:{quantity}")
        if self.current_step == 0:
            reward = 0
        else:
            reward = (self.net_worth - self.prev_net_worth) / self.initial_balance #cambiada recompensa logaritmica

        self.prev_net_worth = self.net_worth
        self.current_step += 1

        # Penalizamos por holdear demasiado:
        if action == 1 and self.shares_held == 0:
            reward -= self.percent_penalize_hold

        #Done: condición de llegar al último paso... o de llegar a la bancarrota (5% del valor inicial)
        done = self.current_step >= len(self.data) - 1 or self.net_worth < self.initial_balance * 0.05

        if done:
            reward += (self.net_worth - self.initial_balance) / self.initial_balance

        info = {
            "net_worth": self.net_worth,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "current_price": current_price,
            "returns": reward
        }

        return self._next_observation(), reward, done, info
