from dataTreatment import load_data
from trading_environment import TradingEnv
from trading_agent import TradingAgent

data = load_data()
env = TradingEnv(data)
agent = TradingAgent(state_size=data.shape[1], action_size=3)
