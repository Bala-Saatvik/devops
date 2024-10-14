import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym import Env
from gym.spaces import Discrete, Box
import random
import plotly.graph_objs as go

# Sample RL Environment (you can replace this with your actual environment)
class TradingEnv(Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.action_space = Discrete(3)  # Buy, Sell, Hold
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)
    
    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step].values

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
        else:
            done = False
        reward = random.uniform(-1, 1)  # Replace with actual reward calculation
        obs = self.data.iloc[self.current_step].values
        return obs, reward, done, {}

    def render(self):
        pass

# RL Agent (Placeholder for actual implementation)
class TradingAgent:
    def __init__(self):
        pass
    
    def train(self, env, episodes=10):
        rewards = []
        for episode in range(episodes):
            obs = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = env.action_space.sample()  # Random action, replace with actual policy
                obs, reward, done, _ = env.step(action)
                total_reward += reward
            rewards.append(total_reward)
        return rewards

    def predict(self, obs):
        return random.choice([0, 1, 2])  # Replace with actual model prediction

# Load or Generate Data
def load_data():
    dates = pd.date_range('2023-01-01', periods=100)
    prices = np.random.rand(100) * 100
    data = pd.DataFrame({'Date': dates, 'Price': prices})
    return data

# Streamlit App
st.title("Reinforcement Learning Based Trading Bot")

# Sidebar options
st.sidebar.header("User Input")
episodes = st.sidebar.slider("Number of training episodes", 1, 100, 10)

# Load Data
data = load_data()
st.write("Stock Price Data")
st.line_chart(data.set_index('Date')['Price'])

# Initialize Environment and Agent
env = TradingEnv(data)
agent = TradingAgent()

# Train the Agent
st.write(f"Training the agent for {episodes} episodes...")
rewards = agent.train(env, episodes=episodes)
st.write("Training complete!")
st.line_chart(rewards)

# Simulate Trading
st.write("Simulating trading based on the trained model...")
obs = env.reset()
actions = []
for _ in range(len(data) - 1):  # Length should be len(data) - 1 to match with actions
    action = agent.predict(obs)
    actions.append(action)
    obs, reward, done, _ = env.step(action)

# Add a placeholder action for the last data point to match the lengths
actions.append(agent.predict(obs))

actions = pd.Series(actions).replace({0: 'Buy', 1: 'Sell', 2: 'Hold'})

# Calculate Profit and Stats
capital = 10000  # Starting capital
position = 0  # Current position (0 means no position, positive means holding a buy position)
profits = []
trade_dates = []
buy_price = 0

for i in range(len(actions)):
    if actions[i] == 'Buy' and position == 0:  # Buy signal and no position
        position = capital / data['Price'][i]  # Number of shares
        buy_price = data['Price'][i]
        trade_dates.append(data['Date'][i])
    elif actions[i] == 'Sell' and position > 0:  # Sell signal and holding a position
        capital = position * data['Price'][i]  # Sell all shares
        profits.append(capital - 10000)  # Calculate profit from the trade
        position = 0
        trade_dates.append(data['Date'][i])

total_profit = sum(profits)
num_trades = len(profits)
win_rate = (len([p for p in profits if p > 0]) / num_trades) * 100 if num_trades > 0 else 0

# Display Stats
st.write(f"Total Profit: ${total_profit:.2f}")
st.write(f"Number of Trades: {num_trades}")
st.write(f"Win Rate: {win_rate:.2f}%")

# Interactive Plot with Plotly
fig = go.Figure()

# Price trace
fig.add_trace(go.Scatter(x=data['Date'], y=data['Price'], mode='lines', name='Price'))

# Buy signals
buy_signals = data['Date'][actions == 'Buy']
fig.add_trace(go.Scatter(x=buy_signals, y=data['Price'][actions == 'Buy'],
                         mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', size=10, color='green')))

# Sell signals
sell_signals = data['Date'][actions == 'Sell']
fig.add_trace(go.Scatter(x=sell_signals, y=data['Price'][actions == 'Sell'],
                         mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', size=10, color='red')))

# Update layout for better visualization
fig.update_layout(title="Interactive Stock Price and Trading Signals",
                  xaxis_title="Date",
                  yaxis_title="Price",
                  legend_title="Legend",
                  hovermode="x unified")

st.plotly_chart(fig)
