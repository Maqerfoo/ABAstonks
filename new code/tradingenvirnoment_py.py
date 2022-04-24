# -*- coding: utf-8 -*-
"""TradingEnvirnoment.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hBGZZk93qv1M_ygGfawYaPprlsFGGw-v
"""

!pip install empyrical

import gym
import pandas as pd
import numpy as np
from numpy import inf
from gym import spaces
from sklearn import preprocessing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from empyrical import sortino_ratio, calmar_ratio, omega_ratio

#from render.BitcoinTradingGraph import BitcoinTradingGraph
#from util.stationarization import log_and_difference
#from util.benchmarks import buy_and_hodl, rsi_divergence, sma_crossover
#from util.indicators import add_indicators

class TradingEnvirnoment(gym.Env):
   
    metadata = {'render.modes': ['human', 'system', 'none']}
    viewer = None

    def __init__(self, df, stationary_df, initial_balance=10000, commission=0.0025, reward_function='sortino', forecast_length=10, confidence_interval=0.95):
        super(TradingEnvirnoment, self).__init__()

        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_function = reward_function

        self.df = df
        self.stationary_df = stationary_df

        benchmarks = kwargs.get('benchmarks', [])
        self.benchmarks = [
            {
                'label': 'Buy and HODL',
                'values': buy_and_hodl(self.df['Close'], initial_balance, commission)
            },
            {
                'label': 'RSI Divergence',
                'values': rsi_divergence(self.df['Close'], initial_balance, commission)
            },
            {
                'label': 'SMA Crossover',
                'values': sma_crossover(self.df['Close'], initial_balance, commission)
            },
            *benchmarks,
        ]

        self.forecast_length = forecast_length
        self.confidence_interval =confidence_interval
        self.obs_shape = (1, 5 + len(self.df.columns) + (self.forecast_length * 3))

        # Actions: from 0 to 3 means "buy" 1 or 0.5 or 0.33 or 0.25 amount
        #          from 4 to 7 means "sell" 1 or 0.5 or 0.33 or 0.25 amount
        #          from 8 to 11 means hold, amount ignored 
        self.action_space = spaces.Discrete(12)

        # Observes the price action, indicators, account action, price forecasts
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float16)

    def _next_observation(self):
        scaler = preprocessing.MinMaxScaler()

        #  features = self.stationary_df[self.stationary_df.columns.difference(['index', 'Date'])]

        scaled =self.stationary_df[:self.current_step + self.forecast_length + 1].values
        scaled[abs(scaled) == inf] = 0
        scaled = scaler.fit_transform(scaled.astype('float32'))
        scaled = pd.DataFrame(scaled, columns=self.stationary_df.columns)

        obs = scaled.values[-1]

        
        past_df = self.stationary_df['Close'][: self.current_step + self.forecast_length + 1]

        forecast_model = SARIMAX(past_df.values, enforce_stationarity=False, simple_differencing=True)
        model_fit = forecast_model.fit(method='bfgs', disp=False)
        forecast = model_fit.get_forecast(steps=self.forecast_length, alpha=(1 - self.confidence_interval))

        obs = np.insert(obs, len(obs), forecast.predicted_mean, axis=0)
        obs = np.insert(obs, len(obs), forecast.conf_int().flatten(), axis=0)

        scaled_history = scaler.fit_transform(self.account_history.astype('float32'))

        obs = np.insert(obs, len(obs), scaled_history[:, -1], axis=0)

        obs = np.reshape(obs.astype('float16'), self.obs_shape)
        #obs[np.bitwise_not(np.isfinite(obs))] = 0 ???

        return obs

    def get_current_price(self):
        return self.df['Close'].values[self.current_step + self.forecast_length] #+0.01??

    def _take_action(self, action):
        current_price = self.get_current_price()
        action_type = int(action / 4)
        amount = 1 / (action % 4 + 1)

        btc_bought = 0
        btc_sold = 0
        cost = 0
        sales = 0

        if action_type == 0: # buy
            price = current_price * (1 + self.commission)
            btc_bought = min(self.balance * amount /price, self.balance / price)
            cost = btc_bought * price

            self.btc_held += btc_bought
            self.balance -= cost
        elif action_type == 1: # sell
            price = current_price * (1 - self.commission)
            btc_sold = self.btc_held * amount
            sales = btc_sold * price

            self.btc_held -= btc_sold
            self.balance += sales

        if btc_sold > 0 or btc_bought > 0:
            self.trades.append({'step': self.current_step,
                                'amount': btc_sold if btc_sold > 0 else btc_bought, 'total': sales if btc_sold > 0 else cost,
                                'type': 'sell' if btc_sold > 0 else 'buy'})

        self.net_worths.append(self.balance + self.btc_held * current_price)

        self.account_history = np.append(self.account_history, [
            [self.balance],
            [btc_bought],
            [cost],
            [btc_sold],
            [sales]
        ], axis=1)

    def _reward(self):
        length = min(self.current_step, self.forecast_length)
        returns = np.diff(self.net_worths[-length:])

        if np.count_nonzero(returns) < 1:
            return 0

        if self.reward_func == 'sortino':
            reward = sortino_ratio(
                returns, annualization=365*24)
        elif self.reward_func == 'calmar':
            reward = calmar_ratio(
                returns, annualization=365*24)
        elif self.reward_func == 'omega':
            reward = omega_ratio(
                returns, annualization=365*24)
        else:
            reward = returns[-1]

        return reward if np.isfinite(reward) else 0

    def _done(self):
       # if we have less than initial_balance /10 or we are in the last row of the dataframe then stop
        return self.net_worths[-1] < self.initial_balance / 10 or self.current_step == len(self.df) - self.forecast_length - 1

    def reset(self):
        self.balance = self.initial_balance
        self.net_worths = [self.initial_balance]
        self.btc_held = 0
        self.current_step = 0

        self.account_history = np.array([
            [self.balance],
            [0],
            [0],
            [0],
            [0]
        ])
        self.trades = []

        return self._next_observation()

    def step(self, action):
        self._take_action(action)

        self.current_step += 1

        obs = self._next_observation()
        reward = self._reward()
        done = self._done()

        return obs, reward, done, {}

    def render(self, mode='human'):
        if mode == 'system':
            print('Price: ' + str(self.get_current_price()))
            print(
                'Bought: ' + str(self.account_history[2][self.current_step]))
            print(
                'Sold: ' + str(self.account_history[4][self.current_step]))
            print('Net worth: ' + str(self.net_worths[-1]))

        elif mode == 'human':
            if self.viewer is None:
                self.viewer = BitcoinTradingGraph(self.df)

            self.viewer.render(self.current_step,
                               self.net_worths, self.benchmarks, self.trades)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

