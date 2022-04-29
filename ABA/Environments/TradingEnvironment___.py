import gym
import pandas as pd
import numpy as np
from numpy import inf
from gym import spaces
from sklearn import preprocessing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from empyrical import sortino_ratio, calmar_ratio, omega_ratio
import importlib

import ABA.Environments.Render.Render_Graph
import ABA.Environments.benchmarks.benchmarks
importlib.reload(ABA.Environments.Render.Render_Graph)
importlib.reload(ABA.Environments.benchmarks.benchmarks)
from ABA.Environments.Render.Render_Graph import BitcoinTradingGraph
from ABA.Environments.benchmarks.benchmarks import buy_and_hold

class TradingEnvironment(gym.Env):

    metadata = {'render.modes': ['human', 'system', 'none']}
    viewer = None

    def __init__(self, data, initial_balance=10000, commission=0.0025, reward_function='sortino', forecast_length=10, confidence_interval=0.95, **kwargs):

        super(TradingEnvironment, self).__init__()

        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_function = reward_function
        data=data.reset_index(level=0) #isws na mh xreiazetai telika edw
        self.df = data.drop(['log_Low_diff_BTC', 'log_Volume_diff_BTC', 'log_High_diff_BTC', 'log_Open_diff_BTC','log_Close_diff_BTC','log_Low_diff_GOLD',
       'log_Volume_diff_GOLD', 'log_High_diff_GOLD', 'log_Open_diff_GOLD',
       'log_Close_diff_GOLD' ],axis=1)
        self.df=self.df.fillna(method='bfill')
        self.df=self.df.fillna(method='ffill')


        self.stationary_df = data.drop(['Date','Open_BTC', 'High_BTC', 'Low_BTC', 'Close_BTC','Volume_BTC','Open_GOLD', 'High_GOLD', 'Low_GOLD', 'Close_GOLD','Volume_GOLD'],axis=1)
        self.stationary_df=self.stationary_df.fillna(method='bfill')
        self.stationary_df=self.stationary_df.fillna(method='ffill')


        benchmarks = kwargs.get('benchmarks', [])

        self.benchmarks = [
            {
                'label': 'Buy and HOLD',
                'values': buy_and_hold(self.df['Close_BTC'], self.df['Close_GOLD'], initial_balance, commission)
            },
            *benchmarks,
        ]
        self.forecast_length = forecast_length
        self.confidence_interval =confidence_interval
        self.obs_shape = (1, 9 + len(self.df.columns)-1 + (self.forecast_length * 6))

        self.action_space = spaces.Discrete(18)
        self.action_array=np.array([[1,0],
          [0,1],
          [0,0],
          [0,0.25],
          [0,0.33],
          [0,0.5],
          [0.25,0],
          [0.25,0.25],
          [0.25,0.33],
          [0.25,0.5],
          [0.33,0],
          [0.33,0.25],
          [0.33,0.33],
          [0.33,0.5],
          [0.5,0],
          [0.5,0.25],
          [0.5,0.33],
          [0.5,0.5]])

        # Observes the price action, indicators, account action, price forecasts
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float16)

    def _next_observation(self):
        scaler = preprocessing.MinMaxScaler()

        #  features = self.stationary_df[self.stationary_df.columns.difference(['index', 'Date'])]

        scaled =self.stationary_df[:self.current_step + self.forecast_length + 1].values
        scaled[abs(scaled) == inf] = 0



        scaled = scaler.fit_transform(scaled.astype('float64'))
        scaled = pd.DataFrame(scaled, columns=self.stationary_df.columns)

        obs = scaled.values[-1]

        #Fit a SARIMMA model to the timeseries of the bitcoin prices
        past_df_BTC = self.stationary_df['log_Close_diff_BTC'][: self.current_step + self.forecast_length + 1]

        forecast_model_BTC = SARIMAX(past_df_BTC.values, enforce_stationarity=False, simple_differencing=True)
        model_fit_BTC = forecast_model_BTC.fit(method='bfgs', disp=False)
        forecast_BTC= model_fit_BTC.get_forecast(steps=self.forecast_length, alpha=(1 - self.confidence_interval))

        obs = np.insert(obs, len(obs), forecast_BTC.predicted_mean, axis=0)
        obs = np.insert(obs, len(obs), forecast_BTC.conf_int().flatten(), axis=0)

        #Fit a SARIMMA model to the timeseries of the gold prices
        past_df_GOLD = self.stationary_df['log_Close_diff_GOLD'][: self.current_step + self.forecast_length + 1]

        forecast_model_GOLD = SARIMAX(past_df_GOLD.values, enforce_stationarity=False, simple_differencing=True)
        model_fit_GOLD = forecast_model_GOLD.fit(method='bfgs', disp=False)
        forecast_GOLD= model_fit_GOLD.get_forecast(steps=self.forecast_length, alpha=(1 - self.confidence_interval))


        obs = np.insert(obs, len(obs), forecast_GOLD.predicted_mean, axis=0)
        obs = np.insert(obs, len(obs), forecast_GOLD.conf_int().flatten(), axis=0)

        scaled_history = scaler.fit_transform(self.account_history.astype('float64'))

        obs = np.insert(obs, len(obs), scaled_history[:, -1], axis=0)

        obs = np.reshape(obs.astype('float64'), self.obs_shape)
        obs[np.bitwise_not(np.isfinite(obs))] = 0
        obs[np.isnan(obs)] = 0

        return obs

    #Get current BTC price
    def get_current_price_BTC(self):
        return self.df['Close_BTC'].values[self.current_step + self.forecast_length]

    #Get current GOLD price
    def get_current_price_GOLD(self):
        return self.df['Close_GOLD'].values[self.current_step + self.forecast_length]

    def _take_action(self, action):

        #Observe current price of BTC and GOLD
        current_price_BTC = self.get_current_price_BTC()
        current_price_GOLD = self.get_current_price_GOLD()

        btc_bought = 0
        btc_sold = 0
        cost_btc = 0
        sales_btc = 0
        gold_bought = 0
        gold_sold = 0
        cost_gold = 0
        sales_gold = 0

        self.current_weights=self.action_array[action]
        self.portfolio_change=self.current_weights-self.previous_weights

        net_worth = (self.balance + self.btc_held * current_price_BTC+ self.gold_held * current_price_GOLD)
        print("net worth:", net_worth)
        if(self.portfolio_change[0]<0): # we sold bitcoins
            price_BTC = current_price_BTC * (1 -self.commission)
            sales_btc = net_worth* abs(self.portfolio_change[0])
            btc_sold = sales_btc / price_BTC
            self.btc_held -= btc_sold
            self.balance += sales_btc

        if(self.portfolio_change[1]<0): # we sold gold
            price_GOLD = current_price_GOLD * (1 -self.commission)
            sales_gold = net_worth * abs(self.portfolio_change[1])
            gold_sold = sales_gold / price_GOLD
            self.gold_held -= gold_sold
            self.balance += sales_gold

        if(self.portfolio_change[0]>0): # we bought bitcoins
            price_BTC = current_price_BTC * (1 + self.commission)
            cost_btc = net_worth* self.portfolio_change[0]
            btc_bought = cost_btc / price_BTC
            self.btc_held += btc_bought
            self.balance -= cost_btc

        if(self.portfolio_change[1]>0): # we bought gold
            price_GOLD = current_price_GOLD * (1 + self.commission)
            cost_gold = net_worth * self.portfolio_change[1]
            gold_bought = cost_gold / price_GOLD
            self.gold_held += gold_bought
            self.balance -= cost_gold

        if btc_sold > 0 or btc_bought > 0:
            self.trades_btc.append({'step': self.current_step,
                                'amount': btc_sold if btc_sold > 0 else btc_bought, 'total': sales_btc if btc_sold > 0 else cost_btc,
                                'type': 'sell' if btc_sold > 0 else 'buy'})

        if gold_sold > 0 or gold_bought > 0:
            self.trades_gold.append({'step': self.current_step,
                                'amount': gold_sold if gold_sold > 0 else gold_bought, 'total': sales_gold if gold_sold > 0 else cost_gold,
                                'type': 'sell' if gold_sold > 0 else 'buy'})

        self.net_worths.append(self.balance + self.btc_held * current_price_BTC+ self.gold_held * current_price_GOLD)
        #print(self.net_worths[self.current_step])

        self.account_history = np.append(self.account_history, [
            [self.balance],
            [btc_bought],
            [cost_btc],
            [btc_sold],
            [sales_btc],
            [gold_bought],
            [cost_gold],
            [gold_sold],
            [sales_gold]], axis=1)

        self.previous_weights=self.current_weights

    def _reward(self):
        length = min(self.current_step, self.forecast_length)
        returns = np.diff(self.net_worths[-length:])

        if np.count_nonzero(returns) < 1:
            return 0

        if self.reward_function == 'sortino':
            reward = sortino_ratio(returns, annualization=365*24)

        elif self.reward_function == 'calmar':
            reward = calmar_ratio(
                returns, annualization=365*24)
        elif self.reward_function == 'omega':
            reward = omega_ratio(
                returns, annualization=365*24)
        else:
            reward = returns[-1]

        return reward if np.isfinite(reward) else 0

    def _done(self):
       # if we have less than initial_balance /10 or we are in the last row of the dataframe then stop
        return self.net_worths[-1] < self.initial_balance / 10 or self.current_step == len(self.df) - self.forecast_length - 1

    def reset(self):
        self.previous_weights=self.action_array[2]
        self.balance = self.initial_balance
        self.net_worths = [self.initial_balance]
        self.btc_held = 0
        self.current_step = 0
        self.gold_held=0


   # account history: balance, btc_bought, cost of btc, btc_sold,sales from btc,gold_bought, cost of gold, gold_sold,sales from gold
        self.account_history = np.array([
            [self.balance],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0]
        ])
    # trades {step: current_step,
    # amount: btc_sold or btc_bought
    # total: sales or cost of btc,
    # type: 'sell' or 'buy' btc}

        self.trades_btc = []

    # trades {step: current_step,
    # amount: gold_sold or gold_bought
    # total: sales or cost,
    # type: 'sell' or 'buy' gold}

        self.trades_gold = []

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
                print(self.viewer)
                self.viewer = BitcoinTradingGraph(self.df)

            self.viewer.render(self.current_step, self.net_worths, self.benchmarks, self.trades_btc,self.trades_gold)

    def return_net_worth(self):
        return self.net_worths

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
