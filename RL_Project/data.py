
import numpy as np
import pandas as pd
import utils as u
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
import rl
from rl.markov_decision_process import MarkovDecisionProcess
from rl.markov_process import State, MarkovProcess, NonTerminal, Terminal

from typing import (Callable, Dict, Generic, Iterator, Iterable, List,
                    Mapping, Optional, Sequence, Tuple, TypeVar, overload)

from rl.distribution import Categorical, Distribution, Constant, Choose
from rl.policy import Policy
from rl.monte_carlo import epsilon_greedy_policy, greedy_policy_from_qvf, glie_mc_control
from rl.function_approx import LinearFunctionApprox, AdamGradient


def generate_ou_process(sigma, mu, kappa, start_date, end_date, S0=100):
    """
    Generates a DataFrame with returns of an Ornstein-Uhlenbeck process over specific dates.

    Parameters:
    - sigma: Volatility of the process.
    - mu: Long-term mean level to which the process reverts.
    - kappa: Rate of reversion to the mean.
    - start_date: Start date of the simulation as a string (YYYY-MM-DD).
    - end_date: End date of the simulation as a string (YYYY-MM-DD).
    - S0: Initial value of the process, default is 100.

    Returns:
    - DataFrame with index as dates and a column 'Value' representing the evolution of the process.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' for business days
    n = len(dates)
    prices = np.zeros(n)
    prices[0] = S0
    dt = 1/252  # assuming 252 trading days in a year

    for t in range(1, n):
        dW = np.random.normal(0, np.sqrt(dt))  # increment of Wiener process
        prices[t] = prices[t-1] + kappa * (mu - prices[t-1]) * dt + sigma * dW

    return pd.DataFrame({
        'Value': prices
    }, index=dates)


def build_simulated_train_test(start='2019-01-01', end='2023-12-31', N = 100):
    #train
    train = []
    for _ in range(N):
        df = generate_ou_process(sigma=0.1, mu=100, kappa=7, start_date=start, end_date=end)
        train.append(df)

    #test
    df = generate_ou_process(sigma=0.1, mu=100, kappa=7, start_date=start, end_date=end)
    return train, df    


class Train_Test_Builder():
    """

    Prepare real data into [[episodes], test] where test is the start of the trading sample. 

    - main attribute is a data array
    - generates the [[episodes], test] needed to feed the AI agent. 
    - start_trading : date at which we want to start the trading (start of OOS)
    - end_trading : date at which we end the trading (end of OOS distribution)
    - start_train : date at which we start fitting the model
    - end_train : date at which we end sampling from training
    - lookback : minimum lookback that needs to be available for the ai agent  
    - length train : length of each training episode
    
    """

    def __init__(self,data, lookback=30, start_trading = None):
        self.data = data
        self.lookback = lookback
        if start_trading is not None:
            self.start_trading = start_trading
        else: #use 70%/30% split 
            index_at_70_percent = int(len(data) * 0.7)
            date_at_70_percent = data.index[index_at_70_percent]
            self.start_trading = date_at_70_percent

    def buil_test(self):
        start_trading = pd.to_datetime(self.start_trading)  # Convert start_date to datetime if it's a string
        lookback_date = start_trading - pd.offsets.BDay(self.lookback)
        self.test = self.data[lookback_date:]


    def build_train(self,N,length_episode):
        self.train = u.sample_dataframes(self.data,self.start_trading,N,length_episode,uniform=True)

    def build_train_test(self,N,length_episode):
        self.buil_test()
        self.build_train(N,length_episode)