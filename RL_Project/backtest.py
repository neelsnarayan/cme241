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

class Backtester():
    """
    This class is to visualize the backtest of a given trading policy
    """

    def __init__(self, trading, policy):
        self.trading = trading #MDP  
        self.policy = policy #policy for the MDP
        self.lookback = None
        self.data = None

    def get_returns(self):

        start_states = self.trading.generate_start_state("test") # we take the test set data of the trading policy
        sequence = self.trading.simulate_actions(start_states, self.policy)

        bt = [] #will be used to build the backtest dataframe

        # Loop through each element in the sequence
        for x in sequence:
            bt.append([x.reward, x.state.state["date"]])
            #as we access x we populate the lookback and data attributes
            self.data = x.state.state["data"]
            self.lookback = x.state.state["lookback"]

        df = pd.DataFrame(bt, columns=['Reward', 'Date'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        self.returns = df


    def summary(self):
        """
        main summary of the backtest
        """
        self.get_returns()
        sharpe = (np.sqrt(252)*self.returns.mean()/self.returns.std())[0]
        cumulative_returns = np.exp(self.returns.cumsum()).iloc[:,0]
        u.plot_backtest_summary( cumulative_returns, self.data, self.lookback ,title=f"Sharpe Ratio {round(sharpe,2)}")
