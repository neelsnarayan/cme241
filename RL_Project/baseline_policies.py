
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

class ThresholdTradingPolicy(Policy[Dict,int]):
    """
    Implements the policy that consists in buying/selling at given thresholds
    """

    def __init__(self, enter_long, exit_long, enter_short, exit_short):
        """
        Setting the thresolds for trading decisions
        """
        self.enter_long = enter_long
        self.enter_short = enter_short
        self.exit_long = exit_long
        self.exit_short = exit_short

    def act(self, state: NonTerminal[Dict])->Distribution[int]:
        St = state.state["Spot"] #current spot, correponds to "t-1" if "t" is the time at the end of the step
        t = state.state["date"]
        pos = state.state["position"] #is +1  -1 or 0 

        action = 0

        if St >= self.enter_short and pos == 0:
            action = -1 #enter short

        if St <= self.exit_short and pos == -1:
            action = 1  #buy back to exit short

        if St <= self.enter_long and pos == 0:
            action = 1 #enter long

        if St >= self.enter_long and pos == 1:
            action = -1 #sell to exit long

        return Constant(action)

class BuyAndHold(Policy[Dict,int]):
    """
    Simply buys the asset at time 0 and holds 
    """
    def act(self, state: NonTerminal[Dict])->Distribution[int]:
        if state.state["time index"]==0:
            action = 1
        else:
            action = 0
        return Constant(action) 