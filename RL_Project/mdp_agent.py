
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



def generate_initial_state_from_data(df):
    """
    generates the initial state dictionnary from the dataframes
    """
    S0 = df.iloc[0][0]
    t = df.index[0]
    pos = 0
    return NonTerminal(
        {
            "Spot" : S0,
            "position" : 0,
            "date" : t,
            "data" : df
        }
    )

class Trading(MarkovDecisionProcess[Dict,int]):
    """
    - train is a list of dataframes representing price processes we want to trade
    - test is a dataframe in which we want to evaluate the policy
    - a state is a dictionnary
        {
            "Spot" : price S_t
            "position" : long/short (-1 or 1)
            "date" : current time step
            "data" : dataframe with price process
        }
    - actions : hold, buy or sell
    """

    def __init__(self,train,test):
        self.train = train
        self.test = test
    
    def actions(self, state):
        return [-1,0,1] #short hold buy
  
    def step(self, state, action)->Distribution[Tuple[State[Dict],float]]:
        #get information about current state
        S_t_1 = state.state["Spot"] #current spot, correponds to "t-1" if "t" is the time at the end of the step
        t_1 = state.state["date"]
        data = state.state["data"]
        pos = state.state["position"] #is +1  -1 or 0 

        #Fetch next spot value and compute the return
        t, is_last = u.get_next(t_1, data)
        S_t = data.loc[t][0]
        r =  pos*(S_t - S_t_1)/S_t

        #build next state
        next_state = {
            "Spot" :  S_t,
            "position" : np.sign(pos+action),
            "date" : t,
            "data" : data
        }
        if is_last:
            next_state = Terminal(next_state)
        else:
            next_state = NonTerminal(next_state)
        return Constant((next_state,r))

    
    def generate_start_state(self,which = "train"):
        """
        Generates the initial distribution of the state from the available training data
        """
        if which == "train":
            return Choose( [generate_initial_state_from_data(train_) for train_ in self.train] )
        elif which == "test":
            return Constant(generate_initial_state_from_data(self.test))  