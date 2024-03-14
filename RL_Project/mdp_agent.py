
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



def generate_initial_state_from_data(df,lookback=30,span=200):
    """
    generates the initial state dictionnary from the dataframes
    - lookback : initial start date requires a minimum of lookback days
    """
    df_ts_feature = u.ts_features(df,span=span)

    if lookback >= len(df):
        print(f"Not enough data in df to incorporate lookback of {lookback} BDays !")

    S0 = df.iloc[lookback][0]
    t = df.index[lookback]
    pos = 0
    mu_t = df_ts_feature.iloc[lookback]["mu_t"]
    sigma_t = df_ts_feature.iloc[lookback]["sigma_t"]


    state_dict = {
            "Spot" : S0,
            "position" : pos,
            "date" : t,
            "data" : df,
            "lookback" : lookback,
            "time index" : 0 ,
            "ts_features" : df_ts_feature,
            "mu_t" : mu_t,
            "sigma_t" : sigma_t
        }



    state =  NonTerminal(state_dict)
    

    return state

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

    def __init__(self,train,test,lookback=30):
        self.train = train
        self.test = test
        self.lookback = lookback
    
    def actions(self, state):
        if state.state["position"] == 1: #we are long
            acts = [-1,0] #we can sell to close long pos. or do nothing
        
        if state.state["position"] == -1: #we are short
            acts =  [0,1] #we can either hold or buy back
        
        else:
            acts = [-1,0,1]
        
        return acts
  
    def step(self, state, action)->Distribution[Tuple[State[Dict],float]]:

        #Get information about current state
        S_t_1 = state.state["Spot"] #current spot, correponds to "t-1" if "t" is the time at the end of the step
        t_1 = state.state["date"]
        data = state.state["data"]
        pos = state.state["position"] #is +1  -1 or 0 
        next_idx = state.state["time index"]+1
        lookback = state.state["lookback"]
        ts_feat = state.state["ts_features"]

        #Fetch next spot value and compute the return
        t, is_last = u.get_next(t_1, data)
        S_t = data.loc[t][0]
        r =  pos*np.log(S_t/S_t_1)#(S_t - S_t_1)/S_t use log returns so it is additive
        mu_t = ts_feat.loc[t]["mu_t"]
        sigma_t = ts_feat.loc[t]["sigma_t"]




        #Build next state
        next_state = {
            "Spot" :  S_t,
            "position" : np.sign(pos+action),
            "date" : t,
            "data" : data,
            "lookback":lookback,
            "time index":next_idx,
            "ts_features":ts_feat,
            "mu_t":mu_t,
            "sigma_t":sigma_t
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
            return Choose( [generate_initial_state_from_data(train_,self.lookback) for train_ in self.train] )
        elif which == "test":
            return Constant(generate_initial_state_from_data(self.test,self.lookback))  


    def build_q_approx(self):

        """
        
        Builds q value approximator
        Q : state -> float

        essential input for the RL algorithms

        """

        ###-- features blocks --##
        def intercept_feature(pos,act):
            return lambda x: 1 if ((x[0].state["position"]==pos)and (x[1] == act )) else 0

        def spot_feature(pos,act):
            return lambda x: x[0].state["Spot"] if  ((x[0].state["position"]==pos)and (x[1] == act )) else 0

        def spot_2_feature(pos, act):
            return lambda x: x[0].state["Spot"]**2 if  ((x[0].state["position"]==pos)and (x[1] == act )) else 0


        ###-- generate features together --##
        def generate_features():
            ffs = []
            #pos = 0
            ffs+=[
                #linear dependency for short
                intercept_feature(pos=0,act=-1),
                spot_feature(pos=0,act=-1),

                #constant dependency for hold
                intercept_feature(pos=0, act=0),
                
                #linear dependency for buy
                intercept_feature(pos=0,act=1),
                spot_feature(pos=0,act=1),
            ]
            #pos = 1s
            ffs+=[
                #linear dependency for close pos. 
                intercept_feature(pos=1,act=-1),
                spot_feature(pos=1,act=-1),

                #constant dependency for hold
                intercept_feature(pos=1, act=0),
            ]

            #pos = -1
            ffs +=[
                #linear dependency for close pos. 
                intercept_feature(pos=-1,act=1),
                spot_feature(pos=-1,act=1),

                #constant dependency for hold
                intercept_feature(pos=-1, act=0),
            ]

            return ffs

        ## -- build linear function approximator -- ##
        return LinearFunctionApprox.create(feature_functions=generate_features())

