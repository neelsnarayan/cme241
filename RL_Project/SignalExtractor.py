#standard imports
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")

#rl book imports
import rl
from rl.markov_decision_process import MarkovDecisionProcess
from rl.markov_process import State, MarkovProcess, NonTerminal, Terminal

from typing import (Callable, Dict, Generic, Iterator, Iterable, List,
                    Mapping, Optional, Sequence, Tuple, TypeVar, overload)


from rl.distribution import Categorical, Distribution, Constant, Choose
from rl.policy import Policy
from rl.monte_carlo import epsilon_greedy_policy, greedy_policy_from_qvf, glie_mc_control, mc_prediction
from rl.function_approx import LinearFunctionApprox, AdamGradient
from rl.td import glie_sarsa, q_learning
from rl.td_lambda import td_lambda_prediction


#custom imports 
import utils as u
import data as dat
import mdp_agent as ag
import baseline_policies as bp
import q_plots as qp
import backtest as btest
import v_predictor as v_true
import v_convergence as v_conv 
import v_plots2 as v2
import pickle

GAMMA = 0.8

class RollingMeanRevPrediction():

    def __init__(self,data,lookback, length_epsiode, num_iter,refit_freq, min_obs = 300):
        self.data = data
        self.length_epsiode = length_epsiode
        self.refit_freq = refit_freq
        self.lookback = lookback
        self.num_iter = num_iter
        self.signals = []
        self.next_split = "1800-01-01"
        self.min_obs  = min_obs

        self.prediction = None


    def fit_rolling(self):
        split  = self.data.index[self.min_obs]
        end_of_data = False
        while not end_of_data:
            end_of_data = self.fit(split)
            split = self.next_split
        
        self.prediction = pd.concat(self.signals)


    def fit(self,split):
        """
        Generates signal for a single OOS period of size self.refit_freq
        """
        
        build_data = dat.Train_Test_Builder(self.data,self.lookback,split)
        build_data.build_train_test(N=30,length_episode=300)
        train,test = build_data.train, build_data.test

        trader = ag.Trading(train, test, self.lookback)
        mrp_buy = trader.apply_policy(bp.BuyAndHold())

        def intercept(x):
            return 1

        def z_score_state(x):
            return (x.state["Spot"] - x.state["mu_t"])/(x.state["sigma_t"])

        def trend_mu(x):
            return (x.state["trend"])

        def ar1(x):
            return (x.state["ar1"])

        adam_g = AdamGradient(
                learning_rate=0.0001,
                decay1=0.9,
                decay2=0.999
            )    

        v_approx = LinearFunctionApprox.create(feature_functions=[intercept,z_score_state,trend_mu,ar1],
                                                adam_gradient=adam_g,
                                                direct_solve=False)

        initial_state = trader.generate_start_state("train")
        traces = mrp_buy.reward_traces(initial_state)
        vfs_td_lambda = td_lambda_prediction(
                traces,
                v_approx,
                GAMMA,
                0.5
        )
        td_l = v_conv.V_Convergence_Analyzer(vfs_td_lambda,trader)

        td_l.get_v(self.num_iter)

        signal = {}

        for i_,state in enumerate(u.get_list_states(trader,"test",start=split)):
            if i_+1 >= self.refit_freq:
                self.next_split = state.state["date"]
                break   
            else:
                signal[state.state["date"]] = td_l.v(state)
        new_signal = pd.DataFrame.from_dict(signal, orient='index', columns=['signal'])
        end_of_data = len(new_signal)+1<self.refit_freq
        self.signals.append(new_signal)

        return end_of_data        