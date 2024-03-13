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


#custom imports 
import utils as u
import data as dat
import mdp_agent as ag
import baseline_policies as bp
import q_plots as qp
import backtest as btest

from sklearn.linear_model import LinearRegression


def get_sde_step(X_t_1,mu,sigma,kappa,time_step=0):
    # This is a placeholder for the actual SDE step function you mentioned.
    # It should return X_{t+dt} given the current X_t.
    dt = (1/252)
    dW = np.random.normal(0, np.sqrt(dt))  # increment of Wiener process
    X_t = X_t_1 + kappa(time_step) * (mu(time_step)  - X_t_1) * dt + sigma(time_step)* dW #step is trend parameter
    return X_t


class Prediction():


    def __init__(self,vfs, params,gamma=0.9):

        self.vfs = vfs #iterator of value functions for which we want to analyze the convergence
        self.params = params  #parameters mu(t), sigma(t), kappa(t) of the OU process can be constants
        self.gamma = gamma


    def monte_carlo(self,x,num_simulations=100,num_steps=1000):

        sigma = self.params["sigma"] #function giving sigma
        kappa = self.params["kappa"] #function giving kappa
        mu = self.params["mu"]

        total_rewards = np.zeros(num_simulations)

        gamma  = self.gamma
    
        for sim in range(num_simulations):
            X_t = x
            reward = 0
            simu = [x]
            for t in range(num_steps):
                X_next = get_sde_step(X_t,mu,sigma=sigma,kappa=kappa,time_step=t)
                reward += ( gamma ** t) * np.log(X_next / X_t)
                X_t = X_next
                simu.append(X_t)

            total_rewards[sim] = reward
        
        expected_reward = np.mean(total_rewards)
        return expected_reward



        




    
