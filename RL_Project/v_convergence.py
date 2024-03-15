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
import datetime
import random

import wandb


class V_Convergence_Analyzer():

    def __init__(self,vf_iterator,train_data,V_star,log_="V_convergence"):

        self.train_data = train_data
        self.vf_iterator = vf_iterator
        self.V_star = V_star
        self.log_ = "V_convergence"
    


    def run(self):

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="convergence of V",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
            }
        )

        # simulate training
        epochs = 10
        offset = random.random() / 5
        for epoch in range(2, epochs):
            acc = 1 - 2 ** -epoch - random.random() / epoch - offset
            loss = 2 ** -epoch + random.random() / epoch + offset
            
            # log metrics to wandb
            wandb.log({"acc": acc, "loss": loss})
            
        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()

            





        

