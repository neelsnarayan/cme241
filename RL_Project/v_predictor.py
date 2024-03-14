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


import numpy as np
import plotly.graph_objects as go

class Prediction:
    def __init__(self, mu, kappa, sigma):
        self.mu = mu
        self.kappa = kappa
        self.sigma = sigma

    def monte_carlo_f(self, x, t_start , gamma, num_paths, N_fixed=100):
        discounts = gamma ** np.arange(N_fixed)
        sums_discounted_log_returns = np.zeros(num_paths)
        
        for i in range(num_paths):
            X_t = x
            for t in range(N_fixed):
                dW = np.random.normal(0, 1)
                X_t_next = X_t + self.kappa(t_start+t) * (self.mu(t_start+t) - X_t)  + self.sigma(t_start+t) * dW
                log_return = np.log(X_t_next / X_t)
                sums_discounted_log_returns[i] += discounts[t] * log_return
                X_t = X_t_next
        return np.mean(sums_discounted_log_returns)


    def sample_path(self, N_fixed=100):
        """Generate a single sample path starting from mu(0) and the first value of t."""
        x = self.mu(0)  # Initial state from mu at t=0
        t = 0  # Starting time
        path = [x]
        
        for _ in range(N_fixed):
            dW = np.random.normal(0, 1)
            x_next = x + self.kappa(t) * (self.mu(t) - x) + self.sigma(t) * dW
            path.append(x_next)
            x = x_next
            t += 1  # Increment time
        
        return path


    def generate_f_heatmap(self, t_range, gamma, num_paths, step_x = 2, N_fixed=100):

        # Define a global x range that covers all possible dynamic ranges
        x_min = min(self.mu(t)*(1 - 0.15*self.sigma(t)) for t in t_range)
        x_max = max(self.mu(t)*(1 + 0.15*self.sigma(t)) for t in t_range)

        #x_range = np.arange(x_min, x_max, step=step_x)  # Adjust the number of points as needed

        x_range = np.linspace(x_min,x_max,num=20)
        
        # Initialize f_values with np.nan
        f_values = np.full((len(t_range),len(x_range)), np.nan)

        for j, t in enumerate(t_range):

            range_inf = self.mu(t)*(1-0.1*self.sigma(t))
            range_sup = self.mu(t)*(1+0.1*self.sigma(t))

            i_inf,i_sup = u.find_indexes_bisect(x_range, range_inf, range_sup)

            for i in range(i_inf,i_sup):
                x = x_range[i]
                f_values[j,i] = self.monte_carlo_f(x, t , gamma, num_paths, N_fixed)
        return x_range, t_range, f_values

    def plot_heatmap(self, x_range, t_range, f_values):
        

        max_abs_value = np.max(np.abs(f_values))

        # Customizing the color scale for correct positive (green) and negative (red) values representation
        color_scale = [
            [0.0, "red"],  # Negative values
            [0.5, "white"],  # Values around zero
            [1.0, "green"]  # Positive values
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=f_values.T,
            x=t_range,
            y=x_range,
            coloraxis="coloraxis",
            zmin=-max_abs_value, zmax=max_abs_value
        ))

        fig.update_layout(
            coloraxis=dict(colorscale=color_scale, cmin=-max_abs_value, cmax=max_abs_value),
            xaxis_title="Time Start (t)",
            yaxis_title="Initial X_t value (x)",
            title=r"Monte - Carlo estimation of f(x,t) = E(sum gamma^t log(X_{t+1}/X_t) | X_t = x"
        )
        

        for _ in range(10):
            sample_path = self.sample_path(len(t_range))
            fig.add_trace(go.Scatter(x=t_range, y=sample_path, mode='lines', line=dict(width=2)))


        fig.show()



    
