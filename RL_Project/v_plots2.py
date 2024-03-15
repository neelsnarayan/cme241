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

class V_Analyzer_Mehdi:
    def __init__(self, vf, states_history):
        self.vf = vf
        self.states_history = states_history #list of list of states
  

    def get_v_value(self, x, t_start):

        v_vals = []

        for state_list in self.states_history:

            s = state_list[t_start]
            
            s_dict=s.state
            s_dict["Spot"] = x

            s_x = NonTerminal(s_dict)

            v_vals.append(self.vf(s_x))

        return np.mean(v_vals)

    def get_t_x_range(self):

        x_min = np.inf
        x_max = -np.inf

        for state_list in self.states_history:

            x_min_new = state_list[0].state["data"].min()[0]
            x_max_new = state_list[0].state["data"].max()[0]
            if x_min_new <=x_min:
                x_min = x_min_new
            if x_max_new >= x_max:
                x_max = x_max_new

        t_range = list(range(len(self.states_history[0])))

        x_range = np.linspace(x_min,x_max,num=20)

        return t_range, x_range

    def get_min_max_spot_t(self,t):

        x_min = np.inf
        x_max = -np.inf

        for state_list in self.states_history:


            x_min_new = state_list[t].state["mu_t"]*(1-0.1*state_list[t].state["sigma_t"])

            x_max_new = state_list[t].state["mu_t"]*(1+0.1*state_list[t].state["sigma_t"])

            if x_min_new <=x_min:
                x_min = x_min_new
            if x_max_new >= x_max:
                x_max = x_max_new

        return x_min, x_max


    def generate_V_heatmap(self):

        t_range , x_range = self.get_t_x_range()
      
        # Initialize f_values with np.nan
        f_values = np.full((len(t_range),len(x_range)), np.nan)

        for j, t in enumerate(t_range):

            range_inf, range_sup = self.get_min_max_spot_t(t)
   

            i_inf,i_sup = u.find_indexes_bisect(x_range, range_inf, range_sup)

            for i in range(i_inf,i_sup):
                x = x_range[i]
                f_values[j,i] = self.get_v_value(x,t)

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
            x=self.states_history[0][0].state["lookback"]+np.array(t_range), #v has started estimating after lookabck
            y=x_range,
            coloraxis="coloraxis",
            zmin=-max_abs_value, zmax=max_abs_value
        ))

        fig.update_layout(
            coloraxis=dict(colorscale=color_scale, cmin=-max_abs_value, cmax=max_abs_value),
            xaxis_title="Time Start (t)",
            yaxis_title="Initial X_t value (x)",
            title=r"V"
        )

        for state_list in self.states_history:

            start_t = state_list[0].state["date"]
            y = state_list[0].state["data"].loc[start_t:].to_numpy().reshape(-1)

            fig.add_trace(go.Scatter(x=t_range, y=y, mode='lines', line=dict(width=2)))
        

        fig.show()
