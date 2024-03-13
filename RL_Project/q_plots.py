#standard imports
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
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
from rl.monte_carlo import epsilon_greedy_policy, greedy_policy_from_qvf, glie_mc_control
from rl.function_approx import LinearFunctionApprox, AdamGradient


import plotly.graph_objs as go



class QAnalyzer():

    """
    class to analyze the q value function
    takes data in input as we do not know a priori what is the distribution of spot values
    """

    def __init__(self,data,qvf):
        self.data = data
        self.qvf = qvf
        self.S_min = data.min()[0]
        self.S_max = data.max()[0]

    def plot_snapshot(self):

        fig = go.Figure()

        for (p,a) in [(-1,1),(-1,0),(0,-1),(0,0),(0,1),(1,-1), (1,0)]:

            qvals = []
            spotPrices = np.arange(self.S_min,self.S_max,(self.S_max-self.S_min)/100)

            for spot in spotPrices:
                state = NonTerminal({
                    "Spot" : spot,
                    "position" : p,
                    "date" : pd.to_datetime("2023-01-01"),
                    "data" : self.data
                })

                qvals.append(self.qvf((state,a)))

            if a == -1:
                line_color = 'red'
            elif a == 1:
                line_color = '#90EE90'
            else:
                line_color = 'black'

            fig.add_trace(go.Scatter(x=spotPrices, y=qvals, mode='lines', name=f'pos : {p}, act : {a}',line=dict(color=line_color)))

        fig.update_layout(title='Q values for different (spot,action,position) sates',
                        xaxis_title='Spot at '+str(state.state["date"]),
                        yaxis_title='Q value')

        fig.show()


    # def line_trace_snapshot(self, f : Callable[[State],float], state, rangeSpot):

    #     spot_min = rangeSpot[0]
    #     spot_max = rangeSpot[1]
    #     spot_prices = np.arange(spot_min,spot_max)
    #     vals = []

    #     for spot_price in range(spot_min,spot_max):
    #         curr_state = f(state,spot_price)
    #         curr_state['Spot'] = spot_price
    #         vals.append(f(curr_state))

    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x=spot_prices, y=vals, mode='lines', name='Q values',line=dict(color='blue')))
    #     fig.update_layout(title='Q values for different spot prices',
    #                     xaxis_title='Spot at '+str(state.state["date"]),
    #                     yaxis_title='Q value')
    #     fig.show()

    # def create_new_state(self, state, spot_price):
    #     new_state = state
    #     new_state['Spot'] = spot_price
    #     return new_state
    
    def create_new_state(self, state, spot_price):
        new_state_data = state.state.copy()
        new_state_data['Spot'] = spot_price
        return NonTerminal(new_state_data)

    def line_trace_snapshot(self, f: Callable[[State], float], state, rangeSpot, action):
        spot_min = rangeSpot[0]
        spot_max = rangeSpot[1]
        spot_prices = np.arange(spot_min,spot_max)
        vals = []

        for spot_price in spot_prices:
            curr_state = self.create_new_state(state, spot_price)
            vals.append(f(curr_state, action))

        line_trace = go.Scatter(x=spot_prices, y=vals, mode='lines', name='Q values', line=dict(color='blue'))
        return vals, line_trace
    
    def change_state(self, state, pos):
        new_state_data = state.state.copy()
        new_state_data['position'] = pos
        return NonTerminal(new_state_data)

    def plot_snapshot(self, mdp_state: State):
        fig = go.Figure()

        for (p, a) in [(-1, 1), (-1, 0), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0)]:
            def q_for_plot(state, action):
                curr_state = self.change_state(state, p)
                return self.qvf((curr_state, action))
            
            vals, line_trace = self.line_trace_snapshot(q_for_plot, mdp_state, [self.S_min, self.S_max], a)

            # Set the line color and name based on the action
            line_color = 'red' if a == -1 else ('#90EE90' if a == 1 else 'black')
            line_trace.update(line=dict(color=line_color))
            line_trace.name = f'pos: {p}, act: {a}'  # Set the name for the trace

            # Add trace for the current (p, a) tuple
            fig.add_trace(line_trace)

        fig.update_layout(title='Q values for different (spot,action,position) states',
                        xaxis_title='Spot at ' + str(mdp_state.state["date"]),
                        yaxis_title='Q value')
        fig.show()

    












# potentially correct?????
        # def plot_snapshot(self, mdp_state: State):
    #     fig = go.Figure()
    #     spotPrices = np.arange(self.S_min,self.S_max,(self.S_max-self.S_min)/100)
    #     functionVals = []

    #     for (p,a) in [(-1,1),(-1,0),(0,-1),(0,0),(0,1),(1,-1), (1,0)]:
    #         def q_for_plot(state, action):
    #             curr_state = self.change_state(state, p)
    #             return self.qvf((curr_state, action))
    #         vals, line_trace = self.line_trace_snapshot(q_for_plot, mdp_state, [self.S_min, self.S_max], a)
    #         functionVals.append(vals)

    #         if a == -1:
    #             line_color = 'red'
    #         elif a == 1:
    #             line_color = '#90EE90'
    #         else:
    #             line_color = 'black'

    #         fig.add_trace(go.Scatter(x=spotPrices, y=functionVals, mode='lines', name=f'pos : {p}, act : {a}',line=dict(color=line_color)))

    #     fig.update_layout(title='Q values for different (spot,action,position) sates',
    #                     xaxis_title='Spot at '+str(mdp_state.state["date"]),
    #                     yaxis_title='Q value')

    #     fig.show()





















# Example time series data
#time = np.linspace(0, 10, 100)  # Replace with your actual time series data
#y_values = np.sin(time)  # Example line plot, replace with your actual data

# Example value function (modify according to your actual function)
#def value_function(y):
#    return 1 - abs(y)  # Example function

# Generate heatmap data based on the value function
#y_range = np.linspace(-1, 1, 100)
#heatmap_z = np.array([[value_function(y) for y in y_range] for _ in time])

# Custom colorscale (white to red, adjust as needed)


# Create the heatmap
#heatmap = go.Heatmap(
#   z=heatmap_z,
#   x=time,
#    y=y_range,
#    colorscale=colorscale,
#    showscale=False  # Hide the heatmap color scale
#)

# Create the line plot
#line_plot = go.Scatter(x=time, y=y_values, mode='lines', name='Line Plot')

# Create the figure and add the heatmap and line plot
#fig = go.Figure(data=[heatmap, line_plot])

# Customize layout
#fig.update_layout(title='Line Plot with Colored Background')

# Show the figure
#fig.show()