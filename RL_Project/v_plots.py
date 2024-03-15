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





class VAnalyzer:
    """
    Class to analyze the v value function
    """

    def __init__(self, vvf, state_dicts):
        self.vvf = vvf
        self.state_dicts = state_dicts

    def createHeatMap(self):
        """
        This function creates a heatmap as follows:
        on the x axis, we have the date
        on the y axis, we have the spot price
        the color of the heatmap is the value of the vvf at the given date
        """

        fig = go.Figure()

        iterationCounter = 0

        for state_dict in self.state_dicts[0:500]:
            # Determine the date range based on the lookback period
            
            end_date = pd.to_datetime(state_dict.state["date"])
            start_date = end_date - pd.Timedelta(days=state_dict.state["lookback"])

            # Filter the data to the lookback period
            lookback_data = state_dict.state["data"].loc[start_date:end_date]

            # Find the min and max spot prices within the lookback period
            spot_min = lookback_data['Value'].min()
            spot_max = lookback_data['Value'].max()


            spotPrices = np.linspace(spot_min, spot_max, 100)
            dates = pd.date_range(start=start_date, end=end_date, periods=100)

            z = np.zeros((len(spotPrices)))

            for i, spot in enumerate(spotPrices):
                state = NonTerminal({
                    "Spot": spot,
                    "position": 1,
                    "date": state_dict.state["date"],
                    "data": state_dict.state["data"],
                    "lookback": state_dict.state["lookback"],
                    "time_index": state_dict.state["time index"],
                })
                z[i] = self.vvf(state)

            fig.add_trace(go.Heatmap(
                z=z,
                x=dates,
                y=spotPrices,
                colorscale='Viridis'
            ))

            iterationCounter += 1
            if iterationCounter % 100 == 0:
                print(f"Completed iteration {iterationCounter} of {len(self.state_dicts)}")

        print("Heatmap creation complete")
        fig.update_layout(
            title="Value function heatmap",
            xaxis_title="Date",
            yaxis_title="Spot price",
            autosize=False,
            width=1000,
            height=500,
            margin=dict(
                l=50,
                r=50,
                b=100,
                t=100,
                pad=4
            ),
            paper_bgcolor="LightSteelBlue",
        )
        print("Heatmap layout complete")

        fig.show()
        print("Heatmap shown")

















































































# class VAnalyzer():

#     """
#     class to analyze the v value function
#     """

#     def __init__(self, vvf, state_dicts):
#         self.vvf = vvf
#         self.state_dicts = state_dicts

#     def createHeatMap(self):
#         """
#         this function creates a heatmap as follows:
#         on the x axis, we have the date
#         on the y axis, we have the spot price
#         the color of the heatmap is the value of the vvf at the given date
#         """

#         fig = go.Figure()

#         for state_dict in self.state_dicts:

#             spotPrices = np.arange(state_dict["Spot_min"],state_dict["Spot_max"],(state_dict["Spot_max"]-state_dict["Spot_min"])/100)
#             dates = pd.date_range(start=state_dict["date_min"],end=state_dict["date_max"],periods=10)

#             z = np.zeros((len(spotPrices),len(dates)))

#             for i,spot in enumerate(spotPrices):
#                 for j,date in enumerate(dates):
#                     state = NonTerminal({
#                         "Spot" : spot,
#                         "date" : date,
#                         "data" : state_dict["data"]
#                     })
#                     z[i,j] = self.vvf(state)

#             fig.add_trace(go.Heatmap(
#                 z=z,
#                 x=dates,
#                 y=spotPrices,
#                 colorscale='Viridis'
#             ))

#         fig.update_layout(
#             title="Value function heatmap",
#             xaxis_title="Date",
#             yaxis_title="Spot price",
#             autosize=False,
#             width=1000,
#             height=500,
#             margin=dict(
#                 l=50,
#                 r=50,
#                 b=100,
#                 t=100,
#                 pad=4
#             ),
#             paper_bgcolor="LightSteelBlue",
#         )

#         fig.show()