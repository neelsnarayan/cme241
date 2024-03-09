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

    def plot_snapshot(self):

        fig = go.Figure()

        for (p,a) in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1), (1,-1),(1,0),(1,1)]:

            qvals = []
            spotPrices = np.arange(100-2,100+2,0.01)

            for spot in spotPrices:
                state = NonTerminal({
                    "Spot" : spot,
                    "position" : p,
                    "date" : pd.to_datetime("2023-01-01"),
                    "data" : self.data
                })

                qvals.append(self.qvf((state,a)))

            fig.add_trace(go.Scatter(x=spotPrices, y=qvals, mode='lines', name=f'pos : {p}, act : {a}'))

        fig.update_layout(title='Q values for different (spot,action,position) sates',
                        xaxis_title='Spot',
                        yaxis_title='Q value')

        fig.show()


    
