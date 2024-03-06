import numpy as np 
import pandas as pd

import pandas as pd
import numpy as np
import plotly.graph_objects as go


# Define the function with comments explaining each step
def get_next(date, data):
    """
    Finds the index date in a time-indexed DataFrame that comes right after a given date.
    Also indicates if the located next date is the last element in the DataFrame.

    Parameters:
    - date: The reference date, as a string or a datetime-like object.
    - data: A Pandas DataFrame with a DatetimeIndex.

    Returns:
    - A tuple containing:
        - The next index date after the given date (None if no such date exists).
        - A boolean indicating whether this next date is the last element in the DataFrame.
    """
    # Convert the input date to pandas datetime, if it's not already
    date = pd.to_datetime(date)
    
    # Ensure the DataFrame's index is in datetime format
    data.index = pd.to_datetime(data.index)
    
    # Find the next index
    next_indices = data.index[data.index > date]
    
    if len(next_indices) == 0:
        # If there are no dates after the given date
        return None, False
    
    # The next date after the given date
    next_date = next_indices[0]
    
    # Check if the next date is the last element in the DataFrame
    is_last = next_date == data.index[-1]
    
    return next_date, is_last

def plot_plotly(df,title=None):
    """
    Plots an Ornstein-Uhlenbeck process using Plotly.

    Parameters:
    - df: DataFrame with index as dates and a column 'Value' representing the Ornstein-Uhlenbeck process.
    """

    # Create a plotly figure
    fig = go.Figure()

    # Add line plot
    fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:,0], mode='lines'))

    # Set titles and labels
    fig.update_layout(title=title, xaxis_title='Date',
                      yaxis_title='Value') # You can change the template as needed

    # Show plot
    fig.show()


def plot_plotly_multiple(dfs):
    # Create a plotly figure
    fig = go.Figure()

    for df in dfs:
    # Add lineplot
        fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:,0], mode='lines'))

        # Set titles and labels
        fig.update_layout(xaxis_title='Date',
                      yaxis_title='Value') # You can change the template as needed

    # Show plot
    fig.show()


