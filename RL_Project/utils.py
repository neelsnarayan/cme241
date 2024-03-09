import numpy as np 
import pandas as pd

import pandas as pd
import numpy as np
import plotly.graph_objects as go


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


def sample_dataframes(df, start_trading, N, n, uniform=False, half_life=None):
    """
    Samples N DataFrames of length n from df with start dates randomly chosen.
    
    Parameters:
    - df: pandas DataFrame with a DateTimeIndex.
    - start_trading: str or datetime, the cut-off date for sampling.
    - N: int, number of DataFrames to sample.
    - n: int, length of each sampled DataFrame.
    - uniform: bool, samples start dates uniformly if True; with exponentially decaying weights if False.
    - half_life: int or float, the half-life for the exponential decay of sampling probabilities. Only used if uniform is False.
    
    Returns:
    - List of N DataFrames each of length n.
    """
    start_trading = pd.to_datetime(start_trading)
    # Ensure we only consider dates far enough back to allow for a window of size n
    valid_starts = df.index[df.index <= start_trading - pd.DateOffset(days=n)]
    
    if uniform:
        weights = None  # Uniform sampling does not require weights
    else:
        if half_life is None:
            raise ValueError("Half-life must be provided for non-uniform sampling.")
        
        latest_date = valid_starts.max()
        days_from_end = np.array([(latest_date - date).days for date in valid_starts])
        decay_rate = np.log(2) / half_life
        weights = np.exp(-decay_rate * days_from_end)
        weights /= np.sum(weights)  # Explicit normalization to ensure sum is precisely 1

    sampled_starts_indices = np.random.choice(range(len(valid_starts)), size=N, replace=True, p=weights)
    sampled_starts = valid_starts[sampled_starts_indices]

    sampled_dfs = []
    for start_index in sampled_starts_indices:
        start = valid_starts[start_index]
        end = start + pd.DateOffset(days=n-1)
        sampled_df = df.loc[start:end]
        sampled_dfs.append(sampled_df)
    
    return sampled_dfs