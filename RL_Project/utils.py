import numpy as np 
import pandas as pd

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import data as dat
import mdp_agent as ag
import baseline_policies as bp
import q_plots as qp
import backtest as btest
import bisect
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


import numpy as np
from scipy.interpolate import RegularGridInterpolator

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



def plot_backtest_summary(cumulative_returns, data, lookback, title=None):
    # Ensure cumulative_returns is a Series and capture its name, or set a default one
    if isinstance(cumulative_returns, pd.DataFrame):
        # Assuming cumulative_returns is the first column if it's accidentally a DataFrame
        cumulative_returns = cumulative_returns.iloc[:, 0]
    series_name = cumulative_returns.name if cumulative_returns.name else 'Value'
    
    # Determine the lookback start date
    t2 = cumulative_returns.index[0]  # Assuming cumulative_returns is sorted
    t1 = t2 - pd.Timedelta(days=lookback)
    
    # Create a series for the lookback period with value 1, with the same name as cumulative_returns
    previous_indexes = pd.date_range(start=t1, end=t2, closed='left')
    lookback_series = pd.Series(1, index=previous_indexes, name=series_name)
    
    # Concatenate lookback_series with cumulative_returns ensuring a single column result
    combined_series = pd.concat([lookback_series, cumulative_returns])
    
    # Plot
    fig = go.Figure()
    
    # Add cumulative returns line
    fig.add_trace(go.Scatter(x=combined_series.index, y=combined_series, mode='lines', name='Cumulative Returns'))
    
    # Add a red vertical line at t2
    fig.add_vline(x=t2, line=dict(color='Red', width=2), name='End of Lookback')
    
    # Add shading for the lookback period
    fig.add_shape(type="rect", x0=t1, y0=min(combined_series), x1=t2, y1=max(combined_series),
                  fillcolor="lightgrey", opacity=0.5, line_width=0)
    
    # Set titles and labels
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Value')
    
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


def find_indexes_bisect(x_range, x_inf, x_sup):
    """
    Find the indexes i_inf and i_sup such that x_range[i_inf:i_sup] is the closest approximation
    of the interval [x_inf, x_sup] within the sorted array x_range using bisect.
    """
    i_inf = bisect.bisect_left(x_range, x_inf)
    i_sup = bisect.bisect_right(x_range, x_sup)
    return i_inf, i_sup


#def ts_features(df_,span=200):

#    df = df_.copy()

#    column_name = df.columns[0]

#    sigma_t = df[column_name].ewm(span=span, adjust=False).std()
#    mu_t = df[column_name].ewm(span=span, adjust=False).mean()

#    # Update the DataFrame with the new computations
#    df['mu_t'] = mu_t
#    df['sigma_t'] = sigma_t
#    df['S_down'] = mu_t - 1.5*sigma_t
#    df['S_up'] = mu_t + 1.5*sigma_t
#    df["z_score"] = (df[column_name] - mu_t)/(sigma_t)
#    return df

#def ts_features(df_, span=200, trend_window=50):
#    df = df_.copy()
#    column_name = df.columns[0]

#    sigma_t = df[column_name].ewm(span=span, adjust=False).std()
#    mu_t = df[column_name].ewm(span=span, adjust=False).mean()

#    # Update the DataFrame with the new computations
#    df['mu_t'] = mu_t
#    df['sigma_t'] = sigma_t
#    df['S_down'] = mu_t - 1.5*sigma_t
#    df['S_up'] = mu_t + 1.5*sigma_t
#    df["z_score"] = (df[column_name] - mu_t) / sigma_t

#    # Compute the trend for mu_t
#    def compute_trend(y):
#        window_size = len(y)  # Adjust window size to the actual number of observations
#        weights = np.exp(np.linspace(-1, 0, window_size))
#        weights /= weights.sum()
        
#        # Adjust linear regression to use the actual window size
#        lr = LinearRegression()
#        X = np.arange(window_size).reshape(-1, 1)  # Time indices as features
#        lr.fit(X, y, sample_weight=weights)
#        return lr.coef_[0]  # The slope of the regression line

#    # Apply trend calculation on a rolling window
#    trend = df['mu_t'].rolling(window=trend_window, min_periods=1).apply(
#        compute_trend, raw=False)
    
#    df['mu_trend'] = trend

#    return df

def ts_features(df_, span=200, trend_window=100,ar1_window=50,last_weight_trend = -0.5, last_weight_ar1 = -1):
    df = df_.copy()
    column_name = df.columns[0]

    sigma_t = df[column_name].ewm(span=span, adjust=False).std()
    mu_t = df[column_name].ewm(span=span, adjust=False).mean()

    # Update the DataFrame with the new computations
    df['mu_t'] = mu_t
    df['sigma_t'] = sigma_t
    df['S_down'] = mu_t - 1.5*sigma_t
    df['S_up'] = mu_t + 1.5*sigma_t
    df["z_score"] = (df[column_name] - mu_t) / sigma_t

    # Compute the trend for mu_t
    def compute_trend(y):
        window_size = len(y)
        weights = np.exp(np.linspace(last_weight_trend, 0, window_size))
        weights /= weights.sum()
        
        lr = LinearRegression()
        X = np.arange(window_size).reshape(-1, 1)
        lr.fit(X, y, sample_weight=weights)
        return lr.coef_[0]

    trend = df['mu_t'].rolling(window=trend_window, min_periods=1).apply(compute_trend, raw=False)
    df['mu_trend'] = trend

    # Compute AR(1) coefficient a
    def compute_ar1(y):
        if len(y) < 2:  # Not enough data to compute AR(1)
            return np.nan
        weights = np.exp(np.linspace(last_weight_ar1, 0, len(y)))
        weights /= weights.sum()

        # Target variable (y) is the series shifted by one, predictor (X) is the original series
        X = y[:-1].reshape(-1, 1)
        y_target = y[1:]
        
        lr = LinearRegression()
        lr.fit(X, y_target, sample_weight=weights[:-1])
        return lr.coef_[0]  # Coefficient a

    ar1_coeff = df[column_name].rolling(window=ar1_window, min_periods=2).apply(
        lambda x: compute_ar1(x.values), raw=False)
    df['ar1_coeff'] = ar1_coeff

    return df

def get_list_states(trd,which="test",start=None):

    mrp = trd.apply_policy(bp.BuyAndHold())

    if start is not None:
        start_states = trd.generate_start_state(which,start) 
    
    else:

        start_states = trd.generate_start_state(which) # we take the test set data of the trading policy
    
    sequence = trd.simulate_actions(start_states, bp.BuyAndHold())

    states = [] #will be used to build the backtest dataframe

    for x in sequence:

        states.append(x.state)

    return states


def interpolate2D_heatmaps(x_range, y_range, x_star, y_star, f_star,f_range):
    # Ensuring that the x_range and y_range values are within x_star and y_star bounds
    x_range_valid = x_range[(x_range >= min(x_star)) & (x_range <= max(x_star))]
    y_range_valid = y_range[(y_range >= min(y_star)) & (y_range <= max(y_star))]

    # Create the interpolator function
    
    print(x_star.shape)
    print(y_star.shape)
    print(f_star.shape)

    interpolator = RegularGridInterpolator((x_star, y_star), f_star)

    # Create a meshgrid for the interpolation coordinates
    x_interp_grid, y_interp_grid = np.meshgrid(x_range_valid, y_range_valid, indexing='ij')
    interp_points = np.vstack((x_interp_grid.ravel(), y_interp_grid.ravel())).T

    # Perform the interpolation
    f_star_interp_values = interpolator(interp_points).reshape(x_interp_grid.shape)


    # Extract the corresponding values from f_range
    x_indices = [np.where(x_range == x)[0][0] for x in x_range_valid]
    y_indices = [np.where(y_range == y)[0][0] for y in y_range_valid]

    f_range_valid = f_range[np.ix_(x_indices, y_indices)]

    x_valid = np.array([x_range[ix] for ix in x_indices])
    y_valid = np.array([y_range[ix] for ix in y_indices])


    return x_valid, y_valid, f_range_valid, f_star_interp_values

def SSE_interpolated(x_range, y_range, x_star, y_star, f_star,f_range):
    # Ensuring that the x_range and y_range values are within x_star and y_star bounds
    x_range_valid = x_range[(x_range >= min(x_star)) & (x_range <= max(x_star))]
    y_range_valid = y_range[(y_range >= min(y_star)) & (y_range <= max(y_star))]

    # Create the interpolator function
    interpolator = RegularGridInterpolator((x_star, y_star), f_star)

    # Create a meshgrid for the interpolation coordinates
    x_interp_grid, y_interp_grid = np.meshgrid(x_range_valid, y_range_valid, indexing='ij')
    interp_points = np.vstack((x_interp_grid.ravel(), y_interp_grid.ravel())).T

    # Perform the interpolation
    f_star_interp_values = interpolator(interp_points).reshape(x_interp_grid.shape)


    # Extract the corresponding values from f_range
    x_indices = [np.where(x_range == x)[0][0] for x in x_range_valid]
    y_indices = [np.where(y_range == y)[0][0] for y in y_range_valid]

    f_range_valid = f_range[np.ix_(x_indices, y_indices)]

    squared_differences = (f_range_valid - f_star_interp_values) ** 2
    sum_of_squared_differences = np.nansum(squared_differences)

    return np.nansum(squared_differences)


def plot_heat( x_range, t_range, f_values, xaxis="Time", yaxis="Spot value",title =None):
        

    max_abs_value = np.max(np.abs(f_values))

    # Customizing the color scale for correct positive (green) and negative (red) values representation
    color_scale = [
        [0.0, "red"],  # Negative values
        [0.5, "white"],  # Values around zero
        [1.0, "green"]  # Positive values
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=f_values.T,
        x=np.array(t_range), #v has started estimating after lookabck
        y=x_range,
        coloraxis="coloraxis",
        zmin=-max_abs_value, zmax=max_abs_value
    ))

    fig.update_layout(
        coloraxis=dict(colorscale=color_scale, cmin=-max_abs_value, cmax=max_abs_value),
        xaxis_title="Time Start (t)",
        yaxis_title="Initial X_t value (x)",
        title=title
    )

    fig.show()



def quick_backtest(price_df, prediction_df):
    # Ensure there's only one column in each dataframe and get their names
    price_col = price_df.columns[0]
    prediction_col = prediction_df.columns[0]
    
    # Merge dataframes on index
    merged_df = prediction_df.join(price_df, how='left')
    
    # Calculate returns for the merged timestamps
    merged_df['next_price'] = merged_df[price_col].shift(-1)
    merged_df['return'] = 10*merged_df[prediction_col] * (merged_df['next_price'] - merged_df[price_col]) / merged_df[price_col]
    return merged_df.dropna(subset=['return'])[['return']]


def multi_plot_compare(df1, df2,h=1000,w=1000):
    df1.index = pd.to_datetime(df1.index)
    df2.index = pd.to_datetime(df2.index)
    
    num_cols = df1.shape[1]
    grid_size = int(np.ceil(np.sqrt(num_cols)))

    fig = make_subplots(rows=grid_size, cols=grid_size,
                        subplot_titles=df1.columns)
    
    for i in range(num_cols):
        
        ccy = list(df1.columns)[i]

        aligned_data = pd.merge(df1.iloc[:, [i]], df2.iloc[:, [i]], left_index=True, right_index=True, suffixes=('_1', '_2'))
        
        x = aligned_data.iloc[:, 0].values.reshape(-1, 1)
        y = aligned_data.iloc[:, 1].values
        
        # Linear regression
        model = LinearRegression().fit(x, y)
        y_pred = model.predict(x)
        
        # R^2 score
        r2 = r2_score(y, y_pred)
        
        # Hit ratio
        hits = np.sum((x.flatten() > x.mean()) == (y > y.mean())) / len(x)
        
        row = i // grid_size + 1
        col = i % grid_size + 1
        fig.add_trace(go.Scatter(x=x.flatten(), y=y, mode='markers', name=ccy),
                      row=row, col=col)
        # Add regression line
        fig.add_trace(go.Scatter(x=x.flatten(), y=y_pred, mode='lines', name=ccy),
                      row=row, col=col)
        # Update subplot title with R^2 and hit ratio
        fig.layout.annotations[i].text = f'{ccy} R2={r2:.2f}, Hit Ratio={hits:.2f}'
    
    fig.update_layout(height=h, width=w, title_text="Aligned Scatter Plots with Regression")
    fig.show()
