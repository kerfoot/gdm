"""
Miscellaneous utilities for gdm file processing
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__file__)


def nearest(items, pivot, direction=None):
    """Return the element in items that is closest in value to pivot"""
    if direction == 'after':
        items = [i for i in items if i > pivot]
        if not items:
            return
        return min(items, key=lambda x: abs(x - pivot))
    elif direction == 'before':
        items = [i for i in items if i > pivot]
        if not items:
            return
        return min(items, key=lambda x: abs(x - pivot))

    return min(items, key=lambda x: abs(x - pivot))


def calculate_series_first_differences(series):
    """
    Calculate the first differences of the pandas series, ignoring NaNs
    Args:
        series: pandas time-series

    Returns:
        series named {:}_first_diffs.format(series.name)
    """
    if not isinstance(series, pd.Series):
        logging.error('Input arg must be pandas Series')
        return np.empty((0,))

    diffs = np.empty(series.shape) * np.nan

    i = ~series.isna()

    diffs[i] = series.dropna().diff()

    return pd.Series(diffs, index=series.index, name='{:}_first_diffs'.format(series.name))


def calculate_series_rate_of_change(series):
    """
    Calculate the rate of change of the pandas series, ignoring NaNs
    Args:
        series: pandas time-series

    Returns:
        series named {:}_roc.format(series.name)
    """
    if not isinstance(series, pd.Series):
        logging.error('Input arg must be pandas Series')
        return np.empty((0,))

    i = ~series.isna()

    roc = np.empty(series.shape) * np.nan

    roc[i] = series[i].diff()/series[i].index.to_frame().diff().time.dt.total_seconds()

    return pd.Series(roc, index=series.index, name='{:}_roc'.format(series.name))


def resample_to(df, frequency='100L', method='pchip'):

    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    if not isinstance(df, pd.DataFrame):
        logging.error('Time series must be a pandas Series or DataFrame')
        return pd.DataFrame()

    resampled = df.resample(frequency).mean().interpolate(method)

    return resampled


def resample_timeseries(series, frequency='100L'):
    """
    See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    for a list of valid DateOffset objects to use for frequency.
    """
    if not isinstance(series, pd.Series):
        logging.error('Input arg must be pandas Series')
        return pd.Series()

    # Resample the index to offset intervals
    resampled = series.resample(frequency)

    # Convert the resampled TimeIndex from datetime, etc. to epoch floats
    new_ts = resampled.asfreq().index.values.astype(float)

    # Interpolate to the new frequency
    series_copy = series.dropna()
    idata = np.interp(new_ts, series_copy.index.values.astype(float), series_copy)

    return pd.Series(idata, index=resampled.asfreq().index, name='interp_{:}'.format(series.name))


def interpolate_timeseries(series, new_time_index):

    if not isinstance(series, pd.Series):
        logging.error('Interpolation can only be performed on a Series')
        return pd.Series()

    new_ts = new_time_index.values.astype(float)

    # Interpolate to the new frequency
    series_copy = series.dropna()
    idata = np.interp(new_ts, series_copy.index.values.astype(float), series_copy)

    return pd.Series(idata, index=new_time_index, name='interp_{:}'.format(series.name))
