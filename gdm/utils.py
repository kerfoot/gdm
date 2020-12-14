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
