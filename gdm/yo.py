"""
Routines for indexing profiles from the depth timeseries
"""
import logging
import numpy as np
import pandas as pd
import datetime
from scipy.signal import convolve, boxcar

logger = logging.getLogger(__file__)


def index_profiles(yo):
    if not isinstance(yo, pd.Series):
        logger.error('Input argument must be a pressure or depth Series')
        return np.empty((0, 2))

    epochs = (yo.index - datetime.datetime(1970, 1, 1, 0, 0, 0)).total_seconds().values

    yo = np.array([epochs, yo.values]).T

    profile_epoch_times = find_profiles(yo)

    return profile_epoch_times


def find_profiles(yo):
    """ Construct the depth time-series from a parsed dba['data'] array and find
    the profile minima and maxima.  The resulting indexed profiles are filtered
    using gnc.yo.filters.default_profiles_filter which filters on defaults for the
    min number of points, min depth span and min time span.

    Arguments
    data: an array of dicts returned from create_llat_dba_reader containing the
        individual rows of a parsed dba file

    Returns
    profile_times: a 2 column array containing the start and stop times of each
        indexed profile.
    """

    profile_times = find_yo_extrema(yo[:, 0], yo[:, 1])
    if len(profile_times) == 0:
        return profile_times

    filter_profile_times = default_profiles_filter(yo, profile_times)

    return filter_profile_times


def binarize_diff(data):
    data[data <= 0] = -1
    data[data >= 0] = 1
    return data


def calculate_delta_depth(interp_data):
    delta_depth = np.diff(interp_data)
    delta_depth = binarize_diff(delta_depth)

    return delta_depth


def find_yo_extrema(timestamps, depth, tsint=10):
    """Returns the start and stop timestamps for every profile indexed from the
    depth timeseries

    Parameters:
        time, depth

    Returns:
        A Nx2 array of the start and stop timestamps indexed from the yo

    Use filter_yo_extrema to remove invalid/incomplete profiles
    """

    # Create Nx2 numpy array of profile start/stop times - kerfoot method
    profile_times = np.empty((0, 2))

    # validate_glider_args(timestamps, depth)

    est_data = np.column_stack((
        timestamps,
        depth
    ))

    # Set negative depth values to NaN
    est_data[np.any(est_data <= 0, axis=1), :] = float('nan')

    # Remove NaN rows
    est_data = clean_dataset(est_data)
    if len(est_data) < 2:
        logger.debug('Skipping yo that contains < 2 rows')
        return np.empty((0, 2))

    # Create the fixed timestamp array from the min timestamp to the max timestamp
    # spaced by tsint intervals
    min_ts = est_data[:, 0].min()
    max_ts = est_data[:, 0].max()
    if max_ts - min_ts < tsint:
        logger.warning('Not enough timestamps for yo interpolation')
        return np.empty((0, 2))

    ts = np.arange(min_ts, max_ts, tsint)
    # Stretch estimated values for interpolation to span entire dataset
    interp_z = np.interp(
        ts,
        est_data[:, 0],
        est_data[:, 1],
        left=est_data[0, 1],
        right=est_data[-1, 1]
    )

    filtered_z = boxcar_smooth_dataset(interp_z, int(tsint / 2))

    delta_depth = calculate_delta_depth(filtered_z)

    # interp_indices = np.argwhere(delta_depth == 0).flatten()

    p_inds = np.empty((0, 2))
    inflections = np.where(np.diff(delta_depth) != 0)[0]
    if not inflections.any():
        return profile_times

    p_inds = np.append(p_inds, [[0, inflections[0]]], axis=0)
    for p in range(len(inflections) - 1):
        p_inds = np.append(p_inds, [[inflections[p], inflections[p + 1]]], axis=0)
    p_inds = np.append(p_inds, [[inflections[-1], len(ts) - 1]], axis=0)

    # profile_timestamps = np.empty((0,2))
    ts_window = tsint * 2

    # Create orig GUTILS return value - lindemuth method
    # Initialize an nx3 numpy array of nans
    profiled_dataset = np.full((len(timestamps), 3), np.nan)
    # Replace 0 column with the original timestamps
    profiled_dataset[:, 0] = timestamps
    # Replace 1 column with the original depths
    profiled_dataset[:, 1] = depth

    # # Create Nx2 numpy array of profile start/stop times - kerfoot method
    profile_times = np.full((p_inds.shape[0], 2), np.nan)

    # Start profile index
    profile_ind = 0
    # Iterate through the profile start/stop indices
    for p in p_inds:
        # Profile start row
        p0 = int(p[0])
        # Profile end row
        p1 = int(p[1])
        # Find all rows in the original yo that fall between the interpolated timestamps
        profile_i = np.flatnonzero(np.logical_and(profiled_dataset[:, 0] >= ts[p0] - ts_window,
                                                  profiled_dataset[:, 0] <= ts[p1] + ts_window))
        # Slice out the profile
        pro = profiled_dataset[profile_i]
        if pro.size == 0:
            continue
        # Find the row index corresponding to the minimum depth
        try:
            min_i = np.nanargmin(pro[:, 1])
        except ValueError as e:
            logger.warning(e)
            continue
        # Find the row index corresponding to the maximum depth
        try:
            max_i = np.nanargmax(pro[:, 1])
        except ValueError as e:
            logger.warning(e)
            continue
        # Sort the min/max indices in ascending order
        sorted_i = np.sort([min_i, max_i])
        # Set the profile index
        profiled_dataset[profile_i[sorted_i[0]]:profile_i[sorted_i[1]], 2] = profile_ind

        # kerfoot method
        profile_times[profile_ind, :] = [timestamps[profile_i[sorted_i[0]]], timestamps[profile_i[sorted_i[1]]]]
        # Increment the profile index
        profile_ind += 1

    # return profiled_dataset
    return profile_times


def default_profiles_filter(yo, profile_times):
    profile_times = filter_profiles_min_points(yo, profile_times)
    profile_times = filter_profiles_min_depthspan(yo, profile_times)
    profile_times = filter_profiles_min_timespan(yo, profile_times)

    return profile_times


def filter_profile_breaks(yo, profile_times):
    filt_p_times = np.empty((0, 2))

    for p in profile_times:

        # Create the profile by finding all timestamps in yo that are included in the
        # window p
        pro = yo[np.logical_and(yo[:, 0] >= p[0], yo[:, 0] <= p[1])]

        pro = pro[np.all(~np.isnan(pro), axis=1)]

        # Diff the timestamps
        tdiff = np.diff(pro[:, 0])
        # Median and stdev
        med = np.median(tdiff)
        std = np.std(tdiff)

        # Find profile time breaks
        p_breaks = np.where(tdiff > med + std)
        if not len(p_breaks):
            filt_p_times = np.append(filt_p_times, p)
            continue

        p0 = np.append(np.array(0), p_breaks[0].copy())
        p1 = np.append(p_breaks[0].copy() + 1, pro.shape[0] - 1)
        p_inds = np.column_stack((p0, p1))

        for i in p_inds:
            t0 = pro[i[0], 0]
            t1 = pro[i[1], 0]
            filt_p_times = np.append(filt_p_times, [[t0, t1]], axis=0)

    return filt_p_times


def filter_profiles_min_points(yo, profile_times, minpoints=3):
    """Returns profile start/stop times for which the indexed profile contains
    at least minpoints number of non-Nan points.

    Parameters:
        yo: Nx2 numpy array containing the timestamp and depth records
        profile_times: Nx2 numpy array containing the start/stop times of indexed
            profiles from gutils.yo.find_yo_extrema

    Options:
        minpoints: minimum number of points an indexed profile must contain to be
            considered valid <Default=2>

    Returns:
        Nx2 numpy array containing valid profile start/stop times
    """

    new_profile_times = np.full((0, 2), np.nan)

    for p in profile_times:

        # Create the profile by finding all timestamps in yo that are included in the
        # window p
        pro = yo[np.logical_and(yo[:, 0] >= p[0], yo[:, 0] <= p[1])]

        # Eliminate NaN rows
        pro = pro[np.all(~np.isnan(pro), axis=1)]

        if pro.shape[0] >= minpoints:
            new_profile_times = np.append(new_profile_times, [p], axis=0)

    return new_profile_times


def filter_profiles_min_depthspan(yo, profile_times, mindepthspan=1):
    """Returns profile start/stop times for which the indexed profile depth range
    is at least mindepthspan.

    Parameters:
        yo: Nx2 numpy array containing the timestamp and depth records
        profile_times: Nx2 numpy array containing the start/stop times of indexed
            profiles from gutils.yo.find_yo_extrema

    Options:
        mindepthspan: minimum depth range (meters, decibars, bars) an indexed
            profile must span to be considered valid <Default=1>

    Returns:
        Nx2 numpy array containing valid profile start/stop times
    """

    new_profile_times = np.full((0, 2), np.nan)

    for p in profile_times:

        pro = yo[np.logical_and(yo[:, 0] >= p[0], yo[:, 0] <= p[1])]

        # Eliminate NaN rows
        pro = pro[np.all(~np.isnan(pro), axis=1)]

        if np.max(pro[:, 1]) - np.min(pro[:, 1]) >= mindepthspan:
            new_profile_times = np.append(new_profile_times, [p], axis=0)

    return new_profile_times


def filter_profiles_min_timespan(yo, profile_times, mintimespan=10):
    """Returns profile start/stop times for which the indexed profile spans at
    least mintimespan seconds.

    Parameters:
        yo: Nx2 numpy array containing the timestamp and depth records
        profile_times: Nx2 numpy array containing the start/stop times of indexed
            profiles from gutils.yo.find_yo_extrema

    Options:
        mintimespan: minimum number of seconds an indexed profile must span to be
            considered valid <Default=10>

    Returns:
        Nx2 numpy array containing valid profile start/stop times
    """

    new_profile_times = np.full((0, 2), np.nan)

    for p in profile_times:

        pro = yo[np.logical_and(yo[:, 0] >= p[0], yo[:, 0] <= p[1])]

        # Eliminate NaN rows
        pro = pro[np.all(~np.isnan(pro), axis=1)]

        if np.max(pro[:, 0]) - np.min(pro[:, 0]) >= mintimespan:
            new_profile_times = np.append(new_profile_times, [p], axis=0)

    return new_profile_times


def clean_dataset(dataset):
    """Remove any row in dataset for which one or more columns is np.nan
    """

    # Get rid of NaNs
    dataset = dataset[~np.isnan(dataset[:, 1:]).any(axis=1), :]

    return dataset


def boxcar_smooth_dataset(dataset, window_size):
    window = boxcar(window_size)
    return convolve(dataset, window, 'same') / window_size
