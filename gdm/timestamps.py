import datetime
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__file__)


def epoch2datetime(t):
    return pd.to_datetime(t, unit='s', errors='coerce')


def datetime2epoch(dtime):
    return (dtime - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')


def time_index_to_iso_resolution(time_index):
    """Calculate the datetime.timedelta for the specified time_index and return a ISO 8601:2004 duration formatted
    string corresponding to the approximate resolution

    Parameters:
        time_index: pandas DateTimeIndex

    :returns ISO 8601:2004 string
    """

    # Calculate the approximate time_coverage_resolution
    num_seconds = time_index.max() - time_index.min()
    time_delta = datetime.timedelta(0, int(num_seconds.total_seconds() / time_index.size))

    return timedelta_to_iso_duration(time_delta)


def timedelta_to_iso_duration(time_delta):
    """Calculate the datetime.timedelta for the specified time_index and return a ISO 8601:2004 duration formatted
    string corresponding to the duration

    Parameters:
        time_delta: datetime.timedelta

    :returns ISO 8601:2004 string
    """

    seconds = time_delta.total_seconds()
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    days, hours, minutes = map(int, (days, hours, minutes))
    seconds = round(seconds, 6)

    # build date
    date = ''
    if days:
        date = '%sD' % days

    # build time
    time = u'T'
    # hours
    bigger_exists = date or hours
    if bigger_exists:
        time += '{:02}H'.format(hours)
    # minutes
    bigger_exists = bigger_exists or minutes
    if bigger_exists:
        time += '{:02}M'.format(minutes)
    # seconds
    if seconds.is_integer():
        seconds = '{:02}'.format(int(seconds))
    else:
        # 9 chars long w/leading 0, 6 digits after decimal
        seconds = '%09.6f' % seconds
    # remove trailing zeros
    seconds = seconds.rstrip('0')
    time += '{}S'.format(seconds)
    return u'P' + date + time
