import math
import pandas as pd
import logging
from xarray import open_dataset

logging.getLogger(__file__)


def load_navo_nc(nc_file, time_variable='scitime'):

    ds = open_dataset(nc_file)

    # Find time indexed variables
    t_vars = []
    for v in ds.variables:
        if ds[v].dims == ('time',):
            t_vars.append(v)

    # Export the time-indexed variables to a panda Dataframe
    df = ds[t_vars].to_dataframe()

    # Convert df.scitime from timedeltas to datetimes and set as the dataframe index
    df.index = [pd.to_datetime(d.total_seconds(), unit='s') for d in df[time_variable]]
    df.index.name = 'time'

    # Create a 2-D array of start and end profiles indices
    profile_inds = list(zip(ds.prof_start_index.values, ds.prof_end_index.values))

    pro_meta = create_profile_metadata(df.depth, profile_inds)

    return df, pro_meta, ds


# def load_ctd_nc(nc_file):
#
#     ds = open_dataset(nc_file)
#
#     # Find time indexed variables
#     t_vars = []
#     for v in ds.variables:
#         if ds[v].dims == ('time',):
#             t_vars.append(v)
#
#     # Export the time-indexed variables to a panda Dataframe
#     df = ds[t_vars].to_dataframe()
#
#     # Convert df.scitime from timedeltas to datetimes and set as the dataframe index
#     df.index = [pd.to_datetime(d.total_seconds(), unit='s') for d in df.scitime]
#     df.index.name = 'time'
#
#     # Create a 2-D array of start and end profiles indices
#     profile_inds = list(zip(ds.prof_start_index.values, ds.prof_end_index.values))
#
#     pro_meta = create_profile_metadata(df.depth, profile_inds)
#
#     return df, pro_meta


def create_profile_metadata(depth, profile_inds):
    profile_cols = ['midpoint_time',
                    'total_seconds',
                    'num_points',
                    'direction',
                    'start_time',
                    'end_time',
                    'start_depth',
                    'end_depth',
                    'segment',
                    'depth_resolution',
                    'sampling_frequency']

    pro_meta = []

    for p0, p1 in profile_inds:
        pro = depth.iloc[p0:p1].dropna()

        if pro.empty:
            logging.warning('Profile for indices {:}:{:} contains no valid rows'.format(p0, p1))
            continue

        profile_metadata = {k: None for k in profile_cols}
        profile_metadata['midpoint_time'] = pro.index.mean()
        profile_metadata['total_seconds'] = pro.index.max().timestamp() - pro.index.min().timestamp()
        profile_metadata['num_points'] = pro.size
        profile_dir = ''
        delta_depth = pro.iloc[0] - pro.iloc[-1]
        if delta_depth == 0:
            logging.warning(
                'Skipping profile. Cannot determine profile direction for profile spanning rows {:}-{:}'.format(p0, p1))
            continue
        if delta_depth < 0:
            profile_dir = 'd'
        else:
            profile_dir = 'u'
        profile_metadata['direction'] = profile_dir
        profile_metadata['start_time'] = pro.index[0]
        profile_metadata['end_time'] = pro.index[-1]
        profile_metadata['start_depth'] = pro[0]
        profile_metadata['end_depth'] = pro[-1]
        profile_metadata['depth_resolution'] = math.fabs((pro[0] - pro[-1]) / profile_metadata['total_seconds'])
        profile_metadata['sampling_frequency'] = pro.size / profile_metadata['total_seconds']

        pro_meta.append(profile_metadata)

    return pd.DataFrame(pro_meta).set_index('midpoint_time')


def get_ds_global_attributes(ds):

    return {'navoceano_{:}'.format(k): str(v) for k, v in ds.attrs.items()}


def get_ds_variable_defs(ds):

    default_attrs = {'units': '',
                     'standard_name': '',
                     'comment': ''}

    var_defs = {}

    for v in ds.variables:

        attrs = {'navoceano_{:}'.format(k): str(v) for k, v in ds[v].attrs.items()}
        attrs.update(default_attrs)

        var_defs[v] = {'nc_var_name': v, 'attrs': attrs, 'type': 'f8'}

    return var_defs
