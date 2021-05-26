#!/usr/bin/env python

import logging
import os
import sys
import argparse
import datetime
from xarray import DataArray
from gdm import GliderDataModel
from gdm.gliders.slocum import load_slocum_dba, build_dbas_data_frame


def main(args):
    """Parse one or more Slocum glider dba files and write time-series based NetCDF file(s)."""

    status = 0

    log_level = args.loglevel
    log_level = getattr(logging, log_level.upper())
    log_format = '%(asctime)s:%(module)s:%(levelname)s:%(message)s [line %(lineno)d]'
    logging.basicConfig(format=log_format, level=log_level)

    debug = args.debug
    config_path = args.config_path
    dba_files = args.dba_files
    drop_missing = args.drop
    profiles = args.profiles
    ngdac = args.ngdac
    nc_dest = args.output_path or os.path.realpath(os.curdir)
    clobber = args.clobber
    nc_format = args.nc_format

    if not os.path.isdir(config_path):
        logging.error('Invalid configuration path: {:}'.format(config_path))
        return 1

    if not dba_files:
        logging.error('No dba files specified')
        return 1

    if not os.path.isdir(nc_dest):
        logging.error('Invalid NetCDF destination specified: {:}'.format(nc_dest))
        return 1

    logging.info('Configuration path: {:}'.format(config_path))
    logging.info('NetCDF destination: {:}'.format(nc_dest))
    logging.info('Processing {:} dba files'.format(len(dba_files)))
    if profiles:
        logging.info('Writing profile-based NetCDFs')
    else:
        logging.info('Writing time-series NetCDFs')

    gdm = GliderDataModel(cfg_dir=config_path)
    logging.debug('{:}'.format(gdm))

    netcdf_count = 0
    for dba_file in dba_files:
        logging.info('Processing {:}'.format(dba_file))

        dba_df, pro_meta = load_slocum_dba(dba_file)

        if dba_df.empty:
            continue

        gdm.data = dba_df
        gdm.profiles = pro_meta

        if debug:
            logging.info('{:}'.format(gdm))
            logging.info('debug switch set so no NetCDF creation')
            continue

        dba_meta = build_dbas_data_frame(dba_file)
        if dba_meta.empty:
            continue

        if not profiles:
            logging.info('Writing time-series...')
            ds = gdm.to_timeseries_dataset(drop_missing=drop_missing)
            fname, ext = os.path.splitext(dba_meta.iloc[0].file)

            # Update history attribute
            if 'history' not in ds.attrs:
                ds.attrs['history'] = ''
            new_history = '{:}: {:} {:}'.format(datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                                                sys.argv[0],
                                                dba_file)
            if ds.attrs['history'].strip():
                ds.attrs['history'] = '{:}\n{:}'.format(ds.attrs['history'], new_history)
            else:
                ds.attrs['history'] = '{:}'.format(ds.attrs['history'], new_history)

            # Update the source global attribute
            profile_ds.attrs['source'] = dba_file

            # Add the source_file variable
            source_file_attrs = dba_meta.to_dict(orient='records')[0]
            source_file_attrs['bytes'] = '{:}'.format(source_file_attrs['bytes'])
            profile_ds['source_file'] = DataArray(source_file_attrs['filename_label'], attrs=source_file_attrs)

            netcdf_path = os.path.join(nc_dest, '{:}.nc'.format(fname))
            logging.info('Writing: {:}'.format(netcdf_path))
            ds.to_netcdf(netcdf_path)
            netcdf_count += 1
        else:
            logging.info('Writing profiles...')
            glider = dba_meta.iloc[0].glider
            dbd_type = dba_meta.iloc[0].filename_extension
            if ngdac:
                dbd_type = 'rt'
                if dba_meta.iloc[0].filename_extension != 'sbd':
                    dbd_type = 'delayed'

            for profile_time, profile_ds in gdm.iter_profiles(drop_missing=drop_missing):
                netcdf_path = os.path.join(nc_dest,
                                           '{:}_{:}_{:}.nc'.format(glider, profile_time.strftime('%Y%m%dT%H%M%SZ'),
                                                                   dbd_type))

                if os.path.isfile(netcdf_path):
                    if not clobber:
                        logging.info('Ignoring existing NetCDF: {:}'.format(netcdf_path))
                        continue
                    else:
                        logging.info('Clobbering existing NetCDF: {:}'.format(netcdf_path))

                # Update history attribute
                if 'history' not in profile_ds.attrs:
                    profile_ds.attrs['history'] = ''

                new_history = '{:}: {:} --profiles {:}'.format(
                    datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                    sys.argv[0],
                    dba_file)
                if profile_ds.attrs['history'].strip():
                    profile_ds.attrs['history'] = '{:}\n{:}'.format(profile_ds.attrs['history'], new_history)
                else:
                    profile_ds.attrs['history'] = '{:}'.format(new_history)

                # Update the source global attribute
                profile_ds.attrs['source'] = dba_file

                # Add the source_file variable
                source_file_attrs = dba_meta.to_dict(orient='records')[0]
                source_file_attrs['bytes'] = '{:}'.format(source_file_attrs['bytes'])
                profile_ds['source_file'] = DataArray(source_file_attrs['filename_label'], attrs=source_file_attrs)

                logging.info('Writing: {:}'.format(netcdf_path))
                profile_ds.to_netcdf(netcdf_path)

                netcdf_count += 1

    logging.info('{:} NetCDF files written'.format(netcdf_count))

    return status


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('config_path',
                            help='Location of deployment configuration files')

    arg_parser.add_argument('dba_files',
                            help='Source ASCII dba files to process',
                            nargs='+')

    arg_parser.add_argument('-p', '--profiles',
                            help='Write profile-based NetCDFs instead of time-series',
                            action='store_true')

    arg_parser.add_argument('-d', '--drop',
                            help='Drop variables for which no sensor definition is found',
                            action='store_true')

    arg_parser.add_argument('--ngdac',
                            help='Profile-based NetCDF files named using IOOS NGDAC naming conventions (rt or delayed)',
                            action='store_true')

    arg_parser.add_argument('-o', '--output_path',
                            help='NetCDF destination directory, which must exist. Current directory if not specified')

    arg_parser.add_argument('-c', '--clobber',
                            help='Clobber existing NetCDF files if they exist',
                            action='store_true')

    arg_parser.add_argument('-f', '--format',
                            dest='nc_format',
                            help='NetCDF file format',
                            choices=['NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT', 'NETCDF3_CLASSIC'],
                            default='NETCDF4_CLASSIC')

    arg_parser.add_argument('-x', '--debug',
                            help='Check configuration and create NetCDF file writer, but does not process any files',
                            action='store_true')

    arg_parser.add_argument('-l', '--loglevel',
                            help='Verbosity level',
                            type=str,
                            choices=['debug', 'info', 'warning', 'error', 'critical'],
                            default='info')

    parsed_args = arg_parser.parse_args()

    sys.exit(main(parsed_args))
