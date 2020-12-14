"""
Glider Data Model class for exporting raw glider data sets/files to XArray Dataset instances
"""
import os
import logging
import yaml
import pandas as pd
import xarray as xr
import numpy as np
import uuid
import datetime
from decimal import Decimal
from netCDF4 import default_fillvals
from gdm.gps import geospatial_bounds_wkt
from gdm.timestamps import timedelta_to_iso_duration, time_index_to_iso_resolution


class GliderDataModel(object):

    def __init__(self, cfg_dir=None):
        """

        :param cfg_dir:
        """

        self._cfg_path = None
        self._logger = logging.getLogger(os.path.basename(__file__))
        self._config_files = {'deployment.yml': {},
                              'global_attributes.yml': {},
                              'instruments.yml': [],
                              'sensor_defs.yml': {}}
        self._config_parameters = {}
        for config_file in self._config_files:
            self._config_parameters[config_file[:-4]] = None

        # Dictionary mapping sensor/variable names to their NetCDF encodings (dtype, zlib, complevel, etc.)
        self._encodings = {}
        self._df = pd.DataFrame([])
        self._profiles_meta = pd.DataFrame([])

        self._ds = xr.Dataset()

        self._default_encoding = {'zlib': True,
                                  'complevel': 1,
                                  'dtype': 'f8'}
        if cfg_dir:
            self.config_path = cfg_dir

    @property
    def config_path(self):
        return self._cfg_path

    @config_path.setter
    def config_path(self, config_path):
        """

        :param config_path:
        :return:
        """
        self._logger.debug('Setting configuration path: {:}'.format(config_path))

        if not os.path.isdir(config_path):
            self._logger.error('Invalid path specified: {:}'.format(config_path))
            return

        self._cfg_path = config_path

        self._load_configs()

    @property
    def data(self):
        return self._df

    @data.setter
    def data(self, df):
        """

        :param df:
        :return:
        """
        self._df = df

    @property
    def profiles(self):
        return self._profiles_meta

    @profiles.setter
    def profiles(self, pro_meta):
        """

        :param pro_meta:
        :return:
        """
        self._profiles_meta = pro_meta

    @property
    def missing_variable_definitions(self):

        if self._df.empty:
            self._logger.warning('No DataFrame found')
            return

        if not self._config_parameters.get('sensor_defs', {}):
            self._logger.warning('All sensor/variable definitions missing.')
            return

        missing_defs = [sensor for sensor in self._df if sensor not in self._config_parameters['sensor_defs']]

        return missing_defs

    def to_timeseries_dataset(self, drop_missing=False):
        """

        :param drop_missing:
        :return:
        """

        if self._df.empty:
            self._logger.warning('No pandas DataFrame found for conversion to XArray DataSet')
            return

        self._ds = self._df.to_xarray()

        # Create the platform variable
        self._set_platform()

        # Add any configured instrument variables
        self._set_instruments()

        # Set the trajectory/deployment
        self._set_trajectory()

        # Set global attributes on the DataSet
        self._set_global_attributes()

        # Rename and add attributes to sensors/variables
        self._finalize_variables(drop_missing=drop_missing)

        # Set NetCDF encodings
        self._set_encodings()

        # Tie up loose ends
        self._finish_dataset()

        # Set the global featureType attribute and the title
        self._ds.attrs['featureType'] = 'trajectory'
        self._ds.attrs['title'] = '{:} {:} trajectory'.format(self._config_parameters['deployment']['glider'],
                                                              self._df.index.min().strftime('%Y%m%dT%H%M%SZ'))

        return self._ds

    def slice_profile_dataset(self, profile_time, lat_sensor='ilatitude', lon_sensor='ilongitude', drop_missing=False):
        """

        :param profile_time:
        :param lat_sensor:
        :param lon_sensor:
        :param drop_missing:
        :return:
        """

        if self._df.empty:
            self._logger.warning('No pandas DataFrame found for conversion to XArray DataSet')
            return

        if self._profiles_meta.empty:
            self._logger.warning('No profile metadata specified with the DataFrame')
            return

        if profile_time not in self._profiles_meta.index:
            self._logger.warning('Invalid profile mid-point time specified: {:}'.format(profile_time))
            return

        row = self._profiles_meta.loc[profile_time]
        pro_df = self._df.loc[row.start_time:row.end_time]

        if pro_df.empty:
            self._logger.warning('No profile found in specified time frame ({:} - {:}'.format(
                row.start_time.strftime('%Y-%m-%dT%H:%M:%S'), row.end_time.strftime('%Y-%m-%dT%H:%M:%S')))

        self._ds = pro_df.to_xarray()

        # Create the platform variable
        self._set_platform()

        # Create the platform_id
        self._set_profile_id(profile_time)

        # Add any configured instrument variables
        self._set_instruments()

        # Set global attributes on the DataSet
        self._set_global_attributes()

        # Set the trajectory/deployment
        self._set_trajectory()

        # Need to create the following scalar variables in the DataSet:
        # profile_lat: mean latitude
        # profile_lon: mean longitude
        # profile_time: mean time
        self._ds['profile_lat'] = np.NaN
        self._ds['profile_lon'] = np.NaN
        if lat_sensor in pro_df:
            self._ds['profile_lat'] = pro_df[lat_sensor].mean()
        if self._ds['profile_lat'].isnull():
            self._logger.warning('variable profile_lat is NaN for profile_time={:}'.format(profile_time))
        if lon_sensor in pro_df:
            self._ds['profile_lon'] = pro_df[lon_sensor].mean()
        if self._ds['profile_lon'].isnull():
            self._logger.warning('variable profile_lon is NaN for profile_time={:}'.format(profile_time))
        self._ds['profile_time'] = profile_time

        # Rename and add attributes to sensors/variables
        self._finalize_variables(drop_missing=drop_missing)

        # Set NetCDF encodings
        self._set_encodings()

        # Tie up loose ends
        self._finish_dataset()

        # Set the global featureType attribute and the title
        self._ds.attrs['featureType'] = 'trajectoryProfile'
        self._ds.attrs['title'] = '{:} {:} trajectoryProfile'.format(
            self._config_parameters['deployment'].get('glider', 'unknownglider'),
            profile_time.strftime('%Y%m%dT%H%M%SZ'))

        return self._ds

    def iter_profiles(self):
        """

        :return:
        """

        for profile_time, row in self._profiles_meta.iterrows():
            yield profile_time, self.slice_profile_dataset(profile_time)

    def _set_global_attributes(self):
        """

        :return:
        """

        if not self._ds:
            self._logger.warning('No dataset available for global attribution')
            return

        global_atts = self._config_parameters.get('global_attributes', {}).copy()

        self._logger.debug('Setting global attributes...')
        self._ds.attrs = global_atts

        self._ds.attrs.update(self._add_temporal_geospatial_attributes())

    def _set_instruments(self):
        """

        :return:
        """

        if not self._ds:
            self._logger.warning('No dataset available for global attribution')
            return

        # Add instruments
        for instrument in self._config_parameters.get('instruments', []):
            self._logger.debug('Creating instrument variable {:}'.format(instrument['nc_var_name']))

            self._ds[instrument['nc_var_name']] = xr.DataArray(attrs=instrument['attrs'])
            self._ds[instrument['nc_var_name']].encoding = {'_FillValue': default_fillvals['i1'],
                                                            'dtype': 'i1',
                                                            'complevel': 1,
                                                            'zlib': True}

    def _set_trajectory(self):
        """

        :return:
        """

        if not self._ds:
            self._logger.warning('No dataset available for creating trajectory/deployment variable')
            return

        deployment_config = self._config_parameters.get('deployment', {})
        if not deployment_config:
            self._logger.warning('No deployment configuration found. Skipping platform variable creation')
            return
        if not deployment_config.get('trajectory_name', ''):
            self._logger.warning('No deployment trajectory name found. Skipping trajectory variable creation')
            return

        trajectory_attrs = {
            'cf_role': 'trajectory_id',
            'long_name': 'Trajectory/Deployment Name',
            'comment': 'A trajectory is a single deployment of a glider and may span multiple data files.'
        }

        self._logger.debug('Creating trajectory variable...')
        self._ds['trajectory'] = xr.DataArray(deployment_config['trajectory_name'], attrs=trajectory_attrs)

    def _set_platform(self):
        """

        :return:
        """

        if not self._ds:
            self._logger.warning('No dataset available for creating platform variable')
            return

        deployment_config = self._config_parameters.get('deployment', {})
        if not deployment_config:
            self._logger.warning('No deployment configuration found. Skipping platform variable creation')
            return
        if not deployment_config.get('platform', {}):
            self._logger.warning('No deployment platform configuration found. Skipping platform variable creation')
            return

        # Add platform variables
        self._logger.debug('Creating platform variable...')
        self._ds['platform'] = xr.DataArray(attrs=deployment_config['platform'])
        self._ds.platform.encoding = {'_FillValue': default_fillvals['i4'], 'dtype': 'i1', 'complevel': 1, 'zlib': True}

    def _set_profile_id(self, timestamp):
        """

        :param timestamp:
        :return:
        """

        if not self._ds:
            self._logger.warning('No dataset available for creating platform_id variable')
            return

        # Add profile_id
        self._logger.debug('Creating profile_id variable...')
        profile_id_attrs = self._config_parameters['sensor_defs']['profile_id'].get('attrs', {})
        self._ds['profile_id'] = xr.DataArray(timestamp, attrs=profile_id_attrs)
        self._ds.profile_id.encoding = {'_FillValue': default_fillvals['f8'], 'dtype': 'f8', 'complevel': 1,
                                        'zlib': True}

    def _finalize_variables(self, drop_missing=True):
        """

        :param drop_missing:
        :return:
        """

        if not self._ds:
            self._logger.warning('No dataset available for variable annotation')
            return

        if not self._config_parameters.get('sensor_defs', {}):
            self._logger.warning('No sensor/variable definitions found. Skipping finalizing of variables')
            return

        to_drop = [sensor for sensor in self._ds if sensor not in self._config_parameters['sensor_defs']]
        if drop_missing:
            self._logger.info('Dropping {:} undefined sensors/variables from Dataset'.format(len(to_drop)))
            if to_drop:
                self._ds = self._ds.drop_vars(to_drop, errors='ignore')
                for bad_var in to_drop:
                    self._logger.debug('Dropped variable {:}'.format(bad_var))
        else:
            self._logger.info('Keeping {:} undefined sensors/variables in Dataset'.format(len(to_drop)))

        # Rename variables and update attributes
        rename_mappings = {}
        for v in self._ds.variables:
            if v.startswith('instrument_'):
                # Skip instrument variables as they and their attributes are found in the instruments.yml config file
                continue

            if v not in self._config_parameters['sensor_defs']:
                self._logger.warning('No sensor definition found for {:}'.format(v))
                continue

            # Assign the attributes in the sensor definition
            self._logger.debug('Assigning attributes to variable {:}'.format(v))
            self._ds[v] = self._ds[v].assign_attrs(**self._config_parameters['sensor_defs'][v].get('attrs', {}))

            if v == self._config_parameters['sensor_defs'][v].get('nc_var_name', v):
                continue

            # Add the original variable name mapped to the new nc_var_name in sensor_definitions
            rename_mappings[v] = self._config_parameters['sensor_defs'][v]['nc_var_name']

        # Rename any sensors in mappings
        if rename_mappings:
            self._ds = self._ds.rename(rename_mappings)
            for renamed in rename_mappings:
                self._logger.debug('Renaming variable {:} to {:}'.format(renamed, rename_mappings[renamed]))

    def _set_encodings(self):
        """

        :return:
        """

        if not self._ds:
            self._logger.warning('No dataset available for encoding')
            return

        for v in self._ds.variables:

            if v.startswith('instrument_') or v == 'platform':
                continue

            if v not in self._encodings:
                self._logger.warning('No encoding set for variable {:}'.format(v))
                continue

            encoding = self._encodings[v].copy()

            self._logger.debug('Encoding variable {:}'.format(v))

            # Special encoding case for datetime64 dtypes
            if self._ds[v].dtype.name == 'datetime64[ns]':
                self._logger.debug('Encoding datetime64[ns] variable: {:}'.format(v))
                encoding['units'] = self._ds[v].attrs.get('units', 'seconds since 1970-01-01T00:00:00Z')
                self._ds[v].attrs.pop('units', None)
                self._ds[v].attrs.pop('calendar', None)
            elif self._ds[v].dtype.name.startswith('str'):
                self._logger.debug('Encoding string variable: {:}'.format(v))
                # Special encoding case for strings
                encoding['dtype'] = 'S1'
            else:
                self._logger.debug('Encoding numeric variable: {:}'.format(v))
                # encoding['dtype'] = self._ds[v].attrs.get('dtype', 'f8')
                encoding['_FillValue'] = default_fillvals[encoding['dtype']]

            # Drop dtype attribute if exists
            self._ds[v].attrs.pop('dtype', None)

            # Set the encoding
            self._ds[v].encoding = encoding

    def _finish_dataset(self):
        """

        :return:
        """
        if not self._ds:
            self._logger.warning('No dataset available for encoding')
            return

        if 'id' not in self._ds.attrs or not self._ds.attrs['id'].strip():
            self._ds.attrs['id'] = self._config_parameters['deployment'].get('trajectory_name', 'uknowntrajectory')

        if 'uuid' not in self._ds.attrs or not self._ds.attrs['uuid'].strip():
            self._ds.attrs['uuid'] = str(uuid.uuid4())

        nc_timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        self._ds.attrs['date_created'] = nc_timestamp
        self._ds.attrs['date_issued'] = nc_timestamp

        self._ds.attrs['source'] = 'local file'

    def _load_configs(self):
        """

        :return:
        """
        self._logger.debug('Loading available configuration files: {:}'.format(self._cfg_path))

        if not self._cfg_path:
            self._logger.warning('No configuration file path set')
            self._logger.warning('Exported datasets will contain minimal metadata')
            return

        if not os.path.isdir(self._cfg_path):
            self._logger.warning('Invalid configuration file path set')
            self._logger.warning('Exported datasets will contain minimal metadata')
            return

        for config_file, item_type in self._config_files.items():
            config_path = os.path.join(self._cfg_path, config_file)
            config_type = config_file[:-4]
            if not os.path.isfile(config_path):
                self._logger.warning('Configuration file does not exist: {:}'.format(config_path))
                self._config_parameters[config_type] = item_type
                continue

            with open(config_path, 'r') as fid:
                config_params = yaml.safe_load(fid)
                config_type = config_file[:-4]
                self._logger.debug('Configuring {:}: {:}'.format(config_type, config_path))
                self._config_parameters[config_type] = config_params

        # Set up the encoding dictionary.  Create an entry for each sensor name as well as the nc_var_name.  This will
        # allow self._finalize_variables to be called before or after self._set_encodings
        for sensor, sensor_desc in self._config_parameters['sensor_defs'].items():
            self._encodings[sensor] = self._default_encoding.copy()
            self._encodings[sensor]['dtype'] = sensor_desc.get('dtype', self._default_encoding['dtype'])
            if sensor != sensor_desc.get('nc_var_name', sensor):
                self._encodings[sensor_desc.get('nc_var_name')] = self._encodings[sensor].copy()

    def _add_temporal_geospatial_attributes(self):
        atts = {'geospatial_bounds': None,
                'geospatial_bounds_crs': 'EPSG:4326',
                'geospatial_bounds_vertical_crs': 'EPSG:5831',
                'geospatial_lat_max': None,
                'geospatial_lat_min': None,
                'geospatial_lat_units': 'degrees_north',
                'geospatial_lon_min': None,
                'geospatial_lon_max': None,
                'geospatial_lon_units': 'degrees_east',
                'geospatial_vertical_min': None,
                'geospatial_vertical_max': None,
                'geospatial_vertical_positive': 'down',
                'geospatial_vertical_resolution': None,
                'geospatial_vertical_units': 'm',
                'time_coverage_duration': None,
                'time_coverage_end': None,
                'time_coverage_resolution': None,
                'time_coverage_start': None}

        if not self._ds:
            self._logger.warning('No dataset available for encoding')
            return

        required_columns = ['latitude',
                            'longitude',
                            'depth']

        has_required = True
        for required_column in required_columns:
            if required_column not in self._ds:
                self._logger.warning('Missing required column: {:}'.format(required_column))
                has_required = False

        if not has_required:
            self._logger.warning('Cannot create temporal and geospatial attributes')
            self._logger.warning('Missing one or more required columns')
            return atts

        depths = self._ds.depth

        time_index = pd.DatetimeIndex(self._ds.time.values)
        min_time = time_index.min()
        max_time = time_index.max()
        min_depth = np.nanmin(depths)
        max_depth = np.nanmax(depths)
        min_lat = np.nanmin(self._ds.ilatitude)
        max_lat = np.nanmax(self._ds.ilatitude)
        min_lon = np.nanmin(self._ds.ilongitude)
        max_lon = np.nanmax(self._ds.ilongitude)

        atts['geospatial_bounds'] = geospatial_bounds_wkt(min_lat, max_lat, min_lon, max_lon)
        atts['geospatial_lat_min'] = min_lat
        atts['geospatial_lat_max'] = max_lat
        atts['geospatial_lon_min'] = min_lon
        atts['geospatial_lon_max'] = max_lon
        atts['geospatial_vertical_min'] = min_depth
        atts['geospatial_vertical_max'] = max_depth
        atts['geospatial_vertical_resolution'] = float(
            Decimal((max_depth - min_depth) / depths.size).quantize(Decimal('0.01')))
        atts['time_coverage_duration'] = timedelta_to_iso_duration(max_time - min_time)
        atts['time_coverage_end'] = max_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        atts['time_coverage_resolution'] = time_index_to_iso_resolution(time_index)
        atts['time_coverage_start'] = min_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        return atts

    def __repr__(self):
        has_data = False
        if not self._df.empty:
            has_data = True
        is_configured = True
        for config_type, item_type in self._config_parameters.items():
            if not self._config_parameters[config_type]:
                is_configured = False
                break
        return '<GliderNetCDF(cfg={:}, data={:}, profiles={:})>'.format(is_configured, has_data,
                                                                        self._profiles_meta.shape[0])
