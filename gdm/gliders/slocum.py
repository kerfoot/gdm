"""
Functions for reading slocum glider dba files into pandas DataFrames and storing the associated metadata
Requirements:
1. Parse a dba file and retrieve file headers
2. Parse a dba file and retrieve the sensor definitions
3. Parse a dba file and load into a pandas DataFrame. Add the following additional columns
    segment
4. Index profiles and add:
    profile_id
    profile_dir
"""
import logging
import os
import math
import pandas as pd
import numpy as np
import re
import glob
import pytz
from operator import itemgetter
from dateutil import parser
from gdm.timestamps import epoch2datetime
from gdm.yo import index_profiles
from gdm.ctd import derive_ctd
from gdm.gps import dm2dd

logger = logging.getLogger(__file__)

# Mappings of native sensor sizes to numpy dtype
slocum_data_types = {'1': 'i1',
                     '2': 'i2',
                     '4': 'f4',
                     '8': 'f8'}

# Mappings of native sensor names to data frame column names
slocum_sensor_mappings = {'sci_water_pressure': 'raw_p',
                          'sci_water_cond': 'raw_c',
                          'sci_water_temp': 'raw_t',
                          'depth': 'raw_z',
                          'salinity': 'raw_s',
                          'density': 'raw_d',
                          'sci_m_present_time': 'science_time',
                          'ilatitude': 'ilat',
                          'ilongitude': 'ilon'}


def load_slocum_dba(dba_file, temp_sensor='sci_water_temp', cond_sensor='sci_water_cond', pres_sensor='sci_water_pressure', keep_gld_dups=False):
    """
    Load an Slocum glider ascii dba file, index any profiles and derive ctd parameters

    Parameters:
        dba_file: A Slocum glider dba file created using TWRC shore-side binary to ascii conversion tools
        temp_sensor: native Slocum glider temperature sensor
        cond_sensor: native Slocum glider conductivity sensor
        pres_sensor: native Slocum glider pressure sensor
        keep_gld_dups: boolean value specifying whether to keep or discard all gld_dup_ sensors

    Returns:
          df: pandas DataFrame containing the file data table
          profiles_meta: pandas DataFrame containing metadata for each indexed profile
    """

    # The DBA file must contain the following sensors
    required_sensors = [cond_sensor,
                        temp_sensor,
                        pres_sensor,
                        'm_gps_lat',
                        'm_gps_lon']

    # Load the DBA file into a data frame and get metadata from all indexed profiles
    dba, profiles_meta = load_dba(dba_file, keep_gld_dups=keep_gld_dups)

    # Make sure the dba contains the required sensors
    has_required = True
    for required_sensor in required_sensors:
        if required_sensor not in dba.columns:
            logger.warning('Missing required sensor {:}'.format(required_sensor))
            has_required = False

    if not has_required:
        logger.warning('Skipping CTD calculations due to 1 or more missing required sensors: {:}'.format(dba_file))
        return dba, profiles_meta

    # Calculate derived CTD data
    ctd_params = derive_ctd(dba[cond_sensor], dba[temp_sensor], dba[pres_sensor], dba.ilatitude, dba.ilongitude)

    # Add the derived CTD parameters to the data frame
    if not ctd_params.empty:
        dba = pd.concat([dba, ctd_params], axis=1)

    return dba, profiles_meta


def load_dba(dba_file, keep_gld_dups=False):
    """
    Parse a Slocum glider DBA file, perform some rough QC and processing and return all sensor data in a pandas
    DataFrame along with indexed profile information.

    Parameters:
         dba_file: A Slocum glider dba file created using TWRC shoreside binary to ascii conversion tools
         keep_gld_dups: boolean value specifying whether to keep or discard all gld_dup_ sensors

    Returns:
          df: pandas DataFrame containing the file data table
          profiles_meta: pandas DataFrame containing metadata for each indexed profile
    """
    # Read the dba file contents into a pandas DataFrame
    df = load_dba_to_df(dba_file, keep_gld_dups=keep_gld_dups)
    if df.empty:
        return df

    # Parse the dba sensor definitions
    dba_sensor_defs = parse_dba_sensor_defs(dba_file)

    # Process GPS:
    # 1. Convert to decimal degrees
    # 2. Remove bad values
    # 3. Fill in missing values
    df['latitude'] = np.NaN
    df['longitude'] = np.NaN
    df['ilatitude'] = np.NaN
    df['ilongitude'] = np.NaN
    if 'm_gps_lat' in df.columns:
        # Replace bad fixes with NaN
        df['latitude'] = dm2dd(df.m_gps_lat)
        df.latitude.where((df.latitude < 90) & (df.latitude > -90), inplace=True)
        df['ilatitude'] = df.latitude.interpolate()
    if 'm_gps_lon' in df.columns:
        df['longitude'] = dm2dd(df.m_gps_lon)
        df.longitude.where((df.longitude < 180) & (df.longitude > -180), inplace=True)
        df['ilongitude'] = df.longitude.interpolate()

    # Process pressure sensors
    # 1. Remove bad values
    # 2. Covert bar values to decibar
    pressure_sensors = [s['native_sensor_name'] for s in dba_sensor_defs if s['units'] == 'bar']
    for pressure_sensor in pressure_sensors:
        if pressure_sensor not in df.columns:
            continue

        df[pressure_sensor].where(df[pressure_sensor] > 0, inplace=True)
        df[pressure_sensor] = df[pressure_sensor] * 10.

    # Index profiles and build the profiles metadata DataFrame
    profiles_meta = build_profiles(df)

    return df, profiles_meta


def load_dba_to_df(dba_file, keep_gld_dups=False):
    if not os.path.isfile(dba_file):
        logger.error('DBA file does not exist: {:}'.format(dba_file))
        return pd.DataFrame([])

    # Parse the file header
    dba_headers = parse_dba_header(dba_file)
    if not dba_headers:
        return pd.DataFrame([])

    # Parse the sensor definitions
    dba_sensor_defs = parse_dba_sensor_defs(dba_file)
    if not dba_sensor_defs:
        return pd.DataFrame([])

    # Create a list of sensor included in the dba file
    sensors = [s['native_sensor_name'] for s in dba_sensor_defs]
    if 'm_present_time' not in sensors:
        logger.error('DBA file is missing m_present_time: {:}'.format(dba_file))
        return pd.DataFrame([])

    # Get the list of timestamp sensors for parsing in the DataFrame
    timestamps = [s['native_sensor_name'] for s in dba_sensor_defs if s['units'] == 'timestamp']
    if not keep_gld_dups:
        timestamps = [t for t in timestamps if not t.startswith('gld_dup_')]

    # Calculate the number of header lines to skip
    num_header_lines = int(dba_headers['num_ascii_tags']) + int(dba_headers['num_label_lines'])

    # Read
    df = pd.read_table(dba_file,
                       delimiter='\s+',
                       names=sensors,
                       on_bad_lines = 'warn', 
                       # error_bad_lines=False,
                       # warn_bad_lines=True,
                       header=None,
                       parse_dates=timestamps,
                       date_parser=epoch2datetime,
                       skiprows=num_header_lines)

    if not keep_gld_dups:
        dup_sensors = [s['native_sensor_name'] for s in dba_sensor_defs if
                       s['native_sensor_name'].startswith('gld_dup')]
        if dup_sensors:
            logging.debug('Dropping {:} duplicate sensors'.format(len(dup_sensors)))
            df = df.drop(dup_sensors, axis=1)

    # Use m_present_time as the index
    df.index = df.m_present_time
    # Rename the index to time
    df.index.name = 'time'

    df.columns.name = 'sensors'

    # Drop rows with duplicate index timestamps
    df = df[~df.index.duplicated(keep='first')]

    # Add the segment name
    df['segment'] = dba_headers['segment_filename_0']
    # Add the 8x3 name
    df['the8x3_filename'] = dba_headers['the8x3_filename']

    return df


def parse_dba_header(dba_file):
    """Parse the header lines of a Slocum dba ascii table file

    Args:
        dba_file: dba file to parse

    Returns:
        An dictionary mapping heading keys to values
    """

    if not os.path.isfile(dba_file):
        logger.error('Invalid DBA file specified: {:}'.format(dba_file))
        return

    try:
        with open(dba_file, 'r') as fid:

            dba_headers = {}

            # Get the first line of the file to make sure it starts with 'dbd_label:'
            f = fid.readline()
            if not f.startswith('dbd_label:'):
                return

            tokens = f.strip().split(': ')
            if len(tokens) != 2:
                logger.error('Invalid dba file {:}'.format(dba_file))
                return

            dba_headers[tokens[0]] = tokens[1]

            for f in fid:

                tokens = f.strip().split(': ')
                if len(tokens) != 2:
                    break

                dba_headers[tokens[0]] = tokens[1]
    except IOError as e:
        logger.error('Error parsing {:s} dba header: {}'.format(dba_file, e))
        return

    if not dba_headers:
        logger.warning('No headers parsed: {:s}'.format(dba_file))

    return dba_headers


def parse_dba_sensor_defs(dba_file):
    """Parse the sensor definitions in a Slocum dba ascii table file.

    Args:
        dba_file: dba file to parse

    Returns:
        An array of dictionaries containing the file sensor definitions
    """

    if not os.path.isfile(dba_file):
        logger.error('Invalid DBA file specified: {:}'.format(dba_file))
        return

    # Parse the file header lines
    dba_headers = parse_dba_header(dba_file)
    if not dba_headers:
        return

    if 'num_ascii_tags' not in dba_headers:
        logger.warning('num_ascii_tags header missing: {:s}'.format(dba_file))
        return

    # Sensor definitions begin on the line number after that contained in the
    # dba_headers['num_ascii_tags']
    num_header_lines = int(dba_headers['num_ascii_tags'])

    try:
        with open(dba_file, 'r') as fid:

            line_count = 0
            while line_count < num_header_lines:
                fid.readline()
                line_count += 1

            # Get the sensor names line
            sensors_line = fid.readline().strip()
            # Get the sensor units line
            units_line = fid.readline().strip()
            # Get the datatype byte storage information
            bytes_line = fid.readline().strip()

            sensors = sensors_line.split()
            units = units_line.split()
            datatype_bytes = bytes_line.split()

            if len(sensors) != len(units) or len(sensors) != len(datatype_bytes):
                logger.warning('Incomplete sensors, units or dtypes definition lines: {:}'.format(dba_file))
                return

            sensor_metadata = [
                {'native_sensor_name': sensors[s],
                 'units': units[s],
                 'dtype': slocum_data_types.get(datatype_bytes[s], 'f8')
                 } for s in
                range(len(sensors))]

            return sensor_metadata

    except IOError as e:
        logger.error('Error parsing {:s} dba header: {:s}'.format(dba_file, e))
        return


def build_profiles(df, z_sensor='sci_water_pressure'):
    if z_sensor not in df:
        logger.warning('Invalid depth/pressure sensor specified: {:}'.format(z_sensor))
        logger.error('Skipping profile creation')
        return pd.DataFrame([])

    indexed_profiles = index_profiles(df[z_sensor])

    profiles = []
    for (p_start, p_end) in indexed_profiles:
        pt0 = pd.to_datetime(p_start, unit='s')
        pt1 = pd.to_datetime(p_end, unit='s')

        # Set the profile mid-point time
        # Interval between dt1 and dt0
        pt_delta = pt1 - pt0

        # 1/2 dt_delta and add to dt0 to get the profile mean time
        pt_mean = pt0 + (pt_delta / 2)

        # Add the profile_dir
        profile = df[z_sensor].loc[pt0:pt1].dropna()
        profile_dir = ''
        if profile[0] - profile[-1] < 0:
            profile_dir = 'd'
        elif profile[0] - profile[-1] > 0:
            profile_dir = 'u'

        profile_info = [pt_mean,
                        pt_delta.total_seconds(),
                        len(profile),
                        profile_dir,
                        pt0,
                        pt1,
                        profile[0],
                        profile[-1],
                        df.segment.unique()[0],
                        math.fabs(profile[0] - profile[-1]) / pt_delta.total_seconds(),
                        (len(profile) / pt_delta.total_seconds()) ** -1]
        profiles.append(profile_info)

    profiles_meta = pd.DataFrame([])
    if profiles:
        # Create a DataFrame containing the indexed profiles information and indexed on midpoint_time
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
        profiles_meta = pd.DataFrame(profiles, columns=profile_cols).set_index('midpoint_time')

    return profiles_meta


def get_dbas(dba_dir, dt0=None, dt1=None):
    """Search for all dba files in dba_dir, optionally filtered by fileopen time.

    Parameters:
        dba_dir: Path to search
        dt0: starting datetime
        dt1: ending datetime

    Returns a pandas data frame indexed by fileopen_time
    """

    dbas = ls_dbas(dba_dir, dt0=dt0, dt1=dt1)

    return build_dbas_data_frame(dbas)


def build_dbas_data_frame(dba_files):

    if not dba_files:
        logger.warning('No valid dba files specified')
        return pd.DataFrame([])

    if not isinstance(dba_files, list):
        dba_files = [dba_files]

    dba_records = []

    dre = re.compile(r'([^_]+)')
    # Regex to pull the glider name
    glider_regexp = re.compile('(^.*)-\d{4}')

    for dba_file in dba_files:

        header = parse_dba_header(dba_file)
        if not header:
            logger.error('Invalid dba file: {:}'.format(dba_file))
            continue

        date_pieces = dre.findall(header['fileopen_time'])
        if not date_pieces:
            logger.warning('Invalid fileopen_time: {:}'.format(header['fileopen_time']))
            continue

        glider_match = glider_regexp.search(header['segment_filename_0'])
        if not glider_match:
            logger.warning('Cannot parse glider name: {:}'.format(header['segment_filename_0']))
            continue

        dt = parser.parse(
            '{:} {:}, {:} {:}'.format(date_pieces[1], date_pieces[2], date_pieces[4], date_pieces[3])).replace(
            tzinfo=pytz.UTC)

        header['created_time'] = dt
        header['file'] = os.path.basename(dba_file)
        header['path'] = os.path.dirname(dba_file)
        header['bytes'] = os.path.getsize(dba_file)
        header['glider'] = glider_match.groups()[0]

        columns = header.keys()

        ordered_columns = ['file']
        for c in columns:
            if c not in ordered_columns:
                ordered_columns.append(c)

        dba_records.append(
            pd.DataFrame([[header[c] for c in ordered_columns]], columns=ordered_columns).set_index('created_time'))

    dbas_df = pd.concat(dba_records)

    dbas_df.sort_index(inplace=True)

    return dbas_df


def ls_dbas(dba_dir, dt0=None, dt1=None):
    """Search for all dba files in dba_dir, optionally filtered by fileopen time.

        Parameters:
            dba_dir: Path to search
            dt0: starting datetime
            dt1: ending datetime

        Returns a list of dba files sorted by fileopen_time
    """

    dbas = []

    dre = re.compile(r'([^_]+)')

    if not os.path.isdir(dba_dir):
        logger.error('Invalid directory specified: {:}'.format(dba_dir))

    all_dbas = glob.glob(os.path.join(dba_dir, '*'))
    if not all_dbas:
        logger.warning('No files found')
        return dbas

    if dt0:
        dt0 = dt0.replace(tzinfo=pytz.UTC)
    if dt1:
        dt1 = dt1.replace(tzinfo=pytz.UTC)

    for dba_file in all_dbas:

        if not os.path.isfile(dba_file):
            continue

        header = parse_dba_header(dba_file)
        if not header:
            continue

        date_pieces = dre.findall(header['fileopen_time'])
        if not date_pieces:
            logger.warning('Invalid fileopen_time: {:}'.format(header['fileopen_time']))
            continue

        dt = parser.parse(
            '{:} {:}, {:} {:}'.format(date_pieces[1], date_pieces[2], date_pieces[4], date_pieces[3])).replace(
            tzinfo=pytz.UTC)

        if dt0:
            if dt < dt0:
                continue

        if dt1:
            if dt > dt1:
                continue

        dba = {'file': dba_file, 'dt0': dt}

        dbas.append(dba)

    if dbas:
        dbas.sort(key=itemgetter('dt0'))
        dbas = [dba['file'] for dba in dbas]

    return dbas
