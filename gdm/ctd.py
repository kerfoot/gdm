"""
Routines for calculating derived CTD parameters
"""
import pandas as pd
import numpy as np
from gsw import SP_from_C, SA_from_SP, CT_from_t, rho, z_from_p
from gsw.density import sound_speed
import logging
from gdm.utils import calculate_series_rate_of_change

logger = logging.getLogger(__file__)


def derive_ctd(c, t, p, lat, lon):
    ctd_params = pd.DataFrame([])

    if not isinstance(c, pd.Series):
        logger.error('c must be a conductivity time series')
        return ctd_params

    if not isinstance(t, pd.Series):
        logger.error('t must be a temperature time series')
        return ctd_params

    if not isinstance(p, pd.Series):
        logger.error('p must be a pressure time series')
        return ctd_params

    if not isinstance(lat, pd.Series):
        logger.error('lat must be a latitude time series')
        return ctd_params

    if not isinstance(lon, pd.Series):
        logger.error('c must be a longitude time series')
        return ctd_params

    if np.any(lat.isna()):
        lat = lat.interpolate()
    if np.any(lon.isna()):
        lon = lon.interpolate()

    raw_ctd = pd.concat([c, t, p, lat, lon], axis=1)

    mappings = {c.name: 'cond',
                t.name: 'temp',
                p.name: 'pres',
                lat.name: 'lat',
                lon.name: 'lon'}

    raw_ctd.rename(columns=mappings, inplace=True)

    depth = calculate_depth(raw_ctd.pres.array, raw_ctd.lat.array)
    salinity = calculate_practical_salinity(raw_ctd.cond, raw_ctd.temp, raw_ctd.pres)
    density = calculate_density(raw_ctd.temp, raw_ctd.pres, salinity, raw_ctd.lat, raw_ctd.lon)
    svel = calculate_sound_speed(raw_ctd.temp, raw_ctd.pres, salinity, raw_ctd.lat, raw_ctd.lon)

    return pd.DataFrame(np.vstack((depth, salinity, density, svel)).T, index=raw_ctd.index,
                        columns=['depth', 'salinity', 'density', 'sound_speed'])


def calculate_practical_salinity(conductivity, temperature, pressure):
    """Calculates practical salinity given glider conductivity, temperature,
    and pressure using Gibbs gsw SP_from_C function.

    Parameters:
        timestamp, conductivity (S/m), temperature (C), and pressure (bar).

    Returns:
        salinity (psu PSS-78).
    """

    # Convert S/m to mS/cm
    ms_conductivity = conductivity * 10

    return SP_from_C(
        ms_conductivity,
        temperature,
        pressure
    )


def calculate_density(temperature, pressure, salinity, latitude, longitude):
    """Calculates density given glider practical salinity, pressure, latitude,
    and longitude using Gibbs gsw SA_from_SP and rho functions.

    Parameters:
        timestamps (UNIX epoch),
        temperature (C), pressure (dbar), salinity (psu PSS-78),
        latitude (decimal degrees), longitude (decimal degrees)

    Returns:
        density (kg/m**3),
    """

    # dBar_pressure = pressure * 10

    absolute_salinity = SA_from_SP(
        salinity,
        pressure,
        longitude,
        latitude
    )

    conservative_temperature = CT_from_t(
        absolute_salinity,
        temperature,
        pressure
    )

    density = rho(
        absolute_salinity,
        conservative_temperature,
        pressure
    )

    return density


def calculate_sound_speed(temperature, pressure, salinity, latitude, longitude):
    """Calculates sound speed given glider practical in-situ temperature, pressure and salinity using Gibbs gsw
    SA_from_SP and rho functions.

    Parameters:
        temperature (C), pressure (dbar), salinity (psu PSS-78), latitude, longitude

    Returns:
        sound speed (m s-1)
    """

    absolute_salinity = SA_from_SP(
        salinity,
        pressure,
        longitude,
        latitude
    )

    conservative_temperature = CT_from_t(
        absolute_salinity,
        temperature,
        pressure
    )

    speed = sound_speed(
        absolute_salinity,
        conservative_temperature,
        pressure
    )

    return speed


def calculate_depth(pressure, latitude):
    """Calculates depth from pressure (dbar) and latitude.  By default, gsw returns depths as negative.  This routine
    returns the absolute values for positive depths.

    Paramters:
        pressure (decibars)
        latitude (decimal degrees)

    Returns:
        depth (meters)
    """

    return abs(z_from_p(pressure, latitude))


def correct_thermistor_response(temperature, tau=0.2):

    temp_roc = calculate_series_rate_of_change(temperature)

    ftr = tau * temp_roc
    corrected_temperature = temperature + ftr
    corrected_temperature.name = 'temperature_ftr_corr'

    ftr = pd.Series(ftr, index=temperature.index, name='finite_thermistor_response')

    return corrected_temperature, ftr


def calculate_ctm(ct, alpha, tau):

    ctm = pd.Series()

    return ctm
