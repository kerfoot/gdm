"""
Routines for parsing, converting and calculating GPS coordinates
"""
import numpy as np
import logging
from shapely.geometry import Polygon

logger = logging.getLogger(__file__)


def get_decimal_degrees(lat_lon):
    """Converts glider gps coordinate ddmm.mmm to decimal degrees dd.ddd

    Arguments:
    lat_lon - A floating point latitude or longitude in the format ddmm.mmm
        where dd's are degrees and mm.mmm is decimal minutes.

    Returns decimal degrees float
    """

    if np.isnan(lat_lon):
        return np.nan

    # Absolute value of the coordinate
    try:
        pos_lat_lon = abs(lat_lon)
    except (TypeError, ValueError) as e:
        return

    # Calculate NMEA degrees as an integer
    nmea_degrees = int(pos_lat_lon / 100) * 100

    # Subtract the NMEA degrees from the absolute value of lat_lon and divide by 60
    # to get the minutes in decimal format
    gps_decimal_minutes = (pos_lat_lon - nmea_degrees) / 60.0

    # Divide NMEA degrees by 100 and add the decimal minutes
    decimal_degrees = (nmea_degrees / 100) + gps_decimal_minutes

    if lat_lon < 0:
        return -decimal_degrees

    return decimal_degrees


def geospatial_bounds_wkt(min_lat, max_lat, min_lon, max_lon):
    """Return the well known text (WKT) representation of the geographic bounding box for the specified latitude and
    longitude bounds

    Parameters:
        min_lat: south latitude in decimal degrees
        max_lat: north latitude in decimal degrees
        min_lon: west longitude in decimal degrees
        max_lon: east longitude in decimal degrees

    Returns:
         WKT string representation of the bounding box
    """
    polygon_wkt = ''

    has_required = True
    if np.isnan(min_lat):
        logger.warning('No valid minimum latitude found')
        has_required = False
    if np.isnan(max_lat):
        logger.warning('No valid maximum latitude found')
        has_required = False
    if np.isnan(min_lon):
        logger.warning('No valid minimum longitude found')
        has_required = False
    if np.isnan(max_lon):
        logger.warning('No valid maximum longitude found')
        has_required = False

    if not has_required:
        logger.warning('Cannot determine geospatial bounds')
        return polygon_wkt

    # Create polygon WKT and set geospatial_bounds
    coords = ((max_lat, min_lon),
              (max_lat, max_lon),
              (min_lat, max_lon),
              (min_lat, min_lon),
              (max_lat, min_lon))

    polygon = Polygon(coords)
    polygon_wkt = polygon.wkt

    return polygon_wkt

def dm2dd(lats_lons):
    return [get_decimal_degrees(lat_lon) for lat_lon in lats_lons]