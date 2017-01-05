'''
This is a utility module for analyzing user-geo data in python
'''
import shapely
import dateutil
import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.font_manager as fm
import pandas as pd
from pytz import timezone
from tzwhere import tzwhere
from math import sin, cos, sqrt, atan2, radians

KM_PER_RADIAN = 6371.0088

# MATPLOTLIB
TITLE_FONT = fm.FontProperties(family='Bitstream Vera Sans', style='normal', size=15, weight='normal', stretch='normal')
LABEL_FONT = fm.FontProperties(family='Bitstream Vera Sans', style='normal', size=12, weight='normal', stretch='normal')
TICKS_FONT = fm.FontProperties(family='Bitstream Vera Sans', style='normal', size=10, weight='normal', stretch='normal')
ANNOTATION_FONT = fm.FontProperties(family='Bitstream Vera Sans', style='normal', size=10, weight='normal', stretch='normal')
AXIS_BGCOLOR = '#f0f0f0'

# TZ
TZ = tzwhere.tzwhere(shapely=True, forceTZ=True)

# Day of Week
DOW_MAP = {
    'Sun': 0,
    'Mon': 1,
    'Tue': 2,
    'Wed': 3,
    'Thu': 4,
    'Fri': 5,
    'Sat': 6
    }


def get_tz(lat, lng, tz=TZ):
    '''
    Given latitude and longitude coordinates, this function gets the timezone.
    '''
    try:
        return tz.tzNameAt(lat, lng, forceTZ=True)
    except:
        print ('no tz')
        return None

def adjust_tz(timestamp, tz):
    '''
    Given a timestamp in UTC and timezone, this function adjusts the datetime.
    '''
    try:
        return dateutil.parser.parse(timestamp).astimezone(timezone(tz))
    except:
        print ('error adjusting tz')
        return timestamp

def get_dt_vars(ts):
    '''
    Given a timestamp, this function extracts the day, month, hour, and day of week.
    '''
    day, month, hour, day_of_week = None, None, None, None
    try:
        day = ts.day
        month = ts.month
        hour = ts.hour
        day_of_week = ts.strftime('%a')
    except:
        pass
    return (day, month, hour, day_of_week)

def haversine_dist(lat1, lon1, lat2, lon2, measure='mi'):
    '''
    This function computes the haversine distance.
    Input: latitude/longitude coordinates
    Computes the great circle distance between two points on the earth.
    '''
    if measure == 'mi':
        R = 3958.75
    elif measure == 'km':
        R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def haversine(u,v, unit='km'):
    '''
    Returns distance between 2 coordinate (lat/long) pairs in km
    '''
    lat1, lon1 = map(radians, u)
    lat2, lon2 = map(radians, v)

    d_lon = lon2 - lon1
    d_lat = lat2 - lat1

    a = sin(d_lat / 2)**2 + cos(lat1) * cos(lat2) * sin(d_lon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    if unit == 'km':
        R = 6373.0
    else:
        R = 1.0
    distance = R * c
    return distance

def hourly_plot(df):
    hours = sorted(df.hour.unique())
    N = len(hours)
    colors = cm.Spectral(np.linspace(0, 1, N))

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 1)
    for idx, hour in enumerate(hours):
        ax.scatter(hour, 0, color=colors[idx])
    ax.set_ylim(-0.005, 0.005)
    ax.set_xlim(-1, 24)

    fig, ax = plt.subplots()
    hour_grp = df.groupby('hour')
    i=0
    for hour, group_df in hour_grp:
        ax.scatter(group_df['long'], group_df['lat'], color=colors[i])
        i+=1


def count_demical_pts(x):
    try:
        return str(float(x))[::-1].find('.')
    except:
        return None

def variance(x):
    return pd.Series(x).dropna().var()

def round_geo(x,n):
    try:
        return np.round(x,n)
    except:
        return None
