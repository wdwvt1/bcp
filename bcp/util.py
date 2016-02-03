#!/usr/bin/env python
from bcp.parse import convert_promethion_date
import datetime
import numpy as np

'''
Utility functions.
'''

def add_seconds(dt, seconds):
    '''Add `seconds` to `dt`.'''
    return dt + datetime.timedelta(seconds=seconds)

def seconds_till(dt, **kwargs):
    '''Seconds from `dt` till kwargs, inheret unspecfied kwargs from `dt`.'''
    tmp = {'year': dt.year, 'month': dt.month, 'day': dt.day, 'hour': dt.hour,
           'minute': dt.minute, 'second': dt.second}
    for k,v in kwargs.items():
        tmp[k] = v
    dt2 = datetime.datetime(tmp['year'], tmp['month'], tmp['day'], tmp['hour'],
                            tmp['minute'], tmp['second'])
    return (dt2 - dt).total_seconds()

def nights(day_start, day_length, start_timestamp, total_exp_seconds):
    '''Return array of times at which cages are dark.

    Parameters
    ----------
    day_start : int
        Time in hours in military time when day begins (e.g. 7).
    day_length : int
        Time in hours of day length.
    start_timestamp : datetime.datetime
        The datetime object corresponding to the first recording of the
        experiment.
    total_exp_seconds : int
        The total number of seconds in the experiment.

    Returns
    -------
    nights : np.array
        2xN array with N nights. The [i,0] entry is the time the ith night
        starts (in seconds since the experiment started) and [i,1] is the time
        the ith night ends.
    '''
    # Ensure that midnight occurs during 'night'; e.g. the day must change
    # during the night cycle or this function will break.
    assert day_start + day_length < 24
    
    day_end = day_start + day_length
    day_length = day_length * 3600 

    # Determine where the first timepoint is. 
    a = seconds_till(start_timestamp, hour = day_start, minute = 0, second = 0)
    b = seconds_till(start_timestamp, hour = day_end, minute = 0, second = 0)
    if (a > 0 and b > 0):
        n0s = 0
        n0e = a
    elif (a < 0 and b <= 0):
        n0s = 0
        n0e = seconds_till(start_timestamp, day = start_timestamp.day + 1,
                           hour = day_start, minute = 0, second = 0 )
    elif (a <= 0 and b > 0):
        n0s = seconds_till(start_timestamp, hour = day_end, minute = 0,
                           second = 0 )
        n0e = n0s + 24*3600 - day_length
    nights = [[n0s, n0e]]
    
    # Add entries to night and day until we surpass the total seconds elapsed 
    # for the experiment. Break and return.
    while nights[-1][1] < total_exp_seconds:
        nns = nights[-1][1] + day_length
        nne = nights[-1][1] + 24*3600
        if nns > total_exp_seconds:
            break
        elif nne > total_exp_seconds:
            nne = total_exp_seconds
        nights.append([nns, nne])
    return np.array(nights)

def days(nights, total_exp_seconds):
    '''Return array of times at which cages are light.

    Parameters
    ----------
    nights : np.array
        Return from the 'nights' function.
    total_exp_seconds : int
        The total number of seconds in the experiment.

    Returns
    -------
    days : np.array
        2xN array with N day. The [i,0] entry is the time the ith day starts
        (in seconds since the experiment started) and [i,1] is the time
        the ith day ends.
    '''
    days = np.vstack((nights[:-1, 1], nights[1:, 0])).T
    if nights[0, 0] > 0:
        days = np.vstack((np.array([0, nights[0, 0]]), days))
    if nights[-1, 1] < total_exp_seconds:
        days = np.vstack((days, np.array([nights[-1, 1], total_exp_seconds])))
    return days

def binary_rearing(data):
    '''Return array indicating if mouse is rearing.

    Notes
    -----
    Z position is 0 if the Z-beam is not being broken, and the position of the
    broken beam otherwise. While not strictly an artifact, the fact that 

    Parameters
    ----------
    data : np.array
        Z-position data.

    Returns
    -------
    np.array
        Data that has been converted into binary characters. 1 for Z-beam being
        broken, i.e. rearing, 0 otherwise.
    '''
    return np.where(data > 0, 1, 0)
