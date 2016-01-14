#!/usr/bin/env python
from bcp.parse import convert_promethion_date
import datetime

'''
Utility functions.
'''

def add_seconds(dt, seconds):
    '''Add `seconds` to `dt`.'''
    return dt + datetime.timedelta(seconds=seconds)


def is_light(dt_str):
    '''assume 7am to 7pm is light'''
    d = convert_promethion_date(dt_str)
    if d.hour >= 7 and d.hour < 19:
        return 1
    else:
        return 0

def light_dark_bounds(timestamps):
    r = []
    light = 0
    k = 0
    while light == 0:
        light = is_light(timestamps[k])
        k+=1
    r.append(k-1)
    # at this point light=1
    for i in range(k, len(timestamps)):
        _light = is_light(timestamps[i])
        if _light != light:
            r.append(i)
            light = _light
    return r

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
