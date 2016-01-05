#!/usr/bin/env python
from bcp.parse import convert_promethion_date

'''
Utility functions.
'''

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