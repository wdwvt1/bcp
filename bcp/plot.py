#!/usr/bin/env python
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import datetime
import copy
from bcp.util import add_seconds


def circadian_rythm_axvspan(day_begin, day_end, start_timestamp, times):
    '''Create list of times when mice experience light.

    Notes
    -----
    This function makes many assumptions; most notably it assumes that daylight
    does not span a change in day and that night does.

    Parameters
    ----------
    day_begin : int
        Time in hours in military time when day begins (e.g. 7).
    day_end : ind
        Time in hours in military time when day ends (e.g. 19).
    start_timestamp : datetime.datetime
        The datetime object corresponding to the first entry in `times`.
    times : np.array
        One dimensional array with the ith entry equal to the number of seconds
        since the 0th entry.

    Returns
    -------
    nights : list
        List of tuples with the 1st tuple, 0th entry, the beginning (in seconds)
        of the first night, and the 1st tuple, 1st entry, the end (in seconds)
        of the first night. The ith row is the beginning and end of the ith
        night.
    '''
    total_time_span = times[-1] - times[0]
    day_period = (day_end - day_begin) * 3600
    night_period = 24 * 3600 - day_period
    cycles = np.ceil(total_time_span / (3600 * 24.))

    if start_timestamp.hour >= day_begin and start_timestamp.hour < day_end:
        _tmp = copy.copy(start_timestamp).replace(hour=day_end, minute=0,
                                                  second=0)
        delta = (_tmp - start_timestamp).total_seconds()
        night_1_start = times[0] + delta
        night_1_end = night_1_start + night_period
    else:
        _tmp = copy.copy(start_timestamp)
        _tmp.replace(day=start_timestamp.day + 1, hour=day_end, minute=0,
                     second=0)
        delta = (_tmp - start_timestamp).total_seconds()

        night_1_start = times[0]
        night_1_end = night_1_start + delta
    
    nights = [(night_1_start, night_1_end)]
    for _ in range(np.int(cycles)):
        n_i_start = nights[-1][1] + day_period
        n_i_end = n_i_start + night_period
        nights.append((n_i_start, n_i_end))
    return nights
