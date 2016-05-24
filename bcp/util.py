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
