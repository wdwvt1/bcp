#!/usr/bin/env python
from __future__ import division

import numpy as np 

'''
This library contains functions for replicating the Ethoscan functionality
provided by the Promethion library.

Promethion provides the following codes for behaviors:

EFODA:  Interaction with food hopper A (significant uptake found)
TFODA:  Interaction with food hopper A (no significant uptake)
DWATR:  Interaction with water dispenser (significant uptake found)
TWATR:  Interaction with water dispenser (no significant uptake)
WHEEL:  Interaction with wheel (>= 1 revolution)
IHOME:  Entered habitat (stable mass reading)
THOME:  Interaction with habitat (no stable mass reading)
LLNGE:  Long lounge (> 60 sec, no non-XY sensor interactions)
SLNGE:  Short lounge (5 - 60 sec, no non-XY sensor interactions)
EFODB:  Interaction with food hopper B (significant uptake found)
TFODB:  Interaction with food hopper B (no significant uptake)

This is the header for the reports from our animals. This may be different
depending on cage setup, software version, etc. This header is actually 
incorrect, Start_Date and Start_Time are separated by a tab, not a comma:

' Sample,Start_Date,Start_Time,End_Time,Durat_Sec,Activity,Amount,Rear%,X_cm,Y_cm,S_cm\r\n'

'''

BEHAVIOR_CODES = ['EFODA', 'TFODA', 'DWATR', 'TWATR', 'WHEEL', 'IHOME', 'THOME',
                  'LLNGE', 'SLNGE', 'EFODB', 'TFODB']
BEHAVIOR_CODES_TO_INT_MAP = {k:v for k, v in zip(BEHAVIOR_CODES,
                                                 range(len(BEHAVIOR_CODES)))}
INT_TO_BEHAVIOR_CODES_MAP = {v:k for k, v in zip(BEHAVIOR_CODES,
                                                 range(len(BEHAVIOR_CODES)))}

def parse_ethoscan_line(line):
    '''Parse a single line of Ethoscan behavior report line.'''
    v = line.strip().split(',')
    return [v[0], v[4], v[3], v[5], v[6], v[7], v[8], v[9]]

def parse_ethoscan_report(lines, start_time=None):
    '''Parse Ethoscan report, casting behaviors to integers to avoid mixed type.

    Parameters
    ----------
    lines : list
        Strings from a parsed Ethoscan file.
    start_time: int, optional
        If provided, a number of seconds to add to the first column of `data`.
        This is useful if trying to match Ethoscan observations with a longer
        time series.

    Returns
    -------
    data: np.array
        Each row is the observation of a single behavior and each column is a
        different piece of information about that behavior. The columns are:
         'time' at which the behavior started (recorded as the number of seconds
         since the beginning of the experiment, possibly with `start_time`
         added).
         'activity' that the mouse is engaging in (encoded as an int).
         'duration' in seconds.
         'amount' or magnitude of measurement (cm, ml, or revs/s).
         'rearing percentage' - time spent rearing during the given behavior.
         'x' - x-location of the behavior (start or end?).
         'y' - y-location of the behavior (start or end?).
         's' - total distance traveled during the behavior.
    '''
    # Based on the files I have seen, and the Promethion software I have worked
    # with, line 78 of the promethion output file is where the data begins.
    data = np.array([parse_ethoscan_line(l) for l in lines[78:] if l!='\r\n'])
    data[:, 1] = [BEHAVIOR_CODES_TO_INT_MAP[i] for i in data[:, 1]]
    data = data.astype(float)
    if start_time == None:
        return data
    else:
        data[:, 0] += start_time
        return data

def long_lounge():
    pass

def short_lounge():
    pass

def eating_from_food_hopper():
    pass

def touching_food_hopper():
    pass

def drinking_from_water_bottle():
    pass

def touching_water_bottle():
    pass

def running_on_wheel():
    pass

def in_home():
    pass

def touching_home():
    pass






















