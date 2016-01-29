#!/usr/bin/env python
import numpy as np
import datetime, time, os

'''
Library functions for dealing with the Promethion data.

# Example Promethion Data Output
Date    Time,XPos_1,YPos_1,ZPos_1,XBreak_1,YBreak_1,ZBreak_1,WheelCount_1,FoodA_1,Water_1,BodyMass_1,XPos_2,YPos_2,ZPos_2,XBreak_2,YBreak_2,ZBreak_2,WheelCount_2,FoodA_2,Water_2,BodyMass_2,XPos_3,YPos_3,ZPos_3,XBreak_3,YBreak_3,ZBreak_3,WheelCount_3,FoodA_3,Water_3,BodyMass_3,XPos_4,YPos_4,ZPos_4,XBreak_4,YBreak_4,ZBreak_4,WheelCount_4,FoodA_4,Water_4,BodyMass_4,XPos_5,YPos_5,ZPos_5,XBreak_5,YBreak_5,ZBreak_5,WheelCount_5,FoodA_5,Water_5,BodyMass_5,XPos_6,YPos_6,ZPos_6,XBreak_6,YBreak_6,ZBreak_6,WheelCount_6,FoodA_6,Water_6,XPos_7,YPos_7,ZPos_7,XBreak_7,YBreak_7,ZBreak_7,WheelCount_7,FoodA_7,Water_7,BodyMass_7,XPos_8,YPos_8,ZPos_8,XBreak_8,YBreak_8,ZBreak_8,WheelCount_8,FoodA_8,Water_8,BodyMass_8,RT_FoodA_1,RT_Water_1,RT_WheelCount_1,RT_FoodA_2,RT_Water_2,RT_WheelCount_2,RT_FoodA_3,RT_Water_3,RT_WheelCount_3,RT_FoodA_4,RT_Water_4,RT_WheelCount_4,RT_FoodA_5,RT_Water_5,RT_WheelCount_5,RT_FoodA_6,RT_Water_6,RT_WheelCount_6,RT_FoodA_7,RT_Water_7,RT_WheelCount_7,RT_FoodA_8,RT_Water_8,RT_WheelCount_8,CommMS,ElSeconds,Markers
6/11/2015 18:41:56,5.75,21.75,18.25,19,19,11,6,370.3022,348.3497,-0.3671426,4.5,7,3,17.5,21,8.5,4,381.5571,333.5468,10.75229,5.25,2,0,0,0,0,5,378.6284,320.0024,-0.4578373,4.75,5.5,0,0,0,0,7,376.9134,336.9136,-8.659991E-03,2,4.5,0,0,0,0,4,380.4408,333.2097,6.576459E-02,2.25,2.5,0,0,0,0,0,384.5407,333.9856,3.75,6.75,0,0,0,0,6,379.3676,323.5154,0.5884944,6.25,9.75,0,0,0,0,5,388.9341,344.4416,-0.1612068,0,0,6,0,0,4,0,0,5,0,0,7,0,0,4,0,0,0,0,0,6,0,0,5,255,0.002,-1
'''

def convert_promethion_date(dt_str):
    '''Convert date of the form M/DD/YYYY HH:MM:SS to datetime object.'''
    d, t = dt_str.split()
    month, day, year = d.split('/')
    return datetime.datetime(*map(int, [year, month, day]+ t.split(':')))

def time_since_start(x, dt_start):
    '''Count total seconds elapsed between dt_start and x'''
    return (x - dt_start).total_seconds()

def promethion_to_array(fp, cages, fields, start_timestamp=None):
    '''Convert combined Promethion files to numpy arrays.

    Paramaters
    ----------
    fp : str
        Path to Promethion binary file to be opened.
    cages : list
        List of strings giving the identifiers for the cages whose data should 
        be read from the Promethion output.
    fields : list
        List of strings of fields within the Promethion output file whose data
        should be read.
    start_timestamp : promethion_timestamp, optional
        If a value is passed for this paramter, the function will use it to
        calculate the time elapsed since each observation in the Promethion file
        being read. Useful if appending additional data to an existing array.

    Returns
    -------
    data : np.array
        Two dimensional array containing as many rows as there are observations
        in the Promethion file and as many columns as were requested by cages
        and fields (maximum of len(cages)*len(fields)).
    timestamps : np.array
        One dimensional array of timestamps.
    '''
    o = open(fp)
    header = o.readline().strip().split(',')

    indices = []
    keys = []
    for c in cages:
        for f in fields:
            k = f + '_%s' % c
            if k in header:
                indices.append(header.index(k))
                keys.append(k)

    timestamps = []
    data = []
    for line in o.readlines():
        values = line.strip().split(',')
        timestamps.append(convert_promethion_date(values[0]))
        data.append([values[i] for i in indices])
    
    o.close()

    if start_timestamp is None:
        start_timestamp = timestamps[0]
    else:
        start_timestamp = convert_promethion_date(start_timestamp)
    times = [time_since_start(i, start_timestamp) for i in timestamps]

    return np.array(data).astype(np.float32), times, keys

def append_to_npy(arr, fp, append=True):
    '''Load array at `fp` and make a new array concatenating it with `arr`.'''
    old_arr = np.load(fp)
    if append:
        start_idx = 0
    else:
        start_idx = arr.shape[0] - old_arr.shape[0]
    return np.hstack((old_arr, arr[start_idx:]))

def _make_fp(base_fp, field, cage):
    '''Return base_fp/cage_field.npy'''
    return os.path.join(base_fp, '%s_%s.npy' % (field, cage))

