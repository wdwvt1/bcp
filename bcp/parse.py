#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import datetime, time

'''
Library functions for dealing with the Promethion data.
'''

def convert_promethion_date(dt_str):
    '''Convert date of the form M/DD/YYYY HH:MM:SS to datetime object.'''
    d, t = dt_str.split()
    month, day, year = d.split('/')
    return datetime.datetime(*map(int, [year, month, day]+ t.split(':')))

def time_since_start(dt_cur, dt_start):
    '''Count total seconds elapsed between dt_start and dt_cur'''
    return (dt_cur - dt_start).total_seconds()

def parse_promethion_output(fp, exp_start=None, exclude_cols_past=None,
                            include_cols_with=None):
    '''Parse Promethion output to array, modify timestamp. 

    Parameters
    ----------
    fp : str
        Promethion output filepath.
    exp_start : datetime.datetime
        If this is passed, it will specify the start tiem of the experiment, and
        thus alter the time passed calculated for each row. Valuable if
        concatenating two experiments in different output files.
    exclude_cols_past : int or False
        If not False, exclude columns in the file past value. Valuable for large
        Promethion files where most columns are not valuable.
    include_cols_with : list or None
        If not None, a list of strings. The strings of the parsed header line
        are compared to these strings, and each header field which does not
        contain at least one of those strings is removed from the final output.

    Returns
    -------
    headers : list
        List of 'features' of the sample X feature matrix of data.
    data : np.array
        Data for each timepoint.
    sds : list
        List of unmodified timestamps from the Promethion file.

    Notes
    -----
    The format for promethion files is:
    Date    Time,XPos_1,...
    7/16/2015 15:58:54,8.75,...

    Notice, there are is a tab between Date and Time, but that is the only tab
    in the file, the rest of the data is comma separated. 

    This function will replace the dates with time since the beginning of the
    experiment.
    '''
    f = open(fp, 'U')
    hl = f.readline().strip().split(',')

    ctk = np.arange(1, len(hl))
    if exclude_cols_past != None:
        ctk = ctk[:exclude_cols_past]

    if include_cols_with != None:
        _ctk = []
        for c in ctk:
            if any([ss in hl[c] for ss in include_cols_with]):
                _ctk.append(c)
        ctk = _ctk

    headers = ['Time'] + [hl[i] for i in ctk]

    dts = []
    data = []
    for line in f:
        vals = line.strip().split(',')
        dts.append(vals[0])
        data.append([vals[i] for i in ctk])

    f.close()

    if exp_start is None:
        exp_start = convert_promethion_date(dts[0])

    times = [time_since_start(convert_promethion_date(i), exp_start)
             for i in dts]

    data = np.hstack((np.array(times).reshape(len(times), 1),
                      np.array(data, dtype=np.float32)))
    return headers, data, dts