#!/usr/bin/env python
from __future__ import division

import numpy as np
from bcp.ethoscan import parse_ethoscan_report
from bcp.parse import convert_promethion_date, time_since_start
from bcp.util import add_seconds
import datetime
import subprocess

'''
Ethoscan dates are separated by a tab, regular promethion output file dates are
separated by a space.
'''

ethoscan_fp = '/Users/wdwvt/Desktop/Sonnenburg/cumnock/bcp/data/exp1/ethoscans/animal_1_07082015_to_07162015.csv'
raw_data_fp = '/Users/wdwvt/Desktop/Sonnenburg/cumnock/bcp/data/exp1_raw_Promethion_061115/070815_Infection_E'
x_fp = '/Users/wdwvt/Desktop/Sonnenburg/cumnock/bcp/data/exp1/XPos_1.npy'
y_fp = '/Users/wdwvt/Desktop/Sonnenburg/cumnock/bcp/data/exp1/YPos_1.npy'
times_fp = '/Users/wdwvt/Desktop/Sonnenburg/cumnock/bcp/data/exp1/time.npy'
exp_start_timestamp='6/23/2015 11:46:44'
ethoscan_start_timestamp='7/8/2015 10:25:48'

o = open(ethoscan_fp)
lines = o.readlines()
o.close()

eth_data = parse_ethoscan_report(lines)

x1 = np.load(x_fp)
y1 = np.load(y_fp)
times = np.load(times_fp)


# Find where in x1 the first record in the ethoscan falls.
exp_start = convert_promethion_date(exp_start_timestamp)
eth_start = convert_promethion_date(ethoscan_start_timestamp)

# number of seconds between promethion raw output start and ethoscan start
idx = (eth_start - exp_start).total_seconds()
estart_idx = (times == idx).nonzero()[0][0]


# Verify that the data produced by our parsing of the Ethoscan matches the 
# data produced by our parsing of the raw input files in terms of X and Y
# position.

# each obs is a row of data = a behavior quantified by ethoscan
for obs in eth_data: 
    obs_start = obs[0]
    obs_duration = obs[2]
    
    # Data parsed from the raw promethion files by our scripts
    x1_data = x1[int(estart_idx + obs_start):
                 int(estart_idx + obs_start + obs_duration)]
    y1_data = y1[int(estart_idx + obs_start):
                 int(estart_idx + obs_start + obs_duration)]

    # Create the datetime at which the observation would be found in the raw
    # file to reparse the data from the Promethion output. Basically a check of
    # our parsing. 
    raw_data = []
    dt_start = add_seconds(eth_start, int(obs_start))
    dt_start_str = '%s/%s/%s %s' % (dt_start.month, dt_start.day,
                                    dt_start.year, dt_start.strftime('%T'))
    dt_end = add_seconds(eth_start, int(obs_start+obs_duration))
    dt_end_str = '%s/%s/%s %s' % (dt_end.month, dt_end.day, dt_end.year,
                                  dt_end.strftime('%T'))

    start_line = subprocess.check_output("grep -m 1 -n '%s' %s" % (dt_start_str,
                                                                   raw_data_fp),
                                         shell=True)
    end_line =  subprocess.check_output("grep -m 1 -n '%s' %s" % (dt_end_str,
                                                                  raw_data_fp),
                                         shell=True)

    start_line = int(start_line.strip().split(':')[0])
    end_line = int(end_line.strip().split(':')[0])
    
    o = open(raw_data_fp)
    for i in range(1, end_line): #the first line of the file is at index 1 for grep
        line = o.readline()
        if i >= start_line:
            raw_vals = line.split(',')
            raw_data.append(map(float, [raw_vals[i] for i in [1, 2, 3, 7, 8, 9,
                                                              10]]))
    o.close()

    rd = np.array(raw_data)

    assert (rd[:, 0] == x1_data).all()
    assert (rd[:, 1] == y1_data).all()
