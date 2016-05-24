#!/usr/bin/env python
from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt
from bcp.parse import convert_promethion_date
from bcp.ethoscan import (parse_ethoscan_report, align_ethoscan_data)
from bcp.plot import circadian_rythm_axvspan

# Data from './Water_1.npy' with drinking events classified by eye.
# The ith entry in w1_drinks_human is the approximate x-axis location
# (seconds since 6/23/2015 11:46:44) of the drink.
w1_drinks_human = np.array([3520, 3613, 24160, 28128, 28393, 29617, 33916,
                            35554, 37072, 37758, 40601, 41508, 42957, 43641,
                            44323, 44858, 46670, 47238, 47680, 47952, 50249,
                            51750, 52170, 53691, 54427, 56702, 58114, 59216,
                            60940, 62283, 63380, 64673, 65788, 66354, 66751,
                            66782, 68374, 69964, 70088, 77538, 108832, 109048,
                            111568, 112151, 114659, 115569, 115591, 116431,
                            116822, 117214, 117317, 117354, 117934, 118200,
                            118231, 118782, 119089, 120675, 121513, 122864,
                            123675, 125436, 126409, 126983, 127089, 127850,
                            128502, 128661, 129694, 130902, 131425, 133887,
                            134227, 135582, 136171, 136204, 136976, 137882,
                            138500, 138683, 141211, 141263, 142697, 142900,
                            143558, 144373, 145476, 145520, 146878, 146894,
                            146928, 147691, 149848, 150044, 150306, 150356,
                            151507, 152895, 153676, 156622, 163415, 195859,
                            195899, 196391, 198619, 198322, 200826, 201139,
                            202525, 202661, 202670, 202677, 203361, 203465,
                            204214, 204581, 205267, 206170, 206777, 207319,
                            208163, 209031, 210059, 210859, 210926, 212015,
                            212944, 214004, 214038, 215023, 216328, 216525,
                            217635, 218699, 219273, 219314, 221219, 221228,
                            221314, 221646, 222235, 223248, 224201, 225462,
                            225474, 225790, 226461, 227828, 228711, 229698,
                            229707, 230116, 231149, 231258, 232441, 232995,
                            233138, 234099, 236156, 236540, 236597, 237451,
                            238396, 238776, 240103, 240756, 241308, 256293,
                            256953, 284098, 285662])

# Plot data showing that we are in fact seeing drinking events. Because the
# data are discontinuous due to the experiment stops, we have to cast the x-axis
# locations back to the event number. In addition, plot drinking events called
# by the Ethoscan. 
w1 = np.load('/Users/wdwvt/Desktop/Sonnenburg/cumnock/bcp/data/exp1/Water_1.npy')
times = np.load('/Users/wdwvt/Desktop/Sonnenburg/cumnock/bcp/data/exp1/time.npy')

plt.plot(times, w1, color='k', lw=1, alpha=.75)

observation_indices = np.searchsorted(times, w1_drinks_human)
for i in observation_indices:
    plt.plot(times[i-15: i+15], w1[i-15: i+15], color='red', lw=3, alpha=.4)

# Plot Ethoscan classifications of behaviors.
exp_start = convert_promethion_date('6/23/2015 11:46:44')

ebase = '/Users/wdwvt/Desktop/Sonnenburg/cumnock/bcp/data/exp1/ethoscans/'
scan_fps = ['animal_1_06232015_to_06242015.csv',
            'animal_1_06242015_to_06252015.csv',
            'animal_1_06252015_to_06262015.csv']
scan_starts = ['06/23/2015 11:46:44', '06/24/2015 09:11:21',
               '06/25/2015 08:56:35']

for fp, st in zip(scan_fps, scan_starts):
    o = open(os.path.join(ebase, fp))
    lines = o.readlines()
    o.close()

    edata = parse_ethoscan_report(lines)
    dwatr = (edata[:, 1] == 2).nonzero()[0]
    twatr = (edata[:, 1] == 3).nonzero()[0]

    eth_start = convert_promethion_date(st)
    for i in edata[twatr]:
        start, end = align_ethoscan_data(exp_start, eth_start, i, times)
        plt.plot(times[start:end], w1[start:end], color='orange', lw=3,
                 alpha=.5)

    for i in edata[dwatr]:
        start, end = align_ethoscan_data(exp_start, eth_start, i, times)
        plt.plot(times[start:end], w1[start:end], color='green', lw=3,
                 alpha=.5)

# Finish the plot with labels etc.
plt.title('Ethoscan vs. Human classification')
plt.xlabel('Seconds elapsed')
plt.ylabel('Water reminaing (mL)')
nights = circadian_rythm_axvspan(7, 19, exp_start, times[:250000])
xticks = []
xticklabels = []
for i, n in enumerate(nights):
    plt.axvspan(n[0], n[1], color='grey', alpha=.1)
    xticks.append(n[0])
    xticklabels.append('D%s 19:00' % i)
plt.xticks(xticks, xticklabels, rotation=90, size=10)
plt.xlim(0, 250000)
plt.ylim(316.0, 330.5)
plt.grid()

colors = ['black', 'green', 'orange', 'red']
plt.legend([plt.Line2D((0,1), (0,0), lw=3, color=i) for i in colors],
           ['Raw Trace', 'Ethoscan: Drink', 'Ethoscan: Touch', 'Human: Drink'])



