#!/usr/bin/env python
import numpy as np
import datetime, time, os
from bcp.parse import _make_fp, promethion_to_array, append_to_npy

base_fp = '/Users/wdwvt/Desktop/Sonnenburg/cumnock/bcp/data/exp1_raw_Promethion_061115/'
fps = [base_fp + '062315_Infection_E',
       base_fp + '062415_Infection_E',
       base_fp + '062515_Infection_E',
       base_fp + '062615_Infection_E',
       base_fp + '062715_Infection_E',
       base_fp + '062815_Infection_E',
       base_fp + '062915_Infection_E',
       base_fp + '063015_Infection_E',
       base_fp + '070115_Infection_E',
       base_fp + '070215_Infection_E',
       base_fp + '070315_Infection_E',
       base_fp + '070415_Infection_E',
       base_fp + '070515_Infection_E',
       base_fp + '070615_Infection_E',
       base_fp + '070715_Infection_E',
       base_fp + '070815_Infection_E',
       base_fp + '071615_Infection_E']

cages =  list(map(str, [1, 2, 3, 4, 5, 6, 7, 8]))
fields = ['XPos', 'YPos', 'ZPos', 'WheelCount', 'FoodA', 'Water', 'BodyMass']

output_base_fp = '/Users/wdwvt/Desktop/Sonnenburg/cumnock/bcp/data/exp1/'

# Create empty arrays to intialize files.
for cage in cages:
    for field in fields:
        fp = _make_fp(output_base_fp, field, cage)
        np.save(fp, np.array([]))
time_fp = os.path.join(output_base_fp, 'time.npy')
np.save(os.path.join(output_base_fp, 'time.npy'), np.array([]))

# Read new data and append it to the arrays.
for fp in fps:
    t0 = time.time()
    f = os.path.join(base_fp, fp)
    data, times, keys = promethion_to_array(fp, cages, fields, 
                                            start_timestamp='6/23/2015 11:46:44')
    new_time_arr = append_to_npy(times, time_fp, append=True)
    np.save(time_fp, new_time_arr)
    for col, key in zip(data.T, keys):
        field, cage = key.split('_')
        _out_fp = _make_fp(output_base_fp, field, cage)
        new_arr = append_to_npy(col, _out_fp, append=True)
        np.save(_out_fp, new_arr)
    t1 = time.time()
    print('took: %s' % (t1 - t0))