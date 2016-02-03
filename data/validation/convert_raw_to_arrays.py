#!/usr/bin/env python
import numpy as np
import datetime, time, os
from bcp.parse import _make_fp, promethion_to_array, append_to_npy

base_fp = '/Users/wdwvt/Desktop/Sonnenburg/cumnock/bcp/data/validation/raw_data/'
fps = [base_fp + 'no_animals_011816_am.txt',
       base_fp + 'no_animals_011816_pm.txt',
       base_fp + 'no_animals_011916.txt']

cages =  list(map(str, [1, 2, 3, 4, 5, 6, 7, 8]))
fields = ['XPos', 'YPos', 'ZPos', 'WheelCount', 'FoodA', 'Water', 'BodyMass']

output_base_fp = '/Users/wdwvt/Desktop/Sonnenburg/cumnock/bcp/data/validation/'

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
                                            start_timestamp='1/18/2016 13:29:09')
    new_time_arr = append_to_npy(times, time_fp, append=True)
    np.save(time_fp, new_time_arr)
    for col, key in zip(data.T, keys):
        field, cage = key.split('_')
        _out_fp = _make_fp(output_base_fp, field, cage)
        new_arr = append_to_npy(col, _out_fp, append=True)
        np.save(_out_fp, new_arr)
    t1 = time.time()
    print('took: %s' % (t1 - t0))