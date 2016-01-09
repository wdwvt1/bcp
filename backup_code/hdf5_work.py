#!/usr/bin/env python
import numpy as np
import h5py
import datetime, time, os

'''
Library functions for dealing with the Promethion data.
'''
'''
# Example Promethion Data Output
Date    Time,XPos_1,YPos_1,ZPos_1,XBreak_1,YBreak_1,ZBreak_1,WheelCount_1,FoodA_1,Water_1,BodyMass_1,XPos_2,YPos_2,ZPos_2,XBreak_2,YBreak_2,ZBreak_2,WheelCount_2,FoodA_2,Water_2,BodyMass_2,XPos_3,YPos_3,ZPos_3,XBreak_3,YBreak_3,ZBreak_3,WheelCount_3,FoodA_3,Water_3,BodyMass_3,XPos_4,YPos_4,ZPos_4,XBreak_4,YBreak_4,ZBreak_4,WheelCount_4,FoodA_4,Water_4,BodyMass_4,XPos_5,YPos_5,ZPos_5,XBreak_5,YBreak_5,ZBreak_5,WheelCount_5,FoodA_5,Water_5,BodyMass_5,XPos_6,YPos_6,ZPos_6,XBreak_6,YBreak_6,ZBreak_6,WheelCount_6,FoodA_6,Water_6,XPos_7,YPos_7,ZPos_7,XBreak_7,YBreak_7,ZBreak_7,WheelCount_7,FoodA_7,Water_7,BodyMass_7,XPos_8,YPos_8,ZPos_8,XBreak_8,YBreak_8,ZBreak_8,WheelCount_8,FoodA_8,Water_8,BodyMass_8,RT_FoodA_1,RT_Water_1,RT_WheelCount_1,RT_FoodA_2,RT_Water_2,RT_WheelCount_2,RT_FoodA_3,RT_Water_3,RT_WheelCount_3,RT_FoodA_4,RT_Water_4,RT_WheelCount_4,RT_FoodA_5,RT_Water_5,RT_WheelCount_5,RT_FoodA_6,RT_Water_6,RT_WheelCount_6,RT_FoodA_7,RT_Water_7,RT_WheelCount_7,RT_FoodA_8,RT_Water_8,RT_WheelCount_8,CommMS,ElSeconds,Markers
6/11/2015 18:41:56,5.75,21.75,18.25,19,19,11,6,370.3022,348.3497,-0.3671426,4.5,7,3,17.5,21,8.5,4,381.5571,333.5468,10.75229,5.25,2,0,0,0,0,5,378.6284,320.0024,-0.4578373,4.75,5.5,0,0,0,0,7,376.9134,336.9136,-8.659991E-03,2,4.5,0,0,0,0,4,380.4408,333.2097,6.576459E-02,2.25,2.5,0,0,0,0,0,384.5407,333.9856,3.75,6.75,0,0,0,0,6,379.3676,323.5154,0.5884944,6.25,9.75,0,0,0,0,5,388.9341,344.4416,-0.1612068,0,0,6,0,0,4,0,0,5,0,0,7,0,0,4,0,0,0,0,0,6,0,0,5,255,0.002,-1
'''

def convert_promethion_date(dt_str):
    '''Convert date of the form M/DD/YYYY HH:MM:SS to datetime object.'''
    d, t = dt_str.split()
    month, day, year = d.split('/')
    return datetime.datetime(*map(int, [year, month, day]+ t.split(':')))

def time_since_start(dt_cur, dt_start):
    '''Count total seconds elapsed between dt_start and dt_cur'''
    return (dt_cur - dt_start).total_seconds()

def make_int8_grid_data(data, grid_spacing):
    '''Convert promethion grid to ints for reduced storage.'''
    #print(data.max())
    print((data.max() * 1./grid_spacing) <= 127)
    assert grid_spacing <= 1.
    return (1./grid_spacing * data).astype(np.int8)

def ceil_127(data, grid_spacing):
    return np.where(data > 127, 127, data)

def create_hdf5_table(fp, cages, fields, dtypes, mode='a'):
    '''Create and hdf5 table with data and fields and dtypes.'''
    with h5py.File(fp, mode) as f:
        f.create_dataset('time', data=[0], dtype=np.int32, maxshape=(None,),
                         compression='gzip')
        f.create_dataset('rows', data=np.array([0]), dtype=np.int32)
        for cage in cages:
            f.create_group(cage)
            for field, _dtype in zip(fields, dtypes):
                f[cage].create_dataset(field, data = [0], dtype=_dtype,
                                       maxshape=(None,), compression='gzip')

def promethion_to_array(fp, cages, fields, start_timestamp=None):
    o = open(fp)
    header = o.readline().strip().split(',')

    indices = []
    hdf5_key_pairs = []
    for c in cages:
        for f in fields:
            try:
                k = f + '_%s' % c
                indices.append(header.index(k))
                hdf5_key_pairs.append((f, c))
            except ValueError:
                pass
                # Header did not contain this entry because a sensor
                # malfunctioned or was not setup.
    timestamps = []
    data = []
    for line in o.readlines():
        values = line.strip().split(',')
        timestamps.append(convert_promethion_date(values[0]))
        data.append([values[i] for i in indices])
    
    o.close()

    if start_timestamp is None:
        start_timestamp = timestamps[0]
    times = [time_since_start(i, start_timestamp) for i in timestamps]

    return np.array(data).astype(np.float32), times, hdf5_key_pairs

def array_to_hdf5(table, overlapping, data, times, cfuncs, hdf5_key_pairs):

    # The shape of all arrays, i.e. the number of records for each value, must
    # all be the same. If they are not, something has gone wrong during parsing.
    # We would catch that error elsewhere, so we are content to query the size
    # of the additions for only one column here. 
    cur_rows = table['rows'].value[0]

    if overlapping:
        new_rows = data.shape[0] - cur_rows
        expand_to = data.shape[0]
        data_start_idx = cur_rows
    else:
        new_rows = data.shape[0]
        expand_to = data.shape[0] + cur_rows
        data_start_idx = 0

    for i in range(len(hdf5_key_pairs)):
        print(i, hdf5_key_pairs[i])
        j = i%len(cfuncs)
        field, cage = hdf5_key_pairs[i]
        table[cage][field].resize(expand_to, axis=0)

        if cfuncs[j] is not None:
            table[cage][field][cur_rows:] = cfuncs[j](data[data_start_idx:, i],
                                                      grid_spacing=.25)
        else:
            table[cage][field][cur_rows:] = data[data_start_idx:, i]

    # Resest the number of rows in the data given that we have added data.
    table['rows'][0] = expand_to
    table['time'].resize(expand_to, axis=0)
    table['time'][cur_rows:] = times[data_start_idx:]

    table.flush()

base_fp = '/Users/wdwvt/Desktop/Sonnenburg/cumnock/Promethion_061115/'
fps = [base_fp + '062315 Infection Export',
       base_fp + '062415 Infection Export',
       base_fp + '062515 Infection Export',
       base_fp + '062615 Infection E',
       base_fp + '062715 Infection E',
       base_fp + '062815 Infection E',
       base_fp + '062915 Infection E',
       base_fp + '063015 Infection E',
       base_fp + '070115 Infection E',
       base_fp + '070215 Infection E',
       base_fp + '070315 Infection E',
       base_fp + '070415 Infection E',
       base_fp + '070515 Infection E',
       base_fp + '070615 Infection E',
       base_fp + '070715 Infection E',
       base_fp + '070815_Infection_E',
       base_fp + '071615 Infection E']

cages = list(map(str, [1, 2, 3, 4, 5, 6, 7, 8]))
fields = ['XPos', 'YPos', 'ZPos', 'WheelCount', 'FoodA', 'Water', 'BodyMass']
dtypes = [np.int8, np.int8, np.int8, np.int8, np.float32, np.float32,
          np.float32]
cfuncs = [make_int8_grid_data, make_int8_grid_data, make_int8_grid_data, 
          ceil_127, None, None, None]

fp = '/Users/wdwvt/Desktop/test_with_code.hdf5'
overlapping = False
table = create_hdf5_table(fp, cages, fields, dtypes, mode='a')

with h5py.File(fp, 'a') as table:

    for pf in fps:
        t0 = time.time()
        fp = os.path.join(base_fp, pf)
        data, times, hdf5_key_pairs = promethion_to_array(fp, cages, fields, 
                                                          start_timestamp=None)
        array_to_hdf5(table, overlapping, data, times, cfuncs, hdf5_key_pairs)
        t1 = time.time()
        print('took: %s' % (t1 - t0))

# f[m].create_dataset('X', data=[0], maxshape=(None,), dtype=np.int8)
# f[m].create_dataset('Y', data=[0], maxshape=(None,), dtype=np.int8)
# f[m].create_dataset('Z', data=[0], maxshape=(None,), dtype=np.int8)
# f[m].create_dataset('WC', data=[0], maxshape=(None,), dtype=np.int8)


# f[m].create_dataset('BM', data=[0], maxshape=(None,), dtype=np.float32)
# f[m].create_dataset('F', data=[0], maxshape=(None,), dtype=np.float32)
# f[m].create_dataset('W', data=[0], maxshape=(None,), dtype=np.float32)








# # Store the time at which the experiment started.
# f['exp_start'] = np.array([2015, 6, 23, 11, 46, 44])

# # Store the mice which are infected.
# f['inf_mice'] = np.array([2,3,5,6,8])

# # Store the mice which are control.
# f['con_mice'] = np.array([1,4,7])


















# def convert_promethion_date(dt_str):
#     '''Convert date of the form M/DD/YYYY HH:MM:SS to datetime object.'''
#     d, t = dt_str.split()
#     month, day, year = d.split('/')
#     return datetime.datetime(*map(int, [year, month, day]+ t.split(':')))

# def time_since_start(dt_cur, dt_start):
#     '''Count total seconds elapsed between dt_start and dt_cur'''
#     return (dt_cur - dt_start).total_seconds()

# def parse_promethion_output(fp, exp_start=None, exclude_cols_past=None,
#                             include_cols_with=None):
#     '''Parse Promethion output to array, modify timestamp. 

#     Parameters
#     ----------
#     fp : str
#         Promethion output filepath.
#     exp_start : datetime.datetime
#         If this is passed, it will specify the start time of the experiment, and
#         thus alter the time passed calculated for each row. Valuable if
#         concatenating two experiments in different output files.
#     exclude_cols_past : int or False
#         If not False, exclude columns in the file past value. Valuable for large
#         Promethion files where most columns are not valuable.
#     include_cols_with : list or None
#         If not None, a list of strings. The strings of the parsed header line
#         are compared to these strings, and each header field which does not
#         contain at least one of those strings is removed from the final output.

#     Returns
#     -------
#     headers : list
#         List of 'features' of the sample X feature matrix of data.
#     data : np.array
#         Data for each timepoint.
#     sds : list
#         List of unmodified timestamps from the Promethion file.

#     Notes
#     -----
#     The format for promethion files is:
#     Date    Time,XPos_1,...
#     7/16/2015 15:58:54,8.75,...

#     Notice, there are is a tab between Date and Time, but that is the only tab
#     in the file, the rest of the data is comma separated. 

#     This function will replace the dates with time since the beginning of the
#     experiment.
#     '''
#     f = open(fp, 'U')
#     hl = f.readline().strip().split(',')

#     ctk = np.arange(1, len(hl))
#     if exclude_cols_past != None:
#         ctk = ctk[:exclude_cols_past]

#     if include_cols_with != None:
#         _ctk = []
#         for c in ctk:
#             if any([ss in hl[c] for ss in include_cols_with]):
#                 _ctk.append(c)
#         ctk = _ctk

#     headers = ['Time'] + [hl[i] for i in ctk]

#     dts = []
#     data = []
#     for line in f:
#         vals = line.strip().split(',')
#         dts.append(vals[0])
#         data.append([vals[i] for i in ctk])

#     f.close()

#     if exp_start is None:
#         exp_start = convert_promethion_date(dts[0])

#     times = [time_since_start(convert_promethion_date(i), exp_start)
#              for i in dts]

#     data = np.hstack((np.array(times).reshape(len(times), 1),
#                       np.array(data, dtype=np.float32)))
#     return headers, data, dts