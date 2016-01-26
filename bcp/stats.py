#!/usr/bin/env python
import numpy as np

'''
Code for calculating statistics on Promethion data.
'''

def centered_moving_average(data, r):
    '''Calculate moving average of data with ma_data[i]=data[i-r:i+r]/(2r+1).

    Parameters
    ----------
    data : np.array
        One dimensional array of data whose moving average is to be calculated.
    r : int
        Radius of moving average.

    Returns
    -------
    ma_data : np.array
        One dimensional array of same shape as `data`, where
        ma_data[i]=data[i-r:i+r]/(2r+1). At the beginning and end of `ma_data`,
        where the number of surrounding points is insufficient, 0's are
        reported.

    Notes
    -----
    The radius of the window (`r`) is not inclusive of the point at which the
    window is centered. For example, if r=3:
    data    = [1, 4, 5,   19,    2, 2, 4, 5]
    ma_data = [0, 0, 0, 37/7, 41/7, 0, 0, 0]
    '''
    l = len(data)
    ma_data = np.zeros(l)
    ma_data[r] = data[:2*r + 1].sum() 
    for i in range(r+1, l - r):
        ma_data[i] = ma_data[i-1] + data[i+r] - data[i-(r+1)]
    return ma_data/float(2*r + 1)

def xy_distance_over_window(xs, ys, window):
    '''Calculate the distance traveled over the given window.

    Note: there will be len(xs) - window distances returned from this
    function. This is because N x coordinates have N-1 differences between them. 
    The practical consequence is that wheel_distance_over_window run on the same
    cage data will result in a vector longer by 1.

    Parameters
    ----------
    xs : np.array
        X coordinates of the mouse.
    ys : np.array
        Y coordinates of the mouse.
    window : int
        Size of window (length of slice) to sum movement distance over.

    Returns
    -------
    rx, ry: np.array
        Distance traveled by mouse in x, y direction over window. rx[i] is the 
        amount of distance traveled between time (i, i + 1 + window).'''
    # if this fails we stop since something has gone wrong with parsing
    assert len(xs) == len(ys)

    rx = np.empty(shape=len(xs) - window, dtype=np.float32)
    ry = np.empty(shape=len(ys) - window, dtype=np.float32)

    xs_first_diff_cs = abs(xs[1:] - xs[:-1]).cumsum()
    ys_first_diff_cs = abs(ys[1:] - ys[:-1]).cumsum()

    rx[0] = xs_first_diff_cs[window - 1]
    ry[0] = ys_first_diff_cs[window - 1]

    for i in range(1, len(xs) - window):
        rx[i] = xs_first_diff_cs[window - 1 + i] - xs_first_diff_cs[i-1]
        ry[i] = ys_first_diff_cs[window - 1 + i] - ys_first_diff_cs[i-1]

    return rx, ry

def wheel_distance_over_window(rs, window, radius=11.43/2.):
    '''Calculate distance mouse moves on wheel during window.

    Parameters
    ----------
    rs : np.array
        Revolutions per second that the mouse achieves.
    window : int
        Size of window (length of slice) to sum running distance over.
    radius : float
        Radius of the wheel in cm's. Default to measured distance. 

    Returns
    -------
    r : np.array
        Distance traveled by mouse over each window period. r[i] is the amount
        of distance the mouse travels in the time frame [i, window+i-1].
    '''
    r = np.empty(shape=len(rs) + 1 - window, dtype=np.float32)
    rc = rs.cumsum()
    r[0] = rc[window - 1]
    for i in range(1, len(rc) + 1 - window):
        r[i] = rc[window - 1 + i] - rc[i -1]
    return r * radius * 2 * np.pi

def rearing_over_window(zs, window):
    '''Calculate total number of rearing events during window.

    Parameters
    ----------
    zs : np.array
        z-vector from Promethion data for a given animal.
    window : int
        Size of window (length of slice) to sum rearing over.

    Returns
    -------
    r : np.array
        Number of rearing events in the given window over the course of the data
        vector. r[i] is the number of rearing events that have occurred within 
        the time frame (i, window+i-1).
    '''
    r = np.empty(shape=len(zs) + 1 - window, dtype=np.int)
    zc = (zs > 0).cumsum()
    r[0] = zc[window - 1]
    for i in range(1, len(zc) + 1 - window):
        r[i] = zc[window - 1 + i] - zc[i -1]
    return r