#!/usr/bin/env python
import numpy as np

'''
Code for calculating statistics on Promethion data.
'''

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