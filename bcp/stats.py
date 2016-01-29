#!/usr/bin/env python
from __future__ import division

import numpy as np

'''
Code for calculating statistics on Promethion data.

Examples
--------
To calculate the number of rearing events over a moving window.
>>> zs_binary = (zs > 0).astype(float)
>>> moving_function(zs_binary, window, 'sum', 1)
'''

def moving_function(data, window, function='sum', boundary=1):
    '''Calculate moving average or sum of data.

    Parameters
    ----------
    data : np.array
        One dimensional array of data whose moving average is to be calculated.
    window : int
        Diameter (window size) of moving average (inclusive of center point).
    function : {'sum' or 'average'}, optional
        The function to compute over the data.
    boundary : {1 or 2}, optional
        If 1, do not alter the convolution calculation at the edges of `data`
        where the function cannot be calculated because there are not enough
        points on both sides of the center. If 2, replace the edges of the
        computed data with values from the raw `data`.

    Returns
    -------
    ma_data : np.array

    Notes
    -----
    If `window` is an even number, the boundaries of the computed data will
    be slightly different. Specficially, an even `window` requires that an
    uneven number of non-overlapping dot-products be taken by the convolution
    resulting in a left edge which has one more point (by numpy convention) that
    is the product of a non-overlapping dot-product.

    This function is easily expressed as both a convolution and a mean
    calculation and then a series of additions and subtractions. For large
    input `data` (~1e6 points), the convolution code is ~2X slower.
    '''
    if function == 'sum':
        ma_data = np.convolve(np.ones(window, dtype=float), data, 'same')
    elif function == 'average':
        ma_data = np.convolve(np.ones(window, dtype=float)/window, data, 'same')
    if boundary == 1:
        return ma_data
    elif boundary == 2:
        full_overlaps = (len(data) - window) + 1
        partial_overlaps = len(data) - full_overlaps
        # If `window` is even we need to have a different amount of boundary 
        # points rescaled as described in notes.
        _left = np.ceil(partial_overlaps/2.)
        _right = np.floor(partial_overlaps/2.)
        ma_data[:_left] = data[:_left]
        ma_data[-_right:] = data[-_right:]
        return ma_data
    # Code which is faster given large enough input. Note that `r` will have to
    # be adjusted given the way the convolution code is written; i.e. this code
    # does not return equivalent results with the function. 
    # l = len(data)
    # ma_data = np.zeros(l)
    # ma_data[r] = data[:2*r + 1].sum() 
    # for i in range(r+1, l - r):
    #     ma_data[i] = ma_data[i-1] + data[i+r] - data[i-(r+1)]
    # return ma_data/float(2*r + 1)

def distance_traveled_1d(data):
    '''Calculate (Manhattan) distance traveled from coordinate `data`.

    Parameters
    ----------
    data : np.array
        X or Y coordinates of mouse.

    Returns
    -------
    np.array
    '''
    return abs(data[1:] - data[:-1]).sum()

def distance_traveled_2d(x_data, y_data):
    '''Calculate (Euclidean) distance traveled from coordinate data.

    Parameters
    ---------- 
    x_data : np.array
        X coordinates of mouse.
    y_data : np.array
        Y coordinates of mouse.
    
    Returns
    -------
    np.array
    '''
    return (((x_data[1:] - x_data[:-1])**2 +
             (y_data[1:] - y_data[:-1])**2)**.5).sum()
