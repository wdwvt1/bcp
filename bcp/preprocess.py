#!/usr/bin/env python
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import time, datetime, imp
from hmmlearn.hmm import MultinomialHMM

def smooth(data, radius, a_thresh, w_thresh):
    '''Smooth data.

    Given an index i in data, taken all indices within radius of i, and compute
    the difference between data[i] and data[i-radius:i+radius]. Calculate the
    number of these points that are more than a_thresh (amplitude) different
    than data[i]. If that number is larger than w_thresh, record
    smoother_data[i] as np.nan. Else, record smoother_data[i] as the median of
    the points within radius of the index i.
    '''
    n = len(data)
    start = radius
    stop = n - radius

    smoothed_data = np.zeros(n)
    for i in range(start, stop):
        p =  (a_thresh < abs(data[i-radius:i+radius] - data[i])).sum()
        if p > w_thresh:
            smoothed_data[i] = np.nan
        else:
            smoothed_data[i] = np.median(data[i-radius:i+radius])
    return smoothed_data

def interpolate_between_nans(smoothed_data):
    '''Take smoothed data with contiguous nan blocks, remove and interpolate.'''
    n = len(smoothed_data)
    interpolated_data = np.empty(n)

    left_index = 0
    right_index = 0
    while (left_index < n) and (right_index < n):
        value = smoothed_data[left_index]
        if np.isnan(value):
            right_index = left_index
            while np.isnan(smoothed_data[right_index]):
                right_index += 1
            # Find points to interpolate between.
            start = smoothed_data[left_index - 1]
            stop = smoothed_data[right_index]
            num_steps = right_index - left_index + 2
            interpolated_data[left_index-1:right_index+1] = \
                np.linspace(start, stop, num = num_steps)
            left_index = right_index + 1
        else:
            interpolated_data[left_index] = value
            left_index += 1
    return interpolated_data

def find_nan_cumsum(smoothed_data):
    '''Return a vector of the cumulative sum of nans in smoother_data.'''
    return np.isnan(smoothed_data).cumsum()

def discretize_observations(observations, n):
    '''Discretize data into n**2 bins, and flatten.

    Parameters
    ----------
    observations : np.array
        2d array with shape 2, n.
    n : int
        Number of bins in each dimension of observations.
    '''
    xmin, ymin = observations.min(1)
    xmax, ymax = observations.max(1)
    xbins = np.linspace(xmin-.01, xmax+.01, n+1)
    ybins = np.linspace(ymin-.01, ymax+.01, n+1)

    x = np.searchsorted(xbins, observations[0, :]) - 1
    y = np.searchsorted(ybins, observations[1, :]) - 1

    d_observations = x*(x+1) + y
    return d_observations.astype(int)

def remove_artifacts_body_mass():
    '''A function which processes body mass.

    The challenege here is that the signal is very noisy at many different
    frequencies. Second to second, when the mouse is in its house, there is
    significant fluctuation in the body mass. Over ~1h time scales there are
    large changes in mass - up to .8g. To get an accurate reading of body mass
    during these events is challenging. During times when the mouse is not
    inhabiting the house some small positive weight is registered. This is easy
    to programatically eliminate, but detecting when it has gone back to in the
    house (and giving an accurate weight reading) is hard. 
    '''
    pass

def remove_artifacts_water():
    '''Remove artifacts from water data.'''
    return data

def remove_artifacts_food():
    '''Remove artifacts from food hopper data.'''
    return data

def remove_artifacts_wheel_running(data, max_rps=10):
    '''Remove artifacts from wheel running data.

    Notes
    -----
    The maximum RPS that we are confident the Promethion cage can detect is 10.

    Parameters
    ----------
    data : np.array
        Wheel count data indicating number of revolutions per second.
    max_rps : int, optional
        Maximum RPS allowed in the data.

    Returns
    -------
    np.array
        Data greater than max_rps will be reduced to max_rps.
    '''
    return np.where(data > max_rps, max_rps, data)

def remove_artifacts_x_position(data):
    '''Remove artifacts from x position data.'''
    return data

def remove_artifacts_y_position(data):
    '''Remove artifacts from y position data.'''
    return data

def remove_artifacts_z_position(data):
    '''Remove artifacts from y position data.'''
    return data

