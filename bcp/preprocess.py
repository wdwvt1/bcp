#!/usr/bin/env python
from __future__ import division

import copy
import numpy as np

def weight_sensor_positive_spikes(data, times, threshold):
    '''Find positive spikes in the weight data due to measurement. 

    Parameters
    ----------
    data : np.array
        One dimensional array with data from a weight sensor (water, food,
        bodymass).
    times : np.array
        One dimensional array with the time in seconds since the start of the
        experiment for each element of `data`.
    threshold : numeric
        Amount that data[i+1] must be greater than data[i] to count as a spike.

    Returns
    -------
    indices : np.array
        Indices of spikes in `data`.

    Notes
    -----
    Frequently the weight sensors (water, food, bodymass) will show positive
    spikes; points adjacent in time (1s) that have vastly different values.
    When these spikes are positive, it is likely that the animal is interacting
    with the sensor. This function identifies these points, while taking into
    account the discontinuous nature of the data, i.e. there are likely to be
    spikes between when the machine is turned on and off due to e.g. water
    drops falling off the water.
    '''
    time_diffs = (times[1:] - times[:-1]) > 1
    data_diffs = (data[1:] - data[:-1]) >= threshold
    return (~time_diffs * data_diffs).nonzero()[0]


def smooth_positive_spikes(data, spikes, backward_window, forward_window):
    '''Replace positive spikes with averages of previous points in data.

    Parameters
    ----------
    data : np.array
        One dimensional array with data from a weight sensor (water, food,
        bodymass).
    spikes : np.array
        Indices of spikes in `data`.
    backward_window : int
        Number of points of data prior to i to use to compute mean.
    forward_window : int
        Number of points of data after i to set to mean.

    Returns
    -------
    s_data : np.array
        Copy of data with spikes smoothed.

    Notes
    -----
    This function attempts to remove the positive spikes seen in weight data.
    The algorithm is as follows:
    1. Make a copy of `data` called `s_data`.
    2. Step through indices of spikes. If a spike occurs at index i, the points
       in the interval [i-backward_window, i] of `s_data` will have their mean
       taken. The points in the interval [i, i+forward_window] will be set to
       the mean value computed via the backward window. 
    Because `spikes` is an ordered list of spikes and we make updates to the
    values of `s_data` in the order of `spikes`, whenever spikes occur at
    intervals shorter than `forward_window`, we will be using already smoothed
    data for the backward window mean calculation.
    '''
    s_data = copy.copy(data)
    if (spikes < backward_window).any():
        raise ValueError('Some spikes occur at indices too close to the left '
                         'edge of the data (i.e. smaller than '
                         '`backward_window`). Not all positive spikes can be '
                         'smoothed.')
    if ((data.shape[0] - spikes) < forward_window).any():
        raise ValueError('Some spikes occur at indices too close to the right '
                         'edge of the data. Not all positive spikes can be '
                         'smoothed.')
    for i in spikes:
        s_data[i:i+forward_window] = s_data[i-backward_window:i].mean()
    return s_data


def stable_sequences(data, diff, stability_duration=1):
    '''Find sequences where consecutive entries are within `diff` of each other.

    Parameters
    ----------
    data : np.array
        One dimensional array containing numeric data.
    diff : numeric
        Float or int indicating the maximum difference that can exist between
        consecutive entries of `data` to be considered bounded. `diff` is
        inclusive, i.e. entries must differ more than `diff` to be excluded from
        a sequence. Note that the absolute value of the difference between
        elements is compared to `diff`.
    stability_duration : int >= 1, optional, default=1
        Duration of bounded span that must occur to be recorded as part of the
        bouned/stable spans. Useful if, e.g., you want to require a signal to
        be stable for a certain amount of time before counting it truly stable.
        Note that a value below 1 for this parameter is meaningless because the
        shortest sequence contains at least one difference (i.e. 2 consecutive
        points with values within `diff` of one another).

    Returns
    -------
    np.array
        Array of size K x 2, where the 0th column contains the start indices of
        sequences, and the 1st column contains the duration of the sequence.

    Notes
    -----
    This function calculates stability as a function of the differences between
    consecutive points. For N input points, there are N-1 differences. This
    means that the output durations are one less than the index of the final
    point of a sequence. 

    Examples
    --------
    data = np.array([0, 0, 1, 2, 2.5, 1, 1, 1])
    diff = 0
    stability_duration = 1
    seqs = stable_sequences(data, diff, stability_duration)
    np.testing.assert_array_equal(seqs, np.array([[0, 1], [5, 2]]))
    # To get back the actual values of the data we must add 1 to the duration.
    np.testing.assert_array_equal(np.array([0, 0]), 
                                  data[seqs[0][0]: seqs[0].sum()+1])
    np.testing.assert_array_equal(np.array([1, 1, 1]), 
                                  data[seqs[1][0]: seqs[1].sum()+1])
    '''
    idx = 1
    stable_spans = []
    n = data.size
    while idx < n:
        if abs(data[idx - 1] - data[idx]) <= diff:
            stable = True
            stable_count = 1
            offset = 1
            while stable and (offset + idx) < n:
                if abs(data[offset + idx - 1] - data[offset + idx]) <= diff:
                    stable_count += 1
                    offset += 1
                else:
                    stable = False
            if stable_count >= stability_duration:
                stable_spans.append((idx - 1, stable_count))
            idx += offset
        idx += 1
    return np.array(stable_spans)


def valued_sequences(data, value, stability_duration=1):
    '''Find sequences where consecutive entries are equal to `value`.

    Parameters
    ----------
    data : np.array
        One dimensional array containing numeric data.
    value : numeric
        Float or int indicating the value that consecutive entries of `data`
        must take to be included in the returned sequence indices.
    stability_duration : int, optional, default=1
        Duration of sequence that must occur to be recorded as part of the
        returned sequences. Useful if, e.g., you want to require a signal to
        be stable for a certain amount of time before counting it truly stable.

    Returns
    -------
    np.array
        Array of size K x 2, where the 0th column contains the start indices of
        spans, and the 1st column contains the end index (exclusive) of the run.

    Notes
    -----
    Unlike `stable_sequences`, this function calculates 'stability' on a
    pointwise fashion (i.e. it doesn't need two entries to determine if a point
    should be included in the sequence). As such, the 1st column of the output
    are the (exclusive) endpoints of runs.
    '''
    idx = 0
    n = data.size
    stable_spans = []
    while idx < n:
        if data[idx] == value:
            offset = 1
            stable_count = 1
            stable = True
            while stable and (idx + offset) < n:
                if data[idx + offset] == value:
                    stable_count += 1
                    offset += 1
                else:
                    stable = False
            if stable_count >= stability_duration:
                stable_spans.append((idx, offset))
            idx += offset
        idx += 1
    return np.array(stable_spans)


def unstable_sequences(data, u_diff, s_diff=None, stability_duration=10):
    '''Find unstable sequences in `data`.

    Extended Summary
    ----------------
    This function finds sequences in `data` such that `stability_duration`
    number of differences between consecutive points at the end of each sequence
    are less than `s_diff`. This function is motivated by finding subsequences
    where a process has exited 'statistical control'. An unstable subsequence
    is started by a difference between consecutive elements of `data` that is
    greater than `u_diff`.

    Parameters
    ----------
    data : np.array
        One dimensional array containing numeric data.
    u_diff : numeric
        Numeric value that is the minimum difference two consecutive elements
        of `data` must have to trigger the start of an unstable sequence. 
        `u_diff` is exclusive, i.e. entries must differ more than `u_diff` to
        be unstable. 
    s_diff : numeric, optional, default=None
        Numeric or none. If none, set to `u_diff`. Max difference that can exist
        between consecutive entries of `data` for those points to be considered
        under control or stable. `s_diff` is inclusive, i.e. entries must differ
        more than `s_diff` to be unstable. Should be smaller than `u_diff`.
    stability_duration : int >= 1, optional, default=10
        Duration of stability that must happen after a signal exits control to
        be called stable once again. Note that a value below 1 for this
        parameter is meaningless because the shortest sequence contains at
        least two difference (i.e. 3 consecutive points).

    Returns
    -------
    np.array
        Array of size K x 2, where the 0th column contains the start indices of
        sequences, and the 1st column contains the duration of the sequence.
        Note that the duration of the sequence will be at least 1 larger than
        `stability_duration`. This is because there must be `stability_duration`
        differences less than `s_diff` after the out of control difference.

    Notes
    -----
    This function calculates instability as a function of the differences
    between consecutive points. For N input points, there are N-1 differences.
    This means that the output durations are one less than the index of the
    final point of a sequence.
    '''
    if s_diff == None:
        s_diff = u_diff
    idx = 1
    unstable_spans = []
    n = data.size
    while idx < n:
        if abs(data[idx - 1] - data[idx]) > u_diff:
            stable = False
            stable_count = 0
            offset = 1
            while not stable and (offset + idx) < n:
                # Because of round off error, we have to use a combination of 
                # testing that s_diff is bigger, and testing that it is very
                # very close. Otherwise things like this happen:
                # assert (not abs(.7 - .9) <= .2)
                v = abs(data[offset + idx - 1] - data[offset + idx]) - s_diff
                if v < 0 or np.isclose(v, 0):
                    stable_count += 1
                else:
                    stable_count = 0
                if stable_count >= stability_duration:
                    unstable_spans.append((idx, offset))
                    stable = True
                else:
                    offset += 1
            idx += offset
        idx += 1
    # If we are out of control when data ends we return first unstable point
    # and duration up to the end of the sequence.
    if not stable:
        unstable_spans.append((idx - (offset + 1), offset - 1))
    return np.array(unstable_spans)


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

