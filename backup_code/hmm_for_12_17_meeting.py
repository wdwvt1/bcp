#!/usr/bin/env python
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import time, datetime, imp
from hmmlearn.hmm import MultinomialHMM

plib = imp.load_source('plib',
                       '/Users/wdwvt/Desktop/Sonnenburg/cumnock/promethion_code/promethion.py')

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

def _axvspan_maker(hs_preds):

    left_index = 0
    right_index = 1
    n = len(hs_preds)
    states = []
    indices = []

    while (left_index < n) and (right_index < n):
        value = hs_preds[left_index]
        right_index = left_index
        while right_index < n and hs_preds[right_index] == value:
            right_index += 1
        indices.append((left_index, right_index))
        states.append(value)
        left_index = right_index

    return states, indices


data, headers, dts = plib.load_exp1(include_cols_with=['FoodA'])


def _data_on_mouse(data, idx, smoothing_time_radius, smoothing_amplitude_radius,
                   smoothing_tolerance, sampling_interval, bins):
    shw = smooth(data[:, idx], smoothing_time_radius, smoothing_amplitude_radius,
                 smoothing_tolerance)
    chw = interpolate_between_nans(shw)
    fa = find_nan_cumsum(shw)

    n = data.shape[0]
    thinned_chw = chw[0:n:sampling_interval]
    thinned_fa = fa[0:n:sampling_interval]

    delta_hw = thinned_chw[1:] - thinned_chw[:-1]
    delta_fa = thinned_fa[1:] - thinned_fa[:-1]

    observations = np.vstack((delta_hw, delta_fa))
    observations = observations[:, 1:]

    disc_obs = discretize_observations(observations, bins)

    if len(set(disc_obs.squeeze())) != bins**2:
        disc_obs[:bins**2] = np.arange(bins**2)

    return disc_obs, delta_hw, delta_fa

# train on every mouse
mice = [1,4,7,2,3,5,8]
n_components = 4
bins = 3
smoothing_time_radius = 30 #seconds
smoothing_amplitude_radius = .1 #grams
smoothing_tolerance = 1 #number of indices
sampling_interval = 3600 #seconds

discrete_obs, delta_hws, delta_fas = [], [], []
for idx in mice:
    d = _data_on_mouse(data, idx, smoothing_time_radius,
                       smoothing_amplitude_radius, smoothing_tolerance, 
                       sampling_interval, bins)
    discrete_obs.append(d[0])
    delta_hws.append(d[1])
    delta_fas.append(d[2])

X = np.array(discrete_obs)


model = MultinomialHMM(n_components = n_components)
predictions = []
for i in range(7):
    held_out_X = np.vstack((X[:i], X[i+1:]))
    model.fit(held_out_X)
    predictions.append(model.decode(X[i].reshape(X[i].shape[0], 1)))

f, axarr = plt.subplots(7, 1)
yranges = np.arange(n_components+1, dtype=float)/n_components
colors = plt.cm.rainbow(np.linspace(0, 1, n_components))
for i in range(7):
    states, indices = _axvspan_maker(predictions[i][1])
    for s, idxs in zip(states, indices): 
        axarr[i].axvspan(idxs[0], idxs[1], ymin=yranges[s], ymax=yranges[s+1], color=colors[s])
plt.show()

# healthy_model = MultinomialHMM(n_components = n_components)
# healthy_model.fit(dos)
# hs_preds = healthy_model.predict(dos.reshape(len(dos), 1))


# # control mouse 7, infected mouse 8
# smoothing_time_radius = 30 #seconds
# smoothing_amplitude_radius = .1 #grams
# smoothing_tolerance = 1 #number of indices
# sampling_interval = 3600 #seconds

# m7_shw = smooth(data[:, 7], smoothing_time_radius, smoothing_amplitude_radius,
#                 smoothing_tolerance)
# chw = interpolate_between_nans(m7_shw)
# fa = find_nan_cumsum(m7_shw)

# # create actual sequence of observations
# n = data.shape[0]
# thinned_chw = chw[0:n:sampling_interval]
# thinned_fa = fa[0:n:sampling_interval]

# delta_hw = thinned_chw[1:] - thinned_chw[:-1]
# delta_fa = thinned_fa[1:] - thinned_fa[:-1]

# observations = np.vstack((delta_hw, delta_fa))

# # remove the first observation, its the full gain in food weight.
# observations = observations[:, 1:]

# # turn the observations vector into the V characters that can be
# # emitted/observed
# bins = 4
# dos = discretize_observations(observations, bins)

# # if there are any missing characters (i.e. set(dos) != V, then the hmm will
# # fail. this is an ugly hack that erases the first bit of the data to solve this
# # issue.
# if len(set(dos.squeeze())) != bins**2:
#     dos[:bins**2] = np.arange(bins**2)

# n_components = 3


# healthy_model = MultinomialHMM(n_components = n_components)
# healthy_model.fit(dos)
# hs_preds = healthy_model.predict(dos.reshape(len(dos), 1))


# xs = np.arange(len(hs_preds))
# f, axarr = plt.subplots(6, 1)
# axarr[0].plot(xs, delta_hw[1:])
# axarr[0].set_xticks([])
# axarr[1].plot(xs, delta_fa[1:])
# axarr[1].set_xticks([])
# states, indices = _axvspan_maker(hs_preds)
# colors = plt.cm.rainbow(np.linspace(0, 1, n_components))
# axarr[2].set_ylim(0,n_components)
# axarr[2].set_yticks(np.arange(n_components)+.5)
# axarr[2].set_yticklabels(['State %s' % i for i in range(n_components)])

# yranges = np.arange(n_components+1, dtype=float)/n_components
# for s, i in zip(states, indices):
#     axarr[2].axvspan(i[0], i[1], ymin=yranges[s], ymax=yranges[s+1], color=colors[s])

# axarr[0].set_ylabel('Delta Food (grams)')
# axarr[1].set_ylabel('Feeding Attempts')
# axarr[2].set_ylabel('State Prediction')
# axarr[2].set_xlabel('Time (units of 600s)')


# # testing on mouse 8
# m8_shw = smooth(data[:, 8], smoothing_time_radius, smoothing_amplitude_radius,
#                 smoothing_tolerance)
# chw = interpolate_between_nans(m8_shw)
# fa = find_nan_cumsum(m8_shw)

# # create actual sequence of observations
# n = data.shape[0]
# thinned_chw = chw[0:n:sampling_interval]
# thinned_fa = fa[0:n:sampling_interval]

# delta_hw = thinned_chw[1:] - thinned_chw[:-1]
# delta_fa = thinned_fa[1:] - thinned_fa[:-1]

# observations = np.vstack((delta_hw, delta_fa))

# # remove the first observation, its the full gain in food weight.
# observations = observations[:, 1:]

# # turn the observations vector into the V characters that can be
# # emitted/observed
# bins = 4
# dos = discretize_observations(observations, bins)

# # if there are any missing characters (i.e. set(dos) != V, then the hmm will
# # fail. this is an ugly hack that erases the first bit of the data to solve this
# # issue.
# if len(set(dos.squeeze())) != bins**2:
#     dos[:bins**2] = np.arange(bins**2)

# n_components = 3

# hs_preds = healthy_model.predict(dos.reshape(len(dos), 1))

# axarr[3].plot(xs, delta_hw[1:])
# axarr[3].set_xticks([])
# axarr[4].plot(xs, delta_fa[1:])
# axarr[4].set_xticks([])
# states, indices = _axvspan_maker(hs_preds)
# colors = plt.cm.rainbow(np.linspace(0, 1, n_components))
# axarr[5].set_ylim(0,n_components)
# axarr[5].set_yticks(np.arange(n_components)+.5)
# axarr[5].set_yticklabels(['State %s' % i for i in range(n_components)])

# yranges = np.arange(n_components+1, dtype=float)/n_components
# for s, i in zip(states, indices):
#     axarr[5].axvspan(i[0], i[1], ymin=yranges[s], ymax=yranges[s+1], color=colors[s])

# axarr[3].set_ylabel('Delta Food (grams)')
# axarr[4].set_ylabel('Feeding Attempts')
# axarr[5].set_ylabel('State Prediction')
# axarr[5].set_xlabel('Time (units of 600s)')













# plot showing that the smoothing function works.
# plt.title('Smoothing Algorithm')
# plt.plot(data[:, 0], data[:, 7], color='black', label='Raw Data', alpha=.33,
#          lw=2)
# plt.plot(data[:, 0], m7_shw, color='brown', label='Smoothed Data', alpha=.33,
#          lw=2)
# plt.plot(data[:, 0], chw, color='green', label='Smoothed/Interpolated Data',
#          alpha=.33, lw = 1)
# plt.legend()
# plt.show()



