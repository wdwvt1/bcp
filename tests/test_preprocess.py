#!/usr/bin/env python

import copy
from unittest import TestCase, main
import numpy as np
from bcp.preprocess import (weight_sensor_positive_spikes,
                            smooth_positive_spikes)


class TestWeightPreprocessing(TestCase):
    '''Test preprocessing for weight data.'''

    def setUp(self):
        data1 = np.zeros(100, dtype=float)
        data1[3] = .1
        data1[11] = .88
        data1[15] = 2.5
        data1[27] = -3.
        data1[50] = .8
        data1[55] = .49
        data1[90] = .9
        self.data1 = data1
        self.times1 = np.hstack((np.arange(50), np.arange(50)+100))
        
    def test_weight_sensor_positive_spikes(self):
        # data[50] occurs at a time discontinuity so even though it will be
        # above the threshold, we won't call it a spike.
        threshold = .5
        obs = weight_sensor_positive_spikes(self.data1, self.times1, threshold)
        exp = np.array([10, 14, 27, 89])
        np.testing.assert_array_equal(obs, exp)

        threshold = .1
        obs = weight_sensor_positive_spikes(self.data1, self.times1, threshold)
        exp = np.array([2, 10, 14, 27, 54, 89])
        np.testing.assert_array_equal(obs, exp)

    def test_smooth_positive_spikes(self):
        backward_window = 10
        forward_window = 5
        spikes = np.array([10, 14, 27, 89])
        obs = smooth_positive_spikes(self.data1, spikes, backward_window, 
                                     forward_window)
        exp = copy.copy(self.data1)
        exp[10:15] = .01
        exp[14:19] = .004
        exp[27:32] = .0008
        exp[90:95] = 0.
        np.testing.assert_array_equal(obs, exp)

# run unit tests if run from command-line
if __name__ == '__main__':
    main()
