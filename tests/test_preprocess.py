#!/usr/bin/env python

import copy
from unittest import TestCase, main
import numpy as np
from bcp.preprocess import (weight_sensor_positive_spikes,
                            smooth_positive_spikes, stable_sequences,
                            valued_sequences, unstable_sequences)


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

    def test_stable_sequences(self):
        # Test with a diff of 0. This tests situations where we want to find
        # sequences of repeated characters within data.
        data = np.concatenate((np.ones(5), 10*np.ones(10), np.array([0]),
                               10*np.ones(3)))
        diff = 0
        stability_duration = 1
        exp = np.array([[0, 4],[5, 9], [16, 2]])
        obs = stable_sequences(data, diff, stability_duration)
        np.testing.assert_array_equal(obs, exp)

        diff = 0
        stability_duration = 2
        exp = np.array([[0, 4],[5, 9], [16, 2]])
        obs = stable_sequences(data, diff, stability_duration)
        np.testing.assert_array_equal(obs, exp)

        diff = 0
        stability_duration = 4
        exp = np.array([[0, 4],[5, 9]])
        obs = stable_sequences(data, diff, stability_duration)
        np.testing.assert_array_equal(obs, exp)

        diff = 0
        stability_duration = 5
        exp = np.array([[5, 9]])
        obs = stable_sequences(data, diff, stability_duration)
        np.testing.assert_array_equal(obs, exp)

        # Test with diffs > 0. This tests situations where we want to allow some
        # variation between elements.
        diff = 1
        stability_duration = 1
        exp = np.array([[0, 4],[5, 9], [16, 2]])
        obs = stable_sequences(data, diff, stability_duration)
        np.testing.assert_array_equal(obs, exp)

        diff = 9
        stability_duration = 1
        exp = np.array([[0, 14], [16, 2]])
        obs = stable_sequences(data, diff, stability_duration)
        np.testing.assert_array_equal(obs, exp)

        diff = 9
        stability_duration = 4
        exp = np.array([[0, 14]])
        obs = stable_sequences(data, diff, stability_duration)
        np.testing.assert_array_equal(obs, exp)

        # Test other cases.
        data = np.concatenate((np.arange(10), np.array([9])))
        diff = .999
        stability_duration = 1
        exp = np.array([[9, 1]])
        obs = stable_sequences(data, diff, stability_duration)
        np.testing.assert_array_equal(obs, exp)

        data = np.concatenate((np.arange(10), np.array([9])))
        diff = .999
        stability_duration = 0
        exp = np.array([[9, 1]])
        obs = stable_sequences(data, diff, stability_duration)
        np.testing.assert_array_equal(obs, exp)

        diff = 0
        stability_duration = 2
        exp = np.array([])
        obs = stable_sequences(data, diff, stability_duration)
        np.testing.assert_array_equal(obs, exp)

    def test_valued_sequences(self):
        # Test with a diff of 0. This tests situations where we want to find
        # sequences of a given repeated character within data.
        data = np.concatenate((np.ones(5), 10*np.ones(10), np.array([0]),
                               10*np.ones(3)))
        value = 1
        stability_duration = 1
        exp = np.array([[0, 5]])
        obs = valued_sequences(data, value, stability_duration)
        np.testing.assert_array_equal(obs, exp)

        value = 10
        stability_duration = 1
        exp = np.array([[5, 10], [16,3]])
        obs = valued_sequences(data, value, stability_duration)
        np.testing.assert_array_equal(obs, exp)

        value = 10
        stability_duration = 3
        exp = np.array([[5, 10], [16,3]])
        obs = valued_sequences(data, value, stability_duration)
        np.testing.assert_array_equal(obs, exp)

        value = 10
        stability_duration = 4
        exp = np.array([[5, 10]])
        obs = valued_sequences(data, value, stability_duration)
        np.testing.assert_array_equal(obs, exp)

        value = 0
        stability_duration = 1
        exp = np.array([[15, 1]])
        obs = valued_sequences(data, value, stability_duration)
        np.testing.assert_array_equal(obs, exp)

        value = 0
        stability_duration = 2
        exp = np.array([])
        obs = valued_sequences(data, value, stability_duration)
        np.testing.assert_array_equal(obs, exp)

    def test_unstable_sequences(self):
        # Test when only u_diff is specified, i.e, instabilty occurs above
        # u_diff, and stability below.
        data = np.concatenate((np.ones(5), 10*np.ones(10), np.array([0]),
                               10*np.ones(3)))
        u_diff = 1
        stability_duration = 1
        exp = np.array([[5, 1],[15, 2]])
        obs = unstable_sequences(data, u_diff=u_diff,
                                 stability_duration=stability_duration)
        np.testing.assert_array_equal(obs, exp)

        u_diff = 9
        stability_duration = 1
        exp = np.array([[15, 2]])
        obs = unstable_sequences(data, u_diff=u_diff,
                                 stability_duration=stability_duration)
        np.testing.assert_array_equal(obs, exp)

        u_diff = 1
        stability_duration = 3
        exp = np.array([[5, 3],[15, 3]])
        obs = unstable_sequences(data, u_diff=u_diff,
                                 stability_duration=stability_duration)
        np.testing.assert_array_equal(obs, exp)

        data = np.linspace(-1, 1, 21)
        data[5:8] = 0.01
        data[11] = 12
        
        u_diff = .11
        stability_duration = 2
        exp = np.array([[5, 2], [8, 2], [11, 3]])
        obs = unstable_sequences(data, u_diff=u_diff,
                                 stability_duration=stability_duration)
        np.testing.assert_array_equal(obs, exp)

        u_diff = .11
        stability_duration = 3
        exp = np.array([[5, 10]])
        obs = unstable_sequences(data, u_diff=u_diff,
                                 stability_duration=stability_duration)
        np.testing.assert_array_equal(obs, exp)

        data[5:8] = [1, 2, 3]
        u_diff = .001
        stability_duration = 1
        exp = np.array([[1, 19]])
        obs = unstable_sequences(data, u_diff=u_diff,
                                 stability_duration=stability_duration)
        np.testing.assert_array_equal(obs, exp)

        # Test when u_diff and s_diff are specified, i.e, instabilty occurs
        # above u_diff, and stability below s_diff.
        data = np.array([0, .5, .7, .9, 1.5, 2.3, 2.1, 2.01, 1.9, 2.1, 6.7, 4.6,
                         4.29, 4.2, 4.1, 4.05, .01])
        u_diff = .5
        s_diff = .3
        stability_duration = 1
        exp = np.array([[4, 2], [10, 3], [16, 0]])
        obs = unstable_sequences(data, u_diff=u_diff, s_diff=s_diff,
                                 stability_duration=stability_duration)
        np.testing.assert_array_equal(obs, exp)

        u_diff = .3
        s_diff = np.float128(.2)
        stability_duration = 2
        exp = np.array([[1, 2], [4, 3], [10, 4], [16, 0]])
        obs = unstable_sequences(data, u_diff=u_diff, s_diff=s_diff,
                                 stability_duration=stability_duration)
        np.testing.assert_array_equal(obs, exp)

# run unit tests if run from command-line
if __name__ == '__main__':
    main()
