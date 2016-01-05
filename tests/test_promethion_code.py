#!/usr/bin/env python

from unittest import TestCase, main
from numpy.testing import assert_equal, assert_almost_equal
import numpy as np
import imp 

plib = imp.load_source('plib',
                       '/Users/wdwvt/Desktop/Sonnenburg/cumnock/promethion_code/promethion.py')

class PromethionParsertTests(TestCase):
    '''Test Promethion parsing.'''

    def setUp(self):
        pass

class PromethionDataFunctionsTest(TestCase):
    '''Test functions that deal with calculations on Promethion data.'''

    def setUp(self):
        pass

    def test_wheel_distance_over_window(self):

        rs = np.array([0, 3, 5, 1, 1, 1, 0, 0, 2, 0, 1, 1], dtype=np.float32)
        window = 4
        radius = 11.43
        exp = np.pi * 2 * radius * np.array([9, 10, 8, 3, 2, 3, 2, 3, 4],
                                            dtype=np.float32)
        obs = plib.wheel_distance_over_window(rs, window, radius)
        assert_almost_equal(obs, exp)

    def test_xy_distance_over_window(self):

        xs = np.array([4, 10, 0, 0, 5, 8, 8, 8, 6, 14, 3, 1], dtype=np.float32)
        ys = np.array([8, 0, 0, 0, 0, 2, 1, 10, 3, 20, 3, 1], dtype=np.float32)
        window = 4

        exp_rx = np.array([21, 18, 8, 8, 5, 10, 21, 23], dtype=np.float32)
        exp_ry = np.array([8, 2, 3, 12, 19, 34, 50, 43], dtype=np.float32)

        obs_rx, obs_ry = plib.xy_distance_over_window(xs, ys, window)

        assert_equal(obs_rx, exp_rx)
        assert_equal(obs_ry, exp_ry)

    def test_rearing_over_window(self):

        zs = np.array([56, 80, 0, 0, 0, 0, 0, 0, 12, 11])
        window = 3
        exp = np.array([2, 1, 0, 0, 0, 0, 1, 2])
        obs = plib.rearing_over_window(zs, window)
        assert_equal(obs, exp)

# run unit tests if run from command-line
if __name__ == '__main__':
    main()
