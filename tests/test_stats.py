#!/usr/bin/env python
from __future__ import division

from unittest import TestCase, main
import numpy as np
from bcp.stats import (moving_function, distance_traveled_1d,
                       distance_traveled_2d)


class TestStats(TestCase):
    '''Test stat functions.'''

    def setUp(self):
        pass

    def test_moving_average(self):
        # Test 6 cases - don't need to duplicate odd/even, even/odd
        # {boundary=1, boundary=2} X {window=even, window=odd} X {data=even, data=odd}
        data = np.array([1, 4, 5, 19, 2, 2, 4, 5])
        function = 'average'
        
        window = 3
        boundary = 1
        obs = moving_function(data, window, function, boundary)
        exp = np.array([5/3., 10/3., 28/3., 26/3., 23/3., 8/3., 11/3., 3])
        np.testing.assert_array_almost_equal(obs, exp)

        window = 3
        boundary = 2
        obs = moving_function(data, window, function, boundary)
        exp = np.array([1., 10/3., 28/3., 26/3., 23/3., 8/3., 11/3., 5])
        np.testing.assert_array_almost_equal(obs, exp)
        
        window = 4
        boundary = 1
        obs = moving_function(data, window, function, boundary)
        exp = np.array([5/4., 10/4., 29/4., 30/4., 28/4., 27/4., 13/4., 11/4.])
        np.testing.assert_array_almost_equal(obs, exp)

        window = 4
        boundary = 2
        obs = moving_function(data, window, function, boundary)
        exp = np.array([1., 4., 29/4., 30/4., 28/4., 27/4., 13/4., 5.])
        np.testing.assert_array_almost_equal(obs, exp)

        # Change data to odd length
        data = np.array([1, 4, 5, 19, 2, 2, 4, 5, 15])
        
        window = 5
        boundary = 1
        obs = moving_function(data, window, function, boundary)
        exp = np.array([10/5., 29/5., 31/5., 32/5., 32/5., 32/5., 28/5.,
                        26/5., 24/5.])
        np.testing.assert_array_almost_equal(obs, exp)

        window = 5
        boundary = 2
        obs = moving_function(data, window, function, boundary)
        exp = np.array([1., 4., 31/5., 32/5., 32/5., 32/5., 28/5., 5., 15.])
        np.testing.assert_array_almost_equal(obs, exp)

    def test_moving_sum(self):
        # Test 6 cases - don't need to duplicate odd/even, even/odd
        # {boundary=1, boundary=2} X {window=even, window=odd} X {data=even, data=odd}
        data = np.array([1, 4, 5, 19, 2, 2, 4, 5])
        function = 'sum'

        window = 3
        boundary = 1
        obs = moving_function(data, window, function, boundary)
        exp = np.array([5, 10, 28, 26, 23, 8, 11, 9])
        np.testing.assert_array_almost_equal(obs, exp)

        window = 3
        boundary = 2
        obs = moving_function(data, window, function, boundary)
        exp = np.array([1., 10, 28, 26, 23, 8, 11, 5])
        np.testing.assert_array_almost_equal(obs, exp)

        window = 4
        boundary = 1
        obs = moving_function(data, window, function, boundary)
        exp = np.array([5., 10., 29., 30., 28., 27., 13., 11.])
        np.testing.assert_array_almost_equal(obs, exp)

        window = 4
        boundary = 2
        obs = moving_function(data, window, function, boundary)
        exp = np.array([1., 4., 29, 30, 28, 27, 13, 5.])
        np.testing.assert_array_almost_equal(obs, exp)

        # Change data to odd length
        data = np.array([1, 4, 5, 19, 2, 2, 4, 5, 15])

        window = 5
        boundary = 1
        obs = moving_function(data, window, function, boundary)
        exp = np.array([10, 29, 31, 32, 32, 32, 28, 26, 24])
        np.testing.assert_array_almost_equal(obs, exp)

        window = 5
        boundary = 2
        obs = moving_function(data, window, function, boundary)
        exp = np.array([1., 4., 31, 32, 32, 32, 28, 5., 15.])
        np.testing.assert_array_almost_equal(obs, exp)

    def test_distance_traveled_1d(self):
        coords = np.array([1., 0, 5, 6, 5, 5, 5, 15, 0, 3])
        obs = distance_traveled_1d(coords)
        exp = 36
        self.assertEqual(obs, exp)

    def test_distance_traveled_2d(self):
        x_coords = np.array([1., 0, 5])
        y_coords = np.array([6, 3.5, 9.1])
        obs = distance_traveled_2d(x_coords, y_coords)
        exp = (2.5**2 + 1**2)**.5 + (5**2 + 5.6**2)**.5
        self.assertEqual(obs, exp)

# run unit tests if run from command-line
if __name__ == '__main__':
    main()
