#!/usr/bin/env python

from unittest import TestCase, main
import numpy as np
from bcp.stats import (centered_moving_average)


class TestStats(TestCase):
    '''Test stat functions.'''

    def setUp(self):
        pass

    def test_centered_moving_average(self):
        data = np.array([1, 4, 5, 19, 2, 2, 4, 5])
        r = 3
        obs = centered_moving_average(data, r)
        exp = np.array([0, 0, 0, 37/7., 41/7., 0, 0, 0])
        np.testing.assert_array_equal(obs, exp)

        data = np.random.uniform(size=10)
        r = 2
        obs = np.zeros(10)
        obs[2] = data[:5].sum()
        obs[3] = data[1:6].sum()
        obs[4] = data[2:7].sum()
        obs[5] = data[3:8].sum()
        obs[6] = data[4:9].sum()
        obs[7] = data[5:10].sum()
        obs = obs/5.
        exp = centered_moving_average(data, r)
        np.testing.assert_almost_equal(obs, exp)

# run unit tests if run from command-line
if __name__ == '__main__':
    main()
