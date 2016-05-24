#!/usr/bin/env python

from unittest import TestCase, main
import numpy as np
from bcp.util import (add_seconds, binary_rearing)
import datetime


class TestUtil(TestCase):
    '''Test util functions.'''

    def setUp(self):
        pass

    def test_add_seconds(self):
        dt = datetime.datetime(2016, 10, 6, 3, 45, 12)
        seconds = 150
        exp = datetime.datetime(2016, 10, 6, 3, 47, 42)
        obs = add_seconds(dt, seconds)
        self.assertEqual(obs, exp)

        seconds = 1350232
        exp = datetime.datetime(2016, 10, 21, 18, 49, 4)
        obs = add_seconds(dt, seconds)
        self.assertEqual(obs, exp)

        dt = datetime.datetime(2016, 2, 24, 7, 45, 00)
        seconds = 638859
        exp = datetime.datetime(2016, 3, 2, 17, 12, 39)
        obs = add_seconds(dt, seconds)
        self.assertEqual(obs, exp)

    def test_binary_rearing(self):
        data = np.arange(1000) - 500
        obs = binary_rearing(data)
        exp = np.zeros(1000)
        exp[:501] = 0
        exp[501:] = 1
        np.testing.assert_array_equal(obs, exp)

# run unit tests if run from command-line
if __name__ == '__main__':
    main()
