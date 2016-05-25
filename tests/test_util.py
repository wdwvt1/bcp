#!/usr/bin/env python

from unittest import TestCase, main
import numpy as np
from bcp.util import (add_seconds, binary_rearing, seconds_till, nights, days)
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

    def test_seconds_till(self):
        dt = datetime.datetime(2016, 2, 24, 7, 45, 00)
        exp = 45
        obs = seconds_till(dt, second=45)
        self.assertEqual(obs, exp)

        dt = datetime.datetime(2016, 2, 24, 7, 45, 00)
        exp = 3645
        obs = seconds_till(dt, second=45, hour=8)
        self.assertEqual(obs, exp)

        dt = datetime.datetime(2016, 2, 24, 7, 45, 00)
        exp = 62145
        obs = seconds_till(dt, second=45, minute=0, hour=1, day=25)
        self.assertEqual(obs, exp)

    def test_nights(self):
        day_start = 7
        day_length = 12
        start_timestamp = datetime.datetime(2016, 10, 6, 3, 45, 12)
        total_exp_seconds = 150000
        #exp_datetime = datetime.datetime(2016, 10, 7, 21, 25, 12)
        exp_nights = np.array([[0, 11688],
                               [11688+12*3600, 11688+24*3600],
                               [141288, 150000]])
        obs_nights = nights(day_start, day_length, start_timestamp,
                            total_exp_seconds)
        np.testing.assert_array_equal(obs_nights, exp_nights)

        day_start = 7
        day_length = 12
        start_timestamp = datetime.datetime(2016, 10, 6, 7, 45, 12)
        total_exp_seconds = 451525
        exp_datetime = datetime.datetime(2016, 10, 11, 13, 10, 37)
        exp_nights = np.array([[40488,  83688],
                               [126888, 170088],
                               [213288, 256488],
                               [299688, 342888],
                               [386088, 429288]])
        obs_nights = nights(day_start, day_length, start_timestamp,
                            total_exp_seconds)
        np.testing.assert_array_equal(obs_nights, exp_nights)

    def test_days(self):
        nights = np.array([[0, 11688],
                           [11688+12*3600, 11688+24*3600],
                           [141288, 150000]])
        total_exp_seconds = 150000
        exp_days = np.array([[11688, 11688+12*3600],
                             [11688+24*3600, 141288]])
        obs_days = days(nights, total_exp_seconds)
        np.testing.assert_array_equal(obs_days, exp_days)

        nights = np.array([[40488,  83688],
                           [126888, 170088],
                           [213288, 256488],
                           [299688, 342888],
                           [386088, 429288]])
        total_exp_seconds = 451525
        exp_days = np.array([[0, 40488],
                             [83688, 126888],
                             [170088, 213288],
                             [256488, 299688],
                             [342888, 386088],
                             [429288, 451525]])
        obs_days = days(nights, total_exp_seconds)
        np.testing.assert_array_equal(obs_days, exp_days)

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
