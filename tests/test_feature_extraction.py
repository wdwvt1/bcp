#!/usr/bin/env python

from unittest import TestCase, main
import numpy as np
from bcp.feature_extraction import (trace_to_signals_matrix)


class TestFeatureExtraction(TestCase):
    '''Test feature extraction functions.'''

    def setUp(self):
        pass

    def test_trace_to_signals_matrix(self):
        # Test when signal_length is a proper divisor of data length.
        signal_length = 10
        data = np.random.uniform(size=1000)
        obs = trace_to_signals_matrix(data, signal_length)
        exp = data.reshape(100, 10)
        exp = exp - exp.mean(1).reshape(100, 1)
        exp = exp / exp.std(0)
        np.testing.assert_array_equal(obs, exp)

        # Test when signal_length is not a proper divisor of data length.
        signal_length = 7
        obs = trace_to_signals_matrix(data, signal_length)
        exp = data[:994].reshape(142, 7)
        exp = exp - exp.mean(1).reshape(142, 1)
        exp = exp / exp.std(0)
        np.testing.assert_array_equal(obs, exp)

# run unit tests if run from command-line
if __name__ == '__main__':
    main()
