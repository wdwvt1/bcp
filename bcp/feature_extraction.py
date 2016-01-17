#!/usr/bin/env python
from __future__ import division

import numpy as np
from scipy.cluster.vq import whiten


def trace_to_signals_matrix(data, signal_length, regularization_value=1e-5):
    '''Create a 2D array of signals from a singal data trace.

    Notes
    -----
    Convert `data` to 2d matrix of ~k rows of length `signal_length`. Each row
    has its mean subtracted and each column is divided by the column standard
    deviation. `regularization_value` is added to the standard deviation of
    each row; this can help data from being rescaled to very large values when
    a signal is passed which has little variance. This seems to occur more with
    shorter signal lengths.

    Parameters
    ----------
    data : np.array
        One dimensional array of data whose entries are consecutive recordings
        of a data stream over time (e.g. water weight).
    signal_length : int
        Length of an individual signal to break `data` into.
    regularization_value : float, optional
        This value is added to the standard deviation of each column before the
        column is divided (rescaled to have variance 1).

    Returns
    -------
    np.array
        Array of shape k X signal_length where k is
        data.shape[0] // signal_length. Data has has had row means subtracted
        and where each column has been rescaled.
    '''
    nrows = data.shape[0] // signal_length
    nd = data[:nrows * signal_length].reshape(nrows, signal_length)
    nd = nd - np.expand_dims(nd.mean(1), axis=1)
    return nd / (nd.std(0) + regularization_value)
