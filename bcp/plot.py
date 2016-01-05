#!/usr/bin/env python
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import time, datetime, imp

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