# -*- coding: utf-8 -*-
"""
Utility functions to compare profiles.
"""

import pandas as pd


def compare_profiles(left, right):
    pd.testing.assert_series_equal(
        left,
        right,
        check_exact=False,
        check_names=False,
        check_freq=False,
    )
