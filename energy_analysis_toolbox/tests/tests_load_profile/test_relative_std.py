# -*- coding: utf-8 -*-
"""
Tests for the ``RelativeSTDThreshold``.
"""

import pandas as pd

from energy_analysis_toolbox.timeseries.profiles.thresholds.relative_std import (
    RelativeSTDThreshold,
)
from .fake_data import sinusoid_history
from .check import compare_profiles


def test_daily_1():
    """Check that 0 relative offset falls back to mean profile"""
    history = sinusoid_history(freq="30min", noise=0, n_days=7)
    profile = RelativeSTDThreshold(period="D", offset_std=0)
    daily = profile.compute(history, history.index[-1].ceil("D"))
    expected = history.iloc[:48].copy()
    expected.index += 7 * pd.Timedelta("1D")
    compare_profiles(daily, expected)


def test_daily_2():
    """Check with 3 srt relative offset on periodic time-series.

    TODO : Result not verified.
    """
    history = sinusoid_history(freq="30min", noise=0.5, n_days=7)
    profile = RelativeSTDThreshold(period="D", offset_std=3)
    profile.compute(history, history.index[-1].ceil("D"))
