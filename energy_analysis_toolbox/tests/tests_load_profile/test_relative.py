# -*- coding: utf-8 -*-
"""
Tests for the ``RelativeThreshold``.
"""

import pandas as pd

from energy_analysis_toolbox.timeseries.profiles.thresholds.relative import (
    RelativeThreshold,
)
from .fake_data import sinusoid_history
from .check import compare_profiles


def test_daily_1():
    """Check that 0 relative offset falls back to mean profile"""
    history = sinusoid_history(freq="30min", noise=0, n_days=7)
    profile = RelativeThreshold(period="D", offset_relative=0)
    daily = profile.compute(history, history.index[-1].ceil("D"))
    expected = history.iloc[:48].copy()
    expected.index += 7 * pd.Timedelta("1D")
    compare_profiles(daily, expected)


def test_daily_2():
    """Check with 30% relative offset on periodic time-series."""
    history = sinusoid_history(freq="30min", noise=0, n_days=7)
    profile = RelativeThreshold(period="D", offset_relative=0.3)
    daily = profile.compute(history, history.index[-1].ceil("D"))
    expected = history.iloc[:48].copy() * 1.3
    expected.index += 7 * pd.Timedelta("1D")
    compare_profiles(daily, expected)
