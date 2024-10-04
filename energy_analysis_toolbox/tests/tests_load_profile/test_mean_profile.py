# -*- coding: utf-8 -*-
"""
Tests for the ``MeanProfile`` base class.
"""

import pandas as pd

from energy_analysis_toolbox.timeseries.profiles.mean_profile import MeanProfile
from .fake_data import sinusoid_history
from .check import compare_profiles


def test_daily_1():
    history = sinusoid_history(freq="30min", noise=0, n_days=7)
    profile = MeanProfile(period="D")
    daily = profile.compute(history, history.index[-1].ceil("D"))
    expected = history.iloc[:48].copy()
    expected.index += 7 * pd.Timedelta("1D")
    compare_profiles(daily, expected)


def test_daily_2():
    history = sinusoid_history(freq="15min", noise=0.05, n_days=30)
    profile = MeanProfile(period="D")
    daily = profile.compute(history, history.index[-1].ceil("D"))
    expected = history.iloc[:96].copy()
    expected.index += 30 * pd.Timedelta("1D")
    assert expected.size == daily.size
