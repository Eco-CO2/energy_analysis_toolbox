# -*- coding: utf-8 -*-
"""
Tests for the ``HybridThreshold``.
"""

import pandas as pd

from energy_analysis_toolbox.timeseries.profiles.thresholds.hybrid_rel_std import (
    HybridThreshold,
)
from .fake_data import sinusoid_history
from .check import compare_profiles


def test_daily_1():
    """Check that 0 relative offsets falls back to mean profile"""
    history = sinusoid_history(freq="30min", noise=0, n_days=7)
    profile = HybridThreshold(
        period="D",
        offset_std=0,
        offset_relative=0,
        window=1,
    )
    daily = profile.compute(history, history.index[-1].ceil("D"))
    expected = history.iloc[:48].copy()
    expected.index += 7 * pd.Timedelta("1D")
    compare_profiles(daily, expected)


def test_daily_2():
    """Check with 3 srt relative offset on periodic time-series.

    TODO : Result not verified.
    """
    history = sinusoid_history(freq="30min", noise=0.5, n_days=7)
    profile = HybridThreshold()
    profile.compute(history, history.index[-1].ceil("D"))
