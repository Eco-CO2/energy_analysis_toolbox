# -*- coding: utf-8 -*-
""" """

import scipy.constants as SK
import pandas as pd
from .fake_data import sinusoid_history_df
from energy_analysis_toolbox.timeseries.profiles.preprocessing.history_filters.weekdays import (
    weekdays_only,
    weekends_only,
    same_day_only,
)


def test_same_day():
    monday = pd.Timestamp("2022-01-03")
    history = sinusoid_history_df(
        start=monday, n_days=21, freq="1D", period_variation=7 * SK.day
    )
    # on df
    filtered = same_day_only(history, date=monday + pd.Timedelta("1D"))
    pd.testing.assert_frame_equal(filtered, history.iloc[[1, 8, 15], :])
    # on series
    filtered_s = same_day_only(history.iloc[:, 0])
    pd.testing.assert_series_equal(filtered_s, history.iloc[[0, 7, 14], 0])


def test_weekends():
    monday = pd.Timestamp("2022-01-03")
    history = sinusoid_history_df(
        start=monday, n_days=21, freq="1D", period_variation=7 * SK.day
    )
    # on df
    filtered = weekends_only(history)
    pd.testing.assert_frame_equal(
        filtered, history.iloc[[5, 6, 12, 13, 19, 20], :]
    )
    # on series
    filtered_s = weekends_only(history.iloc[:, 0])
    pd.testing.assert_series_equal(
        filtered_s, history.iloc[[5, 6, 12, 13, 19, 20], 0]
    )


def test_weekdays():
    monday = pd.Timestamp("2022-01-03")
    history = sinusoid_history_df(
        start=monday, n_days=7, freq="1D", period_variation=7 * SK.day
    )
    # on df
    filtered = weekdays_only(history)
    pd.testing.assert_frame_equal(filtered, history.iloc[0:5, :])
    # on series
    filtered_s = weekdays_only(history.iloc[:, 0])
    pd.testing.assert_series_equal(filtered_s, history.iloc[0:5, 0])


def test_weekdays_weekends_orthogonal():
    """Test that filtering weekends then weekdays only returns an empty history."""
    monday = pd.Timestamp("2022-01-03")
    history = sinusoid_history_df(
        start=monday, n_days=7, freq="1D", period_variation=7 * SK.day
    )
    assert weekdays_only(weekends_only(history)).empty


def test_pipelining():
    """Check that the fileters can be applied through DatafRame.pipeline method."""
    monday = pd.Timestamp("2021-12-27")
    history = sinusoid_history_df(
        start=monday, n_days=14, freq="1D", period_variation=7 * SK.day
    )
    filtered = history.pipe(same_day_only).pipe(weekdays_only)
    pd.testing.assert_frame_equal(
        filtered, history.iloc[[0, 7], :], check_freq=False
    )
    filtered = history.pipe(same_day_only).pipe(weekends_only)
    filtered.empty
