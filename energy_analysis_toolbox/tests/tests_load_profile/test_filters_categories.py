# -*- coding: utf-8 -*-
""" """

import functools
import numpy as np
import scipy.constants as SK
import pandas as pd
from .fake_data import sinusoid_history_df
from energy_analysis_toolbox.timeseries.profiles.preprocessing.history_filters.categories import (
    same_category,
    keep_categories,
    remove_categories,
)


# =============================================================================
# Utility functions for test purpose
# =============================================================================
def is_working(timestamp, off_days=None):
    """Return True iif the day of week of timestamp is in ``off_days``.

    Parameters
    ----------
    timestamp : pd.Timestamp
    off_days : list
        List of integers between 0 and 6 representing, say, the "working days"
        of a company. 0 is monday.
        Default is None wjocj falls back to ``[]``.

    Returns
    -------
    bool:
        True is the day od ``date`` is not in ``off_days``.

    """
    if off_days is None:
        off_days = []
    return timestamp.dayofweek not in off_days


# Name after an italian restaurant in Cagnes-sur-Mer which cooks delicious panzerotti
casa_randazzo = functools.partial(is_working, off_days=[0, 1])


# All in one classification
def single_category(x, **kwargs):
    return 1


# =============================================================================
# same_category
# =============================================================================
def test_same_category_identity():
    """Check that the function is identity with only one category."""
    monday = pd.Timestamp("2022-01-03")
    history = sinusoid_history_df(
        start=monday, n_days=14, freq="1D", period_variation=7 * SK.day
    )
    # on df
    filtered = same_category(
        history, date=monday + pd.Timedelta("1D"), classificator=single_category
    )
    pd.testing.assert_frame_equal(filtered, history)
    # on series
    filtered_s = same_category(
        history.iloc[:, 0],
        date=monday + pd.Timedelta("1D"),
        classificator=single_category,
    )
    pd.testing.assert_series_equal(filtered_s, history.iloc[:, 0])


def test_same_category_monday_tuesday():
    """Check the results with two categories : monday/tuesday and others."""
    monday = pd.Timestamp("2022-01-03")
    history = sinusoid_history_df(
        start=monday, n_days=14, freq="1D", period_variation=7 * SK.day
    )
    # on df
    filtered = same_category(
        history, date=monday + pd.Timedelta("1D"), classificator=casa_randazzo
    )
    pd.testing.assert_frame_equal(filtered, history.iloc[[0, 1, 7, 8], :])
    # on series
    filtered_s = same_category(
        history.iloc[:, 0],
        date=monday + pd.Timedelta("1D"),
        classificator=casa_randazzo,
    )
    pd.testing.assert_series_equal(filtered_s, history.iloc[[0, 1, 7, 8], 0])


def test_same_category_arbitrary():
    """Check the result using an external categorization with a proxy getter."""
    monday = pd.Timestamp("2022-01-03")
    history = sinusoid_history_df(
        start=monday, n_days=14, freq="1D", period_variation=7 * SK.day
    )
    categories = pd.Series(
        np.arange(0, 14) % 2,
        index=history.index,
    ).astype("bool")
    categories.loc[monday + pd.Timedelta("15D")] = False

    def classificator(date):
        return categories.loc[date]

    # on df
    filtered = same_category(
        history, date=monday + pd.Timedelta("1D"), classificator=classificator
    )
    pd.testing.assert_frame_equal(filtered, history.loc[categories.values[:-1]])
    # on series
    filtered_s = same_category(
        history.iloc[:, 0],
        date=monday + pd.Timedelta("1D"),
        classificator=classificator,
    )
    pd.testing.assert_series_equal(
        filtered_s, history.loc[categories.values[:-1]].iloc[:, 0]
    )


# =============================================================================
# Example with a single category
# =============================================================================


def test_keep_categories_constant():
    """Check that the function works as expected with simple case of one category."""
    monday = pd.Timestamp("2022-01-03")
    history = sinusoid_history_df(
        start=monday, n_days=14, freq="1D", period_variation=7 * SK.day
    )
    # on df
    filtered = keep_categories(history, keep=[1], classificator=single_category)
    pd.testing.assert_frame_equal(filtered, history)
    filtered_out = keep_categories(
        history, keep=[0], classificator=single_category
    )
    assert filtered_out.empty
    # on series
    filtered_s = keep_categories(
        history.iloc[:, 0], keep=[1], classificator=single_category
    )
    filtered_out_s = keep_categories(
        history.iloc[:, 0], keep=[0], classificator=single_category
    )
    pd.testing.assert_series_equal(filtered_s, history.iloc[:, 0])
    assert filtered_out_s.empty


def test_remove_categories_constant():
    """Check that the function works as expected with simple case of one category."""
    monday = pd.Timestamp("2022-01-03")
    history = sinusoid_history_df(
        start=monday, n_days=14, freq="1D", period_variation=7 * SK.day
    )
    # on df
    filtered = remove_categories(
        history, remove=[0], classificator=single_category
    )
    pd.testing.assert_frame_equal(filtered, history)
    filtered_out = remove_categories(
        history, remove=[1], classificator=single_category
    )
    assert filtered_out.empty
    # on series
    filtered_s = remove_categories(
        history.iloc[:, 0], remove=[0], classificator=single_category
    )
    filtered_out_s = remove_categories(
        history.iloc[:, 0], remove=[1], classificator=single_category
    )
    pd.testing.assert_series_equal(filtered_s, history.iloc[:, 0])
    assert filtered_out_s.empty


# =============================================================================
# Keep only certain categories
# =============================================================================


def test_keep_categories_None_classificator():
    """Check that None classificator returns the whole histroy"""
    monday = pd.Timestamp("2022-01-03")
    history = sinusoid_history_df(
        start=monday, n_days=3, freq="1D", period_variation=7 * SK.day
    )
    filtered = keep_categories(history, keep=[True], classificator=None)
    pd.testing.assert_frame_equal(filtered, history)
    filtered_2 = keep_categories(history, keep=None, classificator=None)
    pd.testing.assert_frame_equal(filtered_2, history)


def test_keep_categories_working_days():
    """Check with a function wich returns True except on monday and tuesday."""
    monday = pd.Timestamp("2022-01-03")
    history = sinusoid_history_df(
        start=monday, n_days=14, freq="1D", period_variation=7 * SK.day
    )
    # on df keep nothing
    assert keep_categories(history, classificator=casa_randazzo).empty
    # on df keep everything
    pd.testing.assert_frame_equal(
        keep_categories(
            history, classificator=casa_randazzo, keep=[True, False]
        ),
        history,
    )
    # keep working days
    filtered = keep_categories(
        history, keep=[True], classificator=casa_randazzo
    )
    pd.testing.assert_frame_equal(
        filtered, history.drop(labels=history.index[[0, 1, 7, 8]])
    )
    # on series
    filtered_s = keep_categories(
        history.iloc[:, 0], keep=[False], classificator=casa_randazzo
    )
    pd.testing.assert_series_equal(filtered_s, history.iloc[[0, 1, 7, 8], 0])


def test_keep_categories_arbitrary():
    """Check when getting categories from a proxy table."""
    monday = pd.Timestamp("2022-01-03")
    history = sinusoid_history_df(
        start=monday, n_days=14, freq="1D", period_variation=7 * SK.day
    )
    categories = pd.Series(
        ["no", "yes"] * 7,  # no on even indices
        index=history.index,
    )

    def classificator(date):
        return categories.loc[date]

    # on df
    filtered = keep_categories(
        history, keep=["yes"], classificator=classificator
    )
    pd.testing.assert_frame_equal(filtered, history.iloc[1::2])
    # on series
    filtered_s = keep_categories(
        history.iloc[:, 0], keep=["no"], classificator=classificator
    )
    pd.testing.assert_series_equal(filtered_s, history.iloc[0::2, 0])


# =============================================================================
# Remove only certain categories
# =============================================================================


def test_remove_categories_None_classificator():
    """Check that None classificator returns the whole histroy"""
    monday = pd.Timestamp("2022-01-03")
    history = sinusoid_history_df(
        start=monday, n_days=3, freq="1D", period_variation=7 * SK.day
    )
    filtered = remove_categories(history, keep=[True], classificator=None)
    pd.testing.assert_frame_equal(filtered, history)
    filtered_2 = remove_categories(history, keep=None, classificator=None)
    pd.testing.assert_frame_equal(filtered_2, history)


def test_remove_categories_working_days():
    """Check with a function wich returns True except on monday and tuesday."""
    monday = pd.Timestamp("2022-01-03")
    history = sinusoid_history_df(
        start=monday, n_days=14, freq="1D", period_variation=7 * SK.day
    )
    # on df remove everything
    assert remove_categories(
        history, classificator=casa_randazzo, remove=[True, False]
    ).empty
    # on df remove nothing
    pd.testing.assert_frame_equal(
        remove_categories(history, classificator=casa_randazzo, remove=[]),
        history,
    )
    # remove working days
    filtered = remove_categories(
        history, remove=[False], classificator=casa_randazzo
    )
    pd.testing.assert_frame_equal(
        filtered, history.drop(labels=history.index[[0, 1, 7, 8]])
    )
    # on series
    filtered_s = remove_categories(
        history.iloc[:, 0], remove=[True], classificator=casa_randazzo
    )
    pd.testing.assert_series_equal(filtered_s, history.iloc[[0, 1, 7, 8], 0])


def test_remove_categories_arbitrary():
    """Check when getting categories from a proxy table."""
    monday = pd.Timestamp("2022-01-03")
    history = sinusoid_history_df(
        start=monday, n_days=14, freq="1D", period_variation=7 * SK.day
    )
    categories = pd.Series(
        ["no", "yes"] * 7,  # no on even indices
        index=history.index,
    )

    def classificator(date):
        return categories.loc[date]

    # on df
    filtered = remove_categories(
        history, remove=["no"], classificator=classificator
    )
    pd.testing.assert_frame_equal(filtered, history.iloc[1::2])
    # on series
    filtered_s = remove_categories(
        history.iloc[:, 0], remove=["yes"], classificator=classificator
    )
    pd.testing.assert_series_equal(filtered_s, history.iloc[0::2, 0])
