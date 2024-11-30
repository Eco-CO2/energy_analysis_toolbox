"""Module for filtering historical data based on time-based categories.

This module provides utility functions to subset or filter historical data by
categorizing time-indexed rows using a user-defined classificator. The filters
allow for keeping or removing rows based on matching categories or inclusion
in specified category lists.
"""

from collections.abc import Callable

import pandas as pd


def same_category(
    history: pd.DataFrame,
    date: pd.Timestamp | None = None,
    classificator: Callable | None = None,
) -> pd.DataFrame:
    """Return the subset of history for which the category is the same as the date.

    Parameters
    ----------
    history : pd.DataFrame
        History data to be filtered. It is expected that the data is time-indexed,
        with monotonic-increasing labels.
    date : pd.Timestamp, optional
        Reference for which the category should be the same in history.
        In case ``None`` is passed, the start of the day after the one of the
        last entry in history is used.
    classificator : callable or None, optional
        A function mapping a timestamp to a category. If None, return the
        whole history.

    Returns
    -------
    pd.DataFrame :
        Input history in which only the entries for which the returned category
        is the same as which of ``date`` are returned.

    """
    if history.empty or classificator is None:
        return history
    if date is None:
        date = (history.index[-1].floor("D")) + pd.Timedelta("1D")
    categories = history.index.to_series().apply(classificator)
    ref_category = classificator(date)
    return history.loc[categories == ref_category]


def keep_categories(
    history: pd.DataFrame,
    classificator: Callable | None = None,
    keep: list | None = None,
) -> pd.DataFrame:
    """Return the subset of history for which the category is in the list.

    Parameters
    ----------
    history : pd.DataFrame
        History data to be filtered. It is expected that the data is time-indexed,
        with monotonic-increasing labels.
    classificator : callable or None, optional
        A function mapping a timestamp to a category. If None, return the
        whole history.
    keep : list, optional
        A list of categories representation.
        All rows in ``history`` for which index the ``classificator`` returns
        a value which ``not is in keep`` are dumped from the returned history.

    Returns
    -------
    pd.DataFrame :
        Input history in which only the entries for which the returned category
        is in ``keep`` are returned.

    """
    if history.empty or classificator is None:
        return history
    if keep is None:
        keep = []
    categories = history.index.to_series().apply(classificator)
    mask = [(cat_image in keep) for cat_image in categories.to_numpy()]
    return history.loc[mask]


def remove_categories(
    history: pd.DataFrame,
    classificator: Callable | None = None,
    remove: list | None = None,
) -> pd.DataFrame:
    """Return the subset of history for which the category is the same as the date.

    Parameters
    ----------
    history : pd.DataFrame
        History data to be filtered. It is expected that the data is time-indexed,
        with monotonic-increasing labels.
    classificator : callable or None, optional
        A function mapping a timestamp to a category. If None, return the
        whole history.
    remove : list, optional
        A list of categories representation.
        All rows in ``history`` for which index the ``classificator`` returns
        a value which ``is in remove`` are dumped from the returned history.

    Returns
    -------
    pd.DataFrame :
        Input history in which only the entries for which the returned category
        is different from ones in ``remove`` are returned.

    """
    if history.empty or classificator is None:
        return history
    if remove is None:
        remove = []
    categories = history.index.to_series().apply(classificator)
    mask = [(day_cat not in remove) for day_cat in categories.to_numpy()]
    return history.loc[mask]
