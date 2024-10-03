"""This module contains functions which filter history data based on day of week."""

import pandas as pd


def weekdays_only(
    history,
    **kwargs,
):
    """Returns the weekdays in history.

    Parameters
    ----------
    history : pd.DataFrame
        History data to be filtered. It is expected that the data is time-indexed,
        with monotonic-increasing labels.

    Returns
    -------
    pd.DataFrame :
        Input history from which all weekends entry have been removed.

    """
    return history.loc[history.index.day_of_week < 5]


def weekends_only(
    history,
    **kwargs,
):
    """Returns the weekends in history.

    Parameters
    ----------
    history : pd.DataFrame
        History data to be filtered. It is expected that the data is time-indexed,
        with monotonic-increasing labels.

    Returns
    -------
    pd.DataFrame :
        Input history from which all weekdays entry have been removed.

    """
    return history.loc[history.index.day_of_week > 4]


def same_day_only(
    history,
    date=None,
    **kwargs,
):
    """Returns the same days of week in history n_days before date.

    Parameters
    ----------
    history : pd.DataFrame
        History data to be filtered. It is expected that the data is time-indexed,
        with monotonic-increasing labels.
    date : pd.Timestamp, optional
        Reference for which the day-of-week should be the same in history.
        In case ``None`` is passed, the day after the one of the last entry
        in history is used.

    Returns
    -------
    pd.DataFrame :
        Input history in which only the entries for which the day in the week is
        the same as ``date`` have been conserved.

    """
    if history.empty:
        return history
    elif date is None:
        date = (history.index[-1].floor("D")) + pd.Timedelta("1D")
    else:
        pass
    return history.loc[history.index.day_of_week == date.day_of_week]
