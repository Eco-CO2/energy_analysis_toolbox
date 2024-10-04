import pandas as pd


def same_category(
    history,
    date=None,
    classificator=None,
    **kwargs,
):
    """Returns the subset of history for which the category is the same as which of date.

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
    ** kwargs

    Returns
    -------
    pd.DataFrame :
        Input history in which only the entries for which the returned category
        is the same as which of ``date`` are returned.

    """
    if history.empty or classificator is None:
        return history
    elif date is None:
        date = (history.index[-1].floor("D")) + pd.Timedelta("1D")
    categories = history.index.to_series().apply(classificator)
    ref_category = classificator(date)
    return history.loc[categories == ref_category]


def keep_categories(
    history,
    classificator=None,
    keep=None,
    **kwargs,
):
    """Returns the subset of history for which the category is in the list.

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
    elif keep is None:
        keep = []
    categories = history.index.to_series().apply(classificator)
    mask = [(cat_image in keep) for cat_image in categories.values]
    return history.loc[mask]


def remove_categories(
    history,
    classificator=None,
    remove=None,
    **kwargs,
):
    """Returns the subset of history for which the category is the same as which of date.

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
    elif remove is None:
        remove = []
    categories = history.index.to_series().apply(classificator)
    mask = [(day_cat not in remove) for day_cat in categories.values]
    return history.loc[mask]
