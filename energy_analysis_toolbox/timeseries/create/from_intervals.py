import numpy as np
import pandas as pd
from ... import keywords as EATK


def flatten_and_fill(
    data,
    fill_values=None,
    start_f=EATK.start_f,
    end_f=EATK.end_f,
    time_f=EATK.time_f,
):
    """Return data as a table of timeseries.

    Parameters
    ----------
    data : pd.DataFrame
        A table of overconsumption which are each defined by a ``start_f``, used as
        index, and a ``end_f``.
    fill_values : dict or None
        A dict where keys are names of columns of ``data`` and values are
        used to fill the duration between the overconsumption for each of the column
        in the dict. Missing column receive ``np.nan``.
        Default is |None| which is treated as an empty dict.
    start_f : str, default |EATK.start_f|
        The name of the column defining interval starts.
    end_f : str, default |EATK.end_f|
        The name of the column defining interval ends.
    time_f : str, default |EATK.end_f|
        The name of the index of the returned Dataframe.

    Returns
    -------
    pd.DataFrame :
        A table describing the input data as a timeseries.
        The dataframe is indexed with the times of the interval starts and ends,
        such that the rows indexed with an interval end in the dataframe contains
        only filler values.


    .. note::

        An empty table is considered to be flattened as an empty table.


    .. warning::

        The function assumes that the overconsumption are correctly defined meaning
        that :

        - no interval overlap one each other,
        - overconsumption have no common boundary.

        The function does not check these assumptions.

    Example
    -------
    Consider e.g. the following dataframe (from the function tests).

    >>> table
                                        end col_str  col_float
    timestamp
    2022-12-05 18:00:00 2022-12-05 18:05:00    toto          0
    2022-12-05 18:10:00 2022-12-05 18:15:00    toto          2
    2022-12-05 18:20:00 2022-12-05 18:25:00    toto          4

    It can be flattened as follows :

    >>> flatten_and_fill(table)
                        col_str  col_float
    timestamp
    2022-12-05 18:00:00    toto        0.0
    2022-12-05 18:05:00     NaN        NaN
    2022-12-05 18:10:00    toto        2.0
    2022-12-05 18:15:00     NaN        NaN
    2022-12-05 18:20:00    toto        4.0
    2022-12-05 18:25:00     NaN        NaN

    The ``nan`` values can be filled with fixed values for each column during
    the flattening process :

    >>> flatten_and_fill(table, fill_values={'col_str': 'tata', 'col_float': 0})
                        col_str  col_float
    timestamp
    2022-12-05 18:00:00    toto          0
    2022-12-05 18:05:00    tata          0
    2022-12-05 18:10:00    toto          2
    2022-12-05 18:15:00    tata          0
    2022-12-05 18:20:00    toto          4
    2022-12-05 18:25:00    tata          0

    """
    if data.empty:
        return data
    fillers = data.set_index(end_f).drop(columns=[start_f])
    fill_inputs = {col: np.nan for col in fillers.columns}
    if fill_values is not None:
        fill_inputs.update(fill_values)
        for col, value in fill_inputs.items():
            new_col = pd.Series(
                data=value,
                index=fillers.index,
                name=col,
                dtype=fillers.dtypes[col],
            )
            fillers.loc[:, col] = new_col
    else:
        for col in fillers.columns:
            fillers.loc[:, col] = pd.Series(
                data=np.nan,
                index=fillers.index,
                name=col,
            ).astype(fillers.dtypes[col])
    timeseries_table = pd.concat(
        [data.set_index(start_f).drop(columns=[end_f]), fillers]
    ).sort_index()
    timeseries_table.index.name = time_f
    return timeseries_table
