"""Contains basic tools to extract features in timeseries."""

import numpy as np
import pandas as pd

from energy_analysis_toolbox import keywords as eatk
from energy_analysis_toolbox.errors import (
    EATEmptyDataError,
    EATInvalidTimestepDurationError,
    EATUndefinedTimestepError,
)


def intervals_over(
    series: pd.Series,
    low_tshd: float,
    *,
    return_positions: bool = False,
) -> pd.DataFrame:
    """Return the overconsumption limits when the series values are over ``low_tshd``.

    Parameters
    ----------
    series : pd.Series
         The series in which overconsumption of consecutive values over ``low_tshd``
         are searched.
    low_tshd : float
        Lower threshold on the values in the series for interval extractions :
        consecutive values of the series elements over **(strict)** this threshold
        are searched.
    return_positions : bool, default False
        If ``True``, return a second Dataframe with interval bounds as integer
        positions in the provided ``series`` instead of the labels. Default is
        ``False``.

    Returns
    -------
    overconsumption : pd.DataFrame
        The dataframe of overconsumption, with two columns: |eatk.start_f| and
        |eatk.end_f|. Each row contains :

        - in |eatk.start_f| the label of the first instant of an interval when
          the values in the series are > `low_tshd`.
        - in |eatk.end_f| the label of the first instant after this interval

        such that the row describes interval as ``[start, end[``.
    iloc_bounds : pd.Dataframe, optional
        Dataframe with same structure as ``overconsumption``, except that the values
        for the start and ends of overconsumption are integer positions ("ilocs") in
        the original series instead of labels.

    Notes
    -----
    The algorithm used in this function is the following.

    [1] First, locate the instants when the value of ``series`` changes either:

    - from a value ``=< low_tshd`` to value ``> low_tshd``
    - from a value ``> low_tshd`` to value ``=< low_tshd``

    These shifts are interval limits.

    [2] The limits of the timeseries are special cases. These bounds are
    considered as shifts if the value is over the threshold at these positions.

    Recalling the indexation is for overconsumption which are open on the right side :

    - an interval start is the index of a positive change in values
      (first index of value ``> low_tshd``),
    - an interval end is the index of a negative change (first index
      ``=< low_tshd`` value).

    [3] Accordingly, with [1] and [2], shifts with even indices are interval
    starts while those with odd ones are interval ends.

    [4] The dataframe of interval bounds is assembled and returned. The ilocs
    are returned as well in case ``return_positions`` is ``True``.

    Example
    -------
    >>> time_begin = pd.Timestamp("2018-07-06 05:00:00")
    >>> time_range = pd.date_range(
            start=time_begin,
            periods=8,
            inclusive='left',
            freq=pd.DateOffset(seconds=300))
    >>> series = pd.Series(np.array([0, 1, 1.5, 2., 2., 3., 1.5, 0.]), index=time_range)
    >>> intervals_over(series, 1.5)
                    start                 end
        0 2018-07-06 05:15:00 2018-07-06 05:30:00
    >>> intervals_over(series, 42)
        Empty DataFrame
        Columns: [start, end]
        Index: []

    """
    if series.empty:
        return pd.DataFrame([], columns=[eatk.start_f, eatk.end_f])
    # 1 diffs in a Bool series -> True if element differs from previous
    over_status_shift = (series > low_tshd).diff()
    # 2
    over_status_shift.iloc[0] = series.iloc[0] > low_tshd
    over_status_shift.iloc[-1] = over_status_shift.iloc[-1] or (
        series.iloc[-1] > low_tshd
    )
    # indexes of shifts
    actual_times = over_status_shift.index.where(over_status_shift).dropna()
    actual_inds = [series.index.get_loc(time) for time in actual_times]
    # 3 as [[start, end]]
    bound_indexes = [
        [actual_inds[2 * i], actual_inds[2 * i + 1]]
        for i in range(len(actual_inds) // 2)
    ]
    bound_labels = [
        [over_status_shift.index[s], over_status_shift.index[e]]
        for s, e in bound_indexes
    ]
    intervals = pd.DataFrame(bound_labels, columns=[eatk.start_f, eatk.end_f])
    # 4
    if return_positions:
        iloc_bounds = pd.DataFrame(bound_indexes, columns=[eatk.start_f, eatk.end_f])
        return intervals, iloc_bounds
    return intervals


def timestep_durations(
    timeseries: pd.Series,
    last_step: float | None = None,
) -> pd.Series:
    """Return the series of timestep durations of a timeseries.

    Parameters
    ----------
    timeseries : pd.Series
        A timeseries in chronological order.
    last_step : float, optional
        Duration of the last time-step in the series in (s).
        The default is |None| meaning that the same duration as the former-last
        one is used.

    Raises
    ------
    EATEmptyDataError :
        If the series is empty.
    EATUndefinedTimestepError :
        If the series contains only one element and ``last_step``
        is |None|.

    Returns
    -------
    pd.Series
        The series of durations in (s). The element at iloc i is the duration
        before the next index in the series.


    .. important::

        The input timeseries should contain at least two elements (one if the
        duration of the last time-step is provided).

    .. seealso::

        :py:func:`energy_analysis_toolbox.timeseries.extract_features.basics.index_to_timesteps`
        which works directly from the series index.

    """
    if timeseries.empty:
        err = "Interval durations cannot be inferred for empty series."
        raise EATEmptyDataError(err)
    min_timeseries_length = 2
    if timeseries.size < min_timeseries_length and last_step is None:
        err = (
            "The timeseries should contain at least 2 elements to infer a duration"
            "when last_step value is None."
        )
        raise EATUndefinedTimestepError(err)
    return pd.Series(
        index_to_timesteps(timeseries.index, last_step),
        index=timeseries.index,
    )


def index_to_timesteps(
    time_indexes: pd.DatetimeIndex,
    last_step: float | None = None,
) -> np.array:
    """Return the array of interval durations of a DatetimeIndex.

    Parameters
    ----------
    time_indexes : pd.DatetimeIndex
        A sequence of time-steps in chronological order.
    last_step : float, optional
        Duration of the last time-step in the series in (s).
        The default is |None| meaning that the same duration as the former-last
        one is used.

    Raises
    ------
    EATEmptyDataError :
        If ``time_indexes`` is empty.
    EATUndefinedTimestepError :
        If ``time_indexes``  contains only one element and ``last_step``
        is |None|.
    EATInvalidTimestepDurationError :
        If ``last_step < 0``.

    Returns
    -------
    np.array
        The series of durations in (s). The element i is the duration
        before the next index.


    .. important::

        The input ``time_indexes`` should contain at least two elements (one if the
        duration of the last time-step is provided).

    .. seealso::

        :py:func:`energy_analysis_toolbox.timeseries.extract_features.basics.timestep_durations`
        which works on timeseries by applying this function to its index.

    """
    if time_indexes.empty:
        err = "Interval durations cannot be inferred for empty time-sequences."
        raise EATEmptyDataError(err)
    min_time_indices_length = 2
    if time_indexes.size < min_time_indices_length and last_step is None:
        err = (
            "The series should contain at least 2 elements to infer a duration"
            "when last_step value is None."
        )
        raise EATUndefinedTimestepError(err)
    if last_step is not None and last_step < 0:
        err = f"Last step duration must be >=0. Received {last_step} s."
        raise EATInvalidTimestepDurationError(err)
    # better initialize and assign to avoid table extension in the next step
    durations = np.empty(time_indexes.size)
    durations[:-1] = (time_indexes[1:] - time_indexes[0:-1]).total_seconds()
    if last_step is None:
        durations[-1] = durations[-2]
    else:
        durations[-1] = last_step
    return durations
