"""Transforms indices of a time series to a new index according to a given function."""

import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde, mode
from ...errors import EATUndefinedTimestepError
from ..extract_features.basics import (
    index_to_timesteps,
    timestep_durations,
)


def tz_convert_or_localize(
    timeseries: pd.Series | pd.DataFrame,
    tz,
) -> pd.Series | pd.DataFrame:
    """Assign the requested timezone to the index of a timeseries.

    Parameters
    ----------
    timeseries : pd.Series or pd.DataFrame
        Timeseries to convert.
    tz : str or pytz.timezone or None
        Timezone to assign to the index of the timeseries.

    Returns
    -------
    pd.Series
        Timeseries with the requested timezone assigned to its index.

    .. note::

        This function is just syntactic sugar to avoid dealing with the `TypeError`
        when applying `tz_convert` to a time-naive timeseries.


    .. important::

        When localizing a time-naive timeseries, the `ambiguous` and `nonexistent`
        arguments are set to `True` and `'NaT'` respectively. This means that
        ambiguous times are localized to the beginning of the DST period
        and non-existent times are converted to `'NaT'`.

    """
    try:
        return timeseries.tz_convert(tz)
    except TypeError:
        return timeseries.tz_localize(tz, ambiguous=True, nonexistent="NaT")


def index_to_freq(
    index: pd.DatetimeIndex,
    freq,
    origin=None,
    last_step_duration=None,
) -> pd.DatetimeIndex:
    """Returns the expected index resulting from resampling a time series to a given frequency.

    Parameters
    ----------
    index : pd.DatetimeIndex
        the index of the data to resample
    freq :str, pd.Timedelta
        the freq to which the series is resampled. Must be a valid
        pandas frequency.
    origin : {None, 'floor', 'ceil', pd.Timestamp}
        What origin should be used for the target resampling range. The following
        values are possible :

        - |None| : the default. Use the first index as the data a starting point.
        - ``'floor'`` : use the first index of the data, floored to the passed
          ``freq`` resolution.
        - ``'ceil'`` : use the first index of the data, ceiled to the passed
          ``freq`` resolution.
        - a ``pd.Timestamp`` : use the passed timestamp as starting point. The
          code tries to localize the value to the timezone of the first index in
          the data. Accordingly :

          * if the passed value is time-naive, it is localized to the timezone
            of the data;
          * if the data is time-naive, the timezone of the passed value is ignored
            and it is processed as if it were time-naive.

    last_step_duration : float, optional
        the duration of the last step of the resampling in (s).
        If |None|, the duration of the former-last time-step is used.
        Used to deduce the end of the resampling range.

    Returns
    -------
    pd.DatetimeIndex
        The resulting index of the resampling. Empty if the passed index is empty.
    """
    if index.empty:
        return pd.DatetimeIndex([], name=index.name, tz=index.tz, freq=freq)
    if origin is None:
        start = index[0]
    elif origin == "floor":
        start = index[0].floor(freq)
    elif origin == "ceil":
        start = index[0].ceil(freq)
    else:
        start = pd.Timestamp(origin)
        try:
            start = start.tz_localize(index.tz)
        except TypeError:
            try:
                start = start.tz_convert(index.tz)
            except TypeError:
                print(
                    "The passed origin could not be localized or converted to the"
                    " timezone of the original index. It is processed as if it were time-naive."
                )
    if last_step_duration is None:
        try:
            last_step_duration = (index[-1] - index[-2]).seconds
        except IndexError:
            raise EATUndefinedTimestepError(
                "The last step duration could not be determined from the index."
                " Please provide it explicitly."
            )
    actual_end = index[-1] + pd.Timedelta(seconds=last_step_duration)
    target_instants = pd.date_range(
        start=start,
        end=actual_end,
        freq=freq,
        inclusive="left",
        name=index.name,
    )
    return target_instants


def estimate_timestep(
    data: pd.Series | pd.DataFrame | pd.DatetimeIndex,
    method: str = "median",
) -> float:
    """Returns an estimation of the sampling period of a time series.

    .. note::
        Each method has its own advantages and drawbacks. The best method
        depends on the data. For instance:

        - if the data is regularly spaced, the ``mode`` is the best choice.
        - if the data is irregularly spaced, the ``kde`` is the best choice.
        - the ``median`` is not sensitive to outliers, and is a good choice if
            the data is irregularly spaced and has outliers.
        - the ``mean`` is almost never a good choice.

    Parameters
    ----------
    data : pd.Series, pd.DataFrame, pd.DatetimeIndex
        the data to analyse. Must have (or be) a DatetimeIndex.
    method : {'mean', 'median', 'mode', 'kde'}, optional
        the method used to compute the expected timestep. Defaults to 'median'.

    Returns
    -------
    float
        the expected timestep of the data in (s).

    Raises
    ------
    ValueError
        If the method is not one of {'mean', 'median', 'mode', 'kde'}.


    .. seealso::
        - :func:`median_time_step`
        - :func:`mean_time_step`
        - :func:`mode_time_step`
        - :func:`max_kde_time_step`

    """
    if method == "mean":
        return mean_time_step(data)
    if method == "median":
        return median_time_step(data)
    if method == "mode":
        return mode_time_step(data)
    if method == "kde":
        return max_kde_time_step(data)
    else:
        raise ValueError(
            "method must be one of {'mean', 'median', 'mode', 'kde'}"
        )


def median_time_step(
    data: pd.Series | pd.DataFrame | pd.DatetimeIndex,
) -> float:
    """Returns the median timestep of a time series.

    Parameters
    ----------
    data : pd.Series, pd.DataFrame, pd.DatetimeIndex
        the data to analyse. Must have (or be) a DatetimeIndex.

    Returns
    -------
    float
        the median timestep of the data in (s).


    .. seealso::
        - :func:`estimate_timestep`
        - :func:`mode_time_step`

    """
    data = data_to_datetimeindex(data)
    timesteps = index_to_timesteps(data)
    return np.median(timesteps)


def mean_time_step(
    data: pd.Series | pd.DataFrame | pd.DatetimeIndex,
) -> float:
    """Returns the mean timestep of a time series.

    Parameters
    ----------
    data : pd.Series, pd.DataFrame, pd.DatetimeIndex
        the data to analyse. Must have (or be) a DatetimeIndex.

    Returns
    -------
    float
        the mean timestep of the data in (s).


    .. seealso::
        - :func:`estimate_timestep`
        - :func:`mode_time_step`

    """
    data = data_to_datetimeindex(data)
    timesteps = index_to_timesteps(data)
    return np.mean(timesteps)


def mode_time_step(
    data: pd.Series | pd.DataFrame | pd.DatetimeIndex,
) -> float:
    """Returns the mode timestep of a time series.

    .. warning:: The mode is the most frequent value. If there are several
        values with the same frequency, the first one is returned.
        If the values vary slightly around a central value, the mode
        is not representative of the data.

    Parameters
    ----------
    data : pd.Series, pd.DataFrame, pd.DatetimeIndex
        the data to analyse. Must have (or be) a DatetimeIndex.

    Returns
    -------
    float
        the mode timestep of the data in (s).


    .. seealso::
        - :func:`estimate_timestep`
        - :func:`max_kde_time_step`

    """
    data = data_to_datetimeindex(data)
    timesteps = index_to_timesteps(data)
    return mode(timesteps, nan_policy="omit").mode


def max_kde_time_step(
    data: pd.Series | pd.DataFrame | pd.DatetimeIndex,
) -> float:
    """Returns the maximum probable timestep of a time series.

    .. note:: It differs from the Mode as the distribution is first
       estimated using a KDE. Then, the max of this distribution is
       used.

    .. warning:: The KDE cannot be estimated if the data is regularly
        spaced. In this case, use another method.

    Parameters
    ----------
    data : pd.Series, pd.DataFrame, pd.DatetimeIndex
        the data to analyse. Must have (or be) a DatetimeIndex.

    Returns
    -------
    float
        the mode timestep of the data in (s).


    .. seealso::
        - :func:`estimate_timestep`
        - :func:`median_time_step`
        - :func:`mode_time_step`

    """
    data = data_to_datetimeindex(data)
    timesteps = index_to_timesteps(data)
    kde = gaussian_kde(timesteps)
    no_samples = 50
    samples = np.linspace(min(timesteps), max(timesteps), no_samples)
    probs = kde.evaluate(samples)
    maxima_index = probs.argmax()
    maxima = samples[maxima_index]
    return maxima


def data_to_datetimeindex(
    data: pd.Series | pd.DataFrame | pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """Convert the data to DatetimeIndex.

    Used to allow the use of the same functions for Series, DataFrame and
    DatetimeIndex.

    Parameters
    ----------
    data : pd.Series | pd.DataFrame | pd.DatetimeIndex
        A pandas object

    Returns
    -------
    pd.DatetimeIndex
        Return data if already an index, else the index of the data

    Raises
    ------
    ValueError
        If the data cannot be converted to pandas.DateTimeIndex

    """
    try:
        data = data.index
    except AttributeError:
        pass
    if not isinstance(data, pd.DatetimeIndex):
        raise ValueError("The data cannot be converted to pandas.DateTimeIndex")
    return data


def fill_missing_entries(
    data: pd.Series | pd.DataFrame,
    sampling_period,
    security_factor=2,
    fill_value=pd.NA,
):
    """Fill the data with new entries where the interval is too long.

    .. note:: The duration between the last new entry of a hole and the next
        (existing) entry is less or equal than the sampling_period.

    Parameters
    ----------
    data : pd.Series | pd.DataFrame
        The Data to process. Must have a DatetimeIndex
    sampling_period : float,
        The expected sampling period in (s)
    security_factor : float, optional
        The factor used to determine when a timestep is too long compared to
        the ``sampling_period``, which means that ``sampling_period * security_factor``
        is the maximum duration (excluded) between two entries.
        By default 2.
    fill_value : float, optional
        The value of the newly created entries, by default pd.NA

    Returns
    -------
    pd.Series | pd.DataFrame
        A copy of `data` with new created entries, sorted by index.


    .. seealso::

        - :func:`fill_data_holes`
        - :func:`estimate_timestep`

    """
    durations = timestep_durations(data)
    intervals_to_fill = durations[
        durations >= sampling_period * security_factor
    ]
    if intervals_to_fill.empty:
        return data
    new_indexes = []
    for index, duration in intervals_to_fill.items():
        number_missing_entries = int(duration // sampling_period)
        tmp_indexes = [
            index + k * pd.Timedelta(seconds=sampling_period)
            for k in range(1, number_missing_entries)
        ]
        new_indexes += tmp_indexes
    missing_index = pd.DatetimeIndex(new_indexes)
    new_data = data.reindex(
        data.index.append(missing_index).sort_values(), fill_value=fill_value
    )
    return new_data


def fill_data_holes(
    data: pd.Series | pd.DataFrame,
    method="mode",
    security_factor=2,
    fill_value=pd.NA,
):
    """Return the data with new entries where the interval is too long.

    .. note:: the new indexes are created using the expected timestep determined
        by ``method``. The duration between the last new entry of a hole and the next
        (existing) entry is less or equal than the expected timestep.

    Parameters
    ----------
    data : pd.Series | pd.DataFrame
        The Data to process. Must have a DatetimeIndex
    method : {'mean', 'median', 'mode', 'kde'}, optional
        The method to estimate the expected Frequency, by default "mode".
        See :func:`estimate_timestep` for more details.
    security_factor : float, optional
        The factor used to determine a timestep is too long compared to
        the expected frequency, by default 2.
    fill_value : float, optional
        The value of the newly created entries, by default pd.NA

    Returns
    -------
    pd.Series | pd.DataFrame
        A copy of `data` with new created entries, sorted.


    .. seealso::
        - :func:`fill_missing_entries`
        - :func:`estimate_timestep`

    """
    sampling_period = estimate_timestep(data, method=method)
    return fill_missing_entries(
        data,
        sampling_period,
        security_factor=security_factor,
        fill_value=fill_value,
    )
