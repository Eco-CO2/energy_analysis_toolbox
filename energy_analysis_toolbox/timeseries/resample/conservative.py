"""
This module contains functions to power timeseries of flows and volumes per
timestep without breaking conservation laws.
"""

import pandas as pd
import numpy as np
from .index_transformation import index_to_freq
from .interpolate import piecewise_affine
from ..extract_features.basics import timestep_durations, index_to_timesteps
from ...errors import (
    EATEmptySourceError,
    EATEmptyTargetsError,
    EATInvalidTimestepDurationError,
)


# =============================================================================
# Resampling with volume conservation
# =============================================================================
def deduce_flows(
    volumes,
    last_step_duration=None,
):
    """Return a timeseries of "flow-rates" from one of an extensive variable.

    Parameters
    ----------
    volumes : pd.Series
        The timeseries of "volumes" (X). In practice, any extensive variable which
        "flows" during the overconsumption of the series.
        The series is assumed to be indexed such that the value at a certain
        index is the volume which flows *until the next index*.
    last_step_duration : float, optional
        Duration of the last time-step in the series in (s).
        The default is |None| meaning that the same duration as the former-last
        one is used.

    Returns
    -------
    pd.Series
        The series of flow-rates in (X.s-1).

    Notes
    -----
    The flow-rate is simply obtained as the ratio between the volume flowing
    during each time-step VS the time-step duration.

    As the values in the series are volumes consumed until the next index, the
    duration to be associated with the last element in the series is ambiguous.
    The chosen value is controlled by ``last_step_duration`` argument.

    """
    durations = timestep_durations(volumes, last_step_duration)
    frs = volumes / durations
    return frs


def flow_rate_conservative(
    flow_rates,
    target_instants,
    last_step_duration=None,
    last_target_step_duration=None,
):
    """Return the series of flow-rate on target instants.

    In what follows, flow-rates and volumes are considered in a broad meaning
    of these terms :

    - volumes represent amounts of an extensive variable which is subjected
      to a conservation law, such as actual volume of water (under certain
      assumptions), or mass, or energy;
    - flow-rates represent flows of this variable, such as volumic flow-rate,
      mass flow-rate or power.

    Parameters
    ----------
    flow_rates : pd.Series
        Timeseries of flow-rates from which to interpolate in (X.s-1) where X
        is any extensive variable.
    target_instants : pd.DatetimeIndex
        Instants at which the flow-rates values have to be returned.
    last_step_duration : float, optional
        Duration of the last time-step in the series in (s).
        The default is |None| in which case the duration of the former-last
        time-step is used.
    last_target_step_duration : float, optional
        Duration of the last time-step in the returned series in (s).
        The default is |None| in which case the duration of the former-last
        time-step is used.
        This duration is used to deduce, if any, what proportion of the volume
        consumption implicitly defined by the ``flow_rates`` series located
        after the last target instant should be attributed to this instant.


    .. important::

        The ``flow_rates`` series is assumed to be indexed such that the value
        at time ``ti`` is the flow-rate during the interval ``[ti, ti+1[`` until
        the next value in the series.

    .. seealso::

        :py:func:`volume_conservative` which resamples the consumed volumes
        on each interval.

    Raises
    ------
    EATEmptySourceError :
        In case ``flow_rates`` is empty.
    EATEmptyTargetsError :
        In case ``target_instants`` is empty.
    EATInvalidTimestepDurationError :
        In case ``last_step_duration <= 0``
    EATInvalidTimestepDurationError :
        In case ``last_target_step_duration <= 0``


    .. note ::

        Uncaught zero-division error will be encountered in case there is a
        null timestep in the source or target values.


    Returns
    -------
    pd.Series :
        Flow-rates values in interpolated at the target instants.


    Notes
    -----
    The flow-rates are interpolated in such a way that the "volume is conserved".
    This means that the volume obtained when integrating the resampled flow-rate
    between two indices in the series should be the same as when computing this
    volume from the initial series.

    The algorithm used by the function is the following :

    - First, the interval durations are computed. The last interval duration
      ambiguity is removed thanks to  ``last_step_duration``. [1.]
    - A series of "volumes" consumed during each timestep is created. This
      is the series of the flow-rate integrals over the time-step in the
      series assuming the ``flow_rates`` series defines a piecewise constant
      function. [2.]
    - The volumes are resampled in a conservative way using
      :py:func:`energy_analysis_toolbox.timeseries.power.volume_conservative` function. [3.]
    - The resulting flow-rates are deduced as the ratio between these volumes
      and the interval durations of the target series. The duration of the
      last interval is defined by ``last_target_step_duration``. [4.]

    """
    if flow_rates.empty:
        raise EATEmptySourceError(
            "Resampling an empty flow-rates series to new instants is an invalid"
            " (undefined) operation."
        )
    elif target_instants.empty:
        raise EATEmptyTargetsError(
            "Target instants must be provided for the series to be resampled."
        )
    else:
        pass
    durations = timestep_durations(
        flow_rates, last_step=last_step_duration
    )  # [1.]
    volumes = flow_rates * durations  # [2.]
    interp_volumes = volume_conservative(
        volumes,
        target_instants,
        last_step_duration=last_step_duration,
        last_target_step_duration=last_target_step_duration,
    )  # [3.]
    target_durations = index_to_timesteps(
        target_instants, last_target_step_duration
    )
    interp_flow_rates = interp_volumes / target_durations  # [4.]
    interp_flow_rates.name = flow_rates.name
    interp_flow_rates.index.name = flow_rates.index.name
    return interp_flow_rates


def volume_conservative(
    volumes,
    target_instants,
    last_step_duration=None,
    last_target_step_duration=None,
):
    """Resample the volume on target instants assuming it is a conservative variable.

    In what follows, flow-rates and volumes are considered in a broad meaning
    of these terms :

    - volumes represent amounts of an extensive variable such as actual
      volume of water, or mass, or energy;
    - flow-rates represent flows of this variable, such as volumic flow-rate,
      mass flow-rate or power.

    The resampling is conservative in the sense that :

    - assuming that ``volumes`` defines a piecewise constant function defined
      on ``[closed, open[`` overconsumption (i.e. constant flow-rate between
      timesteps),
    - the integral of the function defined by the interpolated volume series
      is always equal to which of the function defined by ``volumes`` on the
      same support.


    Parameters
    ----------
    volumes : pd.Series
        The timeseries of volumes in (X). X is the unit of an "extensive"
        variable which can flow along time.
        The series is assumed to be indexed such that the value at a certain
        index is the volume consumed *until the next index*.
    target_instants : pd.DatetimeIndex
        Instants at which the flow-rates values have to be returned.
    last_step_duration : float, optional
        Duration of the last time-step in the ``volume`` series in (s).
        The default is |None| in which case the duration of the former-last
        time-step is used.
        This duration is used to deduce the flow-rate in the last time-step.
    last_target_step_duration : float, optional
        Duration of the last time-step in the returned series in (s).
        The default is |None| in which case the duration of the former-last
        time-step is used.
        This duration is used to deduce, if any, what proportion of the volume
        in the ``volumes`` series located after the last target instant should
        be attributed to this instant.

    Returns
    -------
    pd.Series
        The volumes series sampled on the overconsumption defined in ``target_instants``.

    Raises
    ------
    EATEmptySourceError :
        In case ``volumes`` is empty.
    EATEmptyTargetsError :
        In case ``target_instants`` is empty.
    EATInvalidTimestepDurationError :
        In case ``last_step_duration <= 0``.
    EATInvalidTimestepDurationError :
        In case ``last_target_step_duration <= 0``.

    Example
    -------
    This function can resample extensive variables (typically, volumes) which
    consumption is spread along time using non-matching indices. It assumes a
    constant flux in order to ensure the conservation of the integrated variable
    on the new sampling overconsumption.

    >>> volumes = pd.Series(
            np.array([0, 0.1, 0.05, 0.08]),
            index=pd.date_range("2021-12-15", periods=4, freq='6h')
            )
    >>> volumes
    2021-12-15 00:00:00    0.00
    2021-12-15 06:00:00    0.10
    2021-12-15 12:00:00    0.05
    2021-12-15 18:00:00    0.08
    Freq: 6H, dtype: float64
    >>> target_index = pd.date_range("2021-12-15", periods=3, freq='8h')
    >>> volumes_obtained = volume_conservative(volumes, target_index)
    >>> volumes_obtained
    2021-12-15 00:00:00    0.033333
    2021-12-15 08:00:00    0.100000
    2021-12-15 16:00:00    0.096667

    As the timeseries of extensive variables defined a consumption spread on
    an interval, the last step duration is ambiguous. An arbitrary duration
    can be enforced using the ``last_step_duration`` argument.

    >>> volumes = pd.Series(
            np.array([0.1]),
            index=pd.date_range("2021-12-15", periods=1, freq='1d')
            )
    2021-12-15    0.1
    Freq: D, dtype: float64
    >>> volumes
    >>> target_index = pd.date_range("2021-12-15", periods=4, freq='6h')
    >>> volumes_obtained = volume_conservative(volumes, target_index,
                                           last_step_duration=SK.day)
    >>> volumes_obtained
    2021-12-15 00:00:00    0.025
    2021-12-15 06:00:00    0.025
    2021-12-15 12:00:00    0.025
    2021-12-15 18:00:00    0.025
    Freq: 6H, dtype: float64

    This is also true and applyable for the last target step duration.

    >>> target_index = pd.date_range("2021-12-15", periods=1, freq='12h')
    >>> volumes_obtained = volume_conservative(
            volumes, target_index, last_step_duration=SK.day,
            last_target_step_duration=SK.day / 2)
    >>> volumes_obtained
    2021-12-15    0.05
    Freq: 12H, dtype: float64

    Padding the source series with zeros until the desired end of resampling can
    also ease removing the ambiguity on the "operative" part of the input data.

    Notes
    -----
    The volumes are resampled in such a way that for ``ti`` and ``ti+1`` two
    consecutive indices in ``target_instants``, the resampled volume associated
    with ``ti`` is the volume "consumed" during ``[ti, ti+1[``, assuming a
    constant flow-rate during the time-overconsumption defined in ``volumes``.

    Accordingly, the function computes the resampled volumes as follows :

    - Compute the cumulated sum of volumes for each time-step. [1.]
    - Add a virtual timestep at the end of the volume series located
      ``last_step_duration`` after the last sample, with dummy value. [2.]
    - As the ``volumes`` series was indexed in a way that element at time
      ``t`` was the volume which flows between ``t`` and the next timestep
      the cumulated series is shifted by one step on the right and left-padded
      with a zero. This way, element at position ``t`` is the total volume
      which has flowed until time ``t``. [3.]
    - Also add a virtual timestep in the target series so that the cumulated
      sum at the implicit end of the target series is computed as well. [4.1]
    - Interpolate the series of cumulated volumes as a piecewise affine
      function. Target timesteps outside the convex span of the source-indices
      (including the fictive one) are assigned the corresponding border value:

      * 0 before the beginning of the ``volumes`` series
      * The total cumulated volume after the end of the volume series

      This is consistent with the fact that no volume flow is considered
      outside the overconsumption defined in the series. [4.2]
    - Compute elementwise differences of the cumulated volume series, which
      returns the series of volume "consumed" between each timestep and the
      previous one. [5.]
    - Shift the obtained result one time-step backward in order to match the
      volume-series sampling convention. [6.]
    - Discard the last fictive time-step which has no meaning in this convention. [7.]

    As the flow-rate is the time-derivative of the cumulated consumption curve,
    using a piecewise linear interpolation is equivalent to assuming a constant
    flow-rate during all the time-steps of the ``volumes`` series.

    .. note::

        **Why does this function work ?** I.e. why does it conserve the volume ?
        **Short explanation "with hands" :**

        The sampling convention of the ``volumes`` series is just an indirect way
        of defining a piecewise constant flow-rate function.
        The current method works by resampling exaclty the integral of the
        flow-rate function, to, then, recover a new volume series (piecewise
        constant flow-rate function). By construction, the integral of the resulting
        piecewise constant flow-rate function on the overconsumption of its support of
        definition matches which of the original one


    .. seealso::

        :py:func:`flow_rate_conservative` which resamples the flow-rate on the
        time-overconsumption in a volume-conservative way.

    """
    if volumes.empty:
        raise EATEmptySourceError(
            "Resampling an empty volumes series to new instants is an invalid operation."
        )
    elif target_instants.empty:
        raise EATEmptyTargetsError(
            "Target instants must be provided for the series to be resampled."
        )
    elif last_step_duration is not None and last_step_duration <= 0:
        raise EATInvalidTimestepDurationError(
            "Last step duration cannot be zero."
        )
    elif (
        last_target_step_duration is not None and last_target_step_duration <= 0
    ):
        raise EATInvalidTimestepDurationError(
            "Last step duration cannot be zero."
        )
    else:
        pass
    vol_index = volumes.cumsum()  # [1.]
    # the function deals with None last_step_duration values
    durations = timestep_durations(
        volumes.iloc[-2:], last_step=last_step_duration
    )
    ghost_right = vol_index.index[-1] + pd.Timedelta(
        seconds=durations.iloc[-1]
    )  # [2.]
    vol_index.loc[ghost_right] = np.nan  # np.nan to not break dtype
    vol_index = vol_index.shift(1, fill_value=0.0)  # [3.]
    # the function deals with None last_target_step_duration
    target_durations = index_to_timesteps(
        target_instants[-2:], last_target_step_duration
    )
    target_instants = target_instants.insert(
        target_instants.size,  # at the end
        target_instants[-1] + pd.Timedelta(seconds=target_durations[-1]),
    )  # [4.1]
    interp_vol_index = piecewise_affine(vol_index, target_instants)  # [4.2]
    interp_volumes = (
        interp_vol_index.diff().shift(-1).dropna()
    )  # [5.] [6.] [7.]
    interp_volumes.name = volumes.name
    interp_volumes.index.name = volumes.index.name
    return interp_volumes


def volume_to_freq(
    series: "pd.Series[float]",
    freq,
    origin=None,
    last_step_duration=None,
) -> "pd.Series[float]":
    """Return a series resampled to freq such that the volume is conserved in all timesteps.

    The last step duration of the resampled series is set to the frequency ``freq``.

    Parameters
    ----------
    series : pd.Series
        A series of values of a volume-alike quantity with a DatetimeIndex.
        The sampling period can be constant or not (in this case, use
        ``last_step_duration`` argument).
    freq : str, pd.Timedelta
        the freq to which the series is resampled. Must be a valid
        pandas frequency.
    origin : {None, 'floor, 'ceil', pd.Timestamp}, optional
        What origin should be used for the target resampling range.
        See :py:func:`.index_to_freq` for details.
    last_step_duration : float, optional
        Duration of the last time-step in the ``series`` in (s).
        The default is |None| in which case the duration of the former-last
        time-step is used.
        See :py:func:`.index_to_freq` for details.

    Returns
    -------
    pd.Series
        The resampled series.


    .. seealso::

        * :py:func:`volume_conservative` which resamples the flow-rate on the target instants.
        * :py:func:`flow_rate_to_freq` which resamples the volume on the giver frequency.

    """
    target_instants = index_to_freq(
        series.index, freq, origin, last_step_duration
    )
    resampled_series = volume_conservative(
        series,
        target_instants,
        last_step_duration=last_step_duration,
        last_target_step_duration=pd.Timedelta(freq).total_seconds(),
    )
    return resampled_series


def flow_rate_to_freq(
    series: "pd.Series[float]",
    freq,
    origin=None,
    last_step_duration=None,
) -> "pd.Series[float]":
    """Return a series resampled to freq such that the flow rate is conserved in all timesteps.

    The last step duration of the resampled series is set to the frequency ``freq``.

    Parameters
    ----------
    series : pd.Series
        A series of values of a flow_rate-alike quantity with a DatetimeIndex.
        The sampling period can be constant or not (in this case, use ``last_step_duration`` argument).
    freq : str, pd.Timedelta
        the freq to which the series is resampled. Must be a valid
        pandas frequency.
    origin : {None, 'floor, 'ceil', pd.Timestamp}, optional
        What origin should be used for the target resampling range.
        See :py:func:`.index_to_freq` for details.
    last_step_duration : float, optional
        Duration of the last time-step in the ``series`` in (s).
        The default is |None| in which case the duration of the former-last
        time-step is used.
        This duration is used to deduce the volume in the last time-step
        as well as the last index of the resampled series.

    Returns
    -------
    pd.Series
        The resampled series.


    .. seealso::

        * :py:func:`flow_rate_conservative` which resamples the flow-rate on the target instants.
        * :py:func:`volume_to_freq` which resamples the volume on the giver frequency.

    """
    target_instants = index_to_freq(
        series.index, freq, origin, last_step_duration
    )
    resampled_series = flow_rate_conservative(
        series,
        target_instants,
        last_step_duration=last_step_duration,
        last_target_step_duration=pd.Timedelta(freq).total_seconds(),
    )
    return resampled_series
