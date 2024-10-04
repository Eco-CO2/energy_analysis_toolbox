"""
This module contains basic mathematical transformations to be
applied to timeseries of physical values.

This module defines utilities used to create fine-sampled timeseries from
coarse sampled one:

    - :py:func:`piecewise_affine`
    - :py:func:`piecewise_constant`

Resampling to coarser resolution may be done as well, but the relevance may
be questioned VS a well-chosen aggregation.

.. seealso:: In case the series to be resampled must satisfy conservation laws

    :py:mod:`energy_analysis_toolbox.timeseries.power.conservative`

"""

import numpy as np
import pandas as pd


def piecewise_affine(
    timeseries,
    target_instants,
    **kwargs,
):
    """Return resampled timeseries assuming a piecewise affine function of time.

    Parameters
    ----------
    timeseries : pd.Series or float
        Series of values of a function of time, indexed using DateTimeIndex.
    target_instants : (n_target,) pd.DatetimeIndex
        Dates at which the series values are required, sorted in ascending
        order.

    Returns
    -------
    new_series : (n_target,) pd.Series
        Values of the function, interpolated at target times, indexed with
        ``target_instants``.


    .. warning::

        The returned values may not be relevant when some target times are
        required outside the convex span of the input samples: the
        corresponding border value is used for these target times.


    .. seealso::

        :py:func:`np.interp` on which the interpolation is based.


    """
    try:
        ref_time = target_instants[0]
    except IndexError:
        return pd.Series(
            [], dtype=timeseries.dtype, index=target_instants.copy()
        )
    target_offsets = (target_instants - ref_time).total_seconds()
    sample_dts = (timeseries.index - ref_time).total_seconds()
    new_values = np.interp(target_offsets, sample_dts, timeseries.values)
    new_series = pd.Series(
        new_values, index=target_instants.copy(), name=timeseries.name
    )
    new_series.index.name = timeseries.index.name
    return new_series


def piecewise_constant(
    timeseries,
    target_instants,
    left_pad=None,
    **kwargs,
):
    """Return resampled timeseries assuming a piecewise constant function of time.

    Parameters
    ----------
    timeseries : pd.Series or float
        Series of values of a function of time, indexed using DateTimeIndex.
    target_instants : (n_target,) pd.DatetimeIndex
        Dates at which the series values are required, sorted in ascending
        order.
    left_pad : float or None, optional
        A value to be used for target instants which are located before the
        first instant in ``timeseries``. The default is |None| in which case, the
        value of the first instant is used.

    Returns
    -------
    new_series : (n_target,) pd.Series
        Values of the series interpolated at target times, indexed with
        ``target_instants``.


    .. seealso::

        :py:func:`np.digitize` on which the function is based.

    """
    try:
        ref_time = target_instants[0]
    except IndexError:
        return pd.Series(
            [], dtype=timeseries.dtype, index=target_instants.copy()
        )
    target_offsets = (target_instants - ref_time).total_seconds()
    sample_dts = (timeseries.index - ref_time).total_seconds()
    ix_select = np.digitize(target_offsets, sample_dts, right=False) - 1
    if left_pad is None:
        ix_select[ix_select < 0] = 0  # index 0
        new_values = timeseries.iloc[ix_select].values
    else:
        new_values = timeseries.iloc[ix_select].values
        new_values[ix_select < 0] = left_pad
    new_series = pd.Series(
        new_values, index=target_instants, name=timeseries.name
    )
    new_series.index.name = timeseries.index.name
    return new_series
