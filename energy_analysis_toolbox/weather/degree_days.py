"""Provides functions for calculating degree days.

The provided functions assumes that:

- the sampling is not regular (missing data possible)
- the temperature is given in Celsius
- the frequency of the data is around hourly (e.g. 1-3 hours), as the computations
  assume multiple temperature values during each day to compute the ``min``, ``max``,
  ``integral``, and so on during each day.

Notation
--------
- :math:`DD` : Degree-days for a given day (heating or cooling)
- :math:`T_{min}` : Minimum temperature for a given day
- :math:`T_{max}` : Maximum temperature for a given day
- :math:`T_{ref}` : Reference temperature for degree-days computation
- :math:`\\Delta t_i` : Duration of the i-th timestep
- :math:`T_{mean}` : Mean temperature for a given day computed as :math:`\\sum_{i=1}^{N} T_i \\cdot \\Delta t_i / \\sum_{i=1}^{N} \\Delta t_i`
- :math:`N` : Number of timesteps in a day
- :math:`T_i` : Temperature at the i-th timestep
- :math:`(x)^+` : Positive part of x, i.e. :math:`max(x, 0)`
- Type of degree-days:

  - Heating : Degree-days for heating season, means that the reference temperature is higher than the outside temperature
  - Cooling : Degree-days for cooling season, means that the reference temperature is lower than the outside temperature

Examples
--------
>>> temp = pd.Series(np.random.randn(100), index=pd.date_range("2020-01-01", periods=100, freq='h'))
>>> dd_compute(temp, 17, method="integral").head()
2020-01-01    17.147638
2020-01-02    17.262956
2020-01-03    16.903173
2020-01-04    17.132581
2020-01-05    16.918075
Freq: D, Name: heating_degree_days, dtype: float64

>>> dd_compute(temp, 17, method="integral", type="cooling").head()
2020-01-01    0.0
2020-01-02    0.0
2020-01-03    0.0
2020-01-04    0.0
2020-01-05    0.0
Freq: D, Name: cooling_degree_days, dtype: float64

"""

import energy_analysis_toolbox as eat
import pandas as pd
from typing import Literal, Callable

literal_type = Literal["heating", "cooling"]
literal_method = Literal["min_max", "mean", "integral", "pro"]


def dd_min_max(
    temperature: pd.Series,
    reference: float,
    clip_tshd: float = 0,
    type: literal_type = "heating",
) -> pd.Series:
    """Return daily degree-days with min-max method.

    .. math::
        DD = (\\frac{T_{min} + T_{max}}{2} - T_{ref})^+

    Parameters
    ----------
    temperature : pd.Series
        The timeseries of temperature measures from which DD data has to be inferred.
    reference : float
        The reference temperature for  degree-days computation.
    clip_tshd : float or None, optional
        A threshold under which the computed degree-days are clipped.
        Using 0 will return only positive DD values. The default is 0.
    type : {'heating', 'cooling'}, optional
        The type of degree-days to compute. The default is 'heating'.

    Returns
    -------
    pd.Series :
        The timeseries of daily degree-days for the period covered by ``temperature``.


    .. warning::

        Days without data receive ``nan`` values, while those with partial data are
        processed as regular ones.


    .. note:: To developers

        The min_count parameter of the sum method is used to return nan when
        one of the values (usually both) is nan. This is the desired behavior,
        as it avoids returning the value of `reference` when the data is not available.

    Examples
    --------
    >>> temp = pd.Series(np.random.randn(100), index=pd.date_range("2020-01-01", periods=100, freq='h'))
    >>> dd_min_max(temp, 17).head()
    2020-01-01    16.766840
    2020-01-02    17.219413
    2020-01-03    17.168598
    2020-01-04    16.263783
    2020-01-05    17.786569
    Freq: D, Name: heating_degree_days, dtype: float64

    """
    _assert_dd_type(type)
    min_max = temperature.resample("D").apply(["min", "max"])
    degree_days = reference - min_max.sum(axis=1, min_count=2) / 2
    if type == "cooling":
        degree_days = -degree_days
    degree_days.name = (
        eat.keywords.heating_dd_f
        if type == "heating"
        else eat.keywords.cooling_dd_f
    )
    return degree_days.clip(lower=clip_tshd)


def dd_pro(
    temperature: pd.Series,
    reference: float,
    clip_tshd: float = 0,
    type: literal_type = "heating",
):
    """Return daily degree-days with pro method.

    .. math::
        DD = (T_{max} - T_{ref}) * (0.08 + 0.42 * (T_{max} - T_{ref}) / (T_{max} - T_{min})

    Use in France to estimate the degree days during the intermediate seasons,
    when the reference temperature is between the min and max temperature of the day.

    Reference: `meteo France <http://climatheque.meteo.fr/Docs/DJC-methode.pdf>`_

    A small modification is made to the original formula to use the true average temperature
    instead of the min and max temperature mean.
    """
    _assert_dd_type(type)
    min_max_mean = temperature.resample("D").apply(["min", "max", "mean"])
    if type == "cooling":
        # Invert all the signs to compute cooling degree days
        min_max_mean = -min_max_mean
        min_max_mean["min"], min_max_mean["max"] = (
            min_max_mean["max"],
            min_max_mean["min"],
        )
        reference = -reference
        clip_tshd = -clip_tshd
    mask_tmin_over_tref = min_max_mean["min"] > reference
    mask_between_tmin_tmax = (min_max_mean["min"] <= reference) & (
        min_max_mean["max"] >= reference
    )
    degree_days = reference - min_max_mean["mean"]
    degree_days[mask_tmin_over_tref] = clip_tshd
    degree_days[mask_between_tmin_tmax] = (reference - min_max_mean["min"]) * (
        0.08
        + 0.42
        * (reference - min_max_mean["min"])
        / (min_max_mean["max"] - min_max_mean["min"])
    )
    degree_days.name = (
        eat.keywords.heating_dd_f
        if type == "heating"
        else eat.keywords.cooling_dd_f
    )
    return degree_days


def dd_mean(
    temperature: pd.Series,
    reference: float,
    clip_tshd: float = 0,
    type: literal_type = "heating",
) -> pd.Series:
    """Return daily degree-days with mean method.

    This uses a better estimate of the daily average temperature than the
    min-max method of :py:func:`dd_min_max`.

    .. math::
        DD = (T_{mean} - T_{ref})^+

    Parameters
    ----------
    temperature : pd.Series
        The timeseries of temperature measures from which DD data has to be inferred.
    reference : float
        The reference temperature for  degree-days computation.
    clip_tshd : float or None, optional
        A threshold under which the computed degree-days are clipped.
        Using 0 will return only positive DD values. The default is 0.
    type : {'heating', 'cooling'}, optional
        The type of degree-days to compute. The default is 'heating'.

    Returns
    -------
    pd.Series :
        The timeseries of daily degree-days for the period covered by ``temperature``.


    .. warning::

        Days without data receive ``nan`` values, while those with partial data are
        processed as regular ones.

    Examples
    --------
    >>> temp = pd.Series(np.random.randn(100), index=pd.date_range("2020-01-01", periods=100, freq='h'))
    >>> dd_mean(temp, 17).head()
    2020-01-01    16.688565
    2020-01-02    17.205562
    2020-01-03    16.898472
    2020-01-04    16.794069
    2020-01-05    16.808996
    Freq: D, Name: heating_degree_days, dtype: float64

    """
    _assert_dd_type(type)
    degree_days = reference - temperature.resample("D").mean()
    if type == "cooling":
        degree_days = -degree_days
    degree_days.name = (
        eat.keywords.heating_dd_f
        if type == "heating"
        else eat.keywords.cooling_dd_f
    )
    return degree_days.clip(lower=clip_tshd)


def dd_integral(
    temperature: pd.Series,
    reference: float,
    clip_tshd: float = 0,
    type: literal_type = "heating",
    intraday_clip_tshd: float = 0,
) -> pd.Series:
    """Return daily degree-days with integral method.

    This is the most accurate method for computing degree-days, as it takes into
    account the temperature variations during the day.

    .. math::
        DD = \\frac{\\sum_{i=1}^{N} (T_{ref} - T_i)^+ \\cdot \\Delta t_i}{\\sum_{i=1}^{N} \\Delta t_i}

    Parameters
    ----------
    temperature : pd.Series
        The timeseries of temperature measures from which DD data has to be inferred.
    reference : float
        The reference temperature for  degree-days computation.
    clip_tshd : float or None, optional
        A threshold under which the computed degree-days are clipped.
        Using 0 will return only positive DD values. The default is 0.
    type : {'heating', 'cooling'}, optional
        The type of degree-days to compute. The default is 'heating'.
    intraday_clip_tshd : float or None, optional
        A threshold under which the instantaneous temperature differences used
        in degree-days computation are clipped.
        If None, no clipping is done, hence resulting in the same results as
        :py:func:`dd_mean`. The default is ``0``.
        Using ``0`` will assign no heating-need (resp. cooling) to moments
        when the reference temperature is lower than the outside one.

    Returns
    -------
    pd.Series :
        The timeseries of daily degree-days for the period covered by ``temperature``.


    .. warning::

        Days without data receive ``nan`` values, while those with partial data are
        processed as regular ones.

    Notes
    -----
    About the parameter ``intraday_clip_tshd``:

    If the temperature is always below or above the reference temperature, the clipping
    has no effect.

    If the temperature is sometimes below and sometimes above the reference temperature,
    The question is : does the cold temperature compensate the warm temperature?
    Meaning: would the user heat (resp. cool) the building if the temperature in the morning,
    even though it is warm in the afternoon? If the answer is yes, then the clipping should be
    set to 0. If the answer is no, then the clipping should be set to ``None``.

    Examples
    --------
    >>> temp = pd.Series(np.random.randn(100), index=pd.date_range("2020-01-01", periods=100, freq='h'))
    >>> dd_integral(temp, 17).head()
    2020-01-01    16.666671
    2020-01-02    17.109244
    2020-01-03    17.087415
    2020-01-04    16.976660
    2020-01-05    16.337652
    Freq: D, Name: heating_degree_days, dtype: float64
    """
    _assert_dd_type(type)
    timesteps = eat.timeseries.extract_features.timestep_durations(temperature)
    sign = 1 if type == "heating" else -1
    degree_days = (
        (sign * (reference - temperature)).clip(lower=intraday_clip_tshd)
        * timesteps
    ).resample("D").sum() / timesteps.resample("D").sum()
    degree_days.name = (
        eat.keywords.heating_dd_f
        if type == "heating"
        else eat.keywords.cooling_dd_f
    )
    return degree_days.clip(lower=clip_tshd)


def dd_compute(
    temperature: pd.Series,
    reference: float,
    clip_tshd: float = 0,
    type: literal_type = "heating",
    method: literal_method = "integral",
    **kwargs,
) -> pd.Series:
    """Return daily degree-days with the specified method.

    Parameters
    ----------
    temperature : pd.Series
        The
    reference : float
        The reference temperature for  degree-days computation.
    clip_tshd : float or None, optional
        A threshold under which the computed degree-days are clipped.
        Using 0 will return only positive DD values. The default is 0.
    type : {'heating', 'cooling'}, optional
        The type of degree-days to compute. The default is 'heating'.
    method : {'min_max', 'mean', 'integral'}, optional
        The method to use for computing the degree-days. The default is 'integral'.
        - 'min_max' : Min-max method :py:func:`dd_min_max`
        - 'mean' : Mean method :py:func:`dd_mean`
        - 'integral' : Integral method :py:func:`dd_integral`
        - 'pro' : Pro method :py:func:`dd_pro`

    Keyword Arguments
    -----------------
    intraday_clip_tshd : float or None, optional
        see :py:func:`dd_integral`

    Returns
    -------
    pd.Series :
        The timeseries of daily degree-days for the period covered by ``temperature``.

    Raises
    ------
    ValueError: If the method is not recognized.

    See Also
    --------
    - :py:func:`dd_min_max` : Min-max method
    - :py:func:`dd_mean` : Mean method
    - :py:func:`dd_integral` : Integral method
    - :py:func:`dd_pro` : Pro method

    Examples
    --------
    >>> temp = pd.Series(np.random.randn(100), index=pd.date_range("2020-01-01", periods=100, freq='h'))
    >>> dd_compute(temp, 17, method="integral").head()
    2020-01-01    16.968044
    2020-01-02    17.147927
    2020-01-03    16.837810
    2020-01-04    16.877662
    2020-01-05    17.453723
    Freq: D, Name: heating_degree_days, dtype: float64

    """
    if method == "min_max":
        func = dd_min_max
    elif method == "mean":
        func = dd_mean
    elif method == "integral":
        func = dd_integral
    elif method == "pro":
        func = dd_pro
    else:
        raise ValueError(f"Unknown method {method}")
    return func(temperature, reference, clip_tshd, type, **kwargs)


def dd_calc_method(
    func: Callable,
) -> Literal["min_max", "mean", "integral", "unknown"]:
    """Return the name of the method used for computing degree-days.

    Parameters
    ----------
    func : function
        The function to check.

    Returns
    -------
    str :
        The name of the method used for computing degree-days.

    """
    if func == dd_min_max:
        return "min_max"
    elif func == dd_mean:
        return "mean"
    elif func == dd_integral:
        return "integral"
    elif func == dd_pro:
        return "pro"
    else:
        return "unknown"


def _assert_dd_type(
    type: str,
) -> None:
    """Check that the degree-days type is valid.

    Parameters
    ----------
    type : str
        the type of degree-days to compute.

    Raises
    ------
    ValueError
        If the type is not recognized, i.e. not 'heating' or 'cooling'.

    """
    if type not in ["heating", "cooling"]:
        raise ValueError(f"Unknown degree_days computation type {type}")
