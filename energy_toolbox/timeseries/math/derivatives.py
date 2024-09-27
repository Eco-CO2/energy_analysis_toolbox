"""
This moduel contains utilities to compute time-derivatives of physical values
in a timeseries :

- :py:func:`time_derivative_fwd`
- :py:func:`time_derivative_second`

These implementations use basic forward time-derivatives schemes, more suited
for the interest of signal analysis in the draw-detection than for numerical
simulation.

"""

import numpy as np
import pandas as pd


def time_derivative_fwd(
    timeseries,
):
    """Return the forward 1st-order time-derivative of a time-series.

    Parameters
    ----------
    timeseries : pd.Series
        A timeseries.

    Returns
    -------
    pd.Series :
        The timeseries of forward time-derivative of the input series


    .. important::

        The function assumes that the provided timeseries has at least 3
        elements and that all the elements are ordered following chronological
        order.


    Notes
    -----
    Let `f` be a function of time sampled for `N` values indexed by
    chronological order. Assume `N > 2`.

    The derivative values of the timeseries representing the ordered sequence
    of :math:`(f(t_{i}))_{i \\in [ 1.. N ]` are computed using the following
    formulas :

    .. math::

        \\frac{df}{dt}(t_i) = \\frac{f(t_{i+1}) - f(t_{i})}{t_{i+1} - t_{i}} \\forall i \\in [1.. N -1]

        \\frac{df}{dt}(t_N) = \\frac{f(t_{N}) - f(t_{N-1})}{t_{N} - t_{N - 1}}


    .. warning::

        This first order forward scheme is very crude. Its used is dicouraged in
        physical numerical simulation, in which the accuracy and stability of the
        scheme are of great importance.


    Example
    -------
    .. code-block:: python

        >>> import pandas as pd
        >>> import numpy as np
        >>> time = pd.date_range(pd.Timestamp("2021-03-14"), freq='1S', periods=10)
        >>> values = np.arange(0, 10)**2
        >>> series = pd.Series(values, index=time)
        >>> series
        2021-03-14 00:00:00     0
        2021-03-14 00:00:01     1
        2021-03-14 00:00:02     4
        ...
        2021-03-14 00:00:07    49
        2021-03-14 00:00:08    64
        2021-03-14 00:00:09    81
        Freq: S, dtype: int64
        >>> time_derivative_fwd(series)
        2021-03-14 00:00:00     1.0
        2021-03-14 00:00:01     3.0
        2021-03-14 00:00:02     5.0
        ...
        2021-03-14 00:00:07    15.0
        2021-03-14 00:00:08    17.0
        2021-03-14 00:00:09    17.0
        Freq: S, dtype: float64


    """
    dts = (timeseries.index - timeseries.index[0]).total_seconds()
    # [1.]
    grads = np.empty(timeseries.index.size)
    grads[0:-1] = np.ediff1d(timeseries.values) / np.ediff1d(dts)
    grads[-1] = (timeseries.values[-1] - timeseries.values[-2]) / (
        dts[-1] - dts[-2]
    )
    grad_ts = pd.Series(grads, index=timeseries.index)
    return grad_ts


def time_derivative_second(
    timeseries,
):
    """Return the forward second order time-derivative of a time-series.

    Parameters
    ----------
    timeseries : pd.Series
        A timeseries.

    Returns
    -------
    pd.Series :
        The timeseries of double time-derivative of the input series.


    .. important::

        The function assumes that the provided timeseries has at least 3
        elements and that all the elements are ordered following chronological
        order.


    Notes
    -----
    The second order time-derivative of the input timeseries is obtained by
    applying twice the formula used in :py:func:`time_derivative_fwd`.

    .. warning::

        This time-derivative scheme is very crude. Its used is dicouraged in
        physical numerical simulation, in which the accuracy and stability of the
        scheme are of great importance.

        In particular, due to the way the derivative is computed, the last two
        values of the series are always 0, as seen in the example below.


    Example
    -------

    .. code-block:: python

        >>> import pandas as pd
        >>> import numpy as np
        >>> time = pd.date_range(pd.Timestamp("2021-03-14"), freq='1S', periods=10)
        >>> values = np.arange(0, 10)**2
        >>> series = pd.Series(values, index=time)
        >>> series
        2021-03-14 00:00:00     0
        2021-03-14 00:00:01     1
        2021-03-14 00:00:02     4
        ...
        2021-03-14 00:00:07    49
        2021-03-14 00:00:08    64
        2021-03-14 00:00:09    81
        Freq: S, dtype: int64
        >>> time_derivative_second(series)
        2021-03-14 00:00:00    2.0
        2021-03-14 00:00:01    2.0
        2021-03-14 00:00:02    2.0
        ...
        2021-03-14 00:00:07    2.0
        2021-03-14 00:00:08    0.0
        2021-03-14 00:00:09    0.0
        Freq: S, dtype: float64

    """
    dts = (timeseries.index - timeseries.index[0]).total_seconds()
    grads = np.empty(timeseries.index.size)
    grads[0:-1] = np.ediff1d(timeseries.values) / np.ediff1d(dts)
    grads[-1] = (timeseries.values[-1] - timeseries.values[-2]) / (
        dts[-1] - dts[-2]
    )
    grads2 = np.empty(timeseries.index.size)
    grads2[0:-1] = np.ediff1d(grads) / np.ediff1d(dts)
    grads2[-1] = (grads[-1] - grads[-2]) / (dts[-1] - dts[-2])
    grad2_ts = pd.Series(grads2, index=timeseries.index)
    return grad2_ts
