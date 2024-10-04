"""Function to resample a power series."""

import pandas as pd
from ..timeseries.resample.conservative import flow_rate_to_freq


def to_freq(
    series: "pd.Series[float]",
    freq,
    origin=None,
    last_step_duration=None,
) -> "pd.Series[float]":
    """Resample a power series to a given frequency.

    The last step duration of the resampled series is set to the frequency ``freq``.

    .. note::
        Experimental: This function may change or be removed in a future release without warning.


    Parameters
    ----------
    series : pd.Series[float]
        a series of power.
    freq : str
        the frequency to resample to.
    origin : {"floor", "ceil", pd.Timestamp}, optional
        the origin of the resampling, by default None.
        See :py:func:`flow_rate_to_freq` for more details.
    last_step_duration : float, optional
        Duration of the last time-step in the ``volume`` series in (s).
        The default is |None| in which case the duration of the former-last
        time-step is used.
        See :py:func:`flow_rate_to_freq` for more details.

    Returns
    -------
    pd.Series[float]
        the resampled power series.


    .. seealso::

        * :py:func:`energy_analysis_toolbox.timeseries.resample.conservative.flow_rate_to_freq`
        * :py:func:`energy_analysis_toolbox.timeseries.resample.conservative.flow_rate_conservative`

    """
    if series.empty:
        return series
    return flow_rate_to_freq(series, freq, origin, last_step_duration)
