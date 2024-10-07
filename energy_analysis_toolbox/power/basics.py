"""
This module contains elementary yet useful operations on power timeseries.
"""

from ..timeseries.extract_features import timestep_durations


def integrate_over(
    intervals,
    power_series,
):
    """Return the energy integrated over slices of the power series.

    .. warning::

         This function only works on slices of the power series.
         It cannot be used to integrate the power with smaller
         granularity.


    Parameters
    ----------
    intervals : pd.DataFrame
        A table of overconsumption defined with 'start' and 'end'
        columns containing timestamps. **These timestamps are used as slice**
        **bounds in the series** with included start and excluded end.
    power_series : pd.Series
        A timeseries of power in (W). The series must have at least
        two elements so that the timestep(s) can be deduced.

    Returns
    -------
    energies : pd.Series
        A series of energy in (J) with same index as ``overconsumption``.


    .. note::

        Proper integration with a resolution independent from the timestep can
        be achieved by transforming the power to energy using :py:func:`to_energy`
        and then using the |volume_conservative| resampling.


    Examples
    --------
    >>> power = _constant_power(); power
    2023-10-29 00:00:00+02:00    1
    2023-10-29 01:00:00+02:00    1
    ...
    2023-10-30 22:00:00+01:00    1
    2023-10-30 23:00:00+01:00    1
    Freq: H, dtype: object
    >>> overconsumption = _intervals()
    >>> overconsumption
                           start                        end
    A  2023-10-29 00:00:00+02:00  2023-10-29 01:00:00+02:00
    B  2023-10-30 02:00:00+02:00  2023-10-30 02:00:00+01:00
    >>> eat.power.integrate_over(overconsumption, power)
    A    3600.0
    B    3600.0
    dtype: float64

    Remind that the interval bounds are used to slice the series (with excluded
    end). Accordingly, an interval is completely accounted for in the integration
    as soon as it falls with the slice :
    >>> overconsumption['end'] += pd.Timedelta('10min')
    >>> eat.power.integrate_over(overconsumption, power)
    A    7200.0
    B    7200.0
    dtype: float64


    """
    timesteps = timestep_durations(power_series)

    def integrate(row):
        """Integrate on a slice with excluded end"""
        i_start = timesteps.index.get_slice_bound(row["start"], side="left")
        i_end = timesteps.index.get_slice_bound(row["end"], side="left")
        res = (
            timesteps.iloc[i_start:i_end] * power_series.iloc[i_start:i_end]
        ).sum()
        return res

    energies = intervals.apply(integrate, axis=1)
    return energies


def to_energy(
    power_series,
):
    """Return a series of energy per timestep.

    Parameters
    ----------
    power_series : pd.Series
        Timeseries of (avg) power per ``[ti, ti+1[`` timestep in (W).
        The series must have at least two elements so that the timestep(s)
        can be deduced. The last timestep is assumed to be the same as the
        former last one.

    Returns
    -------
    pd.Series :
        Timeseries of energy per timestep in (J) with same index
        as input series.

    """
    timesteps = timestep_durations(power_series)
    return timesteps * power_series
