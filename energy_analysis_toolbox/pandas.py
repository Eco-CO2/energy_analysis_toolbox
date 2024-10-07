"""Add an accessor to pandas.DataFrame and pandas.Series for the computation toolbox.

**In order for the accessors to be accessible, this module must be imported.**
The module is not imported by default when importing |eat|::

    import energy_analysis_toolbox as et
    import energy_analysis_toolbox.pandas


Series
------
A pandas series convey only one steam of data. It can be a timeseries of power, energy, temperature, etc.

Hence, the functionalities of the computation toolbox are limited to the following assumptions:

- The index is a datetime index
- The values are numeric
- there is no missing values, i.e. the index is complete, and the value correspond to
  the interval between the index and the next index.
- if not provided as an argument, the last timestep duration is assumed to be
  the same as the previous one.

Knowing this, if the accessor has been enabled, it becomes available on any pandas.Series with
name `eat`. Operations such as the following becom possible::

    my_series.eat.to_freq('1h', method='volume_conservative')
    a_power_series.eat.to_energy()


More examples in :doc:`/user_guide/using_the_accessor`.

Examples of use
~~~~~~~~~~~~~~~

If the accessor has been enabled, it becomes available on any pandas.DataFrame with
name `eat`. Operations such as the following becom possible::

    power_data.eat.power_to_freq('1h')
    power_data.eat.power_to_energy()


More examples in :doc:`/user_guide/using_the_accessor`.

"""

import pandas as pd
from . import (
    power,
    energy,
    timeseries,
)


@pd.api.extensions.register_series_accessor("eat")
class EATAccessorSeries:
    """Define a new namespace for the computation toolbox on pandas.Series."""

    def __init__(
        self,
        pandas_obj,
    ):
        self._obj = pandas_obj


    def to_energy(
        self,
        *args,
        **kwargs,
    ):
        """Convert a power series to an energy series.
        See :func:`energy_analysis_toolbox.power.to_energy` for details.

        Returns
        -------
        pd.Series
            An energy series.
        """
        return power.to_energy(self._obj, *args, **kwargs)


    def to_power(
        self,
        *args,
        **kwargs,
    ):
        """Convert an energy series to a power series.
        See :func:`energy_analysis_toolbox.energy.to_power` for details.

        Returns
        -------
        pd.Series
            A power series.
        """
        raise NotImplementedError("to_power is not implemented yet")


    def power_to_freq(
        self,
        *args,
        **kwargs,
    ):
        """Resample a power series to a fixed frequency.
        See :func:`energy_analysis_toolbox.power.to_freq` for details.

        Returns
        -------
        pd.Series
            A power series resampled to a fixed frequency.
        """
        return power.to_freq(self._obj, *args, **kwargs)


    def energy_to_freq(
        self,
        *args,
        **kwargs,
    ):
        """Resample an energy series to a fixed frequency.
        See :func:`energy_analysis_toolbox.energy.to_freq` for details.

        Returns
        -------
        pd.Series
            An Energy series resampled to a fixed frequency.
        """
        return energy.to_freq(self._obj, *args, **kwargs)


    def to_freq(
        self,
        freq,
        origin=None,
        last_step_duration=None,
        method='piecewise_affine',
        *args,
        **kwargs,
    ):
        """Resample a series to a fixed frequency with various strategies.
        See :func:`energy_analysis_toolbox.timeseries.resample.to_freq` for details.

        

        Here the doc on Parameters

        
        Returns
        -------
        pd.Series
            A series resampled to a fixed frequency.
        """
        return timeseries.resample.to_freq(
            timeseries=self._obj,
            freq=freq,
            origin=origin,
            last_step_duration=last_step_duration,
            method='piecewise_affine',
            *args, 
            **kwargs,
        )


    def intervals_over(
        self,
        *args,
        **kwargs,
    ):
        """Detect intervals over a threshold.
        See :func:`energy_analysis_toolbox.timeseries.extract_features.intervals_over` for details.

        Returns
        -------
        pd.DataFrame
            The intervals over the threshold.
        """
        return timeseries.extract_features.intervals_over(self._obj, *args, **kwargs)


    def timestep_durations(
        self,
        *args,
        **kwargs,
    ):
        """Return the series of timestep durations of a timeseries.
        See :func:`energy_analysis_toolbox.timeseries.timestep_durations` for details.

        Returns
        -------
        pd.Series
            The duration of each timestep.
        """
        return timeseries.extract_features.timestep_durations(self._obj, *args, **kwargs)


    def fill_data_holes(
        self,
        *args,
        **kwargs,
    ):
        """Fill the holes in a timeseries.
        See :func:`energy_analysis_toolbox.timeseries.fill_data_holes` for details.

        Returns
        -------
        pd.Series
            The timeseries with the holes filled.
        """
        return timeseries.resample.fill_data_holes(self._obj, *args, **kwargs)


@pd.api.extensions.register_dataframe_accessor("eat")
class EATAccessorDataFrame:
    """Define a new namespace for the computation toolbox on pandas.Series."""

    def __init__(self, pandas_obj):
        raise NotImplementedError("""EAT accessor for DataFrame is not implemented yet.""")
