"""Add an accessor to pandas.DataFrame and pandas.Series for the computation toolbox.

**In order for the accessors to be accessible, this module must be imported.**
The module is not imported by default when importing |eat|::

    import energy_analysis_toolbox as et
    import energy_analysis_toolbox.pandas


Series
------
A pandas series convey only one steam of data. It can be a timeseries of power, energy,
temperature, etc.

Hence, the functionalities of the computation toolbox are limited to the following
assumptions:

- The index is a datetime index
- The values are numeric
- There is no missing values, i.e. the index is complete, and the value correspond to
  the interval between the index and the next index.
- If not provided as an argument, the last timestep duration is assumed to be
  the same as the previous one.

Knowing this, if the accessor has been enabled, it becomes available on any
pandas.Series with name `eat`. Operations such as the following becom possible::

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

from typing import Literal

import pandas as pd

from energy_analysis_toolbox.timeseries.resample._facade import resampling_methods

from . import (
    energy,
    power,
    timeseries,
)


@pd.api.extensions.register_series_accessor("eat")
class EATAccessorSeries:
    """Define a new namespace for the computation toolbox on pandas.Series."""

    def __init__(
        self,
        data: pd.Series,
    ) -> None:
        if not isinstance(data, pd.Series):
            err = f"Expected the input to be a Series, but got {type(data)}."
            raise TypeError(err)
        self._obj = data

    def to_energy(
        self,
        *args,
        **kwargs,
    ) -> pd.Series:
        """Convert a power series to an energy series.

        See :func:`energy_analysis_toolbox.power.to_energy` for details.

        Returns
        -------
        pd.Series
            An energy series.

        """
        return power.to_energy(
            self._obj,
            *args,
            **kwargs,
        )

    def to_power(
        self,
        *args,  # noqa:ARG002
        **kwargs,  # noqa:ARG002
    ) -> pd.Series:
        """Convert an energy series to a power series.

        See :func:`energy_analysis_toolbox.energy.to_power` for details.

        Returns
        -------
        pd.Series
            A power series.

        """
        err = "to_power method is not implemented yet."
        raise NotImplementedError(err)

    def power_to_freq(
        self,
        *args,
        **kwargs,
    ) -> pd.Series:
        """Resample a power series to a fixed frequency.

        See :func:`energy_analysis_toolbox.power.to_freq` for details.

        Returns
        -------
        pd.Series
            A power series resampled to a fixed frequency.

        """
        return power.to_freq(
            self._obj,
            *args,
            **kwargs,
        )

    def energy_to_freq(
        self,
        *args,
        **kwargs,
    ) -> pd.Series:
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
        freq: str | pd.Timedelta | None,
        origin: None | Literal["floor", "ceil"] | pd.Timestamp = None,
        last_step_duration: float | None = None,
        method: resampling_methods = "piecewise_affine",
        **kwargs,
    ) -> pd.Series:
        """Resample a series to a fixed frequency with various strategies.

        See :func:`energy_analysis_toolbox.timeseries.resample.to_freq` for details.

        Parameters
        ----------
        freq : str, pd.Timedelta
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

        method : str or callable, optional
            Method used to interpolate the values of the resampled series. The accepted
            values are:

            * 'piecewise_affine': uses :py:func:`.piecewise_affine`, assume the values
              a straight line between two points. The default method.
            * 'piecewise_constant': uses :py:func:`.piecewise_constant`, assume the
              values constante until the next point.
            * 'volume_conservative': uses :py:func:`.volume_to_freq`, conserve
              quantity of the values. Best to use it for energy timeseries.
            * 'flow_rate_conservative': uses :py:func:`.flow_rate_to_freq`, conserve
              the values time the duration between two points. Best to use it for
              power timeseries.

            If a callable is passed, it must take a :py:class:`pandas.Series` as first
            argument and a :py:class:`pandas.DatetimeIndex` as second argument.
            See the interface of :py:func:`piecewise_affine` function.
            The default is 'piecewise_affine'.

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
            method=method,
            **kwargs,
        )

    def intervals_over(
        self,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """Detect intervals over a threshold.

        See :func:`energy_analysis_toolbox.timeseries.extract_features.intervals_over`
        for details.

        Returns
        -------
        pd.DataFrame
            The intervals over the threshold.

        """
        return timeseries.extract_features.intervals_over(
            self._obj,
            *args,
            **kwargs,
        )

    def timestep_durations(
        self,
        *args,
        **kwargs,
    ) -> pd.Series:
        """Return the series of timestep durations of a timeseries.

        See :func:`energy_analysis_toolbox.timeseries.timestep_durations` for details.

        Returns
        -------
        pd.Series
            The duration of each timestep.

        """
        return timeseries.extract_features.timestep_durations(
            self._obj,
            *args,
            **kwargs,
        )

    def fill_data_holes(
        self,
        *args,
        **kwargs,
    ) -> pd.Series:
        """Fill the holes in a timeseries.

        See :func:`energy_analysis_toolbox.timeseries.fill_data_holes` for details.

        Returns
        -------
        pd.Series
            The timeseries with the holes filled.

        """
        return timeseries.resample.fill_data_holes(
            self._obj,
            *args,
            **kwargs,
        )


@pd.api.extensions.register_dataframe_accessor("eat")
class EATAccessorDataFrame:
    """Define a new namespace for the computation toolbox on pandas.Series."""

    def __init__(
        self,
        data: pd.DataFrame,  # noqa:ARG002
    ) -> None:
        err = """EAT accessor for DataFrame is not implemented yet."""
        raise NotImplementedError(err)
