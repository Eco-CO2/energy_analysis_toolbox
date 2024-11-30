"""The module defines rolling aggregations from history.

The two following sections explain the main steps of the
computations :

1. the Pivot phase
2. the Rolling phase

1. The Pivot Phase
------------------
The first step is to aggregates the different days
of the history align by time.

.. image:: /_static/illustration_rollingProfile_1.png
  :width: 550
  :alt: the Pivot phase

To do this step, we need the timestamps to be aligned.
For instance, the time should be rounded to the minute,
as trailing microseconds could impact the pivot.

2. The Rolling agg Phase
------------------------

The second step is to convert the 2D array into a 1D
array with a rolling window aggregation.

.. image:: /_static/illustration_rollingProfile_2.png
  :width: 250
  :alt: the rolling agg phase

In contrast with most rolling window implementations, here
the window rolls along the time-of-day (wall clock time), but not
along the days: all the history data which time-of-day falls within
the window around the target time-of-day is included in the aggregation.
E.g., with a 60min centered window, the data at time t in the returned
profile is the aggregation of all the data in history which time-of-day
is 30min before or after t, whatever the date.

The window size is a duration (like 60min), that can manage
missing data via ``pandas``.

.. note::

    This look-up using ``pandas`` slices is significantly slow.

    A solution to improve the performance of the rolling profile
    is to make sure that all the expected rows are present (filled with NaNs if
    necessary).

    Using this pre-formatting step, we can use ``numpy`` to slice
    using row indexes.

"""

from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd

from .mean_profile import MeanProfile


class RollingProfile:
    """Compute a profile by aggregating the history on time-periods rolling windows."""

    def __init__(
        self,
        window: int | str | pd.Timedelta,
        aggregation: Callable[[np.ndarray], float],
        as_mean_offset: bool | None = None,
        offset_factor: float = 1.0,
        column_name: str | int = "value",
    ) -> None:
        """Create a Rolling Agg Profile.

        Parameters
        ----------
        window : int, str, pd.Timedelta
            The rolling window size. Any value which can be passed as argument
            for a `pandas` rolling operation is valid.
            The profile will aggregated the values based on time-of-day for any
            date in history in a window of the chosen size, and associate the
            resulting value to the time-of-day of the *center* of the window in the
            produced profile.
        aggregation : function
            An aggregation function which can work on a 1D numpy array. It is applied to
            a raveled 2D horizontal slice of the pivoted history.
        as_mean_offset : bool, optional
            If True, the resulting profile is multiplied by ``offset_factor`` and
            added to the mean profile obtained with the passed data. Default is False.
        offset_factor : float, optional
            The offset factor, see above. By default 1.
        column_name : str | int, Optional
            The name of the column to process.

        """
        if as_mean_offset is None:
            as_mean_offset = False
        self.window = window
        self.agg = aggregation
        self.as_mean_offset = as_mean_offset
        self.column_name = column_name
        if self.as_mean_offset:
            self.reference = MeanProfile()
            self.offset_factor = offset_factor

    def compute(
        self,
        history: pd.DataFrame,
        time: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        """Compute the rolling aggregation profile.

        Parameters
        ----------
        history : pd.DataFrame
            The timeseries of history data, with at least a ``self.column_name``
            column containing the values to be used to create the profile.
            The data should be sampled homogeneously, such that measures are repeated
            at the same time-of-day for every day in the history.
        time : pd.Timestamp
            The time at which the computed profile should start.

        Returns
        -------
        pd.Series
            The computed profile, with same sampling as the history.

        """
        df_day_by_time = self.daily_pivot(history)
        profile = self.windowed_rolling_agg(df_day_by_time)
        profile.index += time
        profile.index.name = history.index.name
        if self.as_mean_offset:
            ref = self.reference.compute(history, time, **kwargs)
            profile = ref.loc[:, [self.column_name]] + self.offset_factor * profile
        return profile

    def daily_pivot(
        self,
        history: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform history data into a table with time-of-day rows and date columns.

        This function reshapes the historical timeseries data into a pivoted format
        where each column represents a day, and each row corresponds to a specific
        time-of-day. This allows for time-of-day-based aggregations over multiple days.

        Parameters
        ----------
        history : pd.DataFrame
            A DataFrame containing the history data, with a DateTimeIndex. It must
            include the column specified by ``self.column_name``, which represents
            the values to be processed.

        Returns
        -------
        pd.DataFrame
            A pivoted DataFrame where:
            - Each column corresponds to a specific day from the history.
            - Each row represents a specific time-of-day.
            - Values are from the column specified by ``self.column_name``.

        Notes
        -----
        This function handles Daylight Saving Time (DST) changes. Specifically:
          - For winter DST transitions (25-hour days), any times beyond 24 hours
            (e.g., the last hour of the day) are dropped to ensure consistency.
          - Duplicated timestamps due to DST transitions are resolved by keeping the
            first occurrence.

        - Times should be rounded to a consistent resolution (e.g., seconds or minutes)
          to avoid pivoting errors due to small differences in time values.


        .. warning::

            Winter DST creates days lasting 25h, which would lead to 25h profiles.
            This function drops any moment in the day which is more than 24h after
            midnight this day, i.e. drops the last hour of the DST.
            This is not perfect but this is considered satisfactory to begin with.


        """
        history = history.copy()
        history["time"] = history.index - history.index.floor("D")
        history["day"] = history.index.date
        try:
            df_day_by_time = pd.pivot_table(
                history,
                index="time",
                columns=["day"],
                values=[self.column_name],
            )
        except ValueError:
            # Happens on winter DST and time-naive data when the same time happens twice
            df_day_by_time = pd.pivot_table(
                history.loc[~history.index.duplicated(keep="first")],
                index="time",
                columns=["day"],
                values=[self.column_name],
            )
        # Deal with winter DST and time-localized data
        df_day_by_time = df_day_by_time.drop(
            labels=df_day_by_time.index[df_day_by_time.index >= pd.Timedelta("1D")],
        )
        return df_day_by_time[self.column_name]

    def windowed_rolling_agg(
        self,
        pivoted_history: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return rolling aggregation over date/time-of-day-window in a pivoted history.

        Parameters
        ----------
        pivoted_history: pd.DataFrame
            The multi column dataframe on which to compute the aggregation. The
            dataframe columns are series of values (usually a date for each column) and
            the rows are times which can be rolled by a duration window.


        Returns
        -------
        aggregated: pd.Dataframe
            The computed aggregation as a table with
            one column named ``self.column_name``.


        .. note::

            Using the setup allows to be sure that the dataframe ``pivoted_history``
            upon which the agg is computed is the desired one (local scope of
            ``custom_agg`` instead of global scope).

        """

        def custom_agg(
            subseries: pd.Series,
        ) -> float:
            """Compute the ``agg`` function on ``pivoted_history`` using subseries.

            For performance issue, slicing and ``ravel`` are used.
            """
            start, end = subseries.index[0], subseries.index[-1]
            values = pivoted_history.loc[start:end, :].to_numpy().ravel()
            return self.agg(values)

        aggregated = (
            pivoted_history.iloc[:, 0]  # select the first column, as we only roll once.
            .rolling(self.window, center=True)
            .apply(custom_agg, raw=False)  # we need the index passed.
        )
        aggregated.name = self.column_name
        return pd.DataFrame(data=aggregated)


class RollingQuantileProfile(RollingProfile):
    """A profile computed by a Quantile of the history on rolling windows."""

    def __init__(
        self,
        window: int | str | pd.Timedelta,
        threshold_quantile: float,
        as_mean_offset: bool | None = None,
        offset_factor: float = 1.0,
        column_name: str | int = "value",
    ) -> None:
        """Create a Rolling Agg Profile.

        Parameters
        ----------
        window : int, str, pd.Timedelta
            The rolling window size. Any value which can be passed as argument
            for a `pandas` rolling operation is valid.
            The profile will aggregated the values based on time-of-day for any
            date in history in a window of the chosen size, and associate the
            resulting value to the time-of-day of the *center* of the window in the
            produced profile.
        threshold_quantile : float
            The quantile value to compute, in [0:1]
        as_mean_offset : bool, optional
            If True, the resulting profile is multiplied by ``offset_factor`` and
            added to the mean profile obtained with the passed data. Default is False.
        offset_factor : float, optional
            The offset factor, see above. By default 1.
        column_name : str | int, Optional
            The name of the column to process.

        """
        if as_mean_offset is None:
            as_mean_offset = False
        aggregation = partial(np.quantile, q=threshold_quantile)
        super().__init__(
            window,
            aggregation,
            as_mean_offset,
            offset_factor,
            column_name,
        )
