"""Module for computing threshold profiles based on relative deviations.

This module provides tools to calculate profiles that deviate from the mean profile
by a user-defined relative offset. The profiles are designed to scale proportionally
with the baseline consumption pattern derived from historical data.
"""

import pandas as pd

from energy_analysis_toolbox.timeseries.profiles.mean_profile import MeanProfile


class RelativeThreshold(MeanProfile):
    """A class which implements a relative deviation from the mean profile."""

    def __init__(
        self,
        offset_relative: float = 0.5,
        **kwargs,
    ) -> None:
        """Return a threshold profile.

        The threshold profile is obtained using a user-defined relative variation
        from the mean profile built from history.

        Parameters
        ----------
        offset_rel : float, optional
            Relative difference VS the computed reference to obtain
            the threshold profile. Default is 0.5 (profile is 150%
            of reference)

        """
        self.offset_rel = offset_relative
        super().__init__(**kwargs)

    def compute(
        self,
        history: pd.Series,
        time: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        """Return a threshold profile.

        The threshold profile is obtained using a user-defined relative variation
        from the mean profile built from history.

        Parameters
        ----------
        history : pd.Series
            Consumption history used to computed the reference
            profile.
        time : pd.Timestamp
            The time at which the profile is of interest. Only
            the information about the date is used in the passed
            timestamp.

        Returns
        -------
        profile : pd.Series
            Profile threshold with same resolution as the history data.

        Notes
        -----
        The profile threshold is obtained as ``(1 + tshd) *`` the mean
        profile obtained from history.

        """
        reference = super().compute(history, time, **kwargs)
        offset = self.offset_rel * reference
        offset.index = reference.index
        return reference + offset
