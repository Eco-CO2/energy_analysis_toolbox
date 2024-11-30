"""Module for computing threshold profiles based on relative standard deviations.

This module provides tools to calculate profiles that deviate from the mean profile
by a user-defined multiple of the standard deviation derived from historical data.
The profiles aim to capture variability while maintaining a consistent baseline.
"""

import pandas as pd

from energy_analysis_toolbox.timeseries.profiles.mean_profile import MeanProfile


class RelativeSTDThreshold(MeanProfile):
    """A class which implements a statistical deviation from the mean profile."""

    def __init__(
        self,
        offset_std: float = 3,
        **kwargs,
    ) -> None:
        """Initialize RelativeSTDThreshold.

        Parameters
        ----------
        offset_std : float, optional
            Number of standard deviations VS the computed reference to obtain
            the threshold profile. Default is 3 (profile is 3 standard deviations
            from reference)

        """
        self.offset_std = offset_std
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
            Profile threshold with same resolution as the history data

        Notes
        -----
        The profile threshold is obtained as the mean profile obtained
        from history + ``tshd`` times the standard deviation profile.

        """
        profile_groups = self.group(history)
        reference = super().compute(history, time, **kwargs)
        offset = self.offset_std * profile_groups.std()
        offset.index = reference.index
        return reference + offset
