"""A profile based on deviation from the mean profile driven by the standard
deviation of the history on each slot.

"""

from ..mean_profile import MeanProfile


class RelativeSTDThreshold(MeanProfile):
    """A class which implements a statistical deviation from the mean profile."""

    def __init__(
        self,
        offset_std=3,
        **kwargs,
    ):
        """

        Parameters
        ----------
        offset_std : float, optional
            Number of standart deviations VS the computed reference to obtain
            the threshold profile. Default is 3 (profile is 3 standard deviations
            from reference)

        """
        self.offset_std = offset_std
        super().__init__(**kwargs)

    def compute(
        self,
        history,
        time,
        **kwargs,
    ):
        """Return a threshold profile.

        The threshold profile is obtained using a user-defined relative variation
        from the mean profile built from history.

        Parameters
        ----------
        history : pd.Series
            Consumption history used to computed the reference
            profile.
        time_day : pd.Timestamp
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
        profile = reference + offset
        return profile
