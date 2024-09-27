"""
The module defines a base class to compute average load profiles from history.
"""


class MeanProfile:
    """A class which computes a simple mean profile, without special pre/post."""

    def __init__(
        self,
        period="D",
        is_max=True,
        window=1,
        **kwargs,
    ):
        """

        Parameters
        ----------
        period : str, optional
            A pandas period string which specifies the kind of period on which the
            profile is realized.
        is_max : bool, optional
            The type of threshold defined by this profile. If True, defines a
            threshold which defines a maximum value which should not be
            crossed, while False defines a minimum value.
        window : int, optional
            Width (# of slots) of the rolling max window used to transform
            the history used to deduce the mean profile (but not the std).
            Default is 1 slots (meaning no transformation).

        """
        self.period = period
        self.is_max = is_max
        self.window = window

    def group(
        self,
        history,
    ):
        """Group history data by chunks of ``self.period``

        Parameters
        ----------
        history : pd.Series or pd.DataFrame

        Returns
        -------
        pd.GroupBy

        """
        return history.groupby(history.index - history.index.floor(self.period))

    def compute(
        self,
        history,
        time,
        **kwargs,
    ):
        """Return an averaged profile.

        Parameters
        ----------
        history : pd.Series
            Consumption history used to computed the reference
            profile. The history should be sampled homogeneously during the
            aggregated periods.
        time : pd.Timestamp
            The time at which the profile is of interest. Only
            the information about the date floored to ``self.period`` is
            used in the passed timestamp.

        Returns
        -------
        profile : pd.Series
            Profile threshold with same resolution as the history data.

        Notes
        -----
        The profile threshold is obtained as the mean profile obtained
        from history.

        """
        if self.window == 1:
            pass
        elif self.is_max:
            history = history.rolling(self.window, center=True).max()
        else:
            history = history.rolling(self.window, center=True).min()
        mean_profile_compare = self.group(history).mean(numeric_only=True)
        mean_profile_compare.index += time
        return mean_profile_compare
