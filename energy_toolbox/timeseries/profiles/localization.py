"""Define Mixins to use manage localized timeseries."""

import pandas as pd
from .mean_profile import MeanProfile
from .rolling_profile import (
    RollingProfile,
    RollingQuantileProfile,
)


class LocalizedProfileMixin:
    """A profile version where time-zoned data is managed, including DSTs.

    The compute method of the profile class with which this mixin is combined
    is overloaded in such a way that passing time-localized data is possible.
    On and across DSTs, the data remains aligned "on the clock" (VS on the sun).
    """

    def compute(
        self,
        history: pd.DataFrame,
        time: pd.Timestamp,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute the profile at ``time`` from the ``history``.

        The computation is managed as follows :

        - just fall back to the parent `compute` method in case the data is time-naive.
        - Else, unlocalize the history and target date before passing them to the
          parent class' `compute` such that the data remains aligned based on "local time"
          for each day.
        - Relocalize the resulting profile to the history timezone. This part takes care of
          DST days :

          * In case the target date is summer DST (23h long day) the extra profile hours is
            just dumped.
          * In case the target date is winter DST(25h long day), the data for the 2am to 3am
            period is associated with the first occurrence of this time-of-day and a gap is
            let on the second.


        .. warning::

          If winter DST is included in the history, the timestamps from 2am to
          3am on this day appears "twice" in the time-naive history passed to the aggregation.
          Therefore, the aggregation should be robust to this situation.


        .. seealso::

            :py:meth:`.MeanProfile.compute`

        """
        source_tz = history.index.tz
        if source_tz is None:
            profile_ref = super().compute(history, time, **kwargs)
        else:
            profile_ref = (
                super()
                .compute(
                    history.tz_localize(None), time.tz_localize(None), **kwargs
                )
                .tz_localize(source_tz, ambiguous=True, nonexistent="NaT")
            )
        return profile_ref


class LocalizedMeanProfile(
    LocalizedProfileMixin,
    MeanProfile,
):
    """Compute the mean profile with timezone data.

    see :py:class:`.MeanProfile`
    """


class LocalizedRollingProfile(
    LocalizedProfileMixin,
    RollingProfile,
):
    """Compute a rolling profile with timezone data.

    see :py:class:`.RollingProfile`
    """


class LocalizedRollingQuantileProfile(
    LocalizedProfileMixin,
    RollingQuantileProfile,
):
    """Compute a rolling quantile with timezone data.

    see :py:class:`.RollingQuantProfile`
    """
