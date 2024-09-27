"""This module is dedicated to transforming overconsumption of overconsumption after
they have beem located.
"""

from ...timeseries.extract_features import (
    timestep_durations,
    intervals_over,
)
from ...timeseries.create.from_intervals import flatten_and_fill


def merge_by_proximity(
    intervals_overshoot,
    min_interval=600,
):
    """Return a table where the overconsumption separated by less than min_interval seconds are merged.

    Parameters
    ----------
    intervals_overshoot : pd.DataFrame
        A table of overshoot overconsumption with at least 'start', 'end' and 'energy'
        columns.
    min_interval : float, optional
        The minimum duration in (s) to be imposed between two overshoot overconsumption.
        All overconsumption separated by a duration under this threshold are merged.
        Default is 600 seconds corresponding to 10 minutes.

    Returns
    -------
    overconsumption : pd.DataFrame
        A table of overconsumption with 'start', 'end' and 'energy' obtained after
        merging the time-neighboring overconsumption in ``intervals_overshoot``.


    .. note::

        The energy of the result of the merge between two overconsumption is the sum
        of the interval energies.

    Notes
    -----
    The function proceeds as follows :

    - [1] flatten the overconsumption to a table of timeseries for each variable. Fill
      all values between overconsumption with zeros.
    - [2] Recompute the durations so that the "duration" variable has the right
      value between the overshoot overconsumption.
    - [3] Drop all the rows for which the duration is under the defined threshold
      and the energy is 0. By construction, overshoot overconsumption hae a non-zero
      energy (they are overshoots!) while other rows were filled with 0. Overshoots
      which were closed than the threshold are now contiguous in the timeseries.
    - [4] Re-extract the periods during which the energy is > 0 : the contiguous
      overconsumption are merged by this process.
    - [5] Recompute the duration and energy for the new overconsumption.

    Some limit cases are managed :

    - empty data is returned directly
    - contiguous overconsumption (which are a limit limit case) are managed by dropping
      the "interstitial" row with duplicate index which appears in the flattened
      overconsumption. This case should not be encountered.

    """
    if intervals_overshoot.empty:
        return intervals_overshoot
    # [1]
    flat_intervals = flatten_and_fill(intervals_overshoot.sort_index()).fillna(
        0
    )
    # [2]
    flat_intervals["duration"] = timestep_durations(
        flat_intervals["duration"],
        last_step=flat_intervals["duration"].iloc[-1],
    )
    # Adjacent overconsumption create duplicates
    flat_intervals = flat_intervals.loc[
        ~flat_intervals.index.duplicated(keep="first")
    ]
    # [3]
    dropped = flat_intervals.iloc[:-1, :].query(
        f"energy == 0 and duration < {min_interval}"
    )
    flat_intervals.drop(index=dropped.index, inplace=True)
    flat_intervals["duration"] = timestep_durations(flat_intervals["duration"])
    # [4]
    intervals = intervals_over(flat_intervals["energy"], 0.0)
    # [5]
    intervals["duration"] = (
        intervals["end"] - intervals["start"]
    ).dt.total_seconds()
    # [5] inclusive loc but energy filled with 0 between the overshoots.
    intervals["energy"] = intervals.apply(
        lambda x: (flat_intervals.loc[x["start"] : x["end"], "energy"].sum()),
        axis=1,
    )
    return intervals
