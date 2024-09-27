"""This module contains a function to locate the overconsumption from
a power series and a threshold.
"""

import pandas as pd
from ...timeseries.extract_features import intervals_over
from .. import basics as power


def from_power_threshold(
    power_series,
    overshoot_tshd,
    reference_energy_tshd=None,
):
    """Return a table of overconsumption where power_series is above overshoot_tshd.

    The function also computes the overshoot energy as the energy of the difference
    between ``power_series`` and ``reference_energy_tshd`` during the overshoot overconsumption.

    Parameters
    ----------
    power_series : pd.Series
        A timeseries of power measures in (W). The power in each element is
        the averaged power during ``[ti, ti+1[`` where ``ti`` and ``ti+1`` are the
        indices of the considered and next elements.
    overshoot_tshd : pd.Series or float
        The threshold in (W) over which the power is considered as
        over-consumption. In case a series is given, it should have the
        same index as ``power_series``.
    reference_energy_tshd : pd.Series or float or None
        A power in (W) to be substracted from the power series in order
        to compute an "overshoot energy" for each interval. The
        default is |None| in which case ``overshoot_tshd`` is used.
        In case a series is given, it should have the same index as
        ``power_series``.

    Returns
    -------
    pd.DataFrame :
        A table of overconsumption  with the following columns :

        - ``start`` : timestamp of the first instant of an overshoot interval;
        - ``end`` : timestamp of the first instant after an overshoot interval;
        - ``duration`` : duration in (s) of the overshoot interval. This is the
          difference in (s) between the start and end bounds.
        - ``energy`` : the energy associated to the difference between ``power_series``
          and ``reference_energy_tshd`` during the ``[start, end[`` interval.


    .. note:: Why use two thresholds ?

        When looking for overconsumption, the threshold defining the "anomaly" of
        the power may not be the reference VS which the overconsumption is computed.
        E.g., the overconsumption may be computed VS the average consumption while the abnormal
        overconsumption may be identified using variability-related thresholds.

        This function provides both location and "overshoot-energy" computation for convenience.

    """
    if reference_energy_tshd is None:
        reference_energy_tshd = overshoot_tshd
    # Find overshoots of power threshold
    if isinstance(overshoot_tshd, pd.Series):
        power_series_aligned, overshoot_tshd_aligned = power_series.align(
            overshoot_tshd
        )
    else:
        power_series_aligned, overshoot_tshd_aligned = (
            power_series,
            overshoot_tshd,
        )
    intervals_overshoot = intervals_over(
        power_series_aligned > overshoot_tshd_aligned, 0.5
    )
    if intervals_overshoot.empty:
        intervals_overshoot["duration"] = []
        intervals_overshoot["energy"] = []
    else:
        intervals_overshoot["duration"] = (
            intervals_overshoot["end"] - intervals_overshoot["start"]
        ).dt.total_seconds()
        # Compute energy criterion associated to overshoots
        if isinstance(reference_energy_tshd, pd.Series):
            power_series_aligned, reference_energy_tshd_aligned = (
                power_series.align(reference_energy_tshd)
            )
        else:
            power_series_aligned, reference_energy_tshd_aligned = (
                power_series,
                reference_energy_tshd,
            )
        intervals_overshoot["energy"] = power.integrate_over(
            intervals_overshoot, power_series - reference_energy_tshd_aligned
        )
    return intervals_overshoot
