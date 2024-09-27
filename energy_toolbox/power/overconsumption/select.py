"""This module contains functions which can be used to select overconsumption of
overconsumption according to various criteria :

- :py:func:`by_individual_proportion` selects those which energy content (over
  their reference) is beyond a certain proportion of the total overconsumption
  energy. It can be used to keep only "big enough" overconsumptions.
- :py:func:`by_cumulated_proportion` selects the minimum set of overconsumption which
  are necessary to explain a certain proportion of the total overconsumption
  energy. It can be used when one wants to keep all the overconsumption which explain
  "most of the overconsumption", but can include small non-significant overshoots
  in certain cases.
- :py:func:`by_combined_proportions` is the combination of both previous approaches.
  It enables the selection of the most significant overconsumption to explain "most of
  the overconsumption".

"""


def by_individual_proportion(
    intervals_overshoot,
    proportion_tshd=0.05,
    energy_reference=None,
):
    """Return only overconsumption for which the overshoot energy is above a certain proportion of the total.

    Parameters
    ----------
    intervals_overshoot : pd.DataFrame
        A table of overshoot overconsumption with at least 'start', 'end' and 'energy'
        columns.
    proportion_tshd : float, optional
        Proportion (in [0,1]) of the total energy of the intervals_overshoot.
        An interval is conserved by the function only if it represents at least
        (>=) this proportion of the total. Default is 5%.
    energy_reference : float or None, optional
        The total energy (in the same unit as the values in the ``"energy"``
        column, relatively to which the proportions are computed. Default is None,
        in which case the sum of the column values is used.

    Returns
    -------
    pd.DataFrame :
        A copy of the input dataframe with an additional "proportion" column, where only
        the overconsumption which represent at least the specified proportion of the total energy
        are conserved.
        The returned overconsumption are sorted by decreasing order of overshoot energy.

    """
    intervals_overshoot = intervals_overshoot.copy()
    if energy_reference is None:
        energy_reference = intervals_overshoot["energy"].sum()
    # Select overconsumption with prominent overconsumption
    intervals_overshoot["proportion"] = (
        intervals_overshoot["energy"] / energy_reference
    )
    intervals_overshoot.sort_values(by="energy", ascending=False, inplace=True)
    return intervals_overshoot[
        intervals_overshoot["proportion"] >= proportion_tshd
    ]


def by_cumulated_proportion(
    intervals_overshoot,
    proportion_tshd=0.80,
    energy_reference=None,
):
    """Return only overconsumption which total energy represent a certain proportion of the total
    overshoot energy.

    Parameters
    ----------
    intervals_overshoot : pd.DataFrame
        A table of overshoot overconsumption with at least 'start', 'end' and 'energy'
        columns.
    proportion_tshd : float, optional
        Proportion (in [0,1]) of the total energy of the intervals_overshoot.
        Intervals are sorted by decreasing order of energy and conserved until
        this proportion - at least (>=) - of the total energy is reached.
        Default is 0.8, meaning that the minimum set of overconsumption which represent
        at least 80% of the total overshoot energy is conserved.
    energy_reference : float or None, optional
        The total energy (in the same unit as the values in the ``"energy"``
        column), relatively to which the proportions are computed. Default is None,
        in which case the sum of the column values is used.

    Returns
    -------
    pd.DataFrame :
        A copy of the input dataframe with an additional "cum_energy_prop" column, where only
        the overconsumption which represent together at least the specified proportion of the total
        energy are conserved.
        The returned overconsumption are sorted by decreasing order of overshoot energy.

    """
    intervals_overshoot = intervals_overshoot.sort_values(
        by="energy", ascending=False
    )
    if energy_reference is None:
        energy_reference = intervals_overshoot["energy"].sum()
    intervals_overshoot["cum_energy_prop"] = intervals_overshoot[
        "energy"
    ].cumsum()
    intervals_overshoot["cum_energy_prop"] /= energy_reference
    # ensure at least one value
    last_selected = (
        intervals_overshoot["cum_energy_prop"] < proportion_tshd
    ).argmin() + 1
    selected = intervals_overshoot.iloc[:last_selected, :]
    return selected


def by_combined_proportions(
    intervals_overshoot,
    proportion_tshd=0.80,
    proportion_indiv_tshd=0.05,
    energy_reference=None,
):
    """Return only based on cumulated and individual proportions of the total overshoot energy

    Parameters
    ----------
    intervals_overshoot : pd.DataFrame
        A table of overshoot overconsumption with at least 'start', 'end' and 'energy'
        columns.
    proportion_tshd : float, optional
        Proportion (in [0,1]) of the total energy of the intervals_overshoot.
        Intervals are sorted by decreasing order of energy and conserved until
        this proportion - at least - of ``energy_reference`` is reached.
        Default is 0.8, meaning that the minimum set of intervals which represent
        at least 80% of the total overshoot energy is conserved.
    proportion_indiv_tshd : float, optional
        Proportion (in [0,1]) of the total energy of the intervals_overshoot.
        An interval is conserved by the function only if it represents at least
        this proportion of the ``energy_reference``. Default is 5%.
    energy_reference : float or None, optional
        The total energy (in the same unit as the values in the ``"energy"``
        column, relatively to which the proportions are computed. Default is None,
        in which case the sum of the column values is used.

    Returns
    -------
    pd.DataFrame :
        A copy of the input dataframe with additional "proportion" and "cum_energy_prop"
        columns, where only the overconsumption which represent together and individually
        at least the specified proportions of the total energy, are conserved.
        The returned overconsumption are sorted by decreasing order of overshoot energy.

    """
    if energy_reference is None:
        energy_reference = intervals_overshoot["energy"].sum()
    intervals_overshoot = intervals_overshoot.sort_values(
        by="energy", ascending=False
    )
    intervals_overshoot["cum_energy_prop"] = intervals_overshoot[
        "energy"
    ].cumsum()
    intervals_overshoot["cum_energy_prop"] /= energy_reference
    intervals_overshoot["proportion"] = (
        intervals_overshoot["energy"] / energy_reference
    )
    # ensure at least one value
    last_selected = (
        intervals_overshoot["cum_energy_prop"] < proportion_tshd
    ).argmin() + 1
    selected = intervals_overshoot.iloc[:last_selected, :]
    selected = selected[selected["proportion"] >= proportion_indiv_tshd]
    return selected
