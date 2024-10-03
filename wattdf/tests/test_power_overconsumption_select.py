import pytest
import pandas as pd
from ..power.overconsumption.select import (
    by_individual_proportion,
    by_cumulated_proportion,
    by_combined_proportions,
)


def example_intervals():
    """Return 4 overconsumption of overconsumption."""
    s1 = pd.Timestamp("2023-06-21 11:07")
    e1 = pd.Timestamp("2023-06-21 11:14")
    # ------------------------------------
    s2 = pd.Timestamp("2023-06-21 12:00")
    e2 = pd.Timestamp("2023-06-21 13:12")
    # ------------------------------------
    s3 = pd.Timestamp("2023-06-21 13:15")
    e3 = pd.Timestamp("2023-06-21 13:17:30")
    # ------------------------------------
    s4 = pd.Timestamp("2023-06-21 18:42")
    e4 = pd.Timestamp("2023-06-21 18:47")
    # ------------------------------------
    intervals = pd.DataFrame.from_dict(
        {
            "start": [s1, s2, s3, s4],
            "end": [e1, e2, e3, e4],
            "duration": [420.0, 4320.0, 150.0, 300.0],
            "energy": [1500.0, 3000.0, 500.0, 5000.0],  # total 10.000
        }
    )
    return intervals


def test_by_individual_proportions():
    """Check selection by individual proportion on various cases :

    1. proportion equal to the smallest one : all overconsumption selected
    2. intermediate proportion : select only a subset (one here)
    3. proportion higher than the biggest one in overconsumption : empty selection
    4. 3 with custom reference equal to half the total energy : two overconsumption
    """
    intervals = example_intervals()
    # all of them
    case_1 = by_individual_proportion(
        intervals_overshoot=intervals, proportion_tshd=0.05
    )
    case_1_expect = intervals.copy().sort_values(by="energy", ascending=False)
    case_1_expect["proportion"] = [0.5, 0.3, 0.15, 0.05]
    pd.testing.assert_frame_equal(case_1, case_1_expect)
    # only some
    case_2 = by_individual_proportion(
        intervals_overshoot=intervals, proportion_tshd=0.4
    )
    case_2_expect = case_1.copy().iloc[0:1]
    pd.testing.assert_frame_equal(case_2, case_2_expect)
    # None
    case_3 = by_individual_proportion(
        intervals_overshoot=intervals, proportion_tshd=0.6
    )
    # custom reference
    case_4 = by_individual_proportion(
        intervals_overshoot=intervals,
        proportion_tshd=0.6,
        energy_reference=5000.0,
    )
    case_4_expect = case_1_expect.copy().iloc[:2]
    case_4_expect["proportion"] = [1.0, 0.6]
    pd.testing.assert_frame_equal(case_4, case_4_expect)
    assert case_3.empty
    # check not inplace
    with pytest.raises(KeyError):
        intervals["proportion"]


def test_by_cumulated_proportions():
    """Check selection by cumulated proportion on various cases :

    1. proportion equal to 1 : all overconsumption selected
    2. intermediate proportion > biggest: select only a subset (two here)
    3. intermediate proportion < biggest: select biggest overshoot
    4. proportion = 0 : select biggest overshoot anyway
    5. proportion = 0.5 and custom reference equal to twice the total energy :
       select all.

    """
    intervals = example_intervals()
    # all of them
    case_1 = by_cumulated_proportion(
        intervals_overshoot=intervals, proportion_tshd=1.0
    )
    case_1_expect = intervals.copy().sort_values(by="energy", ascending=False)
    case_1_expect["cum_energy_prop"] = [0.5, 0.8, 0.95, 1.0]
    pd.testing.assert_frame_equal(case_1, case_1_expect)
    # only some
    case_2 = by_cumulated_proportion(
        intervals_overshoot=intervals, proportion_tshd=0.6
    )
    case_2_expect = case_1.copy().iloc[0:2]
    pd.testing.assert_frame_equal(case_2, case_2_expect)
    # only one
    case_3 = by_cumulated_proportion(
        intervals_overshoot=intervals, proportion_tshd=0.4
    )
    case_3_expect = case_1.copy().iloc[0:1]
    pd.testing.assert_frame_equal(case_3, case_3_expect)
    # 0 proportion : always at least one value (same as case 3)
    case_4 = by_cumulated_proportion(
        intervals_overshoot=intervals, proportion_tshd=0
    )
    pd.testing.assert_frame_equal(case_4, case_3_expect)
    # custom reference
    case_5 = by_cumulated_proportion(
        intervals_overshoot=intervals,
        proportion_tshd=0.5,
        energy_reference=20000.0,
    )
    case_5_expect = case_1_expect.copy()
    case_5_expect["cum_energy_prop"] /= 2
    pd.testing.assert_frame_equal(case_5, case_5_expect)
    # check not inplace
    with pytest.raises(KeyError):
        intervals["cum_energy_prop"]


def test_by_combines_proportions():
    """Check selection by cumulated proportion on various cases.

    Let (pc, pi) be the combined and individual proportions

    1. (1, 0) : all overconsumption selected
    2. (1, 0.1) : drop the smallest one (0.05)
    3. (0.5, 0.1) : keep only the biggest (0.5)
    4. (0, 1) : no interval
    5. (0, 1) and reference is half of total energy : keep only the biggest (0.5)
    """
    intervals = example_intervals()
    # all of them
    case_1 = by_combined_proportions(
        intervals_overshoot=intervals,
        proportion_tshd=1.0,
        proportion_indiv_tshd=0.0,
    )
    case_1_expect = intervals.copy().sort_values(by="energy", ascending=False)
    case_1_expect["cum_energy_prop"] = [0.5, 0.8, 0.95, 1.0]
    case_1_expect["proportion"] = [0.5, 0.3, 0.15, 0.05]
    pd.testing.assert_frame_equal(case_1, case_1_expect)
    # only some
    case_2 = by_combined_proportions(
        intervals_overshoot=intervals,
        proportion_tshd=1.0,
        proportion_indiv_tshd=0.1,
    )
    case_2_expect = case_1.copy().iloc[0:3]
    pd.testing.assert_frame_equal(case_2, case_2_expect)
    # only one
    case_3 = by_combined_proportions(
        intervals_overshoot=intervals,
        proportion_tshd=0.5,
        proportion_indiv_tshd=0.1,
    )
    case_3_expect = case_1.copy().iloc[0:1]
    pd.testing.assert_frame_equal(case_3, case_3_expect)
    # 0 proportion : always at least one value (same as case 3)
    case_4 = by_combined_proportions(
        intervals_overshoot=intervals,
        proportion_tshd=0.0,
        proportion_indiv_tshd=1.0,
    )
    assert case_4.empty
    # custom reference
    case_5 = by_combined_proportions(
        intervals_overshoot=intervals,
        proportion_tshd=0.0,
        proportion_indiv_tshd=1.0,
        energy_reference=5000.0,
    )
    case_5_expect = case_3_expect.copy()
    case_5_expect["proportion"] = [1.0]
    case_5_expect["cum_energy_prop"] = [1.0]
    pd.testing.assert_frame_equal(case_5, case_5_expect)
    # check not inplace
    with pytest.raises(KeyError):
        intervals["cum_energy_prop"]
