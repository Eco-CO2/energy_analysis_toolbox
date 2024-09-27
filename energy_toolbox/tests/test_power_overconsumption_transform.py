import pandas as pd
from ..power.overconsumption.transform import merge_by_proximity
from .test_power_overconsumption_select import example_intervals


def middle_merged():
    intervals = example_intervals()
    merged = pd.Series(
        [
            intervals["start"].iloc[1],
            intervals["end"].iloc[2],
            (
                intervals["end"].iloc[2] - intervals["start"].iloc[1]
            ).total_seconds(),
            intervals["energy"].iloc[1:3].sum(),
        ],
        index=intervals.columns,
    )
    merged_intervals = pd.DataFrame.from_dict(
        {
            0: intervals.iloc[0, :],
            1: merged,
            2: intervals.iloc[3, :],
        },
        orient="index",
    )
    return merged_intervals


def test_merge_on_empty():
    intervals = example_intervals().iloc[:0, :]
    assert merge_by_proximity(intervals).empty


def test_merge_no_hit():
    """Check case when the threshold is lower than any duration between overshoots"""
    intervals = example_intervals()
    pd.testing.assert_frame_equal(
        intervals,
        merge_by_proximity(intervals, min_interval=60),
        check_like=True,
        check_dtype=False,
    )


def test_merge_all():
    """Check case when the threshold is big enough to merge all overconsumption."""
    intervals = example_intervals()
    merged = merge_by_proximity(intervals, min_interval=3600 * 24)
    expected = pd.DataFrame.from_dict(
        {
            "start": [intervals["start"].iloc[0]],
            "end": [intervals["end"].iloc[-1]],
            "duration": [
                (
                    intervals["end"].iloc[-1] - intervals["start"].iloc[0]
                ).total_seconds()
            ],
            "energy": [intervals["energy"].sum()],
        }
    )
    pd.testing.assert_frame_equal(merged, expected, check_dtype=False)


def test_merge_nominal():
    """Check a normal case with a few hits : middle interval merged"""
    intervals = example_intervals()
    merged = merge_by_proximity(intervals, min_interval=600)
    pd.testing.assert_frame_equal(merged, middle_merged(), check_dtype=False)


def test_merge_touching_interval_zero_tshd():
    """Check robustness in limit case when overconsumption are adjecent."""
    intervals = example_intervals()
    intervals.loc[1, "end"] = intervals.loc[2, "start"]
    merged = merge_by_proximity(intervals, min_interval=0)
    pd.testing.assert_frame_equal(merged, middle_merged(), check_dtype=False)
