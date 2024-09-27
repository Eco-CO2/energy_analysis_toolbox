import pandas as pd
import numpy as np
from ..power.overconsumption.find import from_power_threshold


def example_power():
    """Return an example power series, threshold and overshoot overconsumption.

    The power is sampled with 30min period such that any error on the bounds by
    +/- 1 step is pretty obvious in terms of energy.
    The power is piecewise-constant, equal to 1W everywhere except :

    - during 1h from 10:00 to 11:00 when it is 11W
    - during 1h30min from 14:00 to 15:30 when it is 21W.

    """
    times = pd.date_range(
        start=pd.Timestamp("2023-03-26", tz="Europe/Paris"),  # DST
        end=pd.Timestamp("2023-03-27", tz="Europe/Paris"),
        inclusive="left",
        freq="30min",
    )
    pow = pd.Series(np.ones_like(times), index=times)
    s1 = pd.Timestamp("2023-03-26 10:00", tz="Europe/Paris")
    e1 = pd.Timestamp("2023-03-26 10:30", tz="Europe/Paris")
    s2 = pd.Timestamp("2023-03-26 14:00", tz="Europe/Paris")
    e2 = pd.Timestamp("2023-03-26 15:00", tz="Europe/Paris")
    threshold = pow.copy()
    pow.loc[s1:e1] = 11  # inclusive loc
    pow.loc[s2:e2] = 21  # inclusive loc
    intervals = pd.DataFrame.from_dict(
        {
            "start": [s1, s2],
            "end": [e1, e2],
            "duration": [3600, 5400],
            "energy": [3600 * 10, 5400 * 20],
        }
    )
    intervals["end"] += pd.Timedelta("30min")  # doom pandas inclusive loc
    return pow, threshold, intervals


def test_from_power_threshold_constant():
    """Check the function with constant threshold as float and series"""
    power, threshold, intervals = example_power()
    # constant as series
    found_intervals = from_power_threshold(
        power,
        overshoot_tshd=threshold,
    )
    pd.testing.assert_frame_equal(found_intervals, intervals, check_dtype=False)
    # constant as float
    found_intervals = from_power_threshold(
        power,
        overshoot_tshd=1.0,
    )
    pd.testing.assert_frame_equal(found_intervals, intervals, check_dtype=False)


def test_from_power_threshold_over_power():
    """Check the function with threshold over max power"""
    power, _, _ = example_power()
    assert from_power_threshold(power, 100.0).empty
    assert from_power_threshold(power, power.max()).empty


def test_from_power_threshold_under_power():
    """Check the function with threshold under min power.

    Double limit-case :

    - DST : the day is shorter than 24h
    - Whole day : the end of the overconsumption is the last instant in the
      series, as the moment after is not known. (should be the end of the timestep).
    """
    power, _, intervals = example_power()
    found_intervals = from_power_threshold(power, 0.0)
    intervals_expect = pd.DataFrame.from_dict(
        {
            "start": [power.index[0]],
            "end": [power.index[-1]],
            "duration": 1800 * 45,
            "energy": 1800 * 45 + intervals["energy"].sum(),
        }
    )
    pd.testing.assert_frame_equal(
        found_intervals, intervals_expect, check_dtype=False
    )


def test_from_power_threshold_series():
    """Check that the function works correctly with a variable threshold"""
    power, threshold, intervals = example_power()
    threshold.loc[: pd.Timestamp("2023-03-26 12:00", tz="Europe/Paris"),] = 11
    found_intervals = from_power_threshold(
        power,
        overshoot_tshd=threshold,
    )
    pd.testing.assert_frame_equal(
        found_intervals, intervals[1:].reset_index(drop=True), check_dtype=False
    )


def test_from_power_threshold_custom_ref():
    """Check that the custom energy reference is correctly accounted for"""
    power, threshold, intervals = example_power()
    ref = threshold.copy()
    threshold.loc[:] = 11
    found_intervals = from_power_threshold(
        power,
        overshoot_tshd=threshold,
        reference_energy_tshd=ref,
    )
    pd.testing.assert_frame_equal(
        found_intervals, intervals[1:].reset_index(drop=True), check_dtype=False
    )
