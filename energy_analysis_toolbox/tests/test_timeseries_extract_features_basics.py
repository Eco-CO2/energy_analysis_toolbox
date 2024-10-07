# -*- coding: utf-8 -*-
""" """

import numpy as np
import pandas as pd
import pytest

from ..errors import (
    EATInvalidTimestepDurationError,
    EATUndefinedTimestepError,
    EATEmptyDataError,
)
from .. import keywords as EATK
from ..timeseries.extract_features.basics import (
    timestep_durations,
    intervals_over,
)

# =============================================================================
# timestep_durations
# =============================================================================


def test_evenly_spaced_durations():
    """Check the function on evenly spaced series."""
    series = pd.Series(
        np.ones(100),
        index=pd.date_range(start="2021-01-06", freq="10s", periods=100),
    )
    durs = timestep_durations(series)
    assert (durs == 10).all()
    assert (series.index == durs.index).all()
    durs = timestep_durations(series, last_step=42.0)
    assert durs.iloc[-1] == 42.0
    assert (durs.iloc[:-1] == 10).all()
    assert (series.index == durs.index).all()


def test_randomly_spaced_durations():
    """Check the function on non-evenly spaced series."""
    start = pd.Timestamp.now()
    s = pd.Timedelta("1s")
    series = pd.Series(
        np.ones(4),
        index=pd.DatetimeIndex(
            [
                start,
                start + 10 * s,
                start + 14 * s,
                start + 54 * s,
            ]
        ),
    )
    durs = timestep_durations(series)
    assert np.allclose(durs.values, np.array([10, 4, 40, 40]))
    assert (series.index == durs.index).all()


def test_durations_limit_cases():
    """Check how the functions returns with :

    - series of 0 or one element.
    - < 0s last step duration.

    """
    series = pd.Series(
        np.ones(1),
        index=pd.date_range(start="2021-01-06", freq="10s", periods=1),
    )
    with pytest.raises(EATEmptyDataError):
        timestep_durations(series.iloc[:0])
    with pytest.raises(EATUndefinedTimestepError):
        timestep_durations(series.iloc[:1], last_step=None)
    with pytest.raises(EATInvalidTimestepDurationError):
        timestep_durations(series.iloc[:1], last_step=-42.0)
    durs = timestep_durations(series.iloc[:1], last_step=42.0)
    assert durs.size == 1
    assert durs.loc["2021-01-06"] == 42.0


# =============================================================================
# ----- intervals_over
# =============================================================================


def test_intervals_over():
    """A simple test for intervals_over function."""
    time_begin = pd.Timestamp("2018-07-06 05:00:00")
    time_range = pd.date_range(
        start=time_begin,
        periods=8,
        inclusive="left",
        freq=pd.DateOffset(seconds=300),
    )
    power = pd.Series(
        np.array([0, 1, 1.5, 2.0, 2.0, 3.0, 1.5, 0.0]), index=time_range
    )
    start_f = "start"
    end_f = "end"
    up_loc, up_iloc = intervals_over(power, low_tshd=1.5, return_positions=True)
    assert up_loc.shape[0] == 1
    assert up_iloc.loc[0, start_f] == 3
    assert up_iloc.loc[0, end_f] == 6
    assert up_loc.loc[0, start_f] == pd.Timestamp("2018-07-06 05:15:00")
    assert up_loc.loc[0, end_f] == pd.Timestamp("2018-07-06 05:30:00")


def test_intervals_over_no_measures():
    """Check empty input measures return empty overconsumption."""
    power = pd.Series([])
    assert intervals_over(power, low_tshd=0.0).empty


@pytest.mark.parametrize("timestep", [1.0, 3.0, 5.0])
def test_intervals_over_no_interval(timestep):
    """Check no spurious overconsumption are found"""
    time_begin = pd.Timestamp.now()
    total_duration = 12.0 * 3600.0
    time_range = pd.date_range(
        start=time_begin,
        end=time_begin + pd.Timedelta(seconds=total_duration),
        inclusive="left",
        freq=pd.DateOffset(seconds=timestep),
    )
    power = pd.Series(
        np.zeros((time_range.size), dtype=np.float64), index=time_range
    )
    assert intervals_over(power, low_tshd=0.0).empty


@pytest.mark.parametrize("timestep", [1.0, 3.0, 5.0])
def test_intervals_over_no_interval_noisy(timestep):
    """Check no spurious heatings are found with noisy signal and threshold"""
    time_begin = pd.Timestamp.now()
    total_duration = 12.0 * 3600.0
    time_range = pd.date_range(
        start=time_begin,
        end=time_begin + pd.Timedelta(seconds=total_duration),
        inclusive="left",
        freq=pd.DateOffset(seconds=timestep),
    )
    n_steps = time_range.size
    max_noise = 50.0
    fake_data = np.abs(np.random.normal(0, 10.0, n_steps))
    fake_data = np.where(fake_data < max_noise, fake_data, max_noise)
    fake_series = pd.Series(fake_data, index=time_range)
    assert intervals_over(fake_series, low_tshd=max_noise + 1.0).empty


def test_intervals_over_2_timesteps():
    """Check on a very simple case when the periods last two timesteps"""
    time_begin = pd.Timestamp("2018-07-06 05:00:00")
    time_range = pd.date_range(
        start=time_begin,
        periods=5,
        inclusive="left",
        freq=pd.DateOffset(seconds=300),
    )
    power = pd.Series(np.array([0, 1, 1, 0, 0]) * 1000, index=time_range)
    up_loc = intervals_over(power, low_tshd=0.0)
    start = up_loc[EATK.start_f].iloc[0]
    assert up_loc.shape[0] == 1
    assert start == pd.Timestamp("2018-07-06 05:05:00")
    assert up_loc[EATK.end_f].iloc[0] == pd.Timestamp("2018-07-06 05:15:00")


def test_intervals_over_one_slot_only():
    """Check on a very simple case when the period over threshold lasts one timestep"""
    time_begin = pd.Timestamp("2018-07-06 05:00:00")
    time_range = pd.date_range(
        start=time_begin,
        periods=5,
        inclusive="left",
        freq=pd.DateOffset(seconds=300),
    )
    power = pd.Series(np.array([0, 0, 1, 0, 0]) * 1000, index=time_range)
    up_loc = intervals_over(power, low_tshd=0.0)
    start = up_loc[EATK.start_f].iloc[0]
    assert up_loc.shape[0] == 1
    assert start == pd.Timestamp("2018-07-06 05:10:00")
    assert up_loc[EATK.end_f].iloc[0] == pd.Timestamp("2018-07-06 05:15:00")


def test_intervals_over_last_step_front():
    """Check on a very simple case with period finishing on the last time-step"""
    time_begin = pd.Timestamp("2018-07-06 05:00:00")
    time_range = pd.date_range(
        start=time_begin,
        periods=5,
        inclusive="left",
        freq=pd.DateOffset(seconds=300),
    )
    power = pd.Series(np.array([0, 1, 1, 1, 0]) * 1000, index=time_range)
    up_loc = intervals_over(power, low_tshd=0.0)
    start = up_loc[EATK.start_f].iloc[0]
    assert up_loc.shape[0] == 1
    assert start == pd.Timestamp("2018-07-06 05:05:00")
    assert up_loc[EATK.end_f].iloc[0] == pd.Timestamp("2018-07-06 05:20:00")


def test_intervals_over_left_overlap():
    """Check on a very simple case with period overlaping left bound"""
    time_begin = pd.Timestamp("2018-07-06 05:00:00")
    time_range = pd.date_range(
        start=time_begin,
        periods=5,
        inclusive="left",
        freq=pd.DateOffset(seconds=300),
    )
    power = pd.Series(np.array([1, 1, 1, 0, 0]) * 1000, index=time_range)
    up_loc = intervals_over(power, low_tshd=0.0)
    start = up_loc[EATK.start_f].iloc[0]
    assert up_loc.shape[0] == 1
    assert start == pd.Timestamp("2018-07-06 05:00:00")


def test_intervals_over_right_overlap():
    """Check on a very simple case with period overlaping right bound"""
    time_begin = pd.Timestamp("2018-07-06 05:00:00")
    time_range = pd.date_range(
        start=time_begin,
        periods=5,
        inclusive="left",
        freq=pd.DateOffset(seconds=300),
    )
    power = pd.Series(np.array([0, 0, 1, 1, 1]) * 1000, index=time_range)
    up_loc = intervals_over(power, low_tshd=0.0)
    start = up_loc[EATK.start_f].iloc[0]
    assert up_loc.shape[0] == 1
    assert start == pd.Timestamp("2018-07-06 05:10:00")
    assert up_loc[EATK.end_f].iloc[0] == pd.Timestamp("2018-07-06 05:20:00")


def test_intervals_over_always_over():
    """Check on a very simple case when the value is always over the threshold"""
    time_begin = pd.Timestamp("2018-07-06 05:00:00")
    time_range = pd.date_range(
        start=time_begin,
        periods=5,
        inclusive="left",
        freq=pd.DateOffset(seconds=300),
    )
    power = pd.Series(np.array([1, 1, 1, 1, 1]) * 1000, index=time_range)
    up_loc = intervals_over(power, low_tshd=0.0)
    start = up_loc[EATK.start_f].iloc[0]
    assert up_loc.shape[0] == 1
    assert start == pd.Timestamp("2018-07-06 05:00:00")
    assert up_loc[EATK.end_f].iloc[0] == pd.Timestamp("2018-07-06 05:20:00")
