"""Test the eat.timeseries.resample.index_transformation module"""

import pandas as pd
import numpy as np
from ..timeseries.resample.index_transformation import (
    index_to_freq,
    estimate_timestep,
    fill_missing_entries,
    fill_data_holes,
    tz_convert_or_localize,
)

import pytest


@pytest.mark.parametrize(
    "source_tz, target_tz",
    [
        ("UTC", "Europe/Paris"),
        ("Europe/Paris", "UTC"),
        (None, "Europe/Paris"),
        ("Europe/Paris", None),
        (None, None),
    ],
)
def test_tz_convert_localize(source_tz, target_tz):
    """Check that the function works as expected on a series and dataframe"""
    a_timeseries = pd.Series(
        np.arange(10),
        index=pd.date_range("2020-01-01", periods=10, freq="1h", tz=source_tz),
        name="example_series",
    )
    a_dataframe = pd.DataFrame.from_dict(
        {
            "a": a_timeseries,
            "b": a_timeseries * 2,
        }
    )
    if a_timeseries.index.tz is None:
        expected_series = a_timeseries.tz_localize(target_tz)
        expected_df = a_dataframe.tz_localize(target_tz)
    else:
        expected_series = a_timeseries.tz_convert(target_tz)
        expected_df = a_dataframe.tz_convert(target_tz)
    pd.testing.assert_series_equal(
        tz_convert_or_localize(a_timeseries, tz=target_tz), expected_series
    )
    pd.testing.assert_frame_equal(
        tz_convert_or_localize(a_dataframe, tz=target_tz), expected_df
    )


def test_index_to_freq():
    """test the index to freq nominal use"""
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2022-06-15 12:03"),
            pd.Timestamp("2022-06-15 12:05"),
            pd.Timestamp("2022-06-15 12:08"),
            pd.Timestamp("2022-06-15 12:13"),
            pd.Timestamp("2022-06-15 12:19"),
            pd.Timestamp("2022-06-15 12:20"),
        ]
    )
    freq = "1min"
    expected_index = pd.DatetimeIndex(
        pd.date_range(
            start=pd.Timestamp("2022-06-15 12:03"),
            end=pd.Timestamp("2022-06-15 12:20"),
            freq=freq,
        )
    )
    returned_index = index_to_freq(index, freq=freq)
    pd.testing.assert_index_equal(expected_index, returned_index)


def test_estimate_timestep_regular_index():
    """test that the expected timestep is the same for all method when the
    index is regular
    """
    freq = "1min"
    timestep = pd.Timedelta(freq).total_seconds()
    index = pd.DatetimeIndex(
        pd.date_range(
            start=pd.Timestamp("2022-06-15 12:03"),
            end=pd.Timestamp("2022-06-15 12:20"),
            freq=freq,
        )
    )
    assert estimate_timestep(index, "median") == timestep
    assert estimate_timestep(index, "mean") == timestep
    assert estimate_timestep(index, "mode") == timestep
    # assert estimate_timestep(index, "KDE") == timestep  # KDE doesn't work with regularly spaced data


def test_estimate_timestep_missing_entry():
    """Test the expected timestep when one entry is missing"""
    freq = "1min"
    timestep = pd.Timedelta(freq).total_seconds()
    index = pd.DatetimeIndex(
        pd.date_range(
            start=pd.Timestamp("2022-06-15 12:03"),
            end=pd.Timestamp("2022-06-15 12:20"),
            freq=freq,
        )
    )
    index = index.drop(pd.Timestamp("2022-06-15 12:15"))
    assert estimate_timestep(index, "median") == timestep
    assert estimate_timestep(index, "kde") == timestep
    assert estimate_timestep(index, "mode") == timestep
    # assert estimate_timestep(index, "mean") == timestep # this test doesn't work with mean


def test_estimate_timestep_two_frequency():
    """Test that if 2 frequency are present, then only one is returned.
    From tinkering with this test, it looks like it is the last frequency that is returned.
    """
    freq1 = "1min"
    freq2 = "2min"
    timestamp_1 = list(
        pd.date_range(
            start=pd.Timestamp("2022-06-15 12:03"), periods=10, freq=freq1
        )
    )
    timestamp_2 = list(
        pd.date_range(
            start=timestamp_1[-1], periods=10, freq=freq2, inclusive="right"
        )
    )
    index = pd.DatetimeIndex(timestamp_1 + timestamp_2)
    timestep = pd.Timedelta(freq2).total_seconds()
    assert estimate_timestep(index, "median") == timestep
    assert estimate_timestep(index, "kde") == timestep
    assert estimate_timestep(index, "mode") == timestep


def test_estimate_timestep_random_frequency():
    """Test that if the timestep are truly random, the return value still make sense."""
    timestep = 60  # seconds
    size = 1000
    timesteps = np.random.normal(timestep, 2, size=size)
    deltas = np.cumsum(timesteps)
    start = pd.Timestamp("2022-06-15 12:03")
    index = pd.DatetimeIndex([start + pd.Timedelta(seconds=s) for s in deltas])
    atol = 0.1
    rtol = 0.1
    assert np.isclose(
        estimate_timestep(index, "median"), timestep, rtol=rtol, atol=atol
    )
    assert np.isclose(
        estimate_timestep(index, "kde"), timestep, rtol=rtol, atol=atol
    )
    assert np.isclose(
        estimate_timestep(index, "mode"), timestep, rtol=rtol, atol=atol
    )  # I'm quite surprised that this works !
    assert np.isclose(
        estimate_timestep(index, "mean"), timestep, rtol=rtol, atol=atol
    )


def test_fill_missing_entry_series():
    """Test the fill missing entry function"""
    freq = "1min"
    series = pd.Series(
        data=3,
        index=pd.DatetimeIndex(
            pd.date_range(
                start=pd.Timestamp("2022-06-15 12:03"),
                end=pd.Timestamp("2022-06-15 12:20"),
                freq=freq,
            )
        ),
    )
    missing_series = series.drop(pd.Timestamp("2022-06-15 12:15"))
    expected_series = series.copy()
    expected_series.loc[pd.Timestamp("2022-06-15 12:15")] = 42
    fixed_series = fill_missing_entries(
        missing_series, sampling_period=60, security_factor=2, fill_value=42
    )
    pd.testing.assert_series_equal(
        expected_series, fixed_series, check_freq=False
    )


def test_fill_missing_entry_frame():
    freq = "1min"
    df = pd.DataFrame(
        data=3,
        index=pd.DatetimeIndex(
            pd.date_range(
                start=pd.Timestamp("2022-06-15 12:03"),
                end=pd.Timestamp("2022-06-15 12:20"),
                freq=freq,
            ),
        ),
        columns=["a", "b"],
    )
    missing_frame = df.drop(pd.Timestamp("2022-06-15 12:15"))
    expected_frame = df.copy()
    expected_frame.loc[pd.Timestamp("2022-06-15 12:15")] = 42
    fixed_frame = fill_missing_entries(
        missing_frame, sampling_period=60, security_factor=2, fill_value=42
    )
    pd.testing.assert_frame_equal(expected_frame, fixed_frame, check_freq=False)


def test_fill_data_holes():
    """Test the fill data holes function"""
    freq = "1min"
    series = pd.Series(
        data=3,
        index=pd.DatetimeIndex(
            pd.date_range(
                start=pd.Timestamp("2022-06-15 12:03"),
                end=pd.Timestamp("2022-06-15 12:20"),
                freq=freq,
            )
        ),
    )
    missing_series = series.drop(pd.Timestamp("2022-06-15 12:15"))
    expected_series = series.copy()
    expected_series.loc[pd.Timestamp("2022-06-15 12:15")] = 42
    fixed_series = fill_data_holes(
        missing_series, method="median", security_factor=2, fill_value=42
    )
    pd.testing.assert_series_equal(
        expected_series, fixed_series, check_freq=False
    )
