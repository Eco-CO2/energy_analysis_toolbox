import pytest
import pandas as pd
from ..power import to_freq, to_energy
from ..errors import EATUndefinedTimestepError


def test_to_freq_no_data():
    """
    Test case for the to_freq function when the power_series is empty.
    It verifies that the new_power_series is also empty.
    """
    power_series = pd.Series()
    new_freq = "30s"
    new_power_series = to_freq(power_series, new_freq)
    assert new_power_series.empty


def test_to_freq_one_to_many():
    """Test case when the input series has length 1.

    This test case verifies that the `to_freq` function correctly resamples a power series with only one value to a
    higher frequency.
    """
    power_series = pd.Series([1.0], index=[pd.Timestamp("2020-01-01")])
    freq = "30min"
    new_freq = "1min"
    with pytest.raises(EATUndefinedTimestepError):
        new_power_series = to_freq(power_series, new_freq)
    new_power_series = to_freq(
        power_series, new_freq, last_step_duration=pd.Timedelta(freq).seconds
    )
    assert new_power_series.index.freq == pd.Timedelta(new_freq)
    assert new_power_series.index[0] == pd.Timestamp("2020-01-01")
    assert len(new_power_series) == 30
    assert to_energy(new_power_series).sum() == pd.Timedelta(freq).seconds


def test_to_freq_two_to_many():
    """Test case when the input series has length 2.

    Also test that the last step duration is correctly used.
    """
    power_series = pd.Series(
        [1.0, 1.0],
        index=[pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01 00:30:00")],
    )
    freq = "30min"
    new_freq = "1min"
    new_power_series = to_freq(power_series, new_freq)
    assert new_power_series.index.freq == pd.Timedelta(new_freq)
    assert new_power_series.index[0] == pd.Timestamp("2020-01-01")
    assert len(new_power_series) == 60
    assert to_energy(new_power_series).sum() == 2 * pd.Timedelta(freq).seconds
    # last step duration is force to half the new period
    new_power_series = to_freq(
        power_series,
        new_freq,
        last_step_duration=pd.Timedelta(freq).seconds / 2,
    )
    assert len(new_power_series) == 45
    assert to_energy(new_power_series).sum() == 1.5 * pd.Timedelta(freq).seconds


def test_to_freq_many_to_one():
    """Test case when the input series has a duration of 30 minutes but is resample to one element."""
    freq = "1min"
    power_series = pd.Series(
        [1.0] * 30,
        index=pd.date_range(start="2020-01-01", periods=30, freq=freq),
    )
    new_freq = "1h"
    new_power_series = to_freq(power_series, new_freq)
    assert new_power_series.index.freq == pd.Timedelta(new_freq)
    assert new_power_series.index[0] == pd.Timestamp("2020-01-01")
    assert len(new_power_series) == 1
    assert (
        new_power_series.sum() * pd.Timedelta(new_freq).seconds
        == pd.Timedelta(freq).seconds * 30
    )
    # last step duration is force to half the new period
    new_power_series = to_freq(
        power_series,
        new_freq,
        last_step_duration=pd.Timedelta(freq).seconds / 2,
    )
    assert len(new_power_series) == 1
    assert (
        new_power_series.sum() * pd.Timedelta(new_freq).seconds
        == pd.Timedelta(freq).seconds * 29.5
    )


def test_to_freq_one_to_one():
    """Test case when the input series has length 1 and is resampled to one element."""
    freq = "1min"
    power_series = pd.Series(
        [1.0], index=pd.date_range(start="2020-01-01", periods=1, freq=freq)
    )
    new_freq = "1h"
    new_power_series = to_freq(
        power_series, new_freq, last_step_duration=pd.Timedelta(freq).seconds
    )
    assert new_power_series.index.freq == pd.Timedelta(new_freq)
    assert new_power_series.index[0] == pd.Timestamp("2020-01-01")
    assert len(new_power_series) == 1
    assert (
        new_power_series.sum() * pd.Timedelta(new_freq).seconds
        == pd.Timedelta(freq).seconds
    )
