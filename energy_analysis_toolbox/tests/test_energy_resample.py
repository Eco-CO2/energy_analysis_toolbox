import pytest
import pandas as pd
from ..energy import to_freq
from ..errors import EATUndefinedTimestepError


def test_to_freq_no_data():
    """
    Test case for the to_freq function when the energy series is empty.
    It verifies that the returned energy series is also empty.
    """
    energy_series = pd.Series()
    new_freq = "30s"
    new_energy_series = to_freq(energy_series, new_freq)
    assert new_energy_series.empty


def test_to_freq_one_to_many():
    """Test case when the input series has length 1.

    This test case verifies that the `to_freq` function correctly resamples a energy series with only one value to a
    higher frequency.
    """
    energy_series = pd.Series([1.0], index=[pd.Timestamp("2020-01-01")])
    freq = "30min"
    new_freq = "1min"
    with pytest.raises(EATUndefinedTimestepError):
        new_energy_series = to_freq(energy_series, new_freq)
    new_energy_series = to_freq(
        energy_series, new_freq, last_step_duration=pd.Timedelta(freq).seconds
    )
    assert new_energy_series.index.freq == pd.Timedelta(new_freq)
    assert new_energy_series.index[0] == pd.Timestamp("2020-01-01")
    assert len(new_energy_series) == 30
    assert new_energy_series.sum() == 1.0


def test_to_freq_two_to_many():
    """Test case when the input series has length 2.

    Also test that the last step duration is correctly used.
    """
    energy_series = pd.Series(
        [1.0, 1.0],
        index=[pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01 00:30:00")],
    )
    freq = "30min"
    new_freq = "1min"
    expected_energy = energy_series.sum()
    new_energy_series = to_freq(energy_series, new_freq)
    assert new_energy_series.index.freq == pd.Timedelta(new_freq)
    assert new_energy_series.index[0] == pd.Timestamp("2020-01-01")
    assert len(new_energy_series) == 60
    assert new_energy_series.sum() == expected_energy
    # last step duration is force to half the new period
    new_energy_series = to_freq(
        energy_series,
        new_freq,
        last_step_duration=pd.Timedelta(freq).seconds / 2,
    )
    assert len(new_energy_series) == 45
    assert (
        new_energy_series.sum() == expected_energy
    )  # changing the last step duration does not change the energy


def test_to_freq_many_to_one():
    """Test case when the input series has a duration of 30 minutes but is resampled to one element."""
    freq = "1min"
    energy_series = pd.Series(
        [1.0] * 30,
        index=pd.date_range(start="2020-01-01", periods=30, freq=freq),
    )
    new_freq = "1h"
    expected_energy = energy_series.sum()
    new_energy_series = to_freq(energy_series, new_freq)
    assert new_energy_series.index.freq == pd.Timedelta(new_freq)
    assert new_energy_series.index[0] == pd.Timestamp("2020-01-01")
    assert len(new_energy_series) == 1
    assert new_energy_series.sum() == expected_energy
    # last step duration is force to half the new period
    new_energy_series = to_freq(
        energy_series,
        new_freq,
        last_step_duration=pd.Timedelta(freq).seconds / 2,
    )
    assert len(new_energy_series) == 1
    assert (
        new_energy_series.sum() == expected_energy
    )  # changing the last step duration does not change the energy
