import pytest
import pandas as pd
import numpy as np
from ..errors import EATUndefinedTimestepError
from ..power import integrate_over, to_energy


def _constant_power():
    """Return a power series of constant 1W value across a DST"""
    instants = pd.date_range(
        start=pd.Timestamp("2023-10-29", tz="Europe/Paris"),
        end=pd.Timestamp("2023-10-31", tz="Europe/Paris"),
        freq="1h",
        inclusive="left",
    )
    power_series = pd.Series(np.ones_like(instants), index=instants)
    return power_series


# ==============================================================================
# Test to_energy
# ==============================================================================
def test_to_energy():
    """Check that constant 1W power across DST is correctly transformed to energy."""
    power_series = _constant_power()
    energy = to_energy(power_series)
    expected = power_series * 3600
    pd.testing.assert_series_equal(energy, expected)


def test_to_energy_failure():
    """Check that the function fails on series with only one element."""
    power_series = _constant_power().iloc[:1]
    with pytest.raises(EATUndefinedTimestepError):
        to_energy(power_series)


# ==============================================================================
# Test integrate_over
# ==============================================================================


def _intervals():
    intervals = pd.DataFrame.from_dict(
        {
            "start": [
                pd.Timestamp("2023-10-29 00:00", tz="Europe/Paris"),
                pd.Timestamp("2023-10-30 02:00", tz="UTC+02:00"),
            ],
            "end": [
                pd.Timestamp("2023-10-29 01:00", tz="Europe/Paris"),
                pd.Timestamp("2023-10-30 02:00", tz="UTC+01:00"),
            ],
        }
    )
    intervals.index = ["A", "B"]
    return intervals


def test_integrate_over_empty_intervals():
    """Check that empty overconsumption leads to empty results"""
    no_intervals = _intervals().iloc[:0, :]
    assert integrate_over(no_intervals, _constant_power()).empty


def test_integrate_over():
    """Check that integration works across and DST"""
    intervals = _intervals()
    expected = pd.Series([3600.0, 3600.0], index=["A", "B"])
    pd.testing.assert_series_equal(
        integrate_over(intervals, _constant_power()),
        expected,
    )


def test_integrate_over_not_in_index():
    """Check that the function works when the bounds are not indices of the series"""
    intervals = _intervals()
    intervals["start"] -= pd.Timedelta("10min")
    expected = pd.Series([3600.0, 3600.0], index=["A", "B"])
    pd.testing.assert_series_equal(
        integrate_over(intervals, _constant_power()),
        expected,
    )
    # same on the right. Go one step beyond on the right.
    intervals = _intervals()
    intervals["end"] += pd.Timedelta("10min")
    expected = pd.Series([7200.0, 7200.0], index=["A", "B"])
    pd.testing.assert_series_equal(
        integrate_over(intervals, _constant_power()),
        expected,
    )
