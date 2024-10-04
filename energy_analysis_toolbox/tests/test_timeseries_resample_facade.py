"""Tests for :py:mod:`energy_analysis_toolbox.timeseries.resample._facade` module."""

import pandas as pd

from ..timeseries.resample._facade import to_freq

example_series = pd.Series(
    [0.0, 1.0],
    index=pd.date_range("2020-01-01 01:00", periods=2, freq="1h"),
    name="example_series",
)


def test_to_freq_piecewise_affine():
    """The function checks the default interpolation strategy."""
    # oversampling
    pd.testing.assert_series_equal(
        to_freq(example_series, "30min"),
        pd.Series(
            [0, 0.5, 1, 1],
            index=pd.date_range("2020-01-01 01:00", periods=4, freq="30min"),
            name="example_series",
        ),
    )
    # subsampling
    pd.testing.assert_series_equal(
        to_freq(example_series, "2h"),
        pd.Series(
            [0.0],
            index=pd.date_range("2020-01-01 01:00", periods=1, freq="2h"),
            name="example_series",
        ),
    )
    pd.testing.assert_series_equal(
        to_freq(example_series, "2h", origin="floor"),
        pd.Series(
            [0.0, 1.0],
            index=pd.date_range("2020-01-01 00:00", periods=2, freq="2h"),
            name="example_series",
        ),
    )
    pd.testing.assert_series_equal(
        to_freq(example_series, "3h", origin="ceil"),
        pd.Series(
            [1.0],
            index=pd.date_range("2020-01-01 03:00", periods=1, freq="3h"),
            name="example_series",
        ),
    )
    assert to_freq(example_series.iloc[:0], "1h").empty


def test_to_freq_piecewise_constant():
    """The function checks the default interpolation strategy."""
    # oversampling
    pd.testing.assert_series_equal(
        to_freq(example_series, "30min", method="piecewise_constant"),
        pd.Series(
            [0.0, 0.0, 1, 1],
            index=pd.date_range("2020-01-01 01:00", periods=4, freq="30min"),
            name="example_series",
        ),
    )
    # subsampling
    pd.testing.assert_series_equal(
        to_freq(example_series, "2h", method="piecewise_constant"),
        pd.Series(
            [0.0],
            index=pd.date_range("2020-01-01 01:00", periods=1, freq="2h"),
            name="example_series",
        ),
    )
    pd.testing.assert_series_equal(
        to_freq(
            example_series, "2h", origin="floor", method="piecewise_constant"
        ),
        pd.Series(
            [
                0.0,
                1,
            ],
            index=pd.date_range("2020-01-01 00:00", periods=2, freq="2h"),
            name="example_series",
        ),
    )
    pd.testing.assert_series_equal(
        to_freq(
            example_series, "3h", origin="ceil", method="piecewise_constant"
        ),
        pd.Series(
            [1.0],
            index=pd.date_range("2020-01-01 03:00", periods=1, freq="3h"),
            name="example_series",
        ),
    )
    assert to_freq(example_series.iloc[:0], "1h").empty


def test_to_freq_volume_conservative():
    """The function just checks that the right method is called as it is just a
    wrapper.

    """
    # oversampling
    pd.testing.assert_series_equal(
        to_freq(
            example_series,
            "30min",
            method="volume_conservative",
            last_step_duration=3600 * 2,
        ),
        pd.Series(
            [0.0, 0.0, 0.25, 0.25, 0.25, 0.25],
            index=pd.date_range("2020-01-01 01:00", periods=6, freq="30min"),
            name="example_series",
        ),
    )


def test_to_freq_flow_rate_conservative():
    """The function just checks that the right method is called as it is just a
    wrapper.

    """
    # oversampling
    pd.testing.assert_series_equal(
        to_freq(
            example_series,
            "30min",
            method="flow_rate_conservative",
            last_step_duration=3600 * 2,
            origin=pd.Timestamp("2020-01-01 00:00"),
        ),
        pd.Series(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            index=pd.date_range("2020-01-01 00:00", periods=8, freq="30min"),
            name="example_series",
        ),
    )


def test_to_freq_custom_method():
    """Test the function when passing a custom resamplig function."""

    def custom(timeseries, *args, **kwargs):
        return timeseries.resample(kwargs["freq"], origin="start").mean()

    pd.testing.assert_series_equal(
        to_freq(example_series, "2h", method=custom),
        pd.Series(
            [
                0.5,
            ],
            index=pd.date_range("2020-01-01 01:00", periods=1, freq="2h"),
            name="example_series",
        ),
    )
