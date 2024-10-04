"""Test the degree days module."""

import pandas as pd
import numpy as np
from energy_analysis_toolbox.weather.degree_days import (
    dd_mean,
    dd_min_max,
    dd_integral,
    dd_compute,
    dd_calc_method,
    dd_pro,
)


def test_dd_mean_basic_features():
    """Verifies that, when computing degree-days, the name property is set and that we have 7 days."""
    index = pd.date_range(
        start="2023-01-21", periods=24 * 60 * 7, freq="1min", tz="Europe/Paris"
    )
    data = [0] * len(index)
    temperature = pd.Series(
        data=data,
        index=index,
    )
    dd = dd_mean(temperature, reference=15)
    assert dd.name == "heating_degree_days"
    assert len(dd) == 7
    assert dd.dtype == "float64"


def test_dd_mean_missing_data():
    """Verifies that, even a data is set to ``NaN``, the degree-days goes well."""
    index = pd.date_range(
        start="2023-01-21", periods=12 * 60, freq="1min", tz="Europe/Paris"
    )
    data = [0] * len(index)
    temperature = pd.Series(
        data=data,
        index=index,
    )
    temperature.iloc[0] = None
    dd = dd_mean(temperature, reference=15)
    expected_dd = pd.Series(
        [15.0],
        name="heating_degree_days",
        index=pd.date_range(
            start="2023-01-21", periods=1, freq="1D", tz="Europe/Paris"
        ),
    )
    pd.testing.assert_series_equal(dd, expected_dd)


def test_dd_min_max():
    """Verifies that, when computing degree-days, the name property is set and that we have 7 days."""
    index = pd.date_range(
        start="2023-01-21", periods=24 * 60 * 7, freq="1min", tz="Europe/Paris"
    )
    data = [0] * len(index)
    temperature = pd.Series(
        data=data,
        index=index,
    )
    dd = dd_min_max(temperature, reference=15)
    assert dd.name == "heating_degree_days"
    assert len(dd) == 7
    assert dd.dtype == "float64"


def test_dd_integral():
    """Verifies that, when computing degree-days, the name property is set and that we have 7 days."""
    index = pd.date_range(
        start="2023-01-21", periods=24 * 60 * 7, freq="1min", tz="Europe/Paris"
    )
    data = [0] * len(index)
    temperature = pd.Series(
        data=data,
        index=index,
    )
    dd = dd_integral(temperature, reference=15)
    assert dd.name == "heating_degree_days"
    assert len(dd) == 7
    assert dd.dtype == "float64"


def test_dd_compute():
    """Verifies that, when computing degree-days, the name property is set and that we have 7 days."""
    index = pd.date_range(
        start="2023-01-21", periods=24 * 60 * 7, freq="1min", tz="Europe/Paris"
    )
    data = [0] * len(index)
    temperature = pd.Series(
        data=data,
        index=index,
    )
    dd = dd_compute(temperature, reference=15, method="integral")
    assert dd.name == "heating_degree_days"
    assert len(dd) == 7
    assert dd.dtype == "float64"


def generate_days_sin_data(n_periods=2, frequency="1h"):
    """Generate one week of sinusoidal data."""
    start = pd.Timestamp("2023-01-21 00:00:00", tz="Europe/Paris")
    end = start + pd.Timedelta(days=n_periods, seconds=-1)
    index = pd.date_range(
        start=start,
        end=end,
        freq=frequency,
    )
    number_points = len(index)
    data = np.sin(np.linspace(0, 2 * np.pi * n_periods, number_points))
    temperature = pd.Series(
        data=data,
        index=index,
    )
    return temperature


def reference_sin_dd(mean, spread, reference=15):
    """Compute the degree days with the integral method
    for a sin function.

    .. note::

        This function uses the formula:

        .. math::

            \\int_{t_0}^{t_1} \\left( sin(t) - sin(t_0) \\right() dt = cos(t_0) - cos(t_1) + (t_1 - t_0) sin(t_0)

        The ``mean`` and ``spread`` parameters are used to scale the values.
    """
    if mean - spread > reference:
        return 0
    if mean + spread <= reference:
        return reference - mean
    ref_norm = (reference - mean) / spread
    # first time when temperature is below reference
    t0 = np.arcsin(-ref_norm)
    # second time when temperature is above reference
    t1 = np.pi - t0
    expected_dd = np.cos(t0) - np.cos(t1) + (t1 - t0) * ref_norm
    return expected_dd * spread / (2 * np.pi)


def test_computation_realistic_data_integral(spread=10, mean=15):
    """Test the value of the computation of the degree_days using integral
    for a Sin temperature evolution.

    """
    # using high sampling to reduce the error on the integral
    # as we use the rectangle integral method
    temperature = generate_days_sin_data(frequency="1min") * spread + mean
    ref_temperatures = np.linspace(5, 30, 20)
    for ref in ref_temperatures:
        # using mean as there are multiple days in the data
        computed_dd = dd_compute(
            temperature, reference=ref, method="integral"
        ).mean()
        # putting the reference computation in the assert to better debug the test
        assert np.isclose(
            computed_dd,
            reference_sin_dd(mean, spread, reference=ref),
            rtol=1e-2,
        )


def test_computation_realistic_data_pro(spread=10, mean=15):
    """Test the value of the computation of the degree_days using the pro method
    for a Sin temperature evolution.

    """
    temperature = generate_days_sin_data(frequency="60min") * spread + mean
    ref_temperatures = np.linspace(5, 30, 20)
    for ref in ref_temperatures:
        # using mean as there are multiple days in the data
        computed_dd = dd_compute(
            temperature, reference=ref, method="pro"
        ).mean()
        # putting the reference computation in the assert to better debug the test
        assert np.isclose(
            computed_dd, reference_sin_dd(mean, spread, reference=ref), rtol=0.2
        )  # the pro method is less accurate


def reference_sin_dd_not_integral(mean, spread, reference):
    """Return the expected NOT INTEGRAL DD for a sin function temperature .

    This works for both min-max and mean methods, as both return the same value
    when the temperature is symetric around the mean, as in the sin function."""
    if mean > reference:
        return 0
    if mean <= reference:
        return reference - mean


def test_computation_realistic_data_not_integral(spread=10, mean=15):
    """Test the value of the computation of the degree_days using NOT integral
    for a Sin temperature evolution.

    """
    # using high sampling to reduce the error on the integral
    # as we use the rectangle integral method
    temperature = generate_days_sin_data(frequency="1h") * spread + mean
    ref_temperatures = np.linspace(5, 30, 20)
    for ref in ref_temperatures:
        # using mean as there are multiple days in the data
        computed_dd = dd_compute(
            temperature, reference=ref, method="min_max"
        ).mean()
        # putting the reference computation in the assert to better debug the test
        assert np.isclose(
            computed_dd,
            reference_sin_dd_not_integral(mean, spread, reference=ref),
            rtol=1e-2,
        )
    for ref in ref_temperatures:
        # using mean as there are multiple days in the data
        computed_dd = dd_compute(
            temperature, reference=ref, method="mean"
        ).mean()
        # putting the reference computation in the assert to better debug the test
        assert np.isclose(
            computed_dd,
            reference_sin_dd_not_integral(mean, spread, reference=ref),
            rtol=1e-2,
        )


def test_dd_calc_method():
    """Test the method to compute the degree days."""
    assert dd_calc_method(dd_mean) == "mean"
    assert dd_calc_method(dd_min_max) == "min_max"
    assert dd_calc_method(dd_integral) == "integral"
    assert dd_calc_method(dd_pro) == "pro"
    assert dd_calc_method(dd_compute) == "unknown"
    assert dd_calc_method(np.min) == "unknown"
