# -*- coding: utf-8 -*-
"""
Check the conformity of the utility functions used in draw detection
"""

import pytest
import numpy as np
import pandas as pd

from ..timeseries.math.derivatives import (
    time_derivative_fwd,
    time_derivative_second,
)
from ..timeseries.resample.interpolate import (
    piecewise_affine,
    piecewise_constant,
)
from ..timeseries.resample.index_transformation import tz_convert_or_localize

# =============================================================================
# Derivatives
# =============================================================================


@pytest.fixture
def constant_timeseries():
    """Create a timeseries of constant values"""
    nw = pd.Timestamp.now()
    inds = pd.date_range(nw, nw + pd.Timedelta(seconds=10), freq="2s")
    series = pd.Series(np.ones(inds.size), index=inds)
    return series


@pytest.fixture
def affine_timeseries():
    """Create a timeseries of affine increasing values"""
    nw = pd.Timestamp.now()
    inds = pd.date_range(nw, nw + pd.Timedelta(seconds=10), freq="2s")
    series = pd.Series(np.linspace(1, 11, inds.size), index=inds)
    return series


def test_derivative_constant(constant_timeseries):
    """Check that the derivative computed for a constant function is zero"""
    d1t = time_derivative_fwd(constant_timeseries)
    assert (d1t == 0.0).all()
    assert d1t.size == constant_timeseries.size


def test_derivative_affine(affine_timeseries):
    """Check the derivative computation for an affine function"""
    d1t = time_derivative_fwd(affine_timeseries)
    assert np.allclose(d1t.values, np.ones(affine_timeseries.size))


def test_second_derivative_constant(constant_timeseries):
    """Check that the second order derivative computed for a constant function is zero"""
    d2t = time_derivative_second(constant_timeseries)
    assert (d2t == 0.0).all()


def test_second_derivative_affine(affine_timeseries):
    """Check that the second order derivative computed for an affine function is zero"""
    d2t = time_derivative_second(affine_timeseries)
    assert (d2t == 0.0).all()


# =============================================================================
# ----- BC interpolations ------
# =============================================================================


@pytest.mark.parametrize(
    "source_tz, targets_tz",
    [
        (None, None),
        ("Europe/Paris", None),
        (None, "Europe/Paris"),
        ("Europe/Paris", "Europe/Paris"),
        ("Europe/Paris", "UTC"),
        ("UTC", "Europe/Paris"),
    ],
)
@pytest.mark.parametrize(
    "targets, expected",
    [  # samples
        (np.array([0.0, 5.0, 7.0, 10.0]), np.array([0.0, 1.0, 2.0, -1.0])),
        # between samples
        (np.array([1.0, 6.0, 9.0]), np.array([0.2, 1.5, 0.0])),
        # outside samples
        (np.array([-1, 12.0]), np.array([0.0, -1.0])),
        # mix
        (np.array([-1, 6.0, 7.0, 12.0]), np.array([0.0, 1.5, 2.0, -1.0])),
        # No values
        (np.array([]), np.array([])),
    ],
)
def test_piecewise_affine(targets, expected, source_tz, targets_tz):
    """Check the method to interpolate BC timeseries : piecewise affine

    .. note::

        As the source and target instants are generated as offsets from the
        start time, the expected values are not affected by the timezone.


    """
    start_time = pd.Timestamp.now(tz=source_tz)
    sample_dts = [0.0, 5.0, 7.0, 10.0]
    sample_values = [0.0, 1.0, 2.0, -1.0]
    instants = pd.DatetimeIndex(
        [start_time + pd.Timedelta(seconds=dt) for dt in sample_dts]
    )
    bc = pd.Series(sample_values, instants)
    target_instants = pd.DatetimeIndex(
        [start_time + pd.Timedelta(seconds=dt) for dt in targets]
    )
    tz_convert_or_localize(target_instants, targets_tz)
    affine_interp = piecewise_affine(bc, target_instants)
    assert np.allclose(affine_interp.to_numpy(), expected)


@pytest.mark.parametrize(
    "source_tz, targets_tz",
    [
        (None, None),
        ("Europe/Paris", None),
        (None, "Europe/Paris"),
        ("Europe/Paris", "Europe/Paris"),
        ("Europe/Paris", "UTC"),
        ("UTC", "Europe/Paris"),
    ],
)
@pytest.mark.parametrize(
    "targets, expected",
    [  # samples
        (np.array([0.0, 5.0, 7.0, 10.0]), np.array([0.0, 1.0, 2.0, -1.0])),
        # between samples
        (np.array([1.0, 6.0, 9.0]), np.array([0.0, 1.0, 2.0])),
        # outside samples
        (np.array([-1, 12.0]), np.array([0.0, -1.0])),
        # mix
        (np.array([-1, 6.0, 7.0, 12.0]), np.array([0.0, 1.0, 2.0, -1.0])),
        # No values
        (np.array([]), np.array([])),
    ],
)
def test_piecewise_constant(targets, expected, source_tz, targets_tz):
    """Check the method to interpolate BC timeseries : piecewise constant.

    .. note::

        As the source and target instants are generated as offsets from the
        start time, the expected values are not affected by the timezone.

    """
    start_time = pd.Timestamp.now(tz=source_tz)
    sample_dts = [0.0, 5.0, 7.0, 10.0]
    sample_values = [0.0, 1.0, 2.0, -1.0]
    instants = pd.DatetimeIndex(
        [start_time + pd.Timedelta(seconds=dt) for dt in sample_dts]
    )
    bc = pd.Series(sample_values, instants)
    target_instants = pd.DatetimeIndex(
        [start_time + pd.Timedelta(seconds=dt) for dt in targets]
    )
    tz_convert_or_localize(target_instants, targets_tz)
    stair_interp = piecewise_constant(bc, target_instants)
    assert np.allclose(stair_interp.to_numpy(), expected)
    pd.testing.assert_index_equal(stair_interp.index, target_instants)


def test_piecewise_constant_padding():
    """Check the left-padding"""
    start_time = pd.Timestamp.now()
    sample_dts = [0.0, 5.0, 7.0, 10.0]
    sample_values = [0.0, 1.0, 2.0, -1.0]
    instants = pd.DatetimeIndex(
        [start_time + pd.Timedelta(seconds=dt) for dt in sample_dts]
    )
    bc = pd.Series(sample_values, instants)
    target_instants = pd.DatetimeIndex(
        [
            start_time + pd.Timedelta(seconds=dt)
            for dt in np.array([-2, -1, 12.0])
        ]
    )
    stair_interp = piecewise_constant(bc, target_instants)
    assert np.allclose(stair_interp.to_numpy(), np.array([0.0, 0.0, -1.0]))
    stair_interp = piecewise_constant(bc, target_instants, left_pad=42.0)
    assert np.allclose(stair_interp.to_numpy(), np.array([42.0, 42.0, -1.0]))
