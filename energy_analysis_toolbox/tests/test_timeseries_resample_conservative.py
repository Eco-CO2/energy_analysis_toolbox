# -*- coding: utf-8 -*-
"""
Test resampling functions for extensive variable flows.
"""

import numpy as np
import pandas as pd
from pandas.testing import assert_index_equal, assert_series_equal
import pytest

from ..errors import (
    EATEmptySourceError,
    EATEmptyTargetsError,
    EATInvalidTimestepDurationError,
)
from ..constants import MINUTE, DAY
from ..timeseries.resample.conservative import (
    deduce_flows,
    flow_rate_conservative,
    volume_conservative,
)

from .fake.timeseries import example_volume_one_day

# =============================================================================
# --- deduce_flows
# =============================================================================


def test_deduce_flows_nominal():
    """Check that flow-rates are correctly deduced.

    Adapt the input so that the flow rate is always 0.1 m3.s-1.
    """
    start = pd.Timestamp.now()
    s = pd.Timedelta("1s")
    volumes = pd.Series(
        np.array([1.0, 0.5, 2.0, 1.0]),
        index=pd.DatetimeIndex(
            [
                start,
                start + 10 * s,
                start + 15 * s,
                start + 35 * s,
            ]
        ),
    )
    flow_rates = deduce_flows(volumes, last_step_duration=10.0)
    assert np.allclose(flow_rates.values, np.ones(4) * 0.1)
    assert_index_equal(flow_rates.index, volumes.index)


# =============================================================================
# --- flow_rate_conservative
# =============================================================================
def compare_flow_rates(left, right):
    """Factorize flow-rates series comparisons for the tests below."""
    assert_series_equal(
        left, right, check_dtype=False, check_exact=False, check_freq=False
    )
    return


def test_flow_rate_errors():
    """Check that declared errors are raised."""
    example = pd.Series(
        np.arange(0, 10),
        pd.date_range(pd.Timestamp("2021-12-15"), periods=10, freq="45min"),
    )
    with pytest.raises(EATEmptySourceError):
        flow_rate_conservative(pd.Series([], dtype="float64"), example.index)
    with pytest.raises(EATEmptyTargetsError):
        flow_rate_conservative(example, pd.DatetimeIndex([]))


def test_flow_rate_out_of_bounds_right():
    """Check the out of bounds resampling on the right."""
    sources = pd.Series(
        np.arange(0, 10),
        pd.date_range(pd.Timestamp("2021-12-15"), periods=10, freq="45min"),
    )
    target_instants = pd.date_range(
        pd.Timestamp("2022-12-16"), periods=10, freq="12min"
    )
    # default 0 padding
    interp = flow_rate_conservative(sources, target_instants)
    compare_flow_rates(interp, pd.Series(np.zeros(10), index=target_instants))


def test_flow_rate_right_bound_overlap():
    """Check resampling over the right bound.

    Scenario::

        t0                t0+20min
        |    3. X.s-1        |    0. X.s-1                     fr source
        |  3. X.s-1   |   1. X.s-1     |                       fr target
        t0         t0 +15min          (t0+30min)

    """
    sources = pd.Series(
        [3.0],
        pd.date_range(pd.Timestamp("2021-12-15"), periods=1, freq="20min"),
    )
    target_instants = pd.date_range(
        pd.Timestamp("2021-12-15"), periods=2, freq="15min"
    )
    # implicit last target instant
    interp = flow_rate_conservative(
        sources, target_instants, last_step_duration=20 * MINUTE
    )
    compare_flow_rates(interp, pd.Series([3.0, 1.0], index=target_instants))
    # explicit last target instant
    interp = flow_rate_conservative(
        sources,
        target_instants,
        last_step_duration=20 * MINUTE,
        last_target_step_duration=30 * MINUTE,
    )
    compare_flow_rates(interp, pd.Series([3.0, 0.5], index=target_instants))


def test_flow_rate_out_of_bounds_left():
    """Check the out of bounds resampling on the left."""
    sources = pd.Series(
        np.arange(0, 10),
        pd.date_range(pd.Timestamp("2021-12-15"), periods=10, freq="45min"),
    )
    target_instants = pd.date_range(
        pd.Timestamp("2020-12-14"), periods=10, freq="12min"
    )
    # default 0 padding
    interp = flow_rate_conservative(sources, target_instants)
    compare_flow_rates(interp, pd.Series(np.zeros(10), index=target_instants))


def test_flow_rate_identity():
    """Check that the function is identity when source and targets are the same."""
    sources = pd.Series(
        np.arange(0, 10),
        pd.date_range(pd.Timestamp("2021-12-15"), periods=10, freq="45min"),
    )
    compare_flow_rates(flow_rate_conservative(sources, sources.index), sources)


def test_flow_rate_limit_last_steps():
    """Check that null last timesteps raise errors.

    Checked cases :

        - null last step duration -> Error
        - null last target step duration -> Error
        - null last step and target step durations -> Error
        - negative last step duration -> Error
        - negative last target step duration -> Error

    """
    sources = pd.Series(
        np.arange(0, 10),
        pd.date_range(pd.Timestamp("2021-12-15"), periods=10, freq="45min"),
    )
    with pytest.raises(EATInvalidTimestepDurationError):
        flow_rate_conservative(
            sources.iloc[:8], sources.index, last_step_duration=0.0
        )
    with pytest.raises(EATInvalidTimestepDurationError):
        flow_rate_conservative(
            sources, sources.index, last_target_step_duration=0.0
        )
    with pytest.raises(EATInvalidTimestepDurationError):
        flow_rate_conservative(
            sources,
            sources.index,
            last_step_duration=0.0,
            last_target_step_duration=0.0,
        )
    with pytest.raises(EATInvalidTimestepDurationError):
        flow_rate_conservative(sources, sources.index, last_step_duration=-42.0)
    with pytest.raises(EATInvalidTimestepDurationError):
        flow_rate_conservative(
            sources, sources.index, last_target_step_duration=-42.0
        )


def test_flow_rate_conservative():
    """
    Scenario::

                -30s    t0        t1              t2          t3     t4           t5
        dur               |   60s  |      120s      |   +60s   | 30s |    120s    |
        fr*1e3       0    |   1    |      2         |     3    |  4        |    (0)

        fr*1e3    0   | 0.5   |  1.5   | 2 |   2    |     7/3        |      1     |
               m0     m1      m2        m3  m4      m5               m6           m7

    """
    one_s = pd.Timedelta("1s")
    dur_0 = 60
    dur_1 = 120
    dur_2 = 60  # last step duration inferred from dur_2
    dur_3 = dur_2 / 2
    dur_4 = dur_2 * 2
    t0 = pd.Timestamp("2021-01-06 12:03")
    t1 = t0 + dur_0 * one_s
    t2 = t1 + dur_1 * one_s
    t3 = t2 + dur_2 * one_s
    t4 = t3 + dur_3 * one_s
    t5 = t4 + dur_4 * one_s
    flow_rates = pd.Series(
        np.arange(1, 5) * 1e-3, pd.DatetimeIndex([t0, t1, t2, t3])
    )
    target_instants = pd.DatetimeIndex(
        [
            t0 - 3600 * one_s,
            t0 - dur_0 / 2 * one_s,  # m1
            t0 + dur_0 / 2 * one_s,  # m2
            t1 + dur_0 / 2 * one_s,  # m3
            t1 + dur_1 / 2 * one_s,  # m4
            t2,  # m5
            t4,  # m6
            t5,  # m7
        ]
    )
    expected_fr_values = np.array(
        [
            0.0,
            flow_rates.iloc[0] / 2,
            flow_rates.iloc[0:2].mean(),
            flow_rates.iloc[1],
            flow_rates.iloc[1],
            (flow_rates.iloc[2] * 2 + flow_rates.iloc[3] * 1)
            / 3,  # dur_3 is half dur_2
            flow_rates.iloc[3] * 1 / 4,  # dur_4 is 3x dur_3
            0.0,
        ]
    )
    fr_interp = flow_rate_conservative(flow_rates, target_instants)
    compare_flow_rates(
        fr_interp, pd.Series(expected_fr_values, target_instants)
    )


# =============================================================================
# --- volume_conservative function
# =============================================================================
def compare_volumes(left, right):
    """Factorize volume series comparisons for the tests below."""
    assert_series_equal(
        left, right, check_dtype=False, check_exact=False, check_freq=False
    )
    return


def test_volume_conservative_errors():
    """Check that declared errors are raised."""
    example = pd.Series(
        np.arange(0, 10),
        pd.date_range(pd.Timestamp("2021-12-15"), periods=10, freq="45min"),
    )
    with pytest.raises(EATEmptySourceError):
        volume_conservative(pd.Series([], dtype="float64"), example.index)
    with pytest.raises(EATEmptyTargetsError):
        volume_conservative(example, pd.DatetimeIndex([]))


def test_volume_conservative_out_of_bounds_right():
    """Check the out of bounds resampling on the right."""
    sources = pd.Series(
        np.arange(0, 10),
        pd.date_range(pd.Timestamp("2021-12-15"), periods=10, freq="45min"),
    )
    target_instants = pd.date_range(
        pd.Timestamp("2022-12-15"), periods=10, freq="12min"
    )
    interp = volume_conservative(sources, target_instants)
    compare_volumes(interp, pd.Series(np.zeros(10), index=target_instants))


def test_volume_conservative_out_of_bounds_left():
    """Check the out of bounds resampling on the left."""
    sources = pd.Series(
        np.arange(0, 10),
        pd.date_range(pd.Timestamp("2021-12-15"), periods=10, freq="45min"),
    )
    target_instants = pd.date_range(
        pd.Timestamp("2020-12-15"), periods=10, freq="12min"
    )
    # default 0 padding
    interp = volume_conservative(sources, target_instants)
    compare_volumes(interp, pd.Series(np.zeros(10), index=target_instants))


def test_volume_conservative_identity():
    """Check that the function is identity when source and targets are the same

    Scenario::

        t0       +dt        +2dt        (+3dt)
        |    0    |     50    |    0

    """
    t0 = pd.Timestamp.now()
    dt = pd.Timedelta(seconds=300)
    volumes = pd.Series(
        [0.0, 0.050, 0.0], index=pd.DatetimeIndex([t0, t0 + dt, t0 + 2 * dt])
    )
    compare_volumes(volumes, volume_conservative(volumes, volumes.index))


def test_volume_conservative_limit_last_steps():
    """Check that limit last timesteps are managed consistently.

    Checked cases :

        - null last target step duration -> Error
        - null last step duration -> Error
        - negative last step duration -> Error
        - negative last target step duration -> Error

    """
    t0 = pd.Timestamp.now()
    dt = pd.Timedelta(seconds=300)
    volumes = pd.Series(
        [0.0, 0.050, 0.05], index=pd.DatetimeIndex([t0, t0 + dt, t0 + 2 * dt])
    )
    with pytest.raises(EATInvalidTimestepDurationError):
        volume_conservative(
            volumes, volumes.index, last_target_step_duration=0.0
        )
    with pytest.raises(EATInvalidTimestepDurationError):
        volume_conservative(volumes, volumes.index, last_step_duration=0.0)
    with pytest.raises(EATInvalidTimestepDurationError):
        volume_conservative(volumes, volumes.index, last_step_duration=-1.0)
    with pytest.raises(EATInvalidTimestepDurationError):
        volume_conservative(
            volumes, volumes.index, last_target_step_duration=-1.0
        )


def test_volume_conservative_1():
    """Staggered case.

    Scenario::

        t0       +dt        +2dt        (+3dt)
        |    0    |     50    |    0
             |    25    |    25
           +dt/2      +3dt/2        (+5dt/2)

    """
    t0 = pd.Timestamp.now()
    dt = pd.Timedelta(seconds=300)
    volumes = pd.Series(
        [0.0, 0.050, 0.0], index=pd.DatetimeIndex([t0, t0 + dt, t0 + 2 * dt])
    )
    volumes_expect = pd.Series(
        [0.025, 0.025], index=pd.DatetimeIndex([t0 + dt / 2, t0 + 3 * dt / 2])
    )
    compare_volumes(
        volumes_expect, volume_conservative(volumes, volumes_expect.index)
    )


def test_volume_conservative_2():
    """Subinterval case.

    Scenario::

        t0             +dt                  +2dt            (+3dt)
        |    0          |          60           |    0
        .                     |    20    |   20
                            +4dt/3    +5dt/3   (2dt)
    """
    t0 = pd.Timestamp.now()
    dt = pd.Timedelta(seconds=300)
    volumes = pd.Series(
        [0.0, 0.060, 0.0], index=pd.DatetimeIndex([t0, t0 + dt, t0 + 2 * dt])
    )
    volumes_expect = pd.Series(
        [0.020, 0.020],
        index=pd.DatetimeIndex([t0 + 4 * dt / 3, t0 + 5 * dt / 3]),
    )
    compare_volumes(
        volumes_expect, volume_conservative(volumes, volumes_expect.index)
    )


def test_volume_conservative_finer():
    """Resample volume to a regular finer resolution.

    Just check that the total volume is conserved.
    """
    volumes = example_volume_one_day("2020-01-06")
    fine_ixs = pd.date_range(
        start="2020-01-06", end="2020-01-07", inclusive="left", freq="43s"
    )
    fine_volumes = volume_conservative(volumes, fine_ixs)
    assert fine_volumes.sum() == volumes.sum()


def test_volume_conservative_finer_2():
    """Subsampling case.

    Scenario::

        t0             +dt                  +2dt            (+3dt)
        |       40      |          60           |      50
        |  20   |   20  |    30     |     30    |   25 |   25  |

    """
    t0 = pd.Timestamp.now()
    dt = pd.Timedelta(seconds=300)
    volumes = pd.Series(
        [0.040, 0.060, 0.050],
        index=pd.date_range(t0, t0 + 3 * dt, inclusive="left", freq="300s"),
    )
    volumes_expect = pd.Series(
        [0.020, 0.020, 0.030, 0.030, 0.025, 0.025],
        index=pd.date_range(t0, t0 + 3 * dt, inclusive="left", freq="150s"),
    )
    compare_volumes(
        volumes_expect, volume_conservative(volumes, volumes_expect.index)
    )


def test_volume_conservative_ordinary():
    """Resample volume to an ordinary uneven resolution.

    Scenario::

                 t0.........+10min............//+1h.......//......+1d.................(+1d)
                 |    0 m3     |      3.6 m3  //  |  0m3  //       |     0.864m4......
                 |             |   0.001m3/s  //  |       //       |     1e-5m3/s.....
          x   x             x     x      x                    x       x                (?)
            0        0        0.03   0.6         2.970          0.009       ?

    """
    volumes = pd.Series(
        [0.0, 3.6, 0.0, 0.864],
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2022-02-22 22:20:22"),
                # 0 m3 here
                pd.Timestamp("2022-02-22 22:30:22"),  # 600s later
                # 3.6 m3 here (1L/s)
                pd.Timestamp("2022-02-22 23:30:22"),  # 3600s later
                # 0 m3 here
                pd.Timestamp("2022-02-23 23:30:22"),  # 1 day later
                # 0.864 m3 here
            ]
        ),
    )
    volumes_expect = pd.Series(
        [0.0, 0.0, 0.030, 0.600, 2.970, 0.009, 0.072],
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2022-02-22 20:11:13"),
                # 0 m3 here
                pd.Timestamp("2022-02-22 21:14:32"),
                # 0 m3 here
                pd.Timestamp("2022-02-22 22:29:52"),
                # 0.030 m3 here
                pd.Timestamp(
                    "2022-02-22 22:30:52"
                ),  # 1 min later, 30s with "draw"
                # 0.600 m3 here
                pd.Timestamp("2022-02-22 22:40:52"),  # 600s later
                # 2.970 m3 here, the rest of the 3.6 m3
                pd.Timestamp("2022-02-23 21:45:22"),
                # 0.009 m3 here
                pd.Timestamp(
                    "2022-02-23 23:45:22"
                ),  # 2h later, 15 min after the new consumption
                # Depends on the duration of last target timestep
                # 0.855 m3 here if forced to 1d
                # 0.072 m3 if let implicit (2h)
            ]
        ),
    )
    volumes_obtained = volume_conservative(volumes, volumes_expect.index)
    compare_volumes(volumes_obtained, volumes_expect)
    # explicit last_target_step_duration
    volumes_obtained = volume_conservative(
        volumes,
        volumes_expect.index,
        last_target_step_duration=DAY,
    )
    volumes_expect.iloc[-1] = 0.855
    compare_volumes(volumes_obtained, volumes_expect)
    # explicit last steps durations
    volumes_obtained = volume_conservative(
        volumes,
        volumes_expect.index,
        last_step_duration=30 * MINUTE,
        last_target_step_duration=DAY,
    )
    volumes_expect.iloc[-2:] = 0.864 / 2
    compare_volumes(volumes_obtained, volumes_expect)


def test_volume_conservative_coarser():
    """Resample volume with a coarser resolution"""
    volumes = example_volume_one_day("2020-01-06")
    coarse_ixs = pd.date_range(
        start="2020-01-06", end="2020-01-06 12:00", periods=2
    )
    coarse_volumes = volume_conservative(volumes, coarse_ixs)
    assert coarse_volumes.sum() == pytest.approx(volumes.sum())
    assert coarse_volumes.loc["2020-01-06 12:00":].sum() == pytest.approx(
        volumes.loc["2020-01-06 12:00":].sum()
    )


def test_volume_conservative_as_integral():
    """Resample volume on a big timestep straddling the series

    The expected result is the sum of all volumes.

    """
    volumes = example_volume_one_day("2020-01-06")
    coarse_ixs = pd.date_range(start="2020-01-05", end="2020-01-07", periods=1)
    coarse_volumes = volume_conservative(
        volumes, coarse_ixs, last_target_step_duration=DAY * 3
    )
    assert coarse_volumes.loc["2020-01-05"] == pytest.approx(volumes.sum())
    assert coarse_volumes.size == 1
