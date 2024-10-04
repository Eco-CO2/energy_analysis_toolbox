"""Testing module for rolling_profile"""

from itertools import product
import pytest
import pandas as pd
import numpy as np
from energy_analysis_toolbox.timeseries.profiles.rolling_profile import (
    RollingProfile,
    RollingQuantileProfile,
)


def arange_multicolumn(n_days, step, r_rows=10):
    """Generate a dataframe similar to what the pivot phase would produce.

    The values increases linearly from ``0`` to ``n_days*r_rows*step`` along the rows.
    This means that the rolling standard deviation is constant.

    Examples
    --------

    .. code-block::
        python

        >>> arange_multicolumn(3, 2, 4)
                0	1	2
        0	0	2	4
        1	6	8	10
        2	12	14	16
        3	18	20	22

    """
    return pd.DataFrame(
        np.arange(0, n_days * r_rows * step, step).reshape(r_rows, n_days)
    )


def arange_days_multicolumn(n_days, step, r_rows=10):
    """Generate a dataframe similar to what the pivot phase would produce.

    The values increases linearly from ``0`` to ``n_days*step`` along the rows.
    This means that the all rolling aggregations are constant.

    Examples
    --------

    .. code-block::
        python

        >>> arange_days_multicolumn(3, 2, 4)
                0	1	2
        0	0	2	4
        1	0	2	4
        2	0	2	4
        3	0	2	4

    """
    one_row = np.arange(0, n_days * step, step)
    return pd.DataFrame(np.array([one_row] * (r_rows)).reshape(r_rows, n_days))


def expected_constant_df(value, index, window_size):
    """Return a constant dataframe padded with Nan.

    The padding correspond to a centered rolling window with
    a minimum number of rows equal to ``window_size``.

    Parameters
    ----------
    value : Scalar
        the value to set
    index : Index or array-like
        the index of the returned DataFrame
    window_size : int | str | timedelta | ...
        the window size passed to ``pd.rolling``,
        must be consistent with the index type.

    Returns
    -------
    pd.DataFrame
        the expected Dataframe, with one column named ``"value"``
    """
    expected_df = pd.DataFrame(data=value, index=index, columns=["value"])
    return expected_df.rolling(
        window_size,
        center=True,
    ).median()


@pytest.mark.parametrize(
    "n_rows, step, window_size, ddof",
    product(
        [1, 3, 4],  # number of columns
        [1, 2, 3],  # step in the data range
        [3, 4, 7],  # size of the rolling window
        [0, 1],  # DDoF in the STD computation
    ),
)
def test_windowed_rolling_agg_many_combinations(
    n_rows, step, window_size, ddof
):
    """Test the computation of the standard deviation on a rolling window.
    The dataframe generated as several number of columns (1 to 4),
    with data a np.arange with several step values,
    rolling window size and ddof is also tested.

    The expected value is always the same, as the std isn't affected
    by the rising mean value.
    """
    df = arange_multicolumn(n_rows, step=step)
    expected_std = df.iloc[0:window_size, :].to_numpy().ravel().std(ddof=ddof)
    extected_df = expected_constant_df(expected_std, df.index, window_size)
    rolling_profiler = RollingProfile(
        window=window_size, aggregation=lambda x: np.std(x, ddof=ddof)
    )
    results = rolling_profiler.windowed_rolling_agg(df)
    pd.testing.assert_frame_equal(extected_df, results)


@pytest.mark.parametrize(
    "n_days, step, window_size, freq_minutes", [[3, 2, 5, 6]]
)
def test_mean_offset(n_days, step, window_size, freq_minutes):
    """Check that the option to add the mean profile to the results is effective when toggled"""
    n_rows = int(24 * 60 / freq_minutes)
    data = arange_days_multicolumn(n_days, step, n_rows)
    df = pd.DataFrame(
        data={"value": data.to_numpy().T.ravel()},
        index=pd.date_range(
            start="2022-09-23 00:00",
            periods=n_rows * n_days,
            freq=f"{freq_minutes}min",
        ),
    )
    expected_value = np.mean(data.iloc[0, :]) + np.std(data.iloc[0, :])
    expected_time = pd.date_range(
        start="2022-09-23 00:00",
        end="2022-09-23 23:59",
        freq=f"{freq_minutes}min",
    )
    expected_df = expected_constant_df(
        expected_value, expected_time, window_size
    )
    rolling_profiler = RollingProfile(
        window=window_size, aggregation=np.std, as_mean_offset=True
    )
    results = rolling_profiler.compute(df, time=expected_time[0])
    pd.testing.assert_frame_equal(expected_df, results, check_freq=False)


@pytest.mark.parametrize(
    "n_days, step, window_size, q",
    product(
        [1, 3, 4],  # number of columns
        [1, 2, 3],  # step in the data range
        [3, 4, 7],  # size of the rolling window
        [0.5, 0.8],  # DDoF in the STD computation
    ),
)
def test_RollingQuantileProfile_many_combinations(n_days, step, window_size, q):
    """Test that the rolling Quantile works for many combinations.

    The parameter tested are :

    - number of days
    - window_size
    - quantile computed
    - step of the data generation

    """
    freq_minutes = 6
    r_rows = int(n_days * 24 * 60 / freq_minutes)
    data = (
        np.array(
            np.arange(0, n_days * step, step).tolist() * (r_rows // n_days)
        )
        .reshape(-1, n_days)
        .T.ravel()
    )
    index_name = "timestamp_with_changes"
    index = pd.DatetimeIndex(
        data=pd.date_range(
            start="2022-09-23 00:00",
            periods=r_rows,
            freq=f"{freq_minutes}min",
            name=index_name,
        )
    )
    df = pd.DataFrame(data, index=index, columns=["value"])
    quantile_profile = RollingQuantileProfile(window_size, threshold_quantile=q)
    results = quantile_profile.compute(df, time=df.index[0])
    expected_value = np.quantile(
        np.array(np.arange(0, n_days * step, step).tolist() * (window_size)),
        q=q,
    )
    expected_time = pd.date_range(
        start="2022-09-23 00:00",
        end="2022-09-23 23:59",
        freq=f"{freq_minutes}min",
        name=index_name,
    )
    expected_df = expected_constant_df(
        expected_value, expected_time, window_size
    )
    pd.testing.assert_frame_equal(expected_df, results, check_freq=False)


@pytest.mark.parametrize(
    "n_days, freq, tz",
    product(
        [1, 3, 4],  # number of days
        ["1min", "30min", "3h"],  # timesteps of the data
        ["UTC", None, "Europe/Paris", "Asia/Katmandu"],  # the tz
    ),
)
def test_daily_pivot_many_combinations(n_days, freq, tz):
    """Test that daily_pivot works"""
    start = pd.Timestamp("2023-06-21 00:00:00", tz=tz)
    end = start + pd.DateOffset(days=n_days, milliseconds=-1)
    index = pd.date_range(start=start, end=end, freq=freq)
    history = pd.DataFrame(
        data=np.ones(len(index)), index=index, columns=["value"]
    )
    result = RollingProfile(None, None).daily_pivot(history)
    assert result.shape[1] == n_days
    pd.testing.assert_index_equal(
        result.index,
        pd.timedelta_range(0, "1day", freq=freq, closed="left", name="time"),
    )


def test_daily_pivot_duplicated_index():
    """Test that daily_pivot works"""
    n_days = 3
    freq = "6min"
    tz = "UTC"
    start = pd.Timestamp("2023-06-21 00:00:00", tz=tz)
    end = start + pd.DateOffset(days=n_days, milliseconds=-1)
    index = pd.date_range(start=start, end=end, freq=freq)
    index = index.append(index[5:7])
    history = pd.DataFrame(
        data=np.ones(len(index)), index=index, columns=["value"]
    )
    result = RollingProfile(None, None).daily_pivot(history)
    assert result.shape[1] == n_days
    pd.testing.assert_index_equal(
        result.index,
        pd.timedelta_range(0, "1day", freq=freq, closed="left", name="time"),
    )
