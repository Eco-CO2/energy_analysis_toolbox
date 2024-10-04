"""Testing module for time-localized profiles"""

import pandas as pd
import numpy as np
from energy_analysis_toolbox.timeseries.profiles.localization import (
    LocalizedMeanProfile,
    LocalizedRollingProfile,
    LocalizedRollingQuantileProfile,
    MeanProfile,
    RollingQuantileProfile,
    RollingProfile,
)


def test_localized_mean_profile():
    """Test that the Localize Mixin provide the same results as the
    non-localized profile when no DST happens between the first day of history
    and the target date:

    1. without timezone
    2. with timezone


    """
    n_days = 3
    freq_minutes = 6
    start = pd.Timestamp("2022-09-23 00:00")
    end = start + pd.DateOffset(days=n_days, milliseconds=-1)
    index = pd.DatetimeIndex(
        data=pd.date_range(start=start, end=end, freq=str(freq_minutes) + "min")
    )
    data = np.random.randn(len(index))
    history = pd.DataFrame(
        data={"value": data},
        index=index,
    )
    time = pd.Timestamp("2022-10-23 00:00")
    expected = MeanProfile().compute(history=history, time=time)
    result = LocalizedMeanProfile().compute(history=history, time=time)
    # 1.
    pd.testing.assert_frame_equal(expected, result)
    history.index = index.tz_localize("Europe/Paris")
    result = LocalizedMeanProfile().compute(history=history, time=time)
    expected.index = expected.index.tz_localize("Europe/Paris")
    # 2.
    pd.testing.assert_frame_equal(expected, result)


def test_localized_rolling_profile():
    """Test that the Localize Mixin provide the same results as the
    Non-localized Profile:
    1. without timezone
    2. with timezone
    """
    n_days = 3
    freq_minutes = 6
    start = pd.Timestamp("2022-09-23 00:00")
    window = "30min"
    end = start + pd.DateOffset(days=n_days, milliseconds=-1)
    index = pd.DatetimeIndex(
        data=pd.date_range(start=start, end=end, freq=f"{freq_minutes}min")
    )
    data = np.random.randn(len(index))
    history = pd.DataFrame(
        data={"value": data},
        index=index,
    )
    time = pd.Timestamp("2022-10-23 00:00")
    agg = np.std
    expected = RollingProfile(window, agg).compute(history=history, time=time)
    result = LocalizedRollingProfile(window, agg).compute(
        history=history, time=time
    )
    # 1.
    pd.testing.assert_frame_equal(expected, result)
    history.index = index.tz_localize("Europe/Paris")
    result = LocalizedRollingProfile(window, agg).compute(
        history=history, time=time
    )
    expected.index = expected.index.tz_localize("Europe/Paris")
    # 2.
    pd.testing.assert_frame_equal(expected, result)


def test_localized_rolling_quantile_profile():
    """Test that the Localize Mixin provide the same results as the
    Non-localized Profile:
    1. without timezone
    2. with timezone
    """
    n_days = 3
    freq_minutes = 6
    start = pd.Timestamp("2022-09-23 00:00")
    window = "30min"
    q = 0.8
    end = start + pd.DateOffset(days=n_days, milliseconds=-1)
    index = pd.DatetimeIndex(
        data=pd.date_range(start=start, end=end, freq=str(freq_minutes) + "min")
    )
    data = np.random.randn(len(index))
    history = pd.DataFrame(
        data={"value": data},
        index=index,
    )
    time = pd.Timestamp("2022-10-23 00:00")
    expected = RollingQuantileProfile(window, q).compute(
        history=history, time=time
    )
    result = LocalizedRollingQuantileProfile(window, q).compute(
        history=history, time=time
    )
    # 1.
    pd.testing.assert_frame_equal(expected, result)
    history.index = index.tz_localize("Europe/Paris")
    result = LocalizedRollingQuantileProfile(window, q).compute(
        history=history, time=time
    )
    expected.index = expected.index.tz_localize("Europe/Paris")
    # 2.
    pd.testing.assert_frame_equal(expected, result)
