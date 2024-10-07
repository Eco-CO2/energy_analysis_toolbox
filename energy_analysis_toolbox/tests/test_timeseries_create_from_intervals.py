import numpy as np
import pandas as pd
from ..timeseries.create.from_intervals import flatten_and_fill
from .. import keywords as EATK


# =============================================================================
# flatten overconsumption
# =============================================================================
def test_flatten_empty():
    """Empty dataframe is flattened as empty"""
    assert flatten_and_fill(pd.DataFrame()).empty


def test_flatten_one_interval():
    """Check that flattening is OK in one row table."""
    start = pd.Timestamp("2020-02-29")
    end = start + pd.Timedelta("1min")
    test_data = pd.DataFrame(
        [[start, end, 42.0]], columns=[EATK.start_f, EATK.end_f, "test"]
    )
    flat_consumption = flatten_and_fill(
        test_data, start_f=EATK.start_f, end_f=EATK.end_f, time_f=EATK.time_f
    )
    expect_index = pd.DatetimeIndex([start, end], name=EATK.time_f)
    expect_series = pd.DataFrame(
        np.array([42.0, np.nan]).reshape(2, 1),
        index=expect_index,
        columns=["test"],
    )
    pd.testing.assert_frame_equal(
        expect_series, flat_consumption, check_dtype=False, check_freq=False
    )
    pass


def test_flatten():
    """Check that flattening looks OK."""
    time_f = "timestamp"
    end_f = "end_custom"
    start_f = "start_custom"
    begin = pd.Timestamp("2022-12-05 18:00:00")
    starts = pd.date_range(start=begin, freq="10min", periods=3, name=start_f)
    targets = pd.date_range(start=begin, freq="5min", periods=6, name=time_f)
    data = {
        start_f: starts,
        end_f: starts + pd.Timedelta("5min"),
        "col_str": ["toto"] * 3,
        "col_float": np.arange(0.0, 6.0, 2.0),
    }
    table = pd.DataFrame.from_dict(data)
    # Default filling
    flat_expect = pd.DataFrame.from_dict(
        {
            "col_str": ["toto"] * 6,
            "col_float": np.arange(0, 6),
        }
    )
    flat_expect.index = targets
    flat_expect.iloc[[1, 3, 5], :] = np.nan
    flat_table = flatten_and_fill(table, end_f=end_f, start_f=start_f)
    pd.testing.assert_frame_equal(flat_expect, flat_table, check_freq=False)
    # explicit filling
    flat_filled_table = flatten_and_fill(
        table, fill_values={"col_str": "tata"}, end_f=end_f, start_f=start_f
    )
    flat_expect["col_str"] = ["toto", "tata"] * 3
    pd.testing.assert_frame_equal(
        flat_expect,
        flat_filled_table,
        check_freq=False,
        check_dtype=False,
    )
