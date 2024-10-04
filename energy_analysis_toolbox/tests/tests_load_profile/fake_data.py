# -*- coding: utf-8 -*-
"""
Generation of fake data to test the library.
"""

import scipy.constants as SK
import numpy as np
import pandas as pd


def sinusoid_history(
    start=None,
    n_days=21,
    freq="30min",
    noise=0.05,
    period_variation=SK.day,
    min_value=0.0,
    max_value=1.0,
):
    """Return a series of fake historical data."""
    duration = pd.Timedelta(days=n_days)
    if start is None:
        start = pd.Timestamp.now().floor("D") - duration
    end = start + duration
    index_dates = pd.date_range(
        start=start, freq=freq, end=end, inclusive="left"
    )
    # TODO : factorize using cerebro utils (extracted)
    durations = (index_dates - index_dates[0]).total_seconds()
    omega = (2 * np.pi) / period_variation
    data = (np.sin(durations * omega) + 1 + min_value) / 2 * max_value
    data += np.random.uniform(
        low=-1 * noise, high=noise, size=data.shape[0]
    ) * (max_value - min_value)
    history = pd.Series(data, index=index_dates, name="example")
    return history


def sinusoid_history_df(*args, **kwargs):
    """Return a series of fake historical data as a DataFrame

    The function wraps :py:func:`sinusoid_history` and transforms the result
    to a DataFrame.
    """
    history_series = sinusoid_history(*args, **kwargs)
    return pd.DataFrame.from_dict({history_series.name: history_series})
