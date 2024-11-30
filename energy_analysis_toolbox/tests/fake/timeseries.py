import numpy as np
import pandas as pd

DAY_DEFAULT = pd.Timestamp("2020-03-05")

np.random.seed(42)


def example_volume_one_day(day_start=None, total_volume=1.0):
    """Return an example series of "volume" on a day.

    "volume" means any quantity which follows a conservation law.
    The total volume sums to 1(arbitrary unit) so that it can easily be scaled to
    match a target daily volume. The time location of the consumption as well as
    proportions is summarized in the following diagram::

         0  1  2  3  4  5  6  7  8  9  10 11 12 13  14 15 16 17 18 19 20 21 22 23
         Consumption
         |__|__|__|__|__|__|_3|__|._|__|__|__|__|_1.|__|__|__|__|__|__|_2|_3|__|__|

    Parameters
    ----------
    day_start : pd.Timestamp
        Date for which the data are generated. The default is None in case
        data are generated for 2020-03-05.
    total_volume : float
        Total volume consumed during the example day (used to scale the series
        represented above). Default is 1 , meaning that the series returned by
        the function sums up to 1.

    Returns
    -------
    day_example : pd.Series
        An example consumption of an equivalent volume with 30min timestep.
        The series is sampled such that the value with label ``ti`` is the
        volume associated to the interval ``[ti, ti=1[``.

    """
    day_start = day_start or DAY_DEFAULT
    begin = pd.Timestamp(day_start).floor("D")
    index_day = pd.date_range(begin, periods=48, freq="30min")
    day_example = pd.Series(np.zeros(index_day.size), index=index_day)
    day_example.iloc[13] = 0.3  # 6h30 - 30%
    day_example.iloc[16] = 0.05  # 8h - 5%
    day_example.iloc[27] = 0.15  # 13h30 - 15%
    day_example.iloc[41] = 0.2  # 20h30 - 20%
    day_example.iloc[43] = 0.3  # 21h30 - 30%
    return day_example * total_volume
