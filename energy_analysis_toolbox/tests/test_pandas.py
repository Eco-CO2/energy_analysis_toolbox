
import pandas as pd
import pytest
from  .. import pandas # noqa F401

# Test EATAccessorSeries

def test_series_to_energy():
    # Test the to_energy method of EATAccessorSeries
    from energy_analysis_toolbox.power import to_energy
    series = pd.Series([1, 2, 3], index=pd.date_range('2022-01-01', periods=3, freq='D'))
    ct_series = series.eat.to_energy()
    assert isinstance(ct_series, pd.Series)
    pd.testing.assert_series_equal(ct_series, to_energy(series))

def test_series_to_power():
    # Test the to_power method of EATAccessorSeries
    series = pd.Series([24, 48, 72], index=pd.date_range('2022-01-01', periods=3, freq='D'))
    with pytest.raises(NotImplementedError):
        series.eat.to_power()

def test_series_power_to_freq():
    # Test the power_to_freq method of EATAccessorSeries
    from energy_analysis_toolbox.power import to_freq
    series = pd.Series([1, 2, 3], index=pd.date_range('2022-01-01', periods=3, freq='D'))
    ct_series = series.eat.power_to_freq("2D", last_step_duration=3600)
    assert isinstance(ct_series, pd.Series)
    pd.testing.assert_series_equal(ct_series, to_freq(series, "2D", last_step_duration=3600))

def test_series_energy_to_freq():
    # Test the energy_to_freq method of EATAccessorSeries
    from energy_analysis_toolbox.energy import to_freq
    series = pd.Series([1, 2, 3], index=pd.date_range('2022-01-01', periods=3, freq='D'))
    ct_series = series.eat.energy_to_freq("2D", last_step_duration=3600)
    assert isinstance(ct_series, pd.Series)
    pd.testing.assert_series_equal(ct_series, to_freq(series, "2D", last_step_duration=3600))

def test_series_intervals_over():
    # Test the intervals_over method of EATAccessorSeries
    from energy_analysis_toolbox.timeseries.extract_features import intervals_over
    series = pd.Series([1, 2, 3], index=pd.date_range('2022-01-01', periods=3, freq='D'))
    ct_series = series.eat.intervals_over(2)
    assert isinstance(ct_series, pd.DataFrame)
    pd.testing.assert_frame_equal(ct_series, intervals_over(series, 2))

def test_series_timestep_durations():
    # Test the timestep_durations method of EATAccessorSeries
    from energy_analysis_toolbox.timeseries.extract_features import timestep_durations
    series = pd.Series([1, 2, 3], index=pd.date_range('2022-01-01', periods=3, freq='D'))
    ct_series = series.eat.timestep_durations()
    assert isinstance(ct_series, pd.Series)
    pd.testing.assert_series_equal(ct_series, timestep_durations(series))

def test_series_fill_missing_entries():
    # Test the fill_missing_entries method of EATAccessorSeries
    from energy_analysis_toolbox.timeseries.resample import fill_data_holes
    series = pd.Series([1, 2, 3, 4, 5],
                       index=pd.date_range('2022-01-01 01:00:00',
                                           periods=5,
                                           freq='1h')
                       )
    series = series.drop(pd.Timestamp('2022-01-01 03:00:00'))
    ct_series = series.eat.fill_data_holes()
    assert isinstance(ct_series, pd.Series)
    pd.testing.assert_series_equal(ct_series, fill_data_holes(series))


# Test EATAccessorFrame
def test_frame():
    # Test the to_energy method of EATAccessorFrame
    df = pd.DataFrame({'value': [1, 2, 3], 'duration': [1000, 2000, 3000]}, index=pd.date_range('2022-01-01', periods=3, freq='D'))
    with pytest.raises(NotImplementedError):
        df.eat
