"""Provide a comprehensive suite of tools for energy data analysis.

This module is designed to help users analyze energy consumption data using a variety
of methods, including thermosensitivity modeling, weather data integration, time series
analysis, synthetic data generation, and more.

Main Features
-------------
- **Energy Data Analysis**: Tools to process, transform, and analyze energy consumption
  data from buildings or systems.
- **Thermosensitivity Modeling**: Classes for modeling energy consumption sensitivity
  to temperature, including degree days computations.
- **Categorical Analysis**: Functions to categorize energy consumption data by
  time-based metrics, such as days of the week or custom time intervals.
- **Weather Integration**: Tools for working with weather data to analyze the influence
  of external conditions on energy consumption.
- **Synthetic Data Generation**: Methods for creating synthetic datasets that simulate
  thermosensitive energy consumption patterns.
- **Time Series Profiling**: Features for creating, analyzing, and categorizing energy
  time series profiles to derive meaningful insights.
- **Logger and Utilities**: A set of logging tools for detailed traceability and
  utility functions for everyday energy data operations.

Submodules
----------
- **constants**: Contains shared constants used throughout the module for energy and
  temperature analysis.
- **pandas**: Utilities for working with Pandas dataframes specific to energy data
  processing.
- **logger**: A logging configuration and utility to enable detailed logging and
  diagnostics for energy analysis routines.
- **keywords**: Defines specific keywords used throughout the analysis for consistency.
- **weather**: Includes tools to calculate heating and cooling degree days and methods
  to analyze thermosensitivity against weather conditions.
- **timeseries**: Modules for time series analysis, including profiles for
  localization, rolling means, thresholds, and other profiling methods.
- **synthetic**: Provides utilities for generating synthetic energy consumption
  datasets with thermosensitivity properties.
- **tests**: Includes unit tests for verifying the correctness of different analysis
  tools and ensuring the module's integrity.

Examples
--------
To begin analyzing energy data:

>>> from energy_analysis_toolbox.weather import degree_days
>>> from energy_analysis_toolbox.timeseries.profiles import mean_profile
>>> import pandas as pd
>>> temperature_data = pd.Series([...], index=pd.date_range(start="2023-01-01",
... periods=365, freq="D"))
>>> degree_days_data = degree_days.dd_compute(temperature_data, base_temperature=18)
>>> mean_profile_result = mean_profile.calculate_mean_profile(temperature_data)

This example demonstrates how to compute degree days and create a mean temperature
profile using the module's submodules.

Notes
-----
This module is intended for energy data analysts who need to assess consumption trends,
model thermosensitivity, and understand how weather impacts energy use. The toolbox
contains both statistical tools for detailed analysis and utility functions for
preprocessing and structuring energy data.

"""

from . import (
    constants,
    energy,
    errors,
    keywords,
    power,
    synthetic,
    tests,
    thermosensitivity,
    timeseries,
)

__version__ = "0.1.1"
