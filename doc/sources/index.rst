API reference
=============

|et| library is structured as follows.
The main subpackages in `ct` are the following :

- :py:mod:`watt_df.timeseries` is a subpackage containing generic
  functionalities related to time-series analysis, without assumption on the
  physical nature of the data.
- :py:mod:`watt_df.power` and :py:mod:`watt_df.energy`
  are subpackages containing functionalities related to power and energy
  computation, respectively, working directly on time-series.
- :py:mod:`watt_df.weather` is a subpackage containing functionalities
  related to weather data, such as temperature and degree days.
- :py:mod:`watt_df.synthetic` is a subpackage containing functionalities
  related to the generation of synthetic data.

The most commonly used of these functionalities can directly be accessed as pandas
series/Dataframe methods such as::

   my_power_series.wdf.power_to_freq(...)

by importing the :py:mod:`watt_df.pandas` extension module.

These packages are complemented by :

- :py:mod:`watt_df.constants` containing constants used in the
  computations or useful in daily analysis of energy data.
- :py:mod:`watt_df.keywords` containing keywords used in the lib
  interface (still to be improved).
- :py:mod:`watt_df.errors` containing custom errors used in the lib.

Here is a detailed structure of the library :

.. toctree::
   :maxdepth: 2

   watt_df.timeseries
   watt_df.power
   watt_df.energy
   watt_df.pandas
   watt_df.constants
   watt_df.keywords
   watt_df.errors
   watt_df.tests
   watt_df.synthetic
   watt_df.weather
   watt_df.thermosensitivity
