API reference
=============

|et| library is structured as follows.
The main subpackages in `ct` are the following :

- :py:mod:`wattdf.timeseries` is a subpackage containing generic
  functionalities related to time-series analysis, without assumption on the
  physical nature of the data.
- :py:mod:`wattdf.power` and :py:mod:`wattdf.energy`
  are subpackages containing functionalities related to power and energy
  computation, respectively, working directly on time-series.
- :py:mod:`wattdf.weather` is a subpackage containing functionalities
  related to weather data, such as temperature and degree days.
- :py:mod:`wattdf.synthetic` is a subpackage containing functionalities
  related to the generation of synthetic data.

The most commonly used of these functionalities can directly be accessed as pandas
series/Dataframe methods such as::

   my_power_series.wdf.power_to_freq(...)

by importing the :py:mod:`wattdf.pandas` extension module.

These packages are complemented by :

- :py:mod:`wattdf.constants` containing constants used in the
  computations or useful in daily analysis of energy data.
- :py:mod:`wattdf.keywords` containing keywords used in the lib
  interface (still to be improved).
- :py:mod:`wattdf.errors` containing custom errors used in the lib.

Here is a detailed structure of the library :

.. toctree::
   :maxdepth: 2

   wattdf.timeseries
   wattdf.power
   wattdf.energy
   wattdf.pandas
   wattdf.constants
   wattdf.keywords
   wattdf.errors
   wattdf.tests
   wattdf.synthetic
   wattdf.weather
   wattdf.thermosensitivity
