API reference documentation
===========================

|et| library is structured as follows.
The main subpackages in `ct` are the following :

- :py:mod:`energy_toolbox.timeseries` is a subpackage containing generic
  functionalities related to time-series analysis, without assumption on the
  physical nature of the data.
- :py:mod:`energy_toolbox.power` and :py:mod:`energy_toolbox.energy`
  are subpackages containing functionalities related to power and energy
  computation, respectively, working directly on time-series.
- :py:mod:`energy_toolbox.weather` is a subpackage containing functionalities
  related to weather data, such as temperature and degree days.
- :py:mod:`energy_toolbox.synthetic` is a subpackage containing functionalities
  related to the generation of synthetic data.

The most commonly used of these functionalities can directly be accessed as pandas
series/Dataframe methods such as::

   my_power_series.et.power_to_freq(...)

by importing the :py:mod:`energy_toolbox.pandas` extension module.

These packages are complemented by :

- :py:mod:`energy_toolbox.constants` containing constants used in the
  computations or useful in daily analysis of energy data.
- :py:mod:`energy_toolbox.keywords` containing keywords used in the lib
  interface (still to be improved).
- :py:mod:`energy_toolbox.errors` containing custom errors used in the lib.

Here is a detailed structure of the library :

.. toctree::
   :maxdepth: 2

   energy_toolbox.timeseries
   energy_toolbox.power
   energy_toolbox.energy
   energy_toolbox.pandas
   energy_toolbox.constants
   energy_toolbox.keywords
   energy_toolbox.errors
   energy_toolbox.tests
   energy_toolbox.synthetic
   energy_toolbox.weather
   energy_toolbox.thermosensitivity
