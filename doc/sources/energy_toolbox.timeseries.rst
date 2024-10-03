energy\_toolbox.timeseries package
==================================
.. automodule:: wattdf.timeseries
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

The timeseries subpackage contains functions dedicated to processing timeseries
data, without specific a priori assumption on the physical nature of the data.

It is structured as follows:

- :py:mod:`wattdf.timeseries.create` contains functions to create
  timeseries data in left-closed right-open sampling from other kinds of
  representations of time-dependent data.
- :py:mod:`wattdf.timeseries.extract_features` contains functions
  to extract features from timeseries data. For now, only simple features are
  provided such as time-steps, intervals during which the series is over a threshold
  etc.
- :py:mod:`wattdf.timeseries.math` contains functions to perform
  mathematical operations on timeseries data, such as computing the derivative
  of a timeseries.
- :py:mod:`wattdf.timeseries.resample` contains functions to resample
  timeseries data to regular or arbitrary timesteps. Various interpolations
  methods are provided, depending on the assumptions which are made on the data.
- :py:mod:`wattdf.timeseries.profiles` contains functionalites to
  compute profiles of variations of timeseries data from an history of variations.
  Typically, this can be used to compute typical daily patterns of variations.

See the detailled package structure below.

.. toctree::
   :maxdepth: 4

   wattdf.timeseries.create
   wattdf.timeseries.extract_features
   wattdf.timeseries.math
   wattdf.timeseries.resample
   wattdf.timeseries.profiles
