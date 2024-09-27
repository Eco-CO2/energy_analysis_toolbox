energy\_toolbox.power
=====================

`et.power` subpackage contains functionalities dedicated to the processing of
power data, usually as a timeseries.
Accordingly, all functions in this subpackage are designed to work in a consistent
way with the physical nature of the power : e.g., resampling functions are
implemented in such a way that energy conservation is unsured.

The package is structured as follows:

.. toctree::
    :maxdepth: 2

    watt_df.power.basics
    watt_df.power.resample
    watt_df.power.overconsumption

.. automodule:: watt_df.power
   :members:
   :undoc-members:
   :show-inheritance: