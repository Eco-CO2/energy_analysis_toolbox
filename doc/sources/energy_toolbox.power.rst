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

    energy_analysis_toolbox.power.basics
    energy_analysis_toolbox.power.resample
    energy_analysis_toolbox.power.overconsumption

.. automodule:: energy_analysis_toolbox.power
   :members:
   :undoc-members:
   :show-inheritance: