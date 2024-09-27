energy\_toolbox.energy
======================
.. automodule:: watt_df.energy
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

`et.energy` subpackage contains functionalities dedicated to the processing of
energy data, usually as a timeseries.
Accordingly, all functions in this subpackage are designed to work in a consistent
way with the physical nature of the energy : e.g., resampling functions are
implemented in such a way that energy conservation is unsured.

The subpackage is organized as follows:

.. toctree::
    :maxdepth: 2

    watt_df.energy.resample
