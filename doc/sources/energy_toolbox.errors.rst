energy\_toolbox.errors package
==============================
.. automodule:: watt_df.errors
   :members:
   :undoc-members:
   :show-inheritance:

This package contains all the specific exceptions used and raised by |et| library.

Base class for all custom errors
--------------------------------
All exceptions in |et| derive from the class below. Catching this class will catch
any exception specific to |et| library, whatever its type.

.. automodule:: watt_df.errors.base
   :members:
   :show-inheritance:

Errors related to timeseries consistency
----------------------------------------
.. automodule:: watt_df.errors.invalid_timeseries
   :members:
   :show-inheritance:

Errors related to resampling
----------------------------
.. automodule:: watt_df.errors.resampling
   :members:
   :show-inheritance:
