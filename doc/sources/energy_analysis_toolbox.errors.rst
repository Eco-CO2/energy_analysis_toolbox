energy\_analysis\_toolbox.errors package
========================================
.. automodule:: energy_analysis_toolbox.errors
   :members:
   :undoc-members:
   :show-inheritance:

This package contains all the specific exceptions used and raised by |eat| library.

Base class for all custom errors
--------------------------------
All exceptions in |eat| derive from the class below. Catching this class will catch
any exception specific to |eat| library, whatever its type.

.. automodule:: energy_analysis_toolbox.errors.base
   :members:
   :show-inheritance:

Errors related to timeseries consistency
----------------------------------------
.. automodule:: energy_analysis_toolbox.errors.invalid_timeseries
   :members:
   :show-inheritance:

Errors related to resampling
----------------------------
.. automodule:: energy_analysis_toolbox.errors.resampling
   :members:
   :show-inheritance:
