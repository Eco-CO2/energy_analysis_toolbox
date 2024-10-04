"""
This module contains the definition of "keywords" - also called "magic values" -
to be used in this library's public interface.

All variables in this module are basic python objects such as int, str, list etc.

Note about the naming conventions in this module
------------------------------------------------
In order to have readable yet as light as possible names, the following
conventions are used in this module:

1. Variables names are written in snake-case
2. Names suffixed with "_f" are reserved for variables containing field
   names used to identify values in tables such as `pandas` dataframes
3. Names suffixed with "_l" are reserved for sequences (lists) of keywords

.. important::

    DO NOT use these keywords by typing directly their VALUES. The preferred
    use is to pass the variables defined in this module, so that:

    - your code will not be broken in case the content of the variable
      (the magic values) are modified
    - any misspelling will be caught by Python interpreter and lead to
      a ``NameError``
    - you can benefit from the help of autocompletion tools provide
      by your IDE if you have one

Variables defined by the module
-------------------------------
The following reproduces the variables declarations from the module.
"""

#: The name of a field containing the beginning of a time-interval. Unless stated
#: otherwise, the first instant to be included in the interval, as a timestamp.
start_f = "start"
#: The name of a field containing the end of a time-interval. Unless stated
#: otherwise, the first instant to be excluded in the interval, as a timestamp.
end_f = "end"
#: The name of the time-index in timeseries data.
time_f = "timestamp"
#: The name of the field containing the heating degree-days.
heating_dd_f = "heating_degree_days"
#: The name of the field containing the cooling degree-days.
cooling_dd_f = "cooling_degree_days"
#: The name of the field containing the base consumption,
#: i.e. the consumption that is not affected by the weather.
base_f = "base"
