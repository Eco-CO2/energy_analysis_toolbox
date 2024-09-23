from .base import CTExcept, CTEmptyDataError

class CTResamplingError(CTExcept):
    """A resampling operation is impossible.

    This base class is used when an invalid resampling operation is attempted in |et|.
    Derived classed may be used for more specific resampling errors.
    """
    pass

class CTEmptySourceError(CTResamplingError, CTEmptyDataError):
    """Resampling an empty timeseries to specific instants is impossible.

    This exception is used when :

    - a resampling operation is undefined/meaningless when the source data is empty,
    - empty data is passed as source.

    """
    pass

class CTEmptyTargetsError(CTResamplingError, CTEmptyDataError):
    """Resampling a timeseries to empty targets is meaningless in this situation.

    This exception is used when :

    - a resampling operation is undefined/meaningless when the target instants
      set is empty,
    - empty data is passed as targets.

    """
    pass
