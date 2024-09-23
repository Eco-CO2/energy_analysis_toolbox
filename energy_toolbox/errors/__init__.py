"""This namespace defines the custom except for |et|.
"""

from .base import CTExcept, CTEmptyDataError
from .invalid_timeseries_errors import (
    CTInvalidTimeseriesError,
    CTUndefinedTimestepError,
    CTInvalidTimestepDurationError,
)
from .resampling_errors import (
    CTResamplingError,
    CTEmptySourceError,
    CTEmptyTargetsError,
)
