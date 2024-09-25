"""This namespace defines the custom except for |et|.
"""

from .base import (
    ETExcept,
    ETEmptyDataError,
)
from .invalid_timeseries import (
    ETInvalidTimeseriesError,
    ETUndefinedTimestepError,
    ETInvalidTimestepDurationError,
)
from .resampling import (
    CTResamplingError,
    CTEmptySourceError,
    CTEmptyTargetsError,
)
