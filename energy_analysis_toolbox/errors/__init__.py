"""This namespace defines the custom except for |eat|."""

from .base import (
    EATExcept,
    EATEmptyDataError,
)
from .invalid_timeseries import (
    EATInvalidTimeseriesError,
    EATUndefinedTimestepError,
    EATInvalidTimestepDurationError,
)
from .resampling import (
    EATResamplingError,
    EATEmptySourceError,
    EATEmptyTargetsError,
)
