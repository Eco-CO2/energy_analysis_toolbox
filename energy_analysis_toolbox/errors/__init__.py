"""This namespace defines the custom except for |eat|."""

from .base import (
    EATEmptyDataError,
    EATExcept,
)
from .invalid_timeseries import (
    EATInvalidTimeseriesError,
    EATInvalidTimestepDurationError,
    EATUndefinedTimestepError,
)
from .resampling import (
    EATEmptySourceError,
    EATEmptyTargetsError,
    EATResamplingError,
)
