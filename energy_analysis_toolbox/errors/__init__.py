"""Defines all custom errors used in |eat|."""

from .base import (
    EATEmptyDataError,
    EATError,
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
