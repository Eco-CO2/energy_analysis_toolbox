"""Defines basic custom errors."""


class EATError(Exception):
    """The base class for all except in |eat|.

    All exceptions of |eat| library should inherit this class, such that exceptions
    specific to this lib can easily be caught/identified.
    """


class EATEmptyDataError(EATError):
    """An empty data container is passed, but this case cannot be managed."""
