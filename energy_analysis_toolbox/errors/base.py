"""This module defines custom errors."""


class EATExcept(Exception):
    """The base class for all except in |eat|.

    All exceptions of |eat| library should inherit this class, such that exceptions
    specific to this lib can easily be caught/identified.
    """


class EATEmptyDataError(EATExcept):
    """An empty data container is passed, but this case cannot be managed."""
