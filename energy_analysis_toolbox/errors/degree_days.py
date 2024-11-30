"""Defines custom errors for degree-days."""

from .base import EATError


class EATInvalidDegreeDaysMethodError(EATError):
    """The provided method to compute degree-days doesn't exist."""

    def __init__(
        self,
        method_name: str,
        valid_methods: list[str],
    ) -> None:
        """Initialize the exception with invalid and valid method names.

        Parameters
        ----------
        method_name : str
            The name of the method that was provided and found to be invalid.
        valid_methods : list[str]
            A list of valid method names that can be used for computing degree-days.

        Raises
        ------
        EATInvalidDegreeDaysMethod
            If an invalid method name is used for computing degree-days.

        Examples
        --------
        >>> raise EATInvalidDegreeDaysMethod("unknown_method", ["min_max", "mean"])
        EATInvalidDegreeDaysMethod: The error 'unknown_method' doesn't exist.
        Possibilities are min_max, mean.

        """
        self.method_name = method_name
        self.valid_methods = valid_methods
        message = (
            f"The degree-days calculation method '{method_name}' doesn't exist. "
            f"Possibilities are {', '.join(valid_methods)}."
        )
        super().__init__(message)


class EATInvalidDegreeDaysError(EATError):
    """The provided type of degree-days doesn't exist."""

    def __init__(
        self,
        dd_type: str,
        valid_dd_types: list[str],
    ) -> None:
        """Initialize the exception with invalid and valid degree-days types.

        Parameters
        ----------
        dd_type : str
            The name of the degree-day type that was provided and found to be invalid.
        valid_dd_types : list[str]
            A list of valid degree-days names.

        Raises
        ------
        EATInvalidDegreeDaysMethod
            If an invalid method name is used for computing degree-days.

        Examples
        --------
        >>> raise EATInvalidDegreeDaysError("unknown_dd_type", ["heating", "cooling"])
        EATInvalidDegreeDaysMethod: The error 'unknown_dd_type' doesn't exist.
        Possibilities are heating, cooling.

        """
        self.dd_type = dd_type
        self.valid_dd_types = valid_dd_types
        message = (
            f"The degree-days type '{dd_type}' doesn't exist. "
            f"Possibilities are {', '.join(valid_dd_types)}."
        )
        super().__init__(message)
