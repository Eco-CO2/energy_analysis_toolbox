
class CTExcept(Exception):
    """The base class for all except in |et|.

    All exceptions of |et| library should inherit this class, such that exceptions
    specific to this lib can easily be caught/identified.
    """

class CTEmptyDataError(CTExcept):
    """An empty data container is passed, but this case cannot be managed.
    """
