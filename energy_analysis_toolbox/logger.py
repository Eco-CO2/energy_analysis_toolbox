"""Define the logger for `energy_analysis_toolbox`."""

import inspect
import logging
import sys


def init_logging(
    base_logger_name: str = "energy_analysis_toolbox",
    level: int | str = "INFO",
    min_logger_level_stdout: int | str = "DEBUG",
) -> logging.Logger:
    """Return the project logger.

    Parameters
    ----------
    base_logger_name : str, optional
        The name of the logger. Default is ``energy_analysis_toolbox``.
        If an instance already exists, a number will automatically be
        added to make it unique. That means that if the base_logger_name
        is "cow", the first logger will be called "cow", the second "cow2", ...
    level : str or int, optional
        The level of the logger. Default is ``INFO``.
    min_logger_level_stdout: str, int, optional
        Minimum logger level below which no message is transferred to stdout
        (i.e. not printed). Default is ``"DEBUG"``.

    Returns
    -------
    Logger
        The logger object.


    .. note::

        The usual use is the following:

        .. code-block:: python

            log = init_logging()
            log.debug("a debug message")
            log.info("an info")


    """
    logger_name = _unique_logger_name(base_logger_name)
    log = logging.getLogger(logger_name)
    if not log.hasHandlers():
        log.setLevel(level)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(min_logger_level_stdout)
        logger_max_length = len(base_logger_name) + 2
        message_fmt = f"%(name)-{logger_max_length}s :: %(levelname)-8s :: %(message)s"
        formatter = MultiLineFormatter(message_fmt)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    return log


def _unique_logger_name(base_logger_name: str) -> str:
    """Generate a unique logger name by appending a counter to the base logger name.

    This function checks if a logger with the specified base name already has
    handlers attached. If it does, the function appends a counter (starting from
    1) to the base logger name, incrementing the counter until it finds a
    logger name that does not already exist. This ensures that the returned
    logger name is unique and can be safely used to create a new logger.

    Parameters
    ----------
    base_logger_name : str
        The base name for the logger. This name will be used as the starting point
        for generating the unique logger name.

    Returns
    -------
    str
        A unique logger name derived from the base logger name. If the base name
        is already unique, it returns the base name itself; otherwise, it returns
        a name in the format `{base_logger_name}_{counter}` where `counter`
        is the smallest integer that makes the name unique.

    Example
    -------
    >>> logger_name = _unique_logger_name("my_logger")
    >>> print(logger_name)
    'my_logger_1'  # If 'my_logger' already exists with handlers

    Notes
    -----
    This function is useful for dynamically creating loggers when multiple
    instances of a class or module might be initialized, ensuring each has a
    distinct logger to avoid conflicts in logging output.

    """
    counter = 1
    logger_name = base_logger_name
    max_counter = 10  # to avoid infinite loops
    while logging.getLogger(logger_name).hasHandlers() or counter < max_counter:
        logger_name = f"{base_logger_name}_{counter}"
        counter += 1
    return logger_name


class MultiLineFormatter(logging.Formatter):
    """Multi-line formatter."""

    def get_header_length(self, record: logging.LogRecord) -> int:
        """Calculate the header length of the given log record.

        This method formats a `LogRecord` without the message and calculates the length
        of the formatted string. The header includes information such as the logger
        name, log level, timestamp, and any other metadata included in the log format.

        Parameters
        ----------
        record : logging.LogRecord
            The log record for which the header length is to be calculated. This record
            contains all the information pertinent to the log entry, such as the logger
            name, log level, file name, line number, and timestamp.

        Returns
        -------
        int
            The length of the formatted header string, which is used to determine the
            amount of indentation needed for multi-line log messages.

        Example
        -------
        >>> record = logging.LogRecord(name="test_logger", level=logging.INFO,
        ... pathname="test.py", lineno=10, msg="Test message", args=None,
        ... exc_info=None)
        >>> formatter = MultiLineFormatter()
        >>> length = formatter.get_header_length(record)
        >>> print(length)
        40  # (for example, depending on the log format)

        """
        return len(
            super().format(
                logging.LogRecord(
                    name=record.name,
                    level=record.levelno,
                    pathname=record.pathname,
                    lineno=record.lineno,
                    msg="",
                    args=(),
                    exc_info=None,
                ),
            ),
        )

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with added indentation for multi-line messages.

        This method extends the default formatting of a log record by adding indentation
        to any multi-line log messages. The indentation is based on the length of the
        header of the log record, ensuring that all subsequent lines of the message are
        aligned with the header.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format. This contains all the information necessary to
            produce the log output, including the log level, message, logger name,
            and any other relevant details.

        Returns
        -------
        str
            The formatted log message, with added indentation for multi-line entries.
            The first line contains the standard log header, while subsequent lines
            are indented according to the header's length.

        """
        frame = inspect.currentframe()
        method_name = "unknown"
        max_frames_to_go_up = 8
        frames_up = 0
        while frame is not None and frames_up < max_frames_to_go_up:
            frame = frame.f_back
            frames_up += 1
        if frame is not None:
            method_name = frame.f_code.co_name
        record.name = f"{record.name}/{method_name}"
        indent = " " * self.get_header_length(record)
        head, *trailing = super().format(record).splitlines(keepends=True)
        return head + "".join(indent + line for line in trailing)
