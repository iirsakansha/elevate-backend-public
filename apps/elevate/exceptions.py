# elevate/utils/exceptions.py
from rest_framework.views import exception_handler


def custom_exception_handler(exc, context):
    """Custom exception handler for REST Framework."""
    response = exception_handler(exc, context)
    if response is not None:
        response.data["error"] = str(exc)
    return response


class InsufficientDataError(Exception):
    """Raised when analysis data is incomplete."""

    pass


# elevate/exceptions.py
class InvalidFileError(Exception):
    """Raised when a file is invalid or cannot be processed."""

    pass


class AnalysisProcessingError(Exception):
    """Raised when an error occurs during analysis processing."""

    def __init__(self, message, traceback=None):
        super().__init__(message)
        self.traceback = traceback


class InvalidCategoryError(Exception):
    """Raised when category data is invalid."""

    pass


class InvalidDateError(Exception):
    """Raised when date data is invalid."""

    pass


class InvalidTimeFormatError(Exception):
    """Raised when time format is invalid."""

    pass
