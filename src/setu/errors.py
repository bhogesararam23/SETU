"""Exceptions shared across the package.

Kept in one place so that callers can catch `setu.errors.SetuError` rather
than reaching into submodules for the right exception class.
"""

from __future__ import annotations


class SetuError(Exception):
    """Base class for every error this package raises deliberately.
    
    All errors raised by SETU inherit from this class, allowing callers to
    catch all SETU-specific errors with a single handler.
    """


class ConfigError(SetuError):
    """A configuration file is missing a key, or a value is out of range."""


class UpstreamError(SetuError):
    """A data source answered, but not with something usable.

    Carries the source name so the caller can decide whether a failure is
    fatal or whether a backstop can take over.
    
    Attributes:
        source: The name of the upstream data source that failed.
        message: Human-readable description of what went wrong.
    """

    source: str

    def __init__(self, source: str, message: str) -> None:
        super().__init__(f"{source}: {message}")
        self.source = source
        self.message = message


class UpstreamUnavailable(UpstreamError):
    """A data source could not be reached at all - network, DNS, TLS or 5xx.

    Distinct from UpstreamError on purpose: "unreachable" is a reason to fall
    back to a cached or alternative source, whereas a malformed response is a
    bug that falling back would hide.
    
    This error indicates a connectivity or availability issue, not a data
    format problem. Callers should retry or use cached data when this occurs.
    """


class CalibrationError(SetuError):
    """The calibration fit is not usable, e.g. too few co-located samples.
    
    Raised when calibration cannot be performed due to insufficient data,
    invalid parameters, or numerical instability.
    """
