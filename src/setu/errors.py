"""Exceptions shared across the package.

Kept in one place so that callers can catch `setu.errors.SetuError` rather
than reaching into submodules for the right exception class.
"""


class SetuError(Exception):
    """Base class for every error this package raises deliberately."""


class ConfigError(SetuError):
    """A configuration file is missing a key, or a value is out of range."""


class UpstreamError(SetuError):
    """A data source answered, but not with something usable.

    Carries the source name so the caller can decide whether a failure is
    fatal or whether a backstop can take over.
    """

    def __init__(self, source: str, message: str):
        super().__init__(f"{source}: {message}")
        self.source = source


class UpstreamUnavailable(UpstreamError):
    """A data source could not be reached at all - network, DNS, TLS or 5xx.

    Distinct from UpstreamError on purpose: "unreachable" is a reason to fall
    back to a cached or alternative source, whereas a malformed response is a
    bug that falling back would hide.
    """


class CalibrationError(SetuError):
    """The calibration fit is not usable, e.g. too few co-located samples."""
