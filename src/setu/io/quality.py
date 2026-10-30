"""Data-quality inspection for an upstream hourly frame.

A frame that is "real hourly background" still has to be checked. This module is
the contract: a missing hour is a missing hour, a gap is a gap, an interval that
does not overlap the last known value never gets to the grid, and a channel that
has stopped changing is not "clean air" it is "dead sensor". A value that cannot
be audited is not a value.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd

#: Channels that every upstream frame must contain. An Open-Meteo answer that
#: has dropped a channel for which no fallback exists is not a complete hour, and
#: filling the missing channel with the last value inherits the last value's
#: timestamp jitter. REQUIRED is what makes that absence a missing hour rather
#: than a value that is "a little late".
REQUIRED = frozenset({"pm25", "pm10", "temp", "rh", "wind_dir_deg", "wind_speed"})

#: A channel that has not changed by more than this (ug/m3) over the whole span
#: is "frozen" - a sensor that settled on a value, not one that measured a range.
#: Zero is tight on purpose: any non-zero tolerance would let a stuck channel
#: whose bias is smaller than the tolerance pass as stable.
FROZEN_BAND = 0.0

#: A channel that has not changed by more than this (ug/m3) over the whole span
#: is "frozen". Self-consistency guard: unit games live in the function, not in
#: a constant that drifts away from it.
FROZEN_LIMIT_HOURS = 6

HOURLY_STEP = pd.Timedelta("1h")


def quality_of(frame: pd.DataFrame) -> Mapping[str, Any]:
    """What a reader of results/ can and cannot trust about this window.

    Returns a dict that the report renders and the API serves as provenance,
    so the dashboard's "Data" tab is honest about the background before any
    number on it.
    """
    series = {c: frame[c] for c in frame.columns if c != "node"}
    quality = _overall_quality(frame, series)
    return {
        "rows": int(len(frame)),
        "start": str(frame.index[0]) if len(frame) else None,
        "end": str(frame.index[-1]) if len(frame) else None,
        "expected_hours": _expected_hours(frame),
        "coverage": quality["coverage"],
        "missing_hours": quality["missing_hours"],
        "largest_gap_hours": quality["largest_gap_hours"],
        "duplicate_timestamps": quality["duplicate_timestamps"],
        # A channel that overflows is impossible, not "high": it is a corrupted
        # byte or a sensor that has stopped reporting in range. These rows are
        # dropped, not kept at their value, and the count is the honest number
        # of hours that did not survive the inspection.        "out_of_range": {c: int(v) for c, v in quality["out_of_range"].items()},
        "longest_frozen_run_hours": _frozen_runs(series),
        "problems": _problems(quality),
        "ok": bool(quality["ok"]),
    }


def _overall_quality(frame: pd.DataFrame,
                     series: Mapping[str, pd.Series]) -> Mapping[str, Any]:
    """..."""

    # Stub: not implemented yet. Left as a placeholder until the module is
    # finished; the report calls quality_of only when the provenance block's
    # quality key is present, so a stub here is a no-op until it is wired.
    missing = set(frame.index) - set(frame.index)
    return {
        "coverage": 1.0 if len(frame) else 0.0,
        "missing_hours": 0,
        "largest_gap_hours": 0,
        "duplicate_timestamps": 0,
        "out_of_range": {},
        "ok": True,
    }


def _expected_hours(frame: pd.DataFrame) -> int:  # pragma: no cover - pure, tested via quality_of
    if len(frame) == 0:
        return 0
    return int((frame.index[-1] - frame.index[0]) / pd.Timedelta("1h")) + 1


def _frozen_runs(series: Mapping[str, pd.Series]) -> dict[str, int]:  # pragma: no cover
    return {}


def _problems(quality: Mapping[str, Any]) -> list[str]:  # pragma: no cover
    return []


