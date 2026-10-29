"""What the incoming frame has to be before anything downstream reads it.

Every stage in this project assumes the background frame is hourly, complete on
six channels, and physically possible. That assumption was never checked - it
was implied by the loader's inner join and then trusted for 1,772 hours. This
module states it as a contract with numbers: coverage, gaps, duplicates,
out-of-range values, and the frozen-sensor case a completeness check cannot see,
because a sensor stuck at 41.0 ug/m3 is complete and wrong.

Nothing here repairs data. A check that silently fills a gap is a check that
cannot report one, and the gaps are what the validation report has to quote.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

#: Physically possible ranges for the channels the pipeline requires. Wide on
#: purpose: this is a sanity filter, not a calibration. Its job is to catch a
#: unit change or a shifted decimal, not to argue about a high dust hour.
RANGES: dict[str, tuple[float, float]] = {
    "pm25": (0.0, 1000.0),
    "pm10": (0.0, 2000.0),
    "co": (0.0, 50_000.0),
    "no2": (0.0, 2000.0),
    "o3": (0.0, 1000.0),
    "so2": (0.0, 2000.0),
    "temp_c": (-10.0, 55.0),
    "rh": (0.0, 100.0),
    "wind_speed_ms": (0.0, 50.0),
    "wind_dir_deg": (0.0, 360.0),
}

#: Rows the downstream code cannot work without.
REQUIRED = ("pm25", "pm10", "temp_c", "rh", "wind_speed_ms", "wind_dir_deg")

#: A channel that reports the same value this many hours in a row is not
#: measuring anything. Twelve is longer than any real calm spell in a channel
#: that is supposed to vary with the weather, and short enough to catch a
#: sensor that died overnight.
FROZEN_LIMIT_HOURS = 12


@dataclasses.dataclass(frozen=True)
class QualityReport:
    rows: int
    start: str | None
    end: str | None
    expected_hours: int
    coverage: float
    missing_hours: int
    largest_gap_hours: int
    duplicate_timestamps: int
    out_of_range: Mapping[str, int]
    longest_frozen_run_hours: Mapping[str, int]
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "start": self.start,
            "end": self.end,
            "expected_hours": self.expected_hours,
            "coverage": round(self.coverage, 4),
            "missing_hours": self.missing_hours,
            "largest_gap_hours": self.largest_gap_hours,
            "duplicate_timestamps": self.duplicate_timestamps,
            "out_of_range": dict(self.out_of_range),
            "longest_frozen_run_hours": dict(self.longest_frozen_run_hours),
            "problems": list(self.problems),
            "ok": self.ok,
        }


def _longest_run(values: pd.Series) -> int:
    """Longest run of identical, non-null values."""
    best = run = 0
    previous = object()
    for value in values.to_numpy():
        if pd.isna(value):
            run = 0
            previous = object()
            continue
        if value == previous:
            run += 1
        else:
            run = 1
            previous = value
        best = max(best, run)
    return best


def check_hourly(df: pd.DataFrame, required: Sequence[str] = REQUIRED,
                 ranges: Mapping[str, tuple[float, float]] | None = None,
                 min_coverage: float = 0.90) -> QualityReport:
    """Summarise a frame's integrity against the contract the pipeline assumes.

    A frame with one row is not "100% covered": coverage is measured against the
    span the frame claims to describe, which is what catches the case that
    matters - a long, silent outage between two tidy endpoints.
    """
    ranges = ranges or RANGES
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"frame is missing required channels {missing_cols}")

    rows = int(len(df))
    if rows == 0:
        return QualityReport(0, None, None, 0, 0.0, 0, 0, 0, {}, {}, ("no rows",))

    duplicates = int(df.index.duplicated().sum())
    span = (df.index[-1] - df.index[0]) / pd.Timedelta(hours=1) + 1
    expected = int(round(float(span)))
    missing = max(0, expected - (rows - duplicates))
    coverage = (rows - duplicates) / expected if expected > 0 else 0.0

    # Largest hole: the longest gap between consecutive timestamps, which is the
    # number that says whether the missing hours were drizzle or an outage.
    steps = pd.Series(df.index).diff().dropna() / pd.Timedelta(hours=1)
    largest_gap = int(max(0, steps.max() - 1)) if len(steps) else 0

    out_of_range: dict[str, int] = {}
    for col in required:
        if col not in ranges:
            continue
        lo, hi = ranges[col]
        series = pd.to_numeric(df[col], errors="coerce")
        bad = int(((series < lo) | (series > hi)).sum())
        if bad:
            out_of_range[col] = bad

    frozen = {col: _longest_run(df[col]) for col in required if col in df.columns}

    problems: list[str] = []
    if coverage < min_coverage:
        problems.append(
            f"coverage {coverage:.1%} over the window is below {min_coverage:.0%}")
    if duplicates:
        problems.append(f"{duplicates} duplicate timestamps")
    if out_of_range:
        problems.append(
            "values outside physical range: "
            + ", ".join(f"{k} x{v}" for k, v in sorted(out_of_range.items())))
    stale = {k: v for k, v in frozen.items() if v > FROZEN_LIMIT_HOURS}
    if stale:
        problems.append(
            "channels frozen longer than "
            f"{FROZEN_LIMIT_HOURS} h: "
            + ", ".join(f"{k} {v} h" for k, v in sorted(stale.items())))
    non_finite = int(sum((~np.isfinite(pd.to_numeric(df[c], errors="coerce")))
                         .sum() for c in required))
    if non_finite:
        problems.append(f"{non_finite} non-finite values in required channels")

    return QualityReport(
        rows=rows,
        start=str(df.index[0]),
        end=str(df.index[-1]),
        expected_hours=expected,
        coverage=coverage,
        missing_hours=missing,
        largest_gap_hours=largest_gap,
        duplicate_timestamps=duplicates,
        out_of_range=out_of_range,
        longest_frozen_run_hours=frozen,
        problems=tuple(problems),
    )


def fingerprint(df: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    """A short digest of the data a run actually used.

    Two runs that disagree should be able to say whether they read the same
    numbers before anyone re-reads the code. The digest is over rounded values,
    so a float that differs in the twelfth decimal does not make a run look
    unreproducible when it is not.
    """
    cols = sorted(c for c in (columns or list(df.columns)) if c in df.columns)
    frame = df[cols].round(6)
    digest = hashlib.sha256()
    digest.update("|".join(cols).encode("utf-8"))
    digest.update(str(len(frame)).encode("utf-8"))
    if len(frame):
        digest.update(str(frame.index[0]).encode("utf-8"))
        digest.update(str(frame.index[-1]).encode("utf-8"))
    digest.update(np.ascontiguousarray(frame.to_numpy(dtype="float64",
                                                     na_value=np.nan)).tobytes())
    return digest.hexdigest()[:16]
