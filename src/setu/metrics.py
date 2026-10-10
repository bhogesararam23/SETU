"""Error statistics, and the one that actually decides anything.

MAE and R-squared are reported because reviewers expect them. Threshold
accuracy is reported because SETU's output is a decision, not a number: a
calibration that correlates beautifully but misclassifies the NAAQS exceedance
edge is useless for enforcement, and it is possible to build one by accident.

The Kolkata result this is measured against (Nobell & Majumdar, AAQR, 2023,
doi:10.4209/aaqr.230010) is the honest bar: a cross-calibrated low-cost network
still flagged NAAQS attainment correctly on only about 91% of days.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _as_pair(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {p.shape}")
    keep = ~(np.isnan(a) | np.isnan(p))
    if keep.sum() == 0:
        raise ValueError("no finite pairs to score")
    return a[keep], p[keep]


def mae(actual, predicted) -> float:
    a, p = _as_pair(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual, predicted) -> float:
    a, p = _as_pair(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def bias(actual, predicted) -> float:
    """Signed mean error. Low MAE with a large bias means a systematic
    over-read that will silently inflate every exceedance count."""
    a, p = _as_pair(actual, predicted)
    return float(np.mean(p - a))


def nmae(actual, predicted) -> float:
    """Mean absolute error normalised by the observed mean, as a fraction."""
    a, p = _as_pair(actual, predicted)
    mean = float(np.mean(a))
    if mean <= 0:
        raise ValueError("cannot normalise against a non-positive mean")
    return mae(a, p) / mean


def r2(actual, predicted) -> float:
    a, p = _as_pair(actual, predicted)
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    if ss_tot == 0:
        raise ValueError("actual values are constant; R^2 is undefined")
    return 1.0 - ss_res / ss_tot


def threshold_accuracy(actual, predicted, threshold: float,
                       flag_when: str = "at_or_above") -> float:
    """Fraction of samples where predicted and actual agree about the breach.

    This is the metric a municipal officer is exposed to. `flag_when` is
    'at_or_above' for a standard like the 60 ug/m3 PM2.5 24-hour NAAQS, where
    reaching the value is already non-attainment.
    """
    a, p = _as_pair(actual, predicted)
    if flag_when == "at_or_above":
        agree = (a >= threshold) == (p >= threshold)
    elif flag_when == "above":
        agree = (a > threshold) == (p > threshold)
    else:
        raise ValueError(f"unknown flag_when {flag_when!r}")
    return float(np.mean(agree))


def false_alarm_rate(actual, predicted, threshold: float) -> float:
    """Of the alerts raised, how many were against clean air.

    A dispatcher that is wrong half the time stops being dispatched to. This is
    reported separately from threshold accuracy for exactly that reason: a high
    overall accuracy can still hide an unusable alert stream.
    """
    a, p = _as_pair(actual, predicted)
    predicted_breach = p >= threshold
    if predicted_breach.sum() == 0:
        return 0.0
    return float(np.mean(~(a[predicted_breach] >= threshold)))


def detection_rate(actual, predicted, threshold: float) -> float:
    """Of the real exceedances, how many were flagged. Missing these is the
    more expensive error in a public-health sense."""
    a, p = _as_pair(actual, predicted)
    real_breach = a >= threshold
    if real_breach.sum() == 0:
        return float("nan")
    return float(np.mean(p[real_breach] >= threshold))


def persistence_skill(actual, predicted, baseline) -> float:
    """RMSE reduction against a persistence forecast.

    A model that cannot beat "same value as last hour" should not be deployed,
    however good its R^2 looks on a clean plot.
    """
    a, p = _as_pair(actual, predicted)
    b = np.asarray(baseline, dtype=float)[~np.isnan(np.asarray(baseline, dtype=float))]
    if b.shape != a.shape:
        raise ValueError("baseline must align with actual")
    base_rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    if base_rmse == 0:
        raise ValueError("degenerate baseline")
    return 1.0 - rmse(a, p) / base_rmse


@dataclass(frozen=True)
class Scorecard:
    n: int
    mae: float
    rmse: float
    nmae: float
    r2: float
    bias: float
    threshold_accuracy: float | None = None
    false_alarm_rate: float | None = None
    detection_rate: float | None = None

    def as_dict(self) -> dict[str, object]:
        out = {k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in self.__dict__.items()}
        return out


def score(actual, predicted, threshold: float | None = None) -> Scorecard:
    a, p = _as_pair(actual, predicted)
    card = Scorecard(
        n=int(len(a)),
        mae=mae(a, p),
        rmse=rmse(a, p),
        nmae=nmae(a, p),
        r2=r2(a, p),
        bias=bias(a, p),
    )
    if threshold is None:
        return card
    return Scorecard(
        **{**card.__dict__,
           "threshold_accuracy": threshold_accuracy(a, p, threshold),
           "false_alarm_rate": false_alarm_rate(a, p, threshold),
           "detection_rate": detection_rate(a, p, threshold)}
    )


def attainment_series(daily_mean: Sequence[float], standard: float,
                      max_consecutive: int = 2) -> list[bool]:
    """NAAQS 24-hour compliance as the standard is actually written.

    The footnote to CPCB's 2009 notification says a 24-hour value must be met
    on 98% of days in a year and must not be exceeded on two consecutive days.
    Counting single exceedance days therefore overstates non-attainment, and a
    prototype that ignores the consecutiveness rule will disagree with a
    municipal officer's own register on the first foggy spell of November.
    """
    values = np.asarray(list(daily_mean), dtype=float)
    breach = values > standard
    out: list[bool] = []
    run = 0
    for flag in breach:
        run = run + 1 if flag else 0
        # The non-compliance event is the day the second consecutive breach
        # lands on, so the comparison is inclusive.
        out.append(run >= max_consecutive)
    return out
