"""Drift detection and quarantine.

Low-cost PM sensors drift; that is the documented reason CPCB does not accept
their data for regulatory use. A calibration that is fitted once and never
monitored will quietly become wrong, and a mean-based quality check will not
notice, because drift is slow and monotone and therefore looks like the signal.

This is a two-sided CUSUM on the residual between a node's calibrated reading
and a trusted reference. It is chosen over a control chart on the raw value
because it is insensitive to the air actually getting dirtier: a real episode
moves the signal and the residual together, so only a change in the *relationship*
raises an alarm.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Iterable

import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class DriftAlarm:
    at: pd.Timestamp | None
    statistic: float
    direction: str          # "up" reads high, "down" reads low
    mean_offset_so_far: float

    def as_dict(self) -> dict[str, object]:
        return {
            "at": str(self.at) if self.at is not None else None,
            "statistic": round(self.statistic, 3),
            "direction": self.direction,
            "mean_offset_so_far": round(self.mean_offset_so_far, 3),
        }


class CusumMonitor:
    """Streaming two-sided CUSUM with a warm-up that estimates sigma robustly.

    Parameters
    ----------
    allowance : float
        Tolerance in sigmas. Deviations smaller than this are treated as noise
        and do not accumulate. 0.5 sigma is the usual choice for detecting a
        moderate sustained shift.
    interval : float
        Decision threshold in sigmas. Larger means fewer, later alarms.
        10 sigma is chosen so the average run length on clean data is far
        longer than a season: at 5 sigma the false-alarm rate is roughly one
        per few hundred hours, which would quarantine a healthy node every
        month and teach operators to ignore the quarantine column.
    warm_up : int
        Samples used to estimate sigma before any alarm can fire. Two weeks of
        hourly data, not four days: a MAD scale from 96 samples carries roughly
        a 25% error, and since the decision interval is expressed in sigmas,
        that error goes straight into how often good nodes get quarantined.
    cooldown : int
        Samples after an alarm during which no further alarm is raised. A
        sustained drift is one incident, not one incident per hour.
    reset_on_alarm : bool
        After an alarm the statistic is reset, so a long drift raises a sparse
        sequence of events rather than one per hour.
    """

    def __init__(self, allowance: float = 0.5, interval: float = 10.0,
                 warm_up: int = 336, reset_on_alarm: bool = True,
                 cooldown: int = 168):
        if allowance <= 0:
            raise ValueError("allowance must be positive")
        if interval <= allowance:
            raise ValueError("interval must exceed allowance")
        if warm_up < 8:
            raise ValueError("warm_up is too short to estimate sigma")
        if cooldown < 1:
            raise ValueError("cooldown must be at least 1 sample")
        self.allowance = allowance
        self.interval = interval
        self.warm_up = warm_up
        self.reset_on_alarm = reset_on_alarm
        self.cooldown = cooldown

        self._residuals: list[float] = []
        self._baseline_mean: float = 0.0
        self._pos = 0.0
        self._neg = 0.0
        self._sigma: float | None = None
        self._alarms: list[DriftAlarm] = []
        self._n = 0
        self._since_alarm = 10 ** 9

    # -- internals ---------------------------------------------------------
    def _estimate_sigma(self) -> float:
        """MAD-based sigma from the warm-up window only.

        Re-estimating on the full history lets a developing drift inflate its
        own detection threshold, which is the classic way this monitor fails
        silently: the longer a node has been wrong, the harder it is to notice.
        """
        arr = np.asarray(self._residuals[:self.warm_up], dtype=float)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        if mad <= 1e-9:
            std = float(np.std(arr))
            return max(std, 0.5)
        return max(1.4826 * mad, 0.5)

    # -- public ------------------------------------------------------------
    def update(self, residual: float, at: pd.Timestamp | None = None) -> DriftAlarm | None:
        if not math.isfinite(residual):
            return None
        self._residuals.append(float(residual))
        self._n += 1

        if self._sigma is None:
            if self._n < self.warm_up:
                return None
            # Centre on the in-control mean as well as scaling by sigma. A node
            # that is stably 0.8 ug/m3 high has a calibration offset, not drift;
            # accumulating on the raw residual would quarantine it eventually no
            # matter how well behaved it is, and on a 92-day window that is
            # exactly what happened - 40 of 49 healthy nodes were quarantined.
            # What must be caught is the residual MOVING.
            self._baseline_mean = float(np.mean(
                self._residuals[:self.warm_up]))
            self._sigma = self._estimate_sigma()

        centred = residual - self._baseline_mean
        k = self.allowance * self._sigma
        h = self.interval * self._sigma
        self._pos = max(0.0, self._pos + centred - k)
        self._neg = max(0.0, self._neg - centred - k)

        self._since_alarm += 1
        alarm = None
        if self._since_alarm < self.cooldown:
            return None
        if self._pos > h:
            alarm = DriftAlarm(at, self._pos / self._sigma, "up",
                               float(np.mean(self._residuals[-self._n:])))
        elif self._neg > h:
            alarm = DriftAlarm(at, self._neg / self._sigma, "down",
                               float(np.mean(self._residuals[-self._n:])))
        if alarm is not None:
            self._alarms.append(alarm)
            if self.reset_on_alarm:
                self._pos = 0.0
                self._neg = 0.0
            self._since_alarm = 0
        return alarm

    def feed(self, residuals: Iterable[float], index: pd.DatetimeIndex | None = None
             ) -> list[DriftAlarm]:
        out = []
        values = list(residuals)
        if index is not None and len(index) != len(values):
            raise ValueError("index and residuals must align")
        for i, r in enumerate(values):
            at = index[i] if index is not None else None
            alarm = self.update(float(r), at=at)
            if alarm is not None:
                out.append(alarm)
        return out

    @property
    def alarms(self) -> list[DriftAlarm]:
        return list(self._alarms)

    @property
    def sigma(self) -> float | None:
        return self._sigma

    @property
    def state(self) -> dict[str, object]:
        return {
            "samples": self._n,
            "warm_up_complete": self._sigma is not None,
            "sigma": None if self._sigma is None else round(self._sigma, 3),
            "baseline_mean": round(self._baseline_mean, 3),
            "pos_statistic": round(self._pos / self._sigma, 2) if self._sigma else 0.0,
            "neg_statistic": round(self._neg / self._sigma, 2) if self._sigma else 0.0,
            "alarms": len(self._alarms),
        }


def residual_series(calibrated: pd.Series, reference: pd.Series) -> pd.Series:
    common = calibrated.index.intersection(reference.index)
    return (calibrated.loc[common] - reference.loc[common]).dropna()


def quarantine_nodes(results: dict[str, tuple[CusumMonitor, int]],
                     max_alarms: int = 1) -> list[str]:
    """Nodes whose drift monitor has fired at least `max_alarms` times.

    The engine drops these from the grid rather than continuing to average them
    in. A quarantined node stays on the dashboard with its calibration age,
    because an operator has to be able to see that the system is choosing to
    ignore a sensor, and why.
    """
    return sorted(node for node, (monitor, _) in results.items()
                  if len(monitor.alarms) >= max_alarms)
