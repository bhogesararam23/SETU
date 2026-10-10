"""Closing the loop: from a source posterior to a dispatch, and back again.

Everything upstream produces a number. This is the part that decides whether a
truck gets sent, and the part that has to be able to say afterwards whether it
worked.

The gates are ordered deliberately, cheapest and most certain first:

  1. source posterior must clear the floor           (is this even our problem?)
  2. persistence over k of the last n windows        (or is it one odd reading?)
  3. the uncertainty interval must exclude threshold (could this still be clean?)
  4. dwell since the last dispatch of this lever+zone(have we already done this?)
  5. hysteresis on the clear side                    (stop flapping)

Gate 3 is the one that makes the calibration interval load-bearing rather than
decorative. Gate 5 is the one that keeps a dispatcher from receiving the same
alert forty times in a night, which is how pilot systems like this usually die.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections import deque
from typing import Deque, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import ActionTable, Gates, Lever
from ..errors import ConfigError

_ACTION_IDS = itertools.count(1)


@dataclasses.dataclass(frozen=True)
class Observation:
    node: str
    at: pd.Timestamp
    value: float
    sigma: float
    posterior: Mapping[str, float]
    threshold: float
    coarse_ratio: float = 0.0
    wind_from_deg: float = 0.0
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ConfigError("sigma cannot be negative")
        if self.value < 0:
            raise ConfigError("a concentration cannot be negative")

    @property
    def interval_low(self) -> float:
        return self.value - self.sigma if self.low is None else float(self.low)

    @property
    def top_source(self) -> str:
        if not self.posterior:
            return "unknown"
        return max(self.posterior, key=lambda k: self.posterior[k])

    @property
    def confidence(self) -> float:
        return float(max(self.posterior.values())) if self.posterior else 0.0


@dataclasses.dataclass(frozen=True)
class Action:
    id: str
    node: str
    source: str
    lever: str
    owner: str
    dispatched_at: pd.Timestamp
    recheck_at: pd.Timestamp | None
    trigger_value: float
    trigger_sigma: float
    threshold: float
    evidence: Mapping[str, object]
    requires_signoff: bool

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(
            self, dict_factory=lambda pairs: {
                k: (str(v) if isinstance(v, pd.Timestamp) else v)
                for k, v in pairs})


@dataclasses.dataclass(frozen=True)
class Suppressed:
    node: str
    at: pd.Timestamp
    reason: str
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"node": self.node, "at": str(self.at), "reason": self.reason,
                "detail": self.detail}


@dataclasses.dataclass(frozen=True)
class EffectResult:
    action_id: str
    node: str
    lever: str
    before_mean: float
    after_mean: float
    reduction_pct: float
    expected_pct: float | None
    verdict: str
    n_before: int
    n_after: int

    def as_dict(self) -> dict[str, object]:
        return {k: (round(v, 2) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclasses.dataclass
class _ZoneState:
    windows: Deque[bool] = dataclasses.field(default_factory=deque)
    armed: bool = False
    last_dispatch: dict[str, pd.Timestamp] = dataclasses.field(default_factory=dict)
    last_value: float | None = None

    def __post_init__(self) -> None:
        self.windows = deque(maxlen=8)


class ResponseEngine:
    """Stateful across a run; one instance per monitored zone set."""

    def __init__(self, table: ActionTable, window_capacity: int = 8):
        self.table = table
        self.gates: Gates = table.gates
        self.window_capacity = window_capacity
        self._zones: dict[str, _ZoneState] = {}
        self.actions: list[Action] = []
        self.suppressed: list[Suppressed] = []

    # -- helpers -----------------------------------------------------------
    def _zone(self, node: str) -> _ZoneState:
        if node not in self._zones:
            self._zones[node] = _ZoneState(
                windows=deque(maxlen=self.window_capacity))
        return self._zones[node]

    def _lever_for(self, source: str, zone_value: float) -> Lever | None:
        for lever in self.table.levers_for(source):
            if lever.half_life_min is None:
                # A lever with no measured decay can still be issued once -
                # it just never gets a re-dose schedule.
                return lever
            return lever
        return None

    def _passes_persistence(self, state: _ZoneState) -> bool:
        n = min(self.gates.persistence_of, self.window_capacity)
        recent = list(state.windows)[-n:]
        if len(recent) < n:
            return False
        return sum(recent) >= self.gates.persistence_windows

    def _uncertainty_clears(self, obs: Observation) -> bool:
        if not self.gates.uncertainty_must_exclude_threshold:
            return True
        return obs.interval_low > obs.threshold

    def _within_dwell(self, state: _ZoneState, lever: Lever,
                      at: pd.Timestamp) -> bool:
        last = state.last_dispatch.get(lever.key)
        if last is None:
            return False
        elapsed = (at - last).total_seconds() / 60.0
        return elapsed < self.gates.min_dwell_minutes

    # -- the loop ----------------------------------------------------------
    def update(self, obs: Observation) -> Action | Suppressed:
        """One window for one node. Returns the action, or why there isn't one.

        Never returns None: a suppressed trigger is a fact worth logging, and
        the audit trail has to show what the gates cost as well as what they
        prevented.
        """
        state = self._zone(obs.node)
        breach_now = obs.value >= obs.threshold
        state.windows.append(breach_now)
        state.last_value = obs.value

        # Hysteresis: arm on the threshold, disarm only well below it.
        disarm_at = obs.threshold * (1.0 - self.gates.hysteresis_pct / 100.0)
        if breach_now:
            state.armed = True
        elif obs.value < disarm_at:
            state.armed = False

        if not state.armed:
            return self._log(obs, "not_armed",
                             f"value {obs.value:.1f} below arm level {obs.threshold:.1f}")

        source = obs.top_source
        if source not in self.table.policy:
            return self._log(obs, "unknown_source", f"no policy for {source}")

        if obs.confidence < self.gates.min_source_posterior:
            return self._log(obs, "low_posterior",
                             f"{source} posterior {obs.confidence:.2f} < "
                             f"{self.gates.min_source_posterior:.2f}")

        sig = self.table.policy.get(source, ())
        if not sig:
            return self._log(obs, "refused_by_evidence",
                             f"{source} has no local lever: acting on it would "
                             "spend the budget on advection")

        if not self._passes_persistence(state):
            n = min(self.gates.persistence_of, self.window_capacity)
            return self._log(obs, "persistence",
                             f"{sum(list(state.windows)[-n:])} of last {n} "
                             f"windows breached, need {self.gates.persistence_windows}")

        if not self._uncertainty_clears(obs):
            return self._log(obs, "uncertainty",
                             f"interval low {obs.interval_low:.1f} does not "
                             f"exclude threshold {obs.threshold:.1f}")

        lever = self._lever_for(source, obs.value)
        if lever is None:
            return self._log(obs, "no_lever", f"policy {source} resolved to nothing")

        if self._within_dwell(state, lever, obs.at):
            last = state.last_dispatch[lever.key]
            return self._log(obs, "dwell",
                             f"{lever.key} dispatched {last}, dwell is "
                             f"{self.gates.min_dwell_minutes:.0f} min")

        recheck = (obs.at + pd.Timedelta(minutes=lever.dose_interval_min())
                   if lever.dose_interval_min() is not None else None)
        action = Action(
            id=f"A{next(_ACTION_IDS):05d}", node=obs.node, source=source,
            lever=lever.key, owner=lever.owner, dispatched_at=obs.at,
            recheck_at=recheck, trigger_value=float(obs.value),
            trigger_sigma=float(obs.sigma), threshold=float(obs.threshold),
            requires_signoff=self.gates.require_human_signoff,
            evidence={
                "posterior": {k: round(v, 4) for k, v in obs.posterior.items()},
                "coarse_ratio": round(obs.coarse_ratio, 3),
                "wind_from_deg": round(obs.wind_from_deg, 1),
                "lever_effect_pct": lever.effect_pct,
                "lever_half_life_min": lever.half_life_min,
                "lever_source": lever.source,
                "gate_min_posterior": self.gates.min_source_posterior,
                "provenance": "digital-twin" if obs.sigma >= 0 else None,
            },
        )
        state.last_dispatch[lever.key] = obs.at
        # The zone stays armed. Clearing it here would make the hysteresis band
        # unreachable, because the next window would report "not_armed" before
        # the disarm comparison could ever run. Leaving it armed means a
        # persistent breach re-doses once the dwell expires, which is the
        # intended behaviour for a lever with a measured half-life.
        self.actions.append(action)
        return action

    def _log(self, obs: Observation, reason: str, detail: str) -> Suppressed:
        item = Suppressed(node=obs.node, at=obs.at, reason=reason, detail=detail)
        self.suppressed.append(item)
        return item

    # -- closing the loop --------------------------------------------------
    def measure_effect(self, action: Action, series: pd.Series,
                       min_samples: int = 3) -> EffectResult:
        """Did the dispatch do what the literature said it would?

        The window is the lever's own half-life, not an arbitrary hour count:
        asking whether a 62-minute intervention still shows an effect 12 hours
        later measures nothing and would condemn the lever for the wrong reason.

        `series` must be at the node's native cadence, not aggregated to the
        hour. A 150-minute lever only spans two or three hourly samples, which
        is not enough to score; at the 5-minute cadence the node actually
        reports at, the same window carries thirty.
        """
        lever = self.table.levers[action.lever]
        window_min = lever.half_life_min or 120.0
        window = pd.Timedelta(minutes=window_min)

        before = series[(series.index >= action.dispatched_at - 3 * window)
                        & (series.index < action.dispatched_at)].dropna()
        after = series[(series.index > action.dispatched_at)
                       & (series.index <= action.dispatched_at + window)].dropna()
        if len(before) < min_samples or len(after) < min_samples:
            return EffectResult(
                action.id, action.node, action.lever,
                before_mean=float(before.mean()) if len(before) else float("nan"),
                after_mean=float(after.mean()) if len(after) else float("nan"),
                reduction_pct=float("nan"), expected_pct=lever.effect_pct,
                verdict="insufficient_data", n_before=len(before), n_after=len(after))

        b, a = float(before.mean()), float(after.mean())
        reduction = 100.0 * (1 - a / b) if b > 0 else 0.0
        if lever.effect_pct is None:
            verdict = "no_published_effect_size"
        elif reduction >= 0.5 * lever.effect_pct:
            verdict = "consistent_with_literature"
        elif reduction > 0:
            verdict = "weaker_than_published"
        else:
            verdict = "no_effect_measured"
        return EffectResult(action.id, action.node, action.lever, b, a,
                            reduction, lever.effect_pct, verdict,
                            len(before), len(after))

    # -- reporting ---------------------------------------------------------
    def summary(self) -> dict[str, object]:
        reasons: dict[str, int] = {}
        for s in self.suppressed:
            reasons[s.reason] = reasons.get(s.reason, 0) + 1
        by_lever: dict[str, int] = {}
        for a in self.actions:
            by_lever[a.lever] = by_lever.get(a.lever, 0) + 1
        return {
            "actions": len(self.actions),
            "suppressed": len(self.suppressed),
            "suppressed_by_reason": reasons,
            "actions_by_lever": by_lever,
            "zones_tracked": len(self._zones),
            "signoff_required": self.gates.require_human_signoff,
        }

    def reset(self) -> None:
        self._zones.clear()
        self.actions.clear()
        self.suppressed.clear()


def dose_schedule(lever: Lever, start: pd.Timestamp, hours: int = 24
                  ) -> list[pd.Timestamp]:
    """When a lever should be re-applied if the condition persists.

    This is the concrete difference between SETU and a roster. A two-pass-a-day
    tanker schedule is the same shape in every hour; this is not, and the shape
    comes from the measured half-life rather than from a shift pattern.
    """
    interval = lever.dose_interval_min()
    if interval is None:
        return [start]
    step = pd.Timedelta(minutes=interval)
    horizon = start + pd.Timedelta(hours=hours)
    out, t = [], start
    while t <= horizon and len(out) < 200:
        out.append(t)
        t = t + step
    return out


def coverage_fraction(doses: Sequence[pd.Timestamp], window: tuple[pd.Timestamp,
                                                                  pd.Timestamp],
                      lever: Lever) -> float:
    """Fraction of a window that stays covered, given the lever's decay."""
    lo, hi = window
    span = (hi - lo).total_seconds()
    if span <= 0:
        return 0.0
    if lever.half_life_min is None:
        return 0.0
    covered = 0.0
    for d in doses:
        if d < lo or d >= hi:
            continue
        end = min(d + pd.Timedelta(minutes=lever.half_life_min), hi)
        covered += max(0.0, (end - max(d, lo)).total_seconds())
    return min(1.0, covered / span)


def roster_vs_triggered(breach_hours: Sequence[int], passes_per_day: int,
                        half_life_min: float, roster_start_hour: int = 5
                        ) -> dict[str, float]:
    """The one comparison that justifies triggering, at equal cost.

    Both arms get the same number of passes, so this is not a "spend more"
    argument. The roster spaces them from a plausible shift start - 05:00, when
    Indian tanker crews come on - while the trigger places them on the hours
    that are actually breaching. Coverage is scored against breach hours only:
    covering 03:00 is worth nothing if the dust is at 08:00 and 18:00, and a
    fixed schedule is built on the assumption that it is worth something.

    Returns fractions in [0, 1] of distinct breach-hours still covered by at
    least one pass inside the lever's effective window.
    """
    if half_life_min <= 0:
        raise ConfigError("half_life_min must be positive")
    if passes_per_day < 1:
        raise ConfigError("passes_per_day must be at least 1")
    breaches = sorted({int(h) % 24 for h in breach_hours})
    if not breaches:
        return {"roster_coverage": 0.0, "triggered_coverage": 0.0,
                "breach_hours_per_day": 0.0, "gain": 0.0,
                "passes_per_day": float(passes_per_day)}

    reach = max(1, int(math.ceil(half_life_min / 60.0)))

    def footprint(start_hour: int) -> set[int]:
        return {(start_hour + offset) % 24 for offset in range(reach)}

    def coverage(placement: Sequence[int]) -> float:
        hit: set[int] = set()
        for start in placement:
            hit |= footprint(start)
        return len(hit.intersection(breaches)) / len(breaches)

    roster = [int(round(roster_start_hour + i * 24.0 / passes_per_day)) % 24
              for i in range(passes_per_day)]

    # Greedy placement: each pass goes where it adds the most still-uncovered
    # breach hours. That is what a dispatcher with a live grid actually does,
    # and it is deliberately not optimal - an optimum would overstate the case.
    chosen: list[int] = []
    covered: set[int] = set()
    for _ in range(passes_per_day):
        best, best_gain = breaches[0], -1
        for candidate in breaches:
            gain = len((footprint(candidate) & set(breaches)) - covered)
            if gain > best_gain:
                best, best_gain = candidate, gain
        chosen.append(best)
        covered |= footprint(best)

    return {
        "roster_coverage": round(coverage(roster), 4),
        "triggered_coverage": round(coverage(chosen), 4),
        "breach_hours_per_day": float(len(breaches)),
        "gain": round(coverage(chosen) - coverage(roster), 4),
        "passes_per_day": float(passes_per_day),
    }
