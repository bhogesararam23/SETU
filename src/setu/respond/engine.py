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

#: The gate chain, in evaluation order. The number is what the suppression
#: ledger records, so a run can be audited by gate without matching prose:
#: "how many triggers did the uncertainty gate cost" becomes a groupby.
GATE_CHAIN: tuple[tuple[int, str, str], ...] = (
    (1, "source", "source identified with enough posterior to act on"),
    (2, "persistence", "the breach survives the persistence window"),
    (3, "uncertainty", "the calibrated interval excludes the threshold"),
    (4, "dwell", "the lever's re-dose interval has elapsed"),
    (5, "hysteresis", "the zone is armed"),
)

#: Which gate each refusal belongs to. A gate is one place in the chain, and
#: several reasons can share one (the four ways a source gate can say no are
#: the same gate saying no for different reasons).
_GATE_OF: dict[str, int] = {
    "unknown_source": 1, "low_posterior": 1, "refused_by_evidence": 1,
    "no_lever": 1, "persistence": 2, "uncertainty": 3, "dwell": 4,
    "not_armed": 5,
}


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
    #: Expected concentration shaved per pass, in ug/m3: how far the trigger
    #: clears the threshold, times the lever's published effect, times how much
    #: the posterior is worth. The queue is ordered on this rather than on
    #: arrival, because two breaches in the same hour are not equally worth a
    #: truck.
    priority: float = 0.0

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
    #: Which gate in the ordered chain refused, 1..5. The reason string
    #: already said why in prose; this is what makes the ledger groupable, so
    #: the report can show what each gate cost without a string match.
    gate: int = 0
    #: When the refusing gate could next change its answer - the dwell expiry,
    #: or the next window. None where nothing scheduled will change it, because
    #: an operator needs to know which of these to come back to.
    retry_at: pd.Timestamp | None = None

    def as_dict(self) -> dict[str, object]:
        return {"node": self.node, "at": str(self.at), "reason": self.reason,
                "detail": self.detail, "gate": self.gate,
                "retry_at": str(self.retry_at) if self.retry_at is not None else None}


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
    #: Reduction measured on an untreated node over the same window. The
    #: before/after difference alone cannot separate a lever from the weather:
    #: a sea breeze that clears the whole grid looks exactly like a successful
    #: dispatch. Subtracting the control's change is the smallest correction
    #: that removes the common mode.
    control_pct: float | None = None

    @property
    def adjusted_pct(self) -> float | None:
        """Reduction net of what happened anyway, when a control was given."""
        if self.control_pct is None or not math.isfinite(self.reduction_pct):
            return None
        return self.reduction_pct - self.control_pct

    def as_dict(self) -> dict[str, object]:
        return {k: (round(v, 2) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclasses.dataclass
class DosingController:
    """Per-lever re-dose interval, scaled by what the lever actually did.

    Published half-lives are measured somewhere else, on someone else's road,
    in someone else's weather. The controller starts at the published interval
    and moves it within 0.5x-2x of that as dispatches are measured: a reduction
    well short of the literature means the surface is shedding the suppressant
    faster than advertised, so the next dose comes sooner.

    Bounded on purpose. An unbounded controller chasing one windy afternoon
    would halve the interval until the fleet is parked, and a lever dosed more
    often than its measured decay allows is a lever dosed on noise.
    """

    MIN_SCALE = 0.5
    MAX_SCALE = 2.0
    STEP = 0.25

    scales: dict[str, float] = dataclasses.field(default_factory=dict)

    def scale(self, lever_key: str) -> float:
        return self.scales.get(lever_key, 1.0)

    def observe(self, lever: Lever, measured_pct: float | None) -> float:
        """Move the interval one step on one measurement, and return the scale.

        Between half and all of the published effect the interval is left
        alone: that band is measurement noise around the claim, and stepping on
        it would turn run-to-run variation into a schedule that oscillates.
        """
        current = self.scale(lever.key)
        if (lever.effect_pct is None or measured_pct is None
                or not math.isfinite(measured_pct)):
            return current
        if measured_pct < 0.5 * lever.effect_pct:
            current -= self.STEP
        elif measured_pct >= lever.effect_pct:
            current += self.STEP
        else:
            return current
        current = min(self.MAX_SCALE, max(self.MIN_SCALE, current))
        self.scales[lever.key] = current
        return current

    def interval_min(self, lever: Lever, safety: float = 0.75) -> float | None:
        base = lever.dose_interval_min(safety)
        return None if base is None else base * self.scale(lever.key)


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
        self.dosing = DosingController()

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
            return self._log(
                obs, "persistence",
                f"{sum(list(state.windows)[-n:])} of last {n} windows breached, "
                f"need {self.gates.persistence_windows}",
                retry_at=self._next_window(obs.at))

        if not self._uncertainty_clears(obs):
            return self._log(obs, "uncertainty",
                             f"interval low {obs.interval_low:.1f} does not "
                             f"exclude threshold {obs.threshold:.1f}",
                             retry_at=self._next_window(obs.at))

        lever = self._lever_for(source, obs.value)
        if lever is None:
            return self._log(obs, "no_lever", f"policy {source} resolved to nothing")

        if self._within_dwell(state, lever, obs.at):
            last = state.last_dispatch[lever.key]
            due = last + pd.Timedelta(self.gates.min_dwell_minutes, unit="min")
            return self._log(obs, "dwell",
                             f"{lever.key} dispatched {last}, dwell is "
                             f"{self.gates.min_dwell_minutes:.0f} min",
                             retry_at=due)

        interval_min = self.dosing.interval_min(lever)
        recheck = (obs.at + pd.Timedelta(interval_min, unit="min")
                   if interval_min is not None else None)
        # How far this trigger clears the action level, times what the lever is
        # worth, times how much the posterior is worth. A dispatch from a 0.9
        # posterior on a lever with a measured 24% effect, 8 ug/m3 over the
        # line, is worth more than one from a 0.6 posterior on an advisory.
        excess = max(0.0, obs.value - obs.threshold) / max(obs.threshold, 1e-9)
        priority = round(excess * (lever.effect_pct or 0.0) / 100.0
                         * obs.confidence, 4)
        action = Action(
            id=f"A{next(_ACTION_IDS):05d}", node=obs.node, source=source,
            lever=lever.key, owner=lever.owner, dispatched_at=obs.at,
            recheck_at=recheck, trigger_value=float(obs.value),
            trigger_sigma=float(obs.sigma), threshold=float(obs.threshold),
            requires_signoff=self.gates.require_human_signoff,
            priority=priority,
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

    def _log(self, obs: Observation, reason: str, detail: str,
             retry_at: pd.Timestamp | None = None) -> Suppressed:
        item = Suppressed(node=obs.node, at=obs.at, reason=reason, detail=detail,
                          gate=_GATE_OF.get(reason, 0), retry_at=retry_at)
        self.suppressed.append(item)
        return item

    @staticmethod
    def _next_window(at: pd.Timestamp) -> pd.Timestamp:
        """The next observation that could change a per-window gate's answer.

        The cadence lives with the caller, so an hour is the honest guess and
        a note rather than a promise - the ledger says "come back then", not
        "it will pass then".
        """
        return at + pd.Timedelta(hours=1)

    def register_effect(self, action: Action, result: EffectResult) -> float:
        """Feed one measured effect back into the lever's re-dose interval."""
        lever = self.table.levers.get(action.lever)
        if lever is None:
            return 1.0
        measured = (result.adjusted_pct if result.adjusted_pct is not None
                    else result.reduction_pct)
        return self.dosing.observe(lever, measured)

    # -- closing the loop --------------------------------------------------
    def measure_effect(self, action: Action, series: pd.Series,
                       min_samples: int = 3,
                       control: pd.Series | None = None) -> EffectResult:
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
        window = pd.Timedelta(window_min, unit="min")

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

        # The control is the same two windows on a node that got no dispatch.
        # Without it, a sea breeze that clears the whole grid is scored as a
        # successful intervention, and the verdict below would be measuring the
        # weather.
        control_pct: float | None = None
        if control is not None:
            cb = control[(control.index >= action.dispatched_at - 3 * window)
                         & (control.index < action.dispatched_at)].dropna()
            ca = control[(control.index > action.dispatched_at)
                         & (control.index <= action.dispatched_at + window)].dropna()
            if (len(cb) >= min_samples and len(ca) >= min_samples
                    and float(cb.mean()) > 0):
                control_pct = 100.0 * (1 - float(ca.mean()) / float(cb.mean()))

        net = reduction if control_pct is None else reduction - control_pct
        if lever.effect_pct is None:
            verdict = "no_published_effect_size"
        elif net >= 0.5 * lever.effect_pct:
            verdict = "consistent_with_literature"
        elif net > 0:
            verdict = "weaker_than_published"
        else:
            verdict = "no_effect_measured"
        return EffectResult(action.id, action.node, action.lever, b, a,
                            reduction, lever.effect_pct, verdict,
                            len(before), len(after), control_pct)

    # -- reporting ---------------------------------------------------------
    def summary(self) -> dict[str, object]:
        reasons: dict[str, int] = {}
        for s in self.suppressed:
            reasons[s.reason] = reasons.get(s.reason, 0) + 1
        by_gate: dict[str, int] = {}
        labels = {n: name for n, name, _ in GATE_CHAIN}
        for s in self.suppressed:
            name = labels.get(s.gate, "ungated")
            by_gate[name] = by_gate.get(name, 0) + 1
        by_lever: dict[str, int] = {}
        for a in self.actions:
            by_lever[a.lever] = by_lever.get(a.lever, 0) + 1
        retry = [s for s in self.suppressed if s.retry_at is not None]
        return {
            "actions": len(self.actions),
            "suppressed": len(self.suppressed),
            "suppressed_by_reason": reasons,
            "suppressed_by_gate": by_gate,
            #: Suppressions that a clock will change, as opposed to ones that
            #: only new information can. The difference is what tells a
            #: dispatcher which rows of the ledger to come back to.
            "reschedulable": len(retry),
            "actions_by_lever": by_lever,
            "max_priority": round(max((a.priority for a in self.actions),
                                      default=0.0), 4),
            "dosing_scales": {k: round(v, 3) for k, v in
                              self.dosing.scales.items()},
            "zones_tracked": len(self._zones),
            "signoff_required": self.gates.require_human_signoff,
        }

    def reset(self) -> None:
        self._zones.clear()
        self.actions.clear()
        self.suppressed.clear()


def rank_actions(actions: Iterable[Action]) -> list[Action]:
    """Order dispatches by expected effect, not by arrival.

    Two breaches in the same hour are not equally worth a truck: one may be 4
    ug/m3 over the line with a 0.7 posterior on an advisory lever, the other 20
    over with a 0.9 posterior on a sweeper. The engine appends in time order,
    which is what the audit trail needs; this is the order a human should work.
    """
    return sorted(actions, key=lambda a: (-a.priority, a.dispatched_at, a.node))


def dispatch_plan(actions: Sequence[Action], capacity_per_hour: int | None = None
                  ) -> dict[str, object]:
    """The queue, and what a fleet constraint would defer.

    With no capacity this is only an ordering. With a capacity it becomes a
    schedule: within each hour the highest-priority dispatches go, the rest are
    deferred to the next hour with the reason recorded. The deferral is honest
    about its own limitation - it does not re-evaluate the deferred trigger, so
    a breach that clears before the next hour is dispatched on stale evidence,
    which is why the row is marked rather than simply dropped.
    """
    if capacity_per_hour is not None and capacity_per_hour < 1:
        raise ConfigError("capacity_per_hour must be at least 1")
    queue = rank_actions(actions)
    served: list[dict[str, object]] = []
    deferred: list[dict[str, object]] = []
    per_hour: dict[pd.Timestamp, int] = {}
    for action in queue:
        slot = action.dispatched_at.floor("h")
        used = per_hour.get(slot, 0)
        if capacity_per_hour is not None and used >= capacity_per_hour:
            deferred.append({
                "id": action.id, "node": action.node, "lever": action.lever,
                "priority": action.priority,
                "reason": "crew_capacity",
                "retry_at": str(slot + pd.Timedelta(hours=1)),
            })
            continue
        per_hour[slot] = used + 1
        served.append({"id": action.id, "node": action.node,
                       "lever": action.lever, "priority": action.priority})
    return {
        "queue": served,
        "deferred": deferred,
        "hours_with_demand": len(per_hour),
        "peak_hour_demand": max(per_hour.values(), default=0),
        "capacity_per_hour": capacity_per_hour,
        "deferred_count": len(deferred),
    }


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
    step = pd.Timedelta(interval, unit="min")
    horizon = start + pd.Timedelta(hours, unit="h")
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
        end = min(d + pd.Timedelta(lever.half_life_min, unit="min"), hi)
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
    #
    # Two things stop the far arm being over- or under-credited. A pass cannot
    # be placed on an hour that already has one - it is the same truck at the
    # same place, and it used to be counted twice, which understated the trigger
    # exactly on long-lived levers. And when no remaining candidate adds
    # coverage the lever is saturated: the roster is still being handed its full
    # allocation while the trigger needs nothing further, so the pass count is
    # reported rather than the coverage alone.
    chosen: list[int] = []
    covered: set[int] = set()
    for _ in range(passes_per_day):
        best, best_gain = None, 0
        for candidate in breaches:
            if candidate in chosen:
                continue
            gain = len((footprint(candidate) & set(breaches)) - covered)
            if best is None or gain > best_gain:
                best, best_gain = candidate, gain
        if best is None or best_gain == 0:
            break
        chosen.append(best)
        covered |= footprint(best)

    return {
        "roster_coverage": round(coverage(roster), 4),
        "triggered_coverage": round(coverage(chosen), 4),
        "breach_hours_per_day": float(len(breaches)),
        "gain": round(coverage(chosen) - coverage(roster), 4),
        "passes_per_day": float(passes_per_day),
        "passes_used": float(len(chosen)),
        # True when the lever's own window covers the day in fewer passes than
        # the roster is given - the case where equal cost is not equal effort.
        "saturated": len(chosen) < passes_per_day,
    }
