"""The response engine, including the cases where triggering earns nothing.

Two of these tests assert that SETU's central mechanism is worthless under
particular conditions. That is deliberate. The value of triggering is
conditional on a short-lived lever acting on an episodic source, and a repo
that only ever shows the favourable case will not survive a question from a
reader who has examined the road-dust literature.
"""

import numpy as np
import pandas as pd
import pytest

from setu.config import load_action_table
from setu.errors import ConfigError
from setu.respond import (
    Action,
    DosingController,
    Observation,
    ResponseEngine,
    Suppressed,
    coverage_fraction,
    dispatch_plan,
    dose_schedule,
    rank_actions,
)
from setu.respond.engine import roster_vs_triggered


@pytest.fixture
def table():
    return load_action_table()


@pytest.fixture
def engine(table):
    return ResponseEngine(table)


POST = {"road_dust": 0.72, "secondary": 0.10, "construction": 0.08,
        "open_burning": 0.06, "traffic_combustion": 0.04}
SEC = {"secondary": 0.80, "road_dust": 0.08, "construction": 0.05,
       "open_burning": 0.04, "traffic_combustion": 0.03}
T0 = pd.Timestamp("2026-11-01 08:00")


def obs(value=78.0, sigma=3.0, posterior=None, at=T0, node="r01",
        threshold=60.0):
    return Observation(node=node, at=at, value=value, sigma=sigma,
                       posterior=posterior or POST, threshold=threshold,
                       coarse_ratio=0.55, wind_from_deg=270.0)


def _hours(n):
    return pd.date_range(T0, periods=n, freq="h")


def feed(engine, n=6, start=0, **kw):
    """Push n windows from hour `start` and return every verdict.

    `start` exists because the engine is stateful and dwell is measured in real
    time: replaying overlapping timestamps makes elapsed time negative and lets
    a gate pass that should have blocked.
    """
    hours = _hours(n + start)[start:]
    return [engine.update(obs(at=t, **kw)) for t in hours]


# -- arming and persistence -------------------------------------------------
def test_a_single_breach_window_does_not_dispatch(engine):
    out = engine.update(obs())
    assert isinstance(out, Suppressed)
    assert out.reason == "persistence"
    assert engine.actions == []


def test_k_of_n_consecutive_breaches_dispatches(engine):
    verdicts = feed(engine, 5)
    assert any(isinstance(v, Action) for v in verdicts)
    assert len(engine.actions) == 1


def test_alternating_breaches_never_satisfy_persistence(engine):
    values = [78.0, 40.0] * 6
    verdicts = [engine.update(obs(value=v, at=t))
                for v, t in zip(values, _hours(12))]
    assert not any(isinstance(v, Action) for v in verdicts)
    reasons = {s.reason for s in engine.suppressed}
    assert "persistence" in reasons


def test_a_value_below_the_threshold_never_arms(engine):
    out = engine.update(obs(value=41.0))
    assert isinstance(out, Suppressed) and out.reason == "not_armed"


def test_hysteresis_disarms_only_well_below_the_threshold(engine):
    """Otherwise a value hovering at the limit flaps the actuator all night."""
    feed(engine, 4)                       # arms and dispatches
    engine.actions.clear()
    # 58 is below the 60 threshold but above the 51 disarm level, so the zone
    # must stay armed and be held back by dwell rather than by arming.
    out = engine.update(obs(value=58.0, at=T0 + pd.Timedelta(4, unit="h")))
    assert isinstance(out, Suppressed)
    assert out.reason != "not_armed"
    # 45 is below the disarm level, so the zone is properly clear again.
    out2 = engine.update(obs(value=45.0, at=T0 + pd.Timedelta(5, unit="h")))
    assert isinstance(out2, Suppressed) and out2.reason == "not_armed"


# -- the other gates --------------------------------------------------------
def test_a_weak_posterior_is_refused(engine):
    weak = {"road_dust": 0.30, "secondary": 0.28, "construction": 0.16,
            "open_burning": 0.14, "traffic_combustion": 0.12}
    verdicts = feed(engine, 6, posterior=weak)
    assert not any(isinstance(v, Action) for v in verdicts)
    assert any(s.reason == "low_posterior" for s in engine.suppressed)


def test_uncertainty_that_straddles_the_threshold_blocks_the_dispatch(engine):
    # Value is above 60 but a 1-sigma interval reaches down through it.
    verdicts = feed(engine, 6, value=61.5, sigma=4.0)
    assert not any(isinstance(v, Action) for v in verdicts)
    assert any(s.reason == "uncertainty" for s in engine.suppressed)


def test_the_same_interval_clearing_the_threshold_lets_it_through(engine):
    verdicts = feed(engine, 6, value=68.0, sigma=4.0)
    assert any(isinstance(v, Action) for v in verdicts)


def test_secondary_aerosol_is_refused_by_evidence_not_by_lack_of_a_lever(engine):
    verdicts = feed(engine, 6, posterior=SEC)
    assert not any(isinstance(v, Action) for v in verdicts)
    reasons = [s.reason for s in engine.suppressed]
    assert "refused_by_evidence" in reasons


def test_dwell_spaces_consecutive_dispatches_of_the_same_lever(engine):
    """A persistent breach should re-dose, but never faster than the dwell.

    Asserting "exactly one dispatch" would be wrong: the lever has a measured
    half-life, so a breach that lasts all night genuinely does need re-dosing.
    What must hold is a floor on the spacing.
    """
    verdicts = feed(engine, 12)
    times = [v.dispatched_at for v in verdicts if isinstance(v, Action)]
    assert len(times) >= 2, "a 12-hour breach should re-dose at least once"
    gaps = [(b - a).total_seconds() / 60.0 for a, b in zip(times, times[1:])]
    assert min(gaps) >= engine.gates.min_dwell_minutes
    assert any(s.reason == "dwell" for s in engine.suppressed)


def test_disarming_below_the_hysteresis_level_stops_re_dosing(engine):
    """Once the zone is genuinely clean the re-dosing must stop by itself."""
    feed(engine, 10)
    before = len(engine.actions)
    assert before >= 2, "a 10-hour breach should have re-dosed"
    # Ten quiet hours, continuing forward in time.
    for i, t in enumerate(_hours(11)[1:], start=1):
        engine.update(obs(value=30.0, at=t))
    assert len(engine.actions) == before, "a clean zone must stop re-dosing"
    # A fresh spell, later in time, re-arms from scratch.
    feed(engine, 6, start=11)
    assert len(engine.actions) > before


# -- the action itself ------------------------------------------------------
def test_action_carries_the_evidence_and_the_owner(engine):
    action = next(v for v in feed(engine, 6) if isinstance(v, Action))
    assert action.owner == "ward_sanitation_supervisor"
    assert action.lever == "vacuum_sweeping"
    assert action.requires_signoff is True
    assert action.evidence["lever_source"].startswith("AIRUSE")
    assert action.evidence["posterior"]["road_dust"] == pytest.approx(0.72, abs=1e-6)


def test_recheck_time_follows_the_levers_half_life(engine):
    action = next(v for v in feed(engine, 6) if isinstance(v, Action))
    # Vacuum sweeping: 150 min half-life, dose at 0.75 of it.
    assert action.recheck_at == action.dispatched_at + pd.Timedelta(112.5, unit="min")


def test_input_validation():
    idx = _hours(1)[0]
    with pytest.raises(ConfigError):
        Observation("n", idx, 10.0, -1.0, POST, 60.0)
    with pytest.raises(ConfigError):
        Observation("n", idx, -5.0, 1.0, POST, 60.0)


# -- dosing ----------------------------------------------------------------
def test_dose_schedule_is_derived_from_decay_not_from_a_shift(table):
    start = pd.Timestamp("2026-11-01 06:00")
    sprinkling = dose_schedule(table.levers["water_sprinkling"], start, hours=24)
    sweeping = dose_schedule(table.levers["vacuum_sweeping"], start, hours=24)
    notice = dose_schedule(table.levers["site_notice"], start, hours=24)
    assert len(sprinkling) > len(sweeping)          # shorter life, more doses
    assert len(notice) == 1                          # no measured decay -> one shot
    assert sprinkling[1] - sprinkling[0] == pd.Timedelta(46.5, unit="min")


def test_coverage_rises_with_dose_count(table):
    window = (pd.Timestamp("2026-11-01"), pd.Timestamp("2026-11-01 12:00"))
    lever = table.levers["water_sprinkling"]
    sparse = dose_schedule(lever, window[0], hours=1)
    dense = dose_schedule(lever, window[0], hours=12)
    assert coverage_fraction(dense, window, lever) > coverage_fraction(
        sparse, window, lever)


def test_an_unmeasured_lever_covers_nothing_by_definition(table):
    lever = table.levers["site_notice"]
    doses = dose_schedule(lever, pd.Timestamp("2026-11-01"), hours=24)
    assert coverage_fraction(doses, (pd.Timestamp("2026-11-01"),
                                     pd.Timestamp("2026-11-02")), lever) == 0.0


# -- the honest limit of the whole idea ------------------------------------
def test_triggering_wins_on_a_rush_hour_dust_profile():
    out = roster_vs_triggered([7, 8, 9, 17, 18, 19], passes_per_day=2,
                              half_life_min=62)
    assert out["triggered_coverage"] == pytest.approx(0.6667, abs=1e-3)
    assert out["roster_coverage"] == pytest.approx(0.3333, abs=1e-3)
    assert out["gain"] > 0.3


def test_triggering_buys_nothing_when_the_lever_is_long_lived():
    """Calcium chloride lasts a day. Placement is then almost irrelevant."""
    out = roster_vs_triggered([7, 8, 9, 17, 18, 19], passes_per_day=2,
                              half_life_min=1200)
    assert out["gain"] == pytest.approx(0.0)
    assert out["triggered_coverage"] == 1.0


def test_triggering_buys_nothing_when_the_source_is_not_episodic():
    """A uniform all-day breach is a background problem, not a scheduling one."""
    out = roster_vs_triggered(list(range(24)), passes_per_day=3, half_life_min=62)
    assert out["gain"] == pytest.approx(0.0)


def test_greedy_placement_is_not_claimed_as_optimal():
    """With one pass the greedy choice is the first breach hour, not the best."""
    out = roster_vs_triggered([7, 8, 9, 17, 18, 19], passes_per_day=1,
                              half_life_min=62)
    assert out["triggered_coverage"] <= 0.5


def test_roster_start_is_configurable_because_midnight_is_not_a_shift():
    breaches = [7, 8, 9, 17, 18, 19]
    dawn = roster_vs_triggered(breaches, 2, 62, roster_start_hour=5)
    midnight = roster_vs_triggered(breaches, 2, 62, roster_start_hour=0)
    assert dawn["roster_coverage"] > midnight["roster_coverage"]


def test_bad_inputs_are_refused():
    with pytest.raises(ConfigError):
        roster_vs_triggered([8], 2, 0)
    with pytest.raises(ConfigError):
        roster_vs_triggered([8], 0, 62)
    assert roster_vs_triggered([], 2, 62)["gain"] == 0.0


# -- closing the loop -------------------------------------------------------
def _effect_series(action, cadence="10min", before=80.0, after=58.0):
    """Native-cadence observations either side of a dispatch."""
    idx = pd.date_range(action.dispatched_at - pd.Timedelta(6, unit="h"),
                        action.dispatched_at + pd.Timedelta(3, unit="h"), freq=cadence)
    rng = np.random.default_rng(0)
    values = [before if t < action.dispatched_at else after for t in idx]
    return pd.Series(values, index=idx) + rng.normal(0, 0.6, len(idx))


def test_effect_is_unscorable_at_hourly_cadence(engine):
    """Documents why the series must be native-cadence, not aggregated."""
    action = next(v for v in feed(engine, 6) if isinstance(v, Action))
    idx = pd.date_range(action.dispatched_at - pd.Timedelta(6, unit="h"),
                        action.dispatched_at + pd.Timedelta(3, unit="h"), freq="h")
    series = pd.Series([80.0] * len(idx), index=idx)
    assert engine.measure_effect(action, series).verdict == "insufficient_data"


def test_a_successful_dispatch_is_scored_against_the_literature(engine):
    action = next(v for v in feed(engine, 6) if isinstance(v, Action))
    result = engine.measure_effect(action, _effect_series(action))
    # 27.5% against a published 24% -> consistent.
    assert result.verdict == "consistent_with_literature"
    assert result.reduction_pct == pytest.approx(27.5, abs=3)
    assert result.n_after >= 3


def test_a_dispatch_that_did_nothing_is_reported_as_such(engine):
    action = next(v for v in feed(engine, 6) if isinstance(v, Action))
    series = _effect_series(action, before=80.0, after=80.0)
    assert engine.measure_effect(action, series).verdict == "no_effect_measured"


def test_a_lever_with_no_published_effect_size_cannot_be_graded(engine):
    """A notice has no ambient effect size in the literature, so a flat line
    afterwards is not evidence that it failed."""
    from dataclasses import replace
    action = next(v for v in feed(engine, 6) if isinstance(v, Action))
    series = _effect_series(action, before=80.0, after=40.0)
    notice = replace(action, lever="site_notice")
    result = engine.measure_effect(notice, series)
    assert result.expected_pct is None
    assert result.verdict == "no_published_effect_size"


def test_summary_counts_every_suppression_with_a_reason(engine):
    feed(engine, 6)
    feed(engine, 3, start=6, posterior=SEC)
    s = engine.summary()
    # Two dispatches is correct here: dwell is 90 minutes and the breach ran
    # six hours, so the second is a re-dose rather than a duplicate.
    assert s["actions"] == len(engine.actions) >= 1
    assert s["suppressed"] >= 3
    assert "refused_by_evidence" in s["suppressed_by_reason"]
    assert s["actions_by_lever"]["vacuum_sweeping"] == s["actions"]
    assert s["signoff_required"] is True


def test_reset_clears_state_for_a_fresh_run(engine):
    feed(engine, 6)
    engine.reset()
    assert engine.summary()["actions"] == 0
    assert engine.summary()["zones_tracked"] == 0


# -- the suppression ledger -------------------------------------------------
def test_every_suppression_carries_the_gate_that_refused_it(engine):
    """Grouping the ledger by gate, not by prose, is what makes it auditable."""
    feed(engine, 6)
    feed(engine, 3, start=6, posterior=SEC)
    gates = {s.gate for s in engine.suppressed}
    assert gates <= {n for n, _, _ in __import__(
        "setu.respond.engine", fromlist=["GATE_CHAIN"]).GATE_CHAIN}
    assert 0 not in gates, "a refusal outside the chain is a bug, not a gate"
    s = engine.summary()
    assert sum(s["suppressed_by_gate"].values()) == s["suppressed"]
    assert s["suppressed_by_gate"]["persistence"] > 0


def test_the_ledger_groups_every_refusal_under_the_gate_that_made_it(engine):
    """The console renders the ledger, so the totals have to reconcile here.

    Gate totals that do not add up to the suppression count are the failure mode
    this guards: a row silently dropped during aggregation reads on screen as a
    gate that cost less than it did.
    """
    feed(engine, 6)
    feed(engine, 3, start=6, posterior=SEC)
    summary = engine.summary()
    ledger = summary["ledger"]
    assert [g["gate"] for g in ledger] == sorted(g["gate"] for g in ledger)
    assert sum(g["total"] for g in ledger) == summary["suppressed"]
    assert all(g["name"] and g["description"] for g in ledger)
    for group in ledger:
        assert sum(r["count"] for r in group["reasons"]) == group["total"]
        assert group["retryable"] <= group["total"]
        assert len({r["reason"] for r in group["reasons"]}) == len(group["reasons"])


def test_a_clock_gate_says_how_long_and_a_source_gate_says_new_information(engine):
    """Only the gates with a clock belong in a follow-up list."""
    feed(engine, 6)
    feed(engine, 3, start=6, posterior=SEC)
    ledger = {g["name"]: g for g in engine.summary()["ledger"]}
    dwell = next(r for r in ledger["dwell"]["reasons"] if r["reason"] == "dwell")
    assert dwell["retryable"] is True
    assert 0 < dwell["retry_within_minutes"] <= engine.gates.min_dwell_minutes
    assert dwell["last_retry_at"] is not None

    source = next(r for r in ledger["source"]["reasons"]
                  if r["reason"] == "refused_by_evidence")
    assert source["retryable"] is False
    assert source["retry_within_minutes"] is None
    assert source["last_retry_at"] is None


def test_a_dwell_refusal_says_when_the_lever_can_go_again(engine):
    feed(engine, 6)
    dwell = [s for s in engine.suppressed if s.reason == "dwell"]
    assert dwell, "expected the re-dose inside the dwell window to be refused"
    last = engine.actions[0].dispatched_at
    assert dwell[0].retry_at == last + pd.Timedelta(
        engine.gates.min_dwell_minutes, unit="min")
    # An hourly gate clears on its own; a dwell clears on a clock. Only the
    # second kind belongs in a dispatcher's follow-up list.
    assert engine.summary()["reschedulable"] >= len(dwell)


# -- scheduling and prioritisation -----------------------------------------
def test_the_queue_is_ordered_by_expected_effect_not_by_arrival(engine):
    feed(engine, 5, node="a01", value=70.0)
    feed(engine, 5, node="b01", value=110.0)
    assert len(engine.actions) >= 2
    ranked = rank_actions(engine.actions)
    assert ranked[0].node == "b01", "the deeper breach must outrank the shallower"
    assert ranked[0].priority > ranked[-1].priority


def test_a_fleet_constraint_defers_by_priority_and_names_the_reason(engine):
    for node in ("a01", "b01", "c01"):
        feed(engine, 5, node=node, value=70.0 + 10 * int(node[1:3]))
    plan = dispatch_plan(engine.actions, capacity_per_hour=1)
    assert plan["deferred_count"] >= 1
    assert plan["peak_hour_demand"] >= 1
    deferred = {d["id"] for d in plan["deferred"]}
    served = {r["id"] for r in plan["queue"]}
    assert not deferred & served
    assert all(d["reason"] == "crew_capacity" for d in plan["deferred"])
    # Every deferral hands back a time, not just a refusal.
    assert all(d["retry_at"] for d in plan["deferred"])
    # The plan only reorders; it never invents or loses a dispatch.
    assert len(served) + len(deferred) == len(engine.actions)


def test_without_a_capacity_the_plan_is_only_an_ordering(engine):
    for node in ("a01", "b01"):
        feed(engine, 5, node=node, value=90.0)
    plan = dispatch_plan(engine.actions)
    assert plan["deferred_count"] == 0
    assert len(plan["queue"]) == len(engine.actions)


# -- adaptive dosing --------------------------------------------------------
def test_a_lever_that_underperforms_is_redosed_sooner(engine):
    lever = engine.table.levers["vacuum_sweeping"]
    assert lever.effect_pct and lever.half_life_min
    base = engine.dosing.interval_min(lever)
    assert base is not None
    for _ in range(4):
        engine.dosing.observe(lever, 0.2 * lever.effect_pct)
    assert engine.dosing.scale(lever.key) == DosingController.MIN_SCALE
    assert engine.dosing.interval_min(lever) == pytest.approx(base * 0.5)


def test_a_lever_that_outperforms_is_redosed_later_but_not_freely(engine):
    lever = engine.table.levers["vacuum_sweeping"]
    base = engine.dosing.interval_min(lever)
    for _ in range(20):
        engine.dosing.observe(lever, 3.0 * lever.effect_pct)
    # Bounded: a surfacing that held for one calm afternoon does not license a
    # schedule twice as sparse as the published decay.
    assert engine.dosing.scale(lever.key) == DosingController.MAX_SCALE
    assert engine.dosing.interval_min(lever) == pytest.approx(base * 2.0)


def test_an_effect_inside_the_noise_band_leaves_the_schedule_alone(engine):
    lever = engine.table.levers["vacuum_sweeping"]
    for _ in range(5):
        engine.dosing.observe(lever, 0.7 * lever.effect_pct)
    assert engine.dosing.scale(lever.key) == 1.0


def test_a_dispatch_whose_recheck_reflects_the_adapted_interval(engine):
    lever = engine.table.levers["vacuum_sweeping"]
    for _ in range(4):
        engine.dosing.observe(lever, 0.2 * lever.effect_pct)
    action = next(v for v in feed(engine, 6) if isinstance(v, Action))
    assert action.recheck_at is not None
    elapsed = (action.recheck_at - action.dispatched_at).total_seconds() / 60.0
    assert elapsed == pytest.approx(lever.half_life_min * 0.75 * 0.5)


# -- measuring the effect ---------------------------------------------------
def test_an_effect_is_measured_net_of_what_happened_untreated(engine):
    """A sea breeze that clears the grid is not a successful dispatch."""
    action = next(v for v in feed(engine, 6) if isinstance(v, Action))
    treated = _effect_series(action, before=80.0, after=58.0)   # -27%
    control = _effect_series(action, before=80.0, after=50.0)   # -37% anyway
    result = engine.measure_effect(action, treated, control=control)

    assert result.control_pct == pytest.approx(37.5, abs=3)
    assert result.reduction_pct == pytest.approx(27.5, abs=3)
    assert result.adjusted_pct == pytest.approx(
        result.reduction_pct - result.control_pct)
    # Raw, this looks consistent with the literature. Net of the control it is
    # a reduction smaller than the one nobody intervened to cause.
    assert result.verdict == "no_effect_measured"


def test_without_a_control_the_raw_reduction_is_still_reported(engine):
    action = next(v for v in feed(engine, 6) if isinstance(v, Action))
    result = engine.measure_effect(action, _effect_series(action))
    assert result.control_pct is None
    assert result.adjusted_pct is None


def test_a_measured_effect_moves_the_lever_schedule(engine):
    lever = engine.table.levers["vacuum_sweeping"]
    action = next(v for v in feed(engine, 6) if isinstance(v, Action))
    weaker = _effect_series(action, before=80.0, after=78.0)
    result = engine.measure_effect(action, weaker)
    scale = engine.register_effect(action, result)
    assert scale < 1.0
    assert engine.summary()["dosing_scales"][lever.key] == round(scale, 3)


# -- equal cost, on a lever that lasts most of a day ------------------------
def test_a_long_lived_lever_is_not_given_passes_it_does_not_need():
    """The suppressant row must say saturation, not a fake equal-cost tie."""
    out = roster_vs_triggered([8, 9, 10], passes_per_day=2, half_life_min=1200)
    assert out["saturated"] is True
    assert out["passes_used"] == 1
    assert out["roster_coverage"] == 1.0 and out["triggered_coverage"] == 1.0
    assert out["gain"] == 0.0


def test_a_short_lever_uses_every_pass_it_is_given():
    out = roster_vs_triggered([7, 8, 9, 17, 18, 19], passes_per_day=2,
                              half_life_min=62)
    assert out["passes_used"] == 2
    assert out["saturated"] is False
