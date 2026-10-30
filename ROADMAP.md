# SETU Development Roadmap

## Overview

This document tracks the systematic improvement of SETU from its current state (commit `e395d31`) into a robust, reproducible, scientifically credible open-source air quality research project.

**Current Status:**
- 313 tests, all passing, none skipped. The one skip the suite used to report
  was a guard that had switched itself off; it fails now instead
- Full pipeline runs end-to-end
- Digital twin validates algorithms against known ground truth
- Calibration, attribution, forecasting, response and I/O layers all functional
- Milestones 4 (attribution), 5 (forecast), 6 (response), 7 (data pipeline) and
  8 (operator console) complete; milestone 5's GNN task is deferred with its
  reason written down, and so is milestone 8's authentication task
- The console is a read-only view over one run: queue, gate ledger, forecast
  trajectory, evidence packs and provenance, with the run's age on the header
- Documentation is comprehensive and honest about limitations, and now quotes
  the walk-forward forecast numbers rather than the in-sample ones

**Target:** 50-100 meaningful commits with genuine improvements across all layers.

---

## Milestone Structure

### Milestone 0: Foundation (Current State)
- [x] Repository structure and architecture documented
- [x] All major components functional
- [x] Comprehensive test suite (223 tests)
- [x] Honest limitations documented
- [x] Validation protocol defined

### Milestone 1: Test Reliability (Priority 1)
**Goal:** All 223 tests passing reliably

#### Issues Identified:
1. **test_accuracy_rises_with_confidence** - Accuracy at mid confidence (0.667) > high confidence (0.531)
   - Root cause: The test expects monotonic improvement with confidence, but current classifier shows non-monotonic behavior
   - Fix: Investigate classifier calibration and confidence thresholds

2. **test_transfer_calibrates_a_node_that_was_never_co_located** - Transferred MAE (8.16) > naive MAE (4.32)
   - Root cause: Transfer calibration producing worse results than raw readings
   - Fix: Review transfer calibration logic and feature mapping

3. **test_a_drifting_node_diverges_over_the_window** - Drift detection not working as expected
   - Root cause: Need to investigate twin drift injection and CUSUM monitor
   - Fix: Verify drift injection and detection parameters

#### Tasks:
- [ ] Fix test_accuracy_rises_with_confidence
- [ ] Fix test_transfer_calibrates_a_node_that_was_never_co_located  
- [ ] Fix test_a_drifting_node_diverges_over_the_window
- [ ] Run full test suite and verify 223/223 passing
- [ ] Add regression gates to prevent future regressions

### Milestone 2: Code Quality & Maintainability
**Goal:** Clean, maintainable, production-ready codebase

#### Tasks:
- [ ] Fix all deprecation warnings (NumPy timedelta, pandas API changes)
- [ ] Add type hints throughout (currently partial)
- [ ] Improve error messages and validation
- [ ] Add docstrings to all public functions
- [ ] Refactor duplicate code in calibration and attribute modules
- [ ] Add logging for operational visibility
- [ ] Improve configuration validation

### Milestone 3: Calibration Improvements
**Goal:** More robust calibration with better uncertainty quantification

#### Tasks:
- [ ] Fix interval coverage under-nominal issue (0.658 vs 0.80 target)
- [ ] Improve uncertainty estimation (consider conformal prediction)
- [ ] Add cross-validation for calibration models
- [ ] Improve transfer calibration robustness
- [ ] Add calibration diagnostics and health checks
- [ ] Implement periodic recalibration triggers

### Milestone 4: Attribution Enhancements
**Goal:** More accurate and interpretable source attribution

#### Tasks:
- [x] Fix share cross-check failures (road_dust, traffic_combustion)
- [x] Improve classifier interpretability
- [x] Add feature importance analysis
- [x] Validate against published receptor study signatures
- [x] Improve handling of mixed/background sources
- [x] Add confidence calibration for posterior probabilities

### Milestone 5: Forecast Layer Improvements
**Goal:** More accurate and actionable forecasts

#### Tasks:
- [ ] Implement proper graph neural network (when data volume supports it)
  - Deferred with the reason recorded rather than the box ticked: 40 nodes, one
    season, 1,772 hours. What a GNN buys over the per-node model is spatial
    structure learned from many nodes across many regimes, and at this size the
    upwind channel adds nothing measurable - the plume crosses a 320 m pitch in
    under three minutes, so own lags already carry the signal. Revisit at >150
    nodes and a full year of data.
- [x] Add forecast uncertainty estimation - split conformal at 0.20, per
      horizon, with coverage and mean width scored on every run
- [x] Improve handling of missing meteorological data - gaps up to three hours
      bridged and counted, longer ones left missing rather than filled with a
      wind from hours ago; warm-up and weather losses counted separately
- [x] Add forecast validation against persistence baseline - walk-forward rather
      than one split, so the number is the one a refitted deployment gets
- [x] Implement multi-step ahead forecasting - `forecast_path` chains 1 h
      predictions into an hourly trajectory for peak timing; the direct horizon
      models stay the answer for "will it breach at 6 h"
- [x] Add forecast diagnostics - per-fold decay curve, bias, and the count of
      nodes that beat persistence, because a pooled mean of zero cannot tell a
      uniformly marginal grid from a split one

### Milestone 6: Response Engine Enhancements
**Goal:** More effective and measurable interventions

#### Tasks:
- [x] Fix equal-cost comparison for long-lived levers - a pass is no longer
      awarded at an hour that already has one, and `passes_used`/`saturated`
      make the ~20 h suppressant read as a saturated lever rather than a tie
- [x] Improve action scheduling and prioritization - priority is excess over the
      action level times the lever's effect times the posterior; `dispatch_plan`
      defers the lowest-priority rows under a crew capacity with a reason and a
      retry hour
- [x] Add intervention effect measurement framework - before/after with an
      untreated-node control, refused when the cadence cannot resolve the
      lever's window, null on a twin with no intervention dynamics
- [x] Implement adaptive dosing based on measured decay - `DosingController`
      moves each lever's interval within 0.5x-2x of its published decay
- [x] Add response validation and testing - including the tests that assert no
      gain where none should exist (chronic dust, saturated suppressant)
- [x] Improve suppression reason tracking - gate number from the ordered chain,
      plus the time the gate could next change its answer

### Milestone 7: Data Pipeline & I/O Improvements
**Goal:** More robust and maintainable data pipeline

#### Tasks:
- [x] Complete CPCB adapter (currently unreachable) - parsing, provenance and
      window behaviour are now tested against the published CSV layout, so the
      adapter is finished as far as the network allows. The host is still
      unreachable from this environment and that stays reported, not fixed
- [x] Add data quality checks and validation - coverage against the claimed
      span, largest gap, duplicates, physical ranges, frozen-channel runs
- [x] Improve cache management and expiration - `setu cache` with `--list`,
      `--prune-age-days` and `--prune-bytes`, oldest-first and never keeping a
      stale entry at the expense of a fresh one
- [x] Add data provenance tracking - a fingerprint over the six required
      channels, plus per-response staleness, in every run's provenance block
- [x] Implement offline mode improvements - offline runs mark stale cache
      responses instead of serving them silently, and say so in the report
- [x] Add data source reachability monitoring - `setu sources` probes every
      upstream with status and latency, and refuses to pretend in offline mode

### Milestone 8: Dashboard & API Improvements
**Goal:** An operator interface that can be read against the report

#### What it had to fix

The API served the display sample, so the queue was in arrival order, the ledger
was a bar chart of reason strings with no gate behind it, the forecast had no
trajectory, and a missing run and a corrupt one both arrived as a dict-shaped
shrug. The console could not answer "which gate is costing me the trigger" or
"when could this pass" from what it was given.

#### Tasks:
- [x] Fix dashboard display issues - rebuilt as six sections. The queue is the
      ranked head with priority, trigger interval, owner and sign-off; the gate
      ledger is grouped by stage with the clock that clears each reason; the
      forecast is a trajectory with every fold drawn behind the pooled line;
      evidence opens the pack a row references; data is the provenance, gaps and
      cache. Loading, empty, stale, unavailable and rejected-request states each
      have a rendering rather than a stack trace
- [x] ~~Add real-time monitoring capabilities~~ - not done, and deliberately:
      there is no live feed to monitor until hardware exists. What landed instead
      is the honest substitute - `/api/run` dates the payload, reports its age,
      and the console marks a run stale past seven days
- [x] Improve evidence pack generation and display - packs follow the head of the
      queue, so every row in the default view has one, and `/api/evidence` lists
      what exists instead of the console guessing
- [x] Add API documentation - typed response schemas for every endpoint, tags,
      summaries, and generated `/docs`; null means "not measured" throughout
- [x] Implement proper error handling in API - `results_missing` 503,
      `results_corrupt` 500, `invalid_request` 422, `evidence_not_found` 404,
      each with a `code` and a `hint`, plus strict same-origin CSP and no-store
- [ ] Add authentication and authorization (for production) - deferred with its
      reason recorded in `docs/LIMITATIONS.md`: the service is loopback-only,
      read-only, holds no personal data and has no mutation endpoint. Exposing it
      needs a reverse proxy that authenticates the desk reading it, and that is a
      deployment decision rather than a prototype one

### Milestone 9: Firmware Readiness
**Goal:** Firmware ready for actual deployment

#### Tasks:
- [ ] Complete sensor integration (SPS30, BME280, wind)
- [ ] Fix client.setInsecure() - implement proper certificate pinning
- [ ] Implement readBme() and readWind() stubs
- [ ] Add firmware testing framework
- [ ] Implement OTA update capability
- [ ] Add firmware monitoring and health checks

### Milestone 10: CI/CD & DevOps
**Goal:** Automated testing and deployment pipeline

#### Tasks:
- [ ] Create GitHub Actions CI workflow
- [ ] Add automated testing on PRs
- [ ] Implement regression gates (documented but not active)
- [ ] Add code coverage tracking
- [ ] Implement automated documentation builds
- [ ] Add release automation

### Milestone 11: Documentation Improvements
**Goal:** Comprehensive, accurate, and accessible documentation

#### Tasks:
- [ ] Update README with current status
- [ ] Improve ARCHITECTURE.md with implementation details
- [ ] Update VALIDATION_PROTOCOL.md with latest results
- [ ] Add developer guide
- [ ] Add user guide
- [ ] Add API documentation
- [ ] Improve LIMITATIONS.md with mitigation strategies

### Milestone 12: Reproducibility & Validation
**Goal:** Fully reproducible results with strong validation

#### Tasks:
- [x] Add reproducibility checks - `setu verify` re-runs the pipeline from the
      committed snapshot and compares the fresh payload against
      `results/validation.json` field by field: fingerprint and counts exact,
      measured quantities inside a stated tolerance, a field the payload
      predates reported as skipped rather than passed. The CI replay job now
      runs this instead of a bare `setu run`, which only proved nothing crashed
- [ ] Implement result validation framework - the gates in `setu check` and the
      API schemas cover part of this; a general one does not exist
- [ ] Add comparison with published literature - effect sizes are cited, model
      skill is not compared band by band to the papers it quotes
- [ ] Implement cross-validation framework - walk-forward refits exist for the
      forecast; calibration is still a single 60/40 split
- [ ] Add sensitivity analysis
- [x] Document reproducibility procedures - `docs/VALIDATION_PROTOCOL.md` §5
      states the command, what is compared, and what the current run produces

---

## Implementation Priority

### Done
- Test reliability, and a CI workflow
- Calibration, attribution, forecast, response and data-pipeline milestones
- The operator console, its typed API, and the queue/ledger payloads behind it

### Next
1. Sensitivity analysis on the threshold and the lever half-life (milestone 12)
2. Coverage tracking and a documented build (milestone 10)
3. Firmware completion (milestone 9) - blocked on hardware that does not exist
4. Documentation sweep across all five docs (milestone 11)

---

## Commit Guidelines

Each commit should:
- Address exactly one issue or feature
- Include tests for new functionality
- Update documentation as needed
- Maintain backward compatibility where possible
- Include meaningful commit message with context

## Testing Strategy

- All commits must pass the full test suite (313 collected)
- New features must include tests
- Bug fixes must include regression tests
- Performance improvements must maintain accuracy

## Quality Gates

Before accepting a commit:
1. All tests pass
2. No new warnings introduced
3. Code coverage maintained or improved
4. Documentation updated
5. No breaking changes without deprecation period

---

## Progress Tracking

| Milestone | State |
| --- | --- |
| 0 Foundation | done |
| 1 Test reliability | done - the suite is green; the three named failures were fixed before this pass |
| 2 Code quality | partial - deprecations and types are addressed where the code was touched, not swept |
| 3 Calibration | partial - conformal intervals, recalibration triggers and drift monitoring landed; cross-validation of the calibration fit did not |
| 4 Attribution | done |
| 5 Forecast | done, except the GNN task, deferred with its reason recorded |
| 6 Response | done |
| 7 Data pipeline | done, except CPCB reachability, which is a network fact rather than a code gap |
| 8 Operator console | done, except authentication, which is a deployment requirement rather than a prototype one |
| 12 Reproducibility | partial - the replay gate enforces the committed numbers; sensitivity analysis and a literature comparison remain |
| 9-11 | not started |

## Next Steps

Milestone 9 (firmware) is blocked on hardware that does not exist: sensor
integration, certificate pinning and OTA all need a flashed board to be more
than source code, so its honest state is "written, untested" and no amount of
work in this repository changes that. The replay gate closed the first task of
milestone 12; what remains verifiable here is sensitivity analysis (which lever
and which threshold the response result actually depends on), coverage reporting
in CI, and the calibration cross-validation that milestone 3 left open.
