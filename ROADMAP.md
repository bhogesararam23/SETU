# SETU Development Roadmap

## Overview

This document tracks the systematic improvement of SETU from its current state (commit `e395d31`) into a robust, reproducible, scientifically credible open-source air quality research project.

**Current Status:**
- 223 tests defined, 220 passing, 3 failing
- Full pipeline runs end-to-end
- Digital twin validates algorithms against known ground truth
- Calibration, attribution, forecasting, and response layers all functional
- Documentation is comprehensive and honest about limitations

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
- [ ] Add forecast uncertainty estimation
- [ ] Improve handling of missing meteorological data
- [ ] Add forecast validation against persistence baseline
- [ ] Implement multi-step ahead forecasting
- [ ] Add forecast diagnostics

### Milestone 6: Response Engine Enhancements
**Goal:** More effective and measurable interventions

#### Tasks:
- [ ] Fix equal-cost comparison for long-lived levers
- [ ] Improve action scheduling and prioritization
- [ ] Add intervention effect measurement framework
- [ ] Implement adaptive dosing based on measured decay
- [ ] Add response validation and testing
- [ ] Improve suppression reason tracking

### Milestone 7: Data Pipeline & I/O Improvements
**Goal:** More robust and maintainable data pipeline

#### Tasks:
- [ ] Complete CPCB adapter (currently unreachable)
- [ ] Add data quality checks and validation
- [ ] Improve cache management and expiration
- [ ] Add data provenance tracking
- [ ] Implement offline mode improvements
- [ ] Add data source reachability monitoring

### Milestone 8: Dashboard & API Improvements
**Goal:** Production-ready operator interface

#### Tasks:
- [ ] Fix dashboard display issues
- [ ] Add real-time monitoring capabilities
- [ ] Improve evidence pack generation and display
- [ ] Add API documentation
- [ ] Implement proper error handling in API
- [ ] Add authentication and authorization (for production)

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
- [ ] Add reproducibility checks
- [ ] Implement result validation framework
- [ ] Add comparison with published literature
- [ ] Implement cross-validation framework
- [ ] Add sensitivity analysis
- [ ] Document reproducibility procedures

---

## Implementation Priority

### Immediate (Next 5 Commits)
1. Fix the 3 failing tests
2. Run full test suite to confirm 223/223 passing
3. Add CI workflow file
4. Fix deprecation warnings
5. Create first milestone report

### Short-term (Commits 6-20)
1. Code quality improvements (types, docs, validation)
2. Calibration layer enhancements
3. Attribution layer improvements
4. Add regression gates to CI
5. Second milestone report

### Medium-term (Commits 21-50)
1. Forecast layer improvements
2. Response engine enhancements
3. Data pipeline improvements
4. Dashboard/API improvements
5. Third milestone report

### Long-term (Commits 51-100)
1. Firmware completion
2. Advanced features (GNN, conformal prediction)
3. Production readiness
4. Final milestone report

---

## Commit Guidelines

Each commit should:
- Address exactly one issue or feature
- Include tests for new functionality
- Update documentation as needed
- Maintain backward compatibility where possible
- Include meaningful commit message with context

## Testing Strategy

- All commits must pass full test suite (223 tests)
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

Use the todo list in development sessions to track progress against this roadmap.

## Next Steps

Start with Milestone 1: Fix the 3 failing tests and achieve 223/223 passing test suite.
