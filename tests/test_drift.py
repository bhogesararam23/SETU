"""The drift monitor has to distinguish 'the air changed' from 'the sensor
changed'. Getting that backwards either quarantines good nodes during an
episode or keeps bad ones running through winter."""

import numpy as np
import pandas as pd
import pytest

from setu.calibrate.drift import CusumMonitor, quarantine_nodes, residual_series


def _index(n):
    return pd.date_range("2026-09-01", periods=n, freq="h")


def test_clean_noise_raises_almost_nothing():
    """Asserted as a rate, not as zero.

    Any monitor with a finite threshold will eventually fire on clean data;
    the claim worth testing is that it does so rarely enough that operators
    keep believing it. One false quarantine per season is tolerable, one per
    fortnight is not.
    """
    total_hours = 0
    total_alarms = 0
    for seed in range(12):
        rng = np.random.default_rng(seed)
        m = CusumMonitor()
        hours = 2000
        total_alarms += len(m.feed(rng.normal(0.0, 2.0, hours), _index(hours)))
        total_hours += hours
    assert total_alarms / total_hours < 1e-3, (
        f"{total_alarms} false alarms in {total_hours} clean hours")


def test_a_shorter_baseline_makes_the_scale_estimate_unstable():
    """Why warm_up is two weeks: sigma sets every threshold in this module."""
    estimates = []
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        m = CusumMonitor(warm_up=40)
        m.feed(rng.normal(0.0, 2.0, 40), _index(40))
        estimates.append(m.sigma)
    short_spread = float(np.std(estimates))

    estimates = []
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        m = CusumMonitor(warm_up=336)
        m.feed(rng.normal(0.0, 2.0, 336), _index(336))
        estimates.append(m.sigma)
    assert float(np.std(estimates)) < short_spread


def test_a_step_change_is_caught():
    rng = np.random.default_rng(2)
    res = np.concatenate([rng.normal(0, 2.0, 600), rng.normal(14, 2.0, 600)])
    m = CusumMonitor()
    alarms = m.feed(res, _index(len(res)))
    assert alarms
    assert alarms[0].direction == "up"
    # It should fire after the step, not during the clean stretch.
    assert alarms[0].at >= _index(len(res))[600]


def test_a_slow_ramp_is_caught_even_though_no_single_hour_is_oulier():
    """This is the case that a threshold-on-the-value check always misses."""
    rng = np.random.default_rng(3)
    ramp = np.linspace(0.0, 26.0, 3000) + rng.normal(0, 1.5, 3000)
    m = CusumMonitor()
    alarms = m.feed(ramp, _index(len(ramp)))
    assert alarms, "a monotone 14 ug/m3 ramp must raise a drift alarm"


def test_a_downward_drift_is_caught_too():
    rng = np.random.default_rng(4)
    res = np.concatenate([rng.normal(0, 2.0, 500), rng.normal(-14, 2.0, 500)])
    m = CusumMonitor()
    alarms = m.feed(res, _index(len(res)))
    assert any(a.direction == "down" for a in alarms)


def test_resetting_stops_one_long_drift_becoming_two_thousand_alarms():
    rng = np.random.default_rng(5)
    res = np.concatenate([rng.normal(0, 2.0, 400), rng.normal(16, 2.0, 900)])
    m = CusumMonitor(reset_on_alarm=True)
    m.feed(res, _index(len(res)))
    assert 1 <= len(m.alarms) <= 6


def test_sigma_is_not_inflated_by_the_drift_it_is_trying_to_detect():
    rng = np.random.default_rng(6)
    clean = rng.normal(0, 2.0, 400)
    m = CusumMonitor(warm_up=200)
    m.feed(clean, _index(len(clean)))
    baseline_sigma = m.sigma
    assert baseline_sigma is not None, "warm_up must be satisfied by the clean feed"
    assert baseline_sigma == pytest.approx(2.0, abs=0.8)
    # A MAD estimator survives the contaminated tail; a standard deviation
    # would have grown with it and pushed the threshold out of reach.
    m.feed(rng.normal(30, 2.0, 200), _index(200))
    assert m.sigma is not None and m.sigma < 12.0


def test_warm_up_suppresses_alarms_until_sigma_can_be_estimated():
    m = CusumMonitor()
    out = [m.update(50.0) for _ in range(90)]
    assert all(o is None for o in out)


def test_non_finite_residuals_are_skipped_not_propagated():
    m = CusumMonitor(warm_up=8, interval=3.0, allowance=0.4, cooldown=1)
    for v in [0.0, 1.0, np.nan, np.inf, 0.5]:
        m.update(v)
    assert m.state["samples"] == 3      # nan and inf never entered


def test_residual_series_aligns_on_the_common_index():
    idx = _index(10)
    cal = pd.Series(np.arange(10.0), index=idx)
    ref = pd.Series(np.arange(3.0, 10.0) + 1.0, index=idx[3:])
    res = residual_series(cal, ref)
    assert len(res) == 7
    assert np.allclose(res, -1.0)


def test_monitor_parameter_validation():
    with pytest.raises(ValueError):
        CusumMonitor(allowance=0.0)
    with pytest.raises(ValueError):
        CusumMonitor(allowance=1.0, interval=0.5)
    with pytest.raises(ValueError):
        CusumMonitor(warm_up=2)


def test_quarantine_lists_only_nodes_that_actually_fired():
    good, bad = CusumMonitor(), CusumMonitor()
    rng = np.random.default_rng(7)
    good.feed(rng.normal(0, 2, 1000), _index(1000))
    bad.feed(np.concatenate([rng.normal(0, 2, 400), rng.normal(20, 2, 600)]),
             _index(1000))
    out = quarantine_nodes({"n-good": (good, 0), "n-bad": (bad, 0)})
    assert out == ["n-bad"]


def test_a_stable_offset_is_treated_as_calibration_not_drift():
    """A node that is consistently 1.5 ug/m3 high is mis-calibrated, not drifting.

    Quarantining it would be wrong twice over: the fix is a recalibration, and
    the operator loses a sensor that is still telling the truth about changes in
    the air. Asserted as a rate because any finite threshold eventually fires on
    noise - the measured false-alarm rate is about one per 3,000 hours, roughly
    one quarantine per four months per node, which is survivable.
    """
    alarms = 0
    hours = 0
    for seed in range(6):
        rng = np.random.default_rng(21 + seed)
        res = rng.normal(1.5, 2.0, 3000)
        m = CusumMonitor()
        alarms += len(m.feed(res, _index(len(res))))
        hours += len(res)
    assert alarms / hours < 1e-3, f"{alarms} false quarantines in {hours} hours"


def test_a_step_away_from_a_stable_offset_is_still_caught():
    rng = np.random.default_rng(22)
    res = np.concatenate([rng.normal(1.5, 2.0, 800), rng.normal(14.0, 2.0, 800)])
    m = CusumMonitor()
    alarms = m.feed(res, _index(len(res)))
    assert alarms and alarms[0].direction == "up"


def test_a_real_episode_seen_through_a_correct_calibration_is_not_drift():
    """Both signal and reference move together, so the residual stays flat."""
    rng = np.random.default_rng(8)
    hours = 1200
    episode = np.concatenate([np.zeros(600), 80.0 * np.exp(-np.arange(600) / 120.0)])
    truth = 25.0 + episode + rng.normal(0, 3, hours)
    calibrated = truth + rng.normal(0, 2, hours)
    m = CusumMonitor()
    alarms = m.feed((calibrated - truth), _index(hours))
    assert alarms == []
