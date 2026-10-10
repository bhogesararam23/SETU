"""Every downstream claim is computed by this module, so it is checked by hand."""

import numpy as np
import pytest

from setu import metrics as m


def test_perfect_prediction_is_zero_error_and_full_agreement():
    a = [10.0, 20.0, 30.0, 60.0, 5.0]
    card = m.score(a, a, threshold=60.0)
    assert card.mae == 0.0
    assert card.rmse == 0.0
    assert card.nmae == 0.0
    assert card.r2 == pytest.approx(1.0)
    assert card.bias == 0.0
    assert card.threshold_accuracy == 1.0
    assert card.n == 5


def test_hand_computed_case():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 36.0])
    # |e| = 2, 2, 6 -> MAE 3.333; e^2 = 4, 4, 36 -> RMSE sqrt(44/3) = 3.830
    assert m.mae(actual, predicted) == pytest.approx(10 / 3)
    assert m.rmse(actual, predicted) == pytest.approx(np.sqrt(44 / 3))
    # bias = (2 + -2 + 6)/3 = +2 : an over-read
    assert m.bias(actual, predicted) == pytest.approx(2.0)
    # nMAE against a mean of 20
    assert m.nmae(actual, predicted) == pytest.approx((10 / 3) / 20)


def test_nan_pairs_are_dropped_rather_than_propagated():
    a = [1.0, np.nan, 3.0, 4.0]
    p = [1.0, 2.0, 3.0, np.nan]
    card = m.score(a, p)
    assert card.n == 2
    assert card.mae == 0.0


def test_all_nan_is_refused_not_reported_as_zero_error():
    with pytest.raises(ValueError, match="no finite"):
        m.score([np.nan, np.nan], [1.0, 2.0])


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="shape mismatch"):
        m.mae([1.0, 2.0], [1.0])


def test_constant_actuals_cannot_produce_an_r_squared():
    with pytest.raises(ValueError, match="constant"):
        m.r2([5.0, 5.0, 5.0], [5.1, 4.9, 5.0])


def test_a_high_correlation_can_still_fail_the_decision_it_drives():
    """The failure mode worth a test: a calibration that tracks the shape of
    the series but sits just under the standard, so it never fires."""
    actual = np.array([55.0, 58.0, 61.0, 64.0, 59.0, 62.0, 70.0, 40.0])
    predicted = actual - 2.5
    # A constant 2.5 ug/m3 under-read still scores R^2 above 0.9...
    assert m.r2(actual, predicted) > 0.90
    # ...while missing half of the exceedances the alert exists to raise.
    assert m.detection_rate(actual, predicted, 60.0) == pytest.approx(0.5)
    assert m.threshold_accuracy(actual, predicted, 60.0) == pytest.approx(0.75)


def test_false_alarm_rate_ignores_hours_that_were_never_alerted():
    actual = np.array([10.0, 100.0, 12.0, 130.0])
    predicted = np.array([80.0, 100.0, 5.0, 130.0])
    # Three alerts raised; the one at index 0 was against 10 ug/m3 of clean air.
    assert m.false_alarm_rate(actual, predicted, 60.0) == pytest.approx(1 / 3)


def test_no_alerts_is_zero_false_alarms_not_a_division_error():
    assert m.false_alarm_rate([10.0, 20.0], [5.0, 5.0], 60.0) == 0.0


def test_detection_rate_is_nan_when_there_was_nothing_to_detect():
    assert np.isnan(m.detection_rate([1.0, 2.0], [1.0, 2.0], 60.0))


def test_persistence_skill_is_full_when_the_forecast_is_exact():
    actual = np.array([12.0, 11.0, 20.0])
    predicted = actual.copy()
    baseline = np.array([10.0, 12.0, 11.0])   # yesterday's value, by definition
    assert m.persistence_skill(actual, predicted, baseline) == pytest.approx(1.0)


def test_persistence_skill_is_negative_when_worse_than_doing_nothing():
    a = np.array([10.0, 10.0, 10.0, 50.0])
    baseline = np.array([10.0, 10.0, 10.0, 10.0])
    worse = np.array([10.0, 10.0, 10.0, 5.0])
    assert m.persistence_skill(a, worse, baseline) < 0


def test_a_single_breach_day_is_not_non_attainment():
    """CPCB's 24-hour standard is not met on two consecutive days, and must be
    met on 98% of days; a one-day spike is not a compliance failure."""
    daily = [50, 61, 55, 58]
    flags = m.attainment_series(daily, standard=60)
    assert flags == [False, False, False, False]


def test_two_consecutive_breach_days_is_non_attainment():
    daily = [50, 61, 62, 55]
    assert m.attainment_series(daily, standard=60) == [False, False, True, False]


def test_a_long_breach_spell_flags_every_day_after_the_second():
    daily = [61, 62, 63, 50, 64]
    assert m.attainment_series(daily, standard=60) == [False, True, True, False, False]


def test_threshold_flag_mode_above_is_strict():
    assert m.threshold_accuracy([60.0], [60.0], 60.0, flag_when="above") == 1.0
    assert m.threshold_accuracy([60.0], [60.0], 60.0, flag_when="at_or_above") == 1.0
    assert m.threshold_accuracy([60.0], [61.0], 60.0, flag_when="above") == 0.0


def test_unknown_flag_mode_is_rejected():
    with pytest.raises(ValueError, match="flag_when"):
        m.threshold_accuracy([1.0], [1.0], 1.0, flag_when="whenever")


def test_scorecard_serialises_for_the_report():
    d = m.score([1.0, 2.0, 3.0], [1.1, 2.2, 2.7], threshold=2.5).as_dict()
    assert set(d) >= {"n", "mae", "rmse", "nmae", "r2", "bias",
                      "threshold_accuracy", "false_alarm_rate", "detection_rate"}
    assert isinstance(d["mae"], float)
    assert d["n"] == 3
