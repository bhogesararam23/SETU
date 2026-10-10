"""Does the calibration undo what the twin deliberately broke?

The numbers here are not decorative - they are the figures the validation
report quotes. Two properties matter more than raw accuracy: the interval has
to cover at about its nominal rate (otherwise the engine's uncertainty gate is
guarding nothing), and the target must never appear among its own predictors.
"""

import numpy as np
import pandas as pd
import pytest

from setu import grid, twin
from setu.calibrate.model import (
    FEATURES,
    CalibrationModel,
    transfer_calibrate,
)
from setu.config import load_site
from setu.errors import CalibrationError
from setu.io import openmeteo as om
from setu.io.cache import HttpCache


@pytest.fixture(scope="module")
def pair():
    site = load_site("rajkot")
    df, _ = om.load_hourly(site, HttpCache(), past_days=60, forecast_days=0)
    nodes = grid.instrumented(grid.build_lattice(site))
    t = twin.DigitalTwin(df, nodes, site, twin.TwinConfig(seed=20261010))
    node = nodes[0]
    truth = t.truth(node)
    raw = t.raw(node)
    # Split chronologically. Fitting on the whole window and scoring on part of
    # it would flatter a model that can memorise the seasonal shape.
    cut = len(truth) * 2 // 3
    return (node, truth, raw, truth.iloc[:cut], raw.iloc[:cut],
            truth.iloc[cut:], raw.iloc[cut:])


def test_the_target_is_never_one_of_its_own_predictors():
    model = CalibrationModel(target="pm25")
    assert "pm25" not in model.features
    assert "pm10" in model.features      # the sibling channel is fair game


def test_fitting_refuses_an_underfed_window(pair):
    node, truth, raw, tr_train, raw_train, _, _ = pair
    model = CalibrationModel(target="pm25")
    with pytest.raises(CalibrationError, match="refusing to fit"):
        model.fit(raw_train.iloc[:100], tr_train["pm25"].iloc[:100])


def test_predict_before_fit_is_an_error_not_a_zero():
    with pytest.raises(CalibrationError, match="fit"):
        CalibrationModel(target="pm25").predict(pd.DataFrame(
            {c: [1.0] for c in FEATURES}))


def test_calibration_reduces_error_substantially(pair):
    node, truth, raw, tr_train, raw_train, tr_test, raw_test = pair
    model = CalibrationModel(target="pm25").fit(raw_train, tr_train["pm25"])
    report = model.evaluate(raw_test, tr_test["pm25"], threshold=60.0, node=node.id)

    assert report.after.mae < report.before.mae
    # The twin injects a ~1.8x hygroscopic over-read plus gain and offset, so
    # a working calibration should remove most of it, not a token amount.
    assert report.after.mae < 0.5 * report.before.mae
    assert report.as_dict()["mae_reduction_pct"] > 50.0


def test_the_bias_that_over_reads_the_standard_is_removed(pair):
    node, truth, raw, tr_train, raw_train, tr_test, raw_test = pair
    model = CalibrationModel(target="pm25").fit(raw_train, tr_train["pm25"])
    report = model.evaluate(raw_test, tr_test["pm25"], threshold=60.0, node=node.id)
    assert abs(report.after.bias) < abs(report.before.bias)
    assert abs(report.after.bias) < 3.0


def test_the_interval_covers_at_about_its_nominal_rate(pair):
    """10th-90th quantiles should bracket the truth ~80% of the time.

    If coverage were far below that, the response engine's rule "do not fire
    while the interval straddles the threshold" would be firing on a fiction.
    """
    node, truth, raw, tr_train, raw_train, tr_test, raw_test = pair
    model = CalibrationModel(target="pm25").fit(raw_train, tr_train["pm25"])
    report = model.evaluate(raw_test, tr_test["pm25"], node=node.id)
    assert 0.55 <= report.interval_coverage <= 0.98


def test_sigma_is_positive_and_finite_everywhere(pair):
    node, truth, raw, tr_train, raw_train, _, raw_test = pair
    model = CalibrationModel(target="pm25").fit(raw_train, tr_train["pm25"])
    pred = model.predict(raw_test)
    assert (pred["sigma"] > 0).all()
    assert np.isfinite(pred[["value", "low", "high", "sigma"]].to_numpy()).all()
    assert (pred["low"] <= pred["value"]).all()
    assert (pred["value"] <= pred["high"]).all()


def test_pm10_calibrates_too_because_the_coarse_channel_is_the_point(pair):
    node, truth, raw, tr_train, raw_train, tr_test, raw_test = pair
    model = CalibrationModel(target="pm10").fit(raw_train, tr_train["pm10"])
    report = model.evaluate(raw_test, tr_test["pm10"], node=node.id)
    assert report.after.nmae < 0.35


def test_transfer_calibrates_a_node_that_was_never_co_located(pair):
    """The donor saw the reference; the target did not. That is the only way
    coverage scales without a reference station on every pole."""
    site = load_site("rajkot")
    df, _ = om.load_hourly(site, HttpCache(), past_days=60, forecast_days=0)
    nodes = grid.instrumented(grid.build_lattice(site))
    t = twin.DigitalTwin(df, nodes, site, twin.TwinConfig(seed=20261010))
    donor_node, target_node = nodes[0], nodes[1]

    cut = len(df) * 2 // 3
    donor_truth, donor_raw = t.truth(donor_node), t.raw(donor_node)
    donor = CalibrationModel(target="pm25").fit(donor_raw.iloc[:cut],
                                                donor_truth["pm25"].iloc[:cut])

    target_truth, target_raw = t.truth(target_node), t.raw(target_node)
    transferred, diag = transfer_calibrate(target_raw.iloc[:cut],
                                           donor_raw.iloc[:cut], donor)
    assert diag["overlap_hours"] > 48
    assert diag["r2"] > 0.5, f"poor transfer fit: {diag}"

    report = transferred.evaluate(target_raw.iloc[cut:], target_truth["pm25"].iloc[cut:],
                                 node=target_node.id)
    naive = np.mean(np.abs(target_raw["pm25"].iloc[cut:]
                           - target_truth["pm25"].iloc[cut:]))
    assert report.after.mae < naive


def test_transfer_refuses_a_window_too_short_to_trust():
    site = load_site("rajkot")
    df, _ = om.load_hourly(site, HttpCache(), past_days=60, forecast_days=0)
    nodes = grid.instrumented(grid.build_lattice(site))
    t = twin.DigitalTwin(df, nodes, site, twin.TwinConfig())
    donor_truth, donor_raw = t.truth(nodes[0]), t.raw(nodes[0])
    donor = CalibrationModel(target="pm25").fit(donor_truth, donor_truth["pm25"])
    with pytest.raises(CalibrationError, match="overlap"):
        transfer_calibrate(t.raw(nodes[1]).iloc[:20], donor_raw.iloc[:20], donor)


def test_a_constant_target_channel_cannot_be_transferred():
    site = load_site("rajkot")
    df, _ = om.load_hourly(site, HttpCache(), past_days=60, forecast_days=0)
    nodes = grid.instrumented(grid.build_lattice(site))
    t = twin.DigitalTwin(df, nodes, site, twin.TwinConfig())
    donor_truth, donor_raw = t.truth(nodes[0]), t.raw(nodes[0])
    donor = CalibrationModel(target="pm25").fit(donor_truth, donor_truth["pm25"])
    stuck = t.raw(nodes[1]).copy()
    stuck["pm25"] = 0.0
    with pytest.raises(CalibrationError, match="constant"):
        transfer_calibrate(stuck.iloc[:200], donor_raw.iloc[:200], donor)
