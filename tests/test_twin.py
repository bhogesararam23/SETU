"""The twin is only useful if its physics is checkable.

The test that matters most here is test_dust_hours_are_coarse_heavy_and_burning
_hours_are_fine_heavy: the entire attribution claim rests on those two source
classes being separable in the size distribution. If the generator does not
produce that separation, no classifier trained on it means anything.
"""

import numpy as np
import pandas as pd
import pytest

from setu import grid, twin
from setu.config import load_site
from setu.io import openmeteo as om


@pytest.fixture(scope="module")
def background():
    """Real background, served from the on-disk cache when it is present.

    A 30-day window is enough to contain a humid spell, a dust campaign and a
    few burning nights, and it keeps the suite quick on a first clone.
    """
    from setu.io.cache import HttpCache

    site = load_site("rajkot")
    df, _ = om.load_hourly(site, cache=HttpCache(), past_days=30, forecast_days=0)
    return site, df


@pytest.fixture(scope="module")
def twins(background):
    site, df = background
    nodes = grid.instrumented(grid.build_lattice(site))
    cfg = twin.TwinConfig()
    return site, df, nodes, twin.DigitalTwin(df, nodes, site, cfg)


# -- hygroscopic growth ----------------------------------------------------
@pytest.mark.parametrize("rh", [0.0, 20.0, 40.0, 60.0, 80.0, 90.0])
def test_growth_factor_is_monotonic_in_humidity(rh):
    assert twin.growth_factor(rh + 1.0, 0.20) > twin.growth_factor(rh, 0.20)


def test_dry_air_is_not_inflated():
    assert twin.growth_factor(0.0, 0.20) == pytest.approx(1.0)


def test_saturated_air_over_reads_by_a_plausible_amount():
    # ~1.8x at 95% RH, which is the order reported for uncorrected OPCs.
    assert 1.5 < twin.growth_factor(95.0, 0.20) < 2.2


def test_humidity_is_clipped_short_of_one_so_nothing_becomes_infinite():
    assert np.isfinite(twin.growth_factor(100.0, 0.20))
    assert np.isfinite(twin.growth_factor(-5.0, 0.20))


# -- raw vs truth ----------------------------------------------------------
def test_raw_over_reads_truth_on_a_humid_window(twins):
    site, df, nodes, t = twins
    node = nodes[0]
    truth = t.truth(node)
    raw = t.raw(node)
    humid = truth.index[truth["rh"] > 85]
    if len(humid) < 24:
        pytest.skip("this window has no humid stretch")
    ratio = float(raw.loc[humid, "pm25"].mean() / truth.loc[humid, "pm25"].mean())
    assert ratio > 1.05


def test_a_dead_node_reports_zero_rather_than_missing(twins):
    site, df, nodes, t = twins
    health = t.health()
    dead = [n for n in nodes if bool(health.loc[n.id, "dead"])]
    assert dead, "the seeded twin should contain a dead node"
    raw = t.raw(dead[0])
    assert (raw["pm25"] == 0.0).all()
    assert (raw["pm10"] == 0.0).all()


def test_an_obstructed_node_reads_low(twins):
    site, df, nodes, t = twins
    health = t.health()
    obs = [n for n in nodes if bool(health.loc[n.id, "obstructed"])]
    assert obs
    node = obs[0]
    assert t.raw(node)["pm10"].mean() < t.truth(node)["pm10"].mean()


def test_a_drifting_node_diverges_over_the_window(twins):
    site, df, nodes, t = twins
    health = t.health()
    driftery = [n for n in nodes if health.loc[n.id, "drift_fraction"] > 0]
    assert driftery
    node = driftery[0]
    truth, raw = t.truth(node), t.raw(node)
    early = float((raw["pm10"].iloc[:48] / truth["pm10"].iloc[:48]).mean())
    late = float((raw["pm10"].iloc[-48:] / truth["pm10"].iloc[-48:]).mean())
    assert late > early


def test_the_same_seed_reproduces_the_same_twin(background):
    site, df = background
    nodes = grid.instrumented(grid.build_lattice(site))
    a = twin.DigitalTwin(df, nodes, site, twin.TwinConfig(seed=7))
    b = twin.DigitalTwin(df, nodes, site, twin.TwinConfig(seed=7))
    assert np.allclose(a.raw(nodes[3])["pm25"], b.raw(nodes[3])["pm25"])
    c = twin.DigitalTwin(df, nodes, site, twin.TwinConfig(seed=8))
    assert not np.allclose(a.raw(nodes[3])["pm25"], c.raw(nodes[3])["pm25"])


# -- the physics that carries the argument --------------------------------
def test_dust_hours_are_coarse_heavy_and_burning_hours_are_fine_heavy(twins):
    site, df, nodes, t = twins
    coarse_heavy, fine_heavy = [], []
    for node in nodes:
        truth = t.truth(node)
        ratio = truth["coarse"] / truth["pm10"].clip(lower=1e-9)
        dust = truth[truth["dominant_source"] == "road_dust"]
        burn = truth[truth["dominant_source"] == "open_burning"]
        if len(dust) > 20 and len(burn) > 20:
            coarse_heavy.append(float((dust["coarse"] / dust["pm10"].clip(lower=1e-9)).mean()))
            fine_heavy.append(float((burn["coarse"] / burn["pm10"].clip(lower=1e-9)).mean()))
    assert coarse_heavy, "no node produced both dust and burning hours"
    assert np.mean(coarse_heavy) > np.mean(fine_heavy) + 0.10


def test_number_counts_shift_to_the_coarse_bins_for_dust(twins):
    site, df, nodes, t = twins
    dust_ratio, burn_ratio = [], []
    for node in nodes:
        truth = t.truth(node)
        coarse_bins = truth["n5.0"] + truth["n10.0"]
        fine_bins = truth["n0.3"] + truth["n0.5"] + 1e-9
        d = truth[truth["dominant_source"] == "road_dust"]
        b = truth[truth["dominant_source"] == "open_burning"]
        if len(d) > 20 and len(b) > 20:
            dust_ratio.append(float((coarse_bins / fine_bins)[d.index].mean()))
            burn_ratio.append(float((coarse_bins / fine_bins)[b.index].mean()))
    # The measured separation is about 5x. It is not larger because the
    # regional fine-mode background is present in every hour, which is exactly
    # the confounding the classifier has to see through.
    assert np.mean(dust_ratio) > np.mean(burn_ratio) * 3


def test_hours_with_no_injection_are_labelled_as_background_not_forced(twins):
    site, df, nodes, t = twins
    labels = pd.concat([t.truth(n)["dominant_source"] for n in nodes[:10]])
    counts = labels.value_counts()
    assert "regional_background" in counts.index
    assert counts["regional_background"] > 0
    # A class the classifier is allowed to abstain with.
    assert "mixed" in counts.index


def test_open_burning_hours_carry_a_co_spike(twins):
    site, df, nodes, t = twins
    prone = [n for n in nodes
             if bool(t.health().loc[n.id, "burning_prone"])]
    assert prone
    truth = t.truth(prone[0])
    burn = truth[truth["dominant_source"] == "open_burning"]
    other = truth[truth["dominant_source"] != "open_burning"]
    assert len(burn) > 0
    assert burn["co"].mean() > other["co"].mean()


def test_every_frame_is_marked_as_simulated(twins):
    site, df, nodes, t = twins
    assert t.raw(nodes[0]).attrs["provenance"] == "digital-twin"


def test_injected_summary_reports_the_background_it_used(twins):
    site, df, nodes, t = twins
    s = t.injected_summary()
    assert s["hours"] == len(df)
    assert "open-meteo" in s["background"]
    assert s["dead_nodes"] >= 1
    assert s["seed"] == 20261010
