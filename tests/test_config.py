"""The config files carry the citations, so they are load-bearing, not prose.

These tests fail if someone edits a YAML into something the pipeline would
silently mis-read.
"""

import pytest

from setu.config import (
    load_action_table,
    load_signatures,
    load_site,
)
from setu.errors import ConfigError


def test_site_lattice_holds_the_asked_for_node_count():
    site = load_site("rajkot")
    assert site.candidate_sites >= site.instrumented_nodes
    # 2 km at 320 m -> 7 per side -> 49 sites, 40 instrumented, 9 held out.
    assert site.cell_count_per_side == 7
    assert site.candidate_sites == 49
    assert site.candidate_sites - site.instrumented_nodes == 9


def test_thresholds_are_the_notified_indian_standards():
    naaqs = load_site("rajkot").naaqs
    assert naaqs["pm25_24h"] == 60
    assert naaqs["pm25_annual"] == 40
    assert naaqs["pm10_24h"] == 100
    assert naaqs["pm10_annual"] == 60


def test_who_reference_is_stricter_than_naaqs():
    site = load_site("rajkot")
    assert site.who_2021["pm25_annual"] < site.naaqs["pm25_annual"]


def test_signature_priors_are_normalised():
    sigs, window = load_signatures()
    assert window >= 48
    assert abs(sum(s.prior_weight for s in sigs.values()) - 1.0) < 1e-9


def test_secondary_is_the_only_non_actionable_source():
    sigs, _ = load_signatures()
    actionable = {k for k, s in sigs.items() if s.actionable_locally}
    assert "secondary" not in actionable
    assert actionable == {
        "road_dust", "construction", "open_burning", "traffic_combustion"
    }


def test_every_source_has_a_policy_entry_even_if_empty():
    table = load_action_table()
    sigs, _ = load_signatures()
    missing = set(sigs) - set(table.policy)
    assert not missing, f"policy is silent about {missing}"
    # Silence is only acceptable when it is an explicit empty list.
    assert table.policy["secondary"] == ()


def test_dose_interval_follows_the_measured_half_life():
    table = load_action_table()
    sprinkling = table.levers["water_sprinkling"]
    sweeping = table.levers["vacuum_sweeping"]
    assert sprinkling.half_life_min == pytest.approx(62.0)
    assert sprinkling.dose_interval_min() == pytest.approx(46.5)
    # The whole argument for triggering rests on sprinkling being short-lived:
    # it must need re-dosing far sooner than sweeping does.
    assert sprinkling.dose_interval_min() < sweeping.dose_interval_min()


def test_unmeasured_levers_refuse_to_invent_a_dose_interval():
    table = load_action_table()
    notice = table.levers["site_notice"]
    assert notice.half_life_min is None
    assert notice.dose_interval_min() is None
    assert not notice.has_measured_effect


def test_refusals_are_present_and_specific():
    table = load_action_table()
    assert set(table.refusals) >= {"smog_tower", "blanket_odd_even"}
    assert "NGT" in table.refusals["smog_tower"]


def test_a_broken_prior_sum_is_rejected_not_renormalised(monkeypatch, tmp_path):
    """Guard the loader itself, not just the shipped file."""
    import setu.config as cfg

    bad = tmp_path / "signatures.yaml"
    bad.write_text(
        "sources:\n"
        "  dust:\n"
        "    label: x\n"
        "    prior_weight: 0.4\n"
        "    actionable_locally: true\n"
        "    features: {coarse_ratio: {direction: up, strength: 0.5}}\n"
        "    channels_simulated: []\n"
        "    corroborates: none\n"
        "    published_share: [0.0, 0.5]\n"
        "baseline: {window_hours: 336}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "_read", lambda name: cfg.__dict__["_read"](name)
                        if name != "signatures.yaml" else
                        __import__("yaml").safe_load(bad.read_text()))
    with pytest.raises(ConfigError, match="sum to"):
        cfg.load_signatures()


def test_a_signature_without_a_share_band_is_rejected_not_guessed(monkeypatch,
                                                                  tmp_path):
    """The cross-check against receptor studies checks against these bands;
    a source that declares none would silently be checked against nothing."""
    import setu.config as cfg

    bad = tmp_path / "signatures.yaml"
    bad.write_text(
        "sources:\n"
        "  dust:\n"
        "    label: x\n"
        "    prior_weight: 1.0\n"
        "    actionable_locally: true\n"
        "    features: {coarse_ratio: {direction: up, strength: 0.5}}\n"
        "    channels_simulated: []\n"
        "    corroborates: none\n"
        "baseline: {window_hours: 336}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "_read",
                        lambda name: (__import__("yaml").safe_load(bad.read_text())
                                      if name == "signatures.yaml"
                                      else cfg.__dict__["_read"](name)))
    with pytest.raises(ConfigError, match="published_share"):
        cfg.load_signatures()


def test_every_published_band_is_a_band():
    sigs, _ = load_signatures()
    for key, sig in sigs.items():
        lo, hi = sig.published_share
        assert 0.0 <= lo < hi <= 1.0, f"{key}: {sig.published_share}"
    # Secondary has to start above zero: a band containing zero would let a
    # classifier that never names it pass the cross-check.
    assert sigs["secondary"].published_share[0] > 0.0
    assert sigs["traffic_combustion"].published_share[0] > 0.0
