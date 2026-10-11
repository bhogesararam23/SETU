"""Transfer must be scored by distance, not averaged over it.

`_distance_bands` decides whether the headline "38 nodes calibrated without ever
seeing a reference" survives the obvious objection: maybe they only work because
their donor is next door. These tests pin the banding arithmetic so the reported
answer cannot drift.
"""

import math

from setu.pipeline import _distance_bands, _summarise_transfer

SPACING = 320.0


def diag(distances, mae_after, mae_before=None, r2=None):
    n = len(distances)
    return {
        "nodes": [f"n{i}" for i in range(n)],
        "distance_m": list(distances),
        "mae_before": list(mae_before if mae_before is not None else [10.0] * n),
        "mae_after": list(mae_after),
        "r2": list(r2 if r2 is not None else [0.9] * n),
    }


def bands(d, spacing=SPACING):
    """Just the rows; the correlation has its own test."""
    return _distance_bands(d, spacing)[0]


def test_every_node_lands_in_exactly_one_band():
    distances = [100.0, 400.0, 700.0, 1000.0, 1300.0, 1600.0, 2500.0]
    rows = bands(diag(distances, [1.0] * len(distances)))
    assert sum(b["nodes"] for b in rows) == len(distances)
    assert rows, "no bands produced"


def test_band_labels_come_back_in_near_to_far_order():
    # 320 m pitch -> edges at 480 / 800 / 1120 m.
    distances = [2500.0, 100.0, 1000.0, 700.0]
    rows = bands(diag(distances, [1.0] * len(distances)))
    assert [b["band"] for b in rows] == ["within 1.5 pitches", "1.5-2.5 pitches",
                                         "2.5-3.5 pitches", "over 3.5 pitches"]
    assert [b["mean_distance_m"] for b in rows] == sorted(
        b["mean_distance_m"] for b in rows)


def test_a_far_band_reports_worse_mae_rather_than_hiding_it():
    """If skill decays with distance the report has to show the decay; averaging
    the two bands together is how a reviewer would catch us."""
    d = diag([150.0, 1500.0], [1.0, 6.0])
    rows = bands(d)
    near = [b for b in rows if b["band"] == "within 1.5 pitches"][0]
    far = [b for b in rows if b["band"] == "over 3.5 pitches"][0]
    assert near["mean_mae_after"] < far["mean_mae_after"]
    assert near["mean_distance_m"] < far["mean_distance_m"]


def test_distance_mae_correlation_is_one_number_for_the_whole_set():
    """It is a statistic about all 38 nodes; repeating it inside every band would
    invite someone to read it as a per-band measurement."""
    d = diag([100.0, 500.0, 900.0, 1400.0], [1.0, 2.0, 3.0, 4.0])
    bands, corr = _distance_bands(d, SPACING)
    assert corr is not None and corr > 0.9
    assert all("distance_mae_correlation" not in b for b in bands)
    assert _summarise_transfer(d, SPACING)["distance_mae_correlation"] == round(corr, 3)


def test_correlation_is_refused_when_distance_does_not_vary():
    """Every node the same distance from its donor carries no information about
    distance decay. Returning 0.0 there would be a fabricated result."""
    bands, corr = _distance_bands(diag([300.0, 300.0, 300.0], [1.0, 2.0, 3.0]), SPACING)
    assert bands and corr is None


def test_empty_transfer_produces_no_bands_and_no_crash():
    bands, corr = _distance_bands({"distance_m": [], "mae_after": [],
                                   "mae_before": [], "r2": []}, SPACING)
    assert bands == [] and corr is None


def test_summary_carries_the_reach_of_the_donor_network():
    d = diag([150.0, 1500.0], [1.0, 2.0])
    s = _summarise_transfer(d, spacing_m=SPACING)
    assert s["max_donor_distance_m"] == 1500.0
    assert s["nodes_transferred"] == 2
    assert {"band", "nodes", "mean_mae_after"} <= set(s["by_distance_band"][0])


def test_spacing_drives_the_band_edges():
    """A denser grid should push more nodes into the nearer bands - the bands are
    in lattice pitches, not in absolute metres."""
    d = diag([500.0], [1.0])
    # 320 m pitch puts 500 m in the second band; a 400 m pitch (edge 600 m) does not.
    assert bands(d, SPACING)[0]["band"] == "1.5-2.5 pitches"
    assert bands(d, 400.0)[0]["band"] == "within 1.5 pitches"


def test_bands_are_contiguous_with_no_gaps():
    """A node at a band edge must not be silently dropped from the count."""
    edges = [0.0, 1.5 * SPACING, 2.5 * SPACING, 3.5 * SPACING, math.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if math.isinf(hi):
            continue
        probe = [hi - 1e-6, hi + 1e-6]
        rows = bands(diag(probe, [1.0, 1.0]))
        assert sum(b["nodes"] for b in rows) == 2
