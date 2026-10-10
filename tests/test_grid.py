"""Geometry and transport graph. The physics here is the novelty claim."""

import math

import pytest

from setu import grid
from setu.config import load_site
from setu.grid import Node


@pytest.fixture(scope="module")
def site():
    return load_site("rajkot")


@pytest.fixture(scope="module")
def nodes(site):
    return grid.build_lattice(site)


def test_lattice_delivers_the_instrumented_and_holdout_split(nodes, site):
    assert len(nodes) == 49
    assert len(grid.instrumented(nodes)) == site.instrumented_nodes == 40
    assert len(grid.holdout(nodes)) == 9
    assert len({n.id for n in nodes}) == 49


def test_neighbour_spacing_matches_the_configured_grid(nodes):
    a, b = nodes[0], nodes[1]
    assert grid.haversine_m(a, b) == pytest.approx(320.0, abs=1.0)


def test_the_lattice_is_centred_on_the_site(nodes, site):
    mid_lat = sum(n.lat for n in nodes) / len(nodes)
    mid_lon = sum(n.longitude for n in nodes) / len(nodes)
    assert mid_lat == pytest.approx(site.latitude, abs=0.005)
    assert mid_lon == pytest.approx(site.longitude, abs=0.005)


def test_holdout_sites_are_spread_across_the_district(nodes):
    """Held-out sites must not cluster, or the transfer test is spatially
    biased toward one corner of the district and the generalisation number is
    meaningless."""
    h = grid.holdout(nodes)
    xs = [n.x_m for n in h]
    ys = [n.y_m for n in h]
    # Span most of the lattice in both axes rather than sitting in one quadrant.
    assert (max(xs) - min(xs)) > 0.8 * (max(n.x_m for n in nodes)
                                        - min(n.x_m for n in nodes))
    assert (max(ys) - min(ys)) > 0.8 * (max(n.y_m for n in nodes)
                                        - min(n.y_m for n in nodes))
    # And no two of them share a site.
    assert len({n.id for n in h}) == len(h)


def test_bearing_is_clockwise_from_north():
    a = Node("a", 22.0, 70.0, 0, 0, "instrumented")
    north = Node("n", 22.01, 70.0, 0, 11100, "instrumented")
    east = Node("e", 22.0, 70.01, 11100, 0, "instrumented")
    south = Node("s", 21.99, 70.0, 0, -11100, "instrumented")
    assert grid.bearing_deg(a, north) == pytest.approx(0.0, abs=0.5)
    assert grid.bearing_deg(a, east) == pytest.approx(90.0, abs=1.0)
    assert grid.bearing_deg(a, south) == pytest.approx(180.0, abs=0.5)


def test_upwind_score_is_one_directly_upwind_and_zero_downwind():
    centre = Node("c", 22.0, 70.0, 0, 0, "instrumented")
    west = Node("w", 22.0, 69.99, -1000, 0, "instrumented")
    east = Node("e", 22.0, 70.01, 1000, 0, "instrumented")
    # Wind FROM the west: the western node is upwind of the centre.
    assert grid.upwind_score(centre, west, 270.0) == pytest.approx(1.0, abs=1e-6)
    assert grid.upwind_score(centre, east, 270.0) == pytest.approx(0.0, abs=1e-6)
    # Reversing the wind reverses the answer.
    assert grid.upwind_score(centre, east, 90.0) == pytest.approx(1.0, abs=1e-6)


def test_a_flank_neighbour_contributes_nothing():
    centre = Node("c", 22.0, 70.0, 0, 0, "instrumented")
    north = Node("n", 22.01, 70.0, 0, 1000, "instrumented")
    assert grid.upwind_score(centre, north, 270.0) == pytest.approx(0.0, abs=1e-9)


def test_adjacency_is_directed_and_reverses_with_the_wind(nodes):
    subset = grid.instrumented(nodes)[:16]
    west_wind = grid.adjacency(subset, 270.0)
    east_wind = grid.adjacency(subset, 90.0)
    assert west_wind != east_wind
    assert grid.graph_density(west_wind)["mean_edges"] > 0


def test_isolated_nodes_are_reported_not_silently_dropped(nodes):
    """A node with no upwind neighbour is a real state, and the forecaster has
    to know it is flying blind rather than receive an empty row."""
    subset = grid.instrumented(nodes)[:9]
    graph = grid.adjacency(subset, 270.0)
    assert len(graph) == 9
    assert all(v is not None for v in graph.values())


def test_nearest_neighbours_respects_the_radius(nodes):
    a = nodes[0]
    found = grid.nearest_neighbours(a, nodes, k=20, max_distance_m=400.0)
    assert found
    assert all(d <= 400.0 for _, d in found)
    assert len(found) <= 4      # a 320 m lattice has at most 4 nodes in 400 m


def test_an_impossible_node_count_is_refused():
    import dataclasses
    site = dataclasses.replace(load_site("rajkot"), instrumented_nodes=99)
    with pytest.raises(ValueError, match="requested"):
        grid.build_lattice(site)
