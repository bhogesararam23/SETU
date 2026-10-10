"""Node placement and the wind-aware adjacency graph.

Two separate jobs, deliberately kept apart:

  * placement is geometry - where the 40 nodes physically go;
  * adjacency is physics - which nodes can deliver air to which, which changes
    hour by hour with the wind.

The reason adjacency is not just "the four nearest nodes": 84% of Delhi's
winter secondary inorganic aerosol arrives from more than 30 km away
(AQuaIM / DPCC, 2023). A graph that treats every neighbour as symmetric will
keep recommending local action against an imported problem, which is the most
expensive mistake this system can make.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Iterable, Sequence

import numpy as np

from .config import Site

EARTH_RADIUS_M = 6_371_000.0


@dataclasses.dataclass(frozen=True)
class Node:
    id: str
    lat: float
    longitude: float
    x_m: float
    y_m: float
    role: str  # "instrumented" | "holdout" | "reference"

    @property
    def is_instrumented(self) -> bool:
        return self.role == "instrumented"

    def as_dict(self) -> dict[str, object]:
        return {"id": self.id, "lat": round(self.lat, 5),
                "lon": round(self.longitude, 5),
                "x_m": round(self.x_m, 1), "y_m": round(self.y_m, 1),
                "role": self.role}


def haversine_m(a: Node, b: Node) -> float:
    d_lat = math.radians(b.lat - a.lat)
    d_lon = math.radians(b.longitude - a.longitude)
    s = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(a.lat)) * math.cos(math.radians(b.lat))
         * math.sin(d_lon / 2) ** 2)
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(s))


def bearing_deg(a: Node, b: Node) -> float:
    """Initial great-circle bearing from a to b, clockwise from north."""
    d_lon = math.radians(b.longitude - a.longitude)
    y = math.sin(d_lon) * math.cos(math.radians(b.lat))
    x = (math.cos(math.radians(a.lat)) * math.sin(math.radians(b.lat))
         - math.sin(math.radians(a.lat)) * math.cos(math.radians(b.lat))
         * math.cos(d_lon))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _offset_to_latlon(lat0: float, lon0: float, dx: float, dy: float
                      ) -> tuple[float, float]:
    """Local metres to degrees. Fine for a 2 km lattice."""
    lat = lat0 + dy / EARTH_RADIUS_M * 180.0 / math.pi
    lon = lon0 + dx / (EARTH_RADIUS_M * math.cos(math.radians(lat0))) * 180.0 / math.pi
    return lat, lon


def build_lattice(site: Site) -> list[Node]:
    """A square lattice centred on the site, rotated to the prevailing axis.

    Every `holdout_pattern`-th site is left uninstrumented. Those nodes are
    never used to fit the calibration model, which is what makes the transfer
    result in results/validation.json an honest generalisation number rather
    than an in-sample one.
    """
    per_side = site.cell_count_per_side
    step = site.spacing_m
    half = (per_side - 1) * step / 2.0
    theta = math.radians(site.rotation_deg)

    nodes: list[Node] = []
    index = 0
    for row in range(per_side):
        for col in range(per_side):
            ex = col * step - half
            ey = row * step - half
            x = ex * math.cos(theta) - ey * math.sin(theta)
            y = ex * math.sin(theta) + ey * math.cos(theta)
            lat, lon = _offset_to_latlon(site.latitude, site.longitude, x, y)
            holdout = (site.holdout_pattern == "every_5th" and index % 5 == 4)
            role = "holdout" if holdout else "instrumented"
            nodes.append(Node(id=f"{site.id}-{index:02d}", lat=lat, longitude=lon,
                              x_m=x, y_m=y, role=role))
            index += 1

    # The caller asked for a specific instrumented count; demote from the
    # corners inward so the surviving lattice stays spatially compact.
    wanted = site.instrumented_nodes
    current = sum(1 for n in nodes if n.role == "instrumented")
    if current < wanted:
        raise ValueError(
            f"lattice yields {current} instrumented sites but {wanted} requested"
        )
    excess = current - wanted
    if excess:
        by_distance = sorted(
            (i for i, n in enumerate(nodes) if n.role == "instrumented"),
            key=lambda i: -(nodes[i].x_m ** 2 + nodes[i].y_m ** 2),
        )
        for i in by_distance[:excess]:
            nodes[i] = dataclasses.replace(nodes[i], role="holdout")
    return nodes


def instrumented(nodes: Sequence[Node]) -> list[Node]:
    return [n for n in nodes if n.role == "instrumented"]


def holdout(nodes: Sequence[Node]) -> list[Node]:
    return [n for n in nodes if n.role == "holdout"]


def nearest_neighbours(node: Node, pool: Iterable[Node], k: int = 4,
                       max_distance_m: float = 900.0) -> list[tuple[Node, float]]:
    out = [(n, haversine_m(node, n)) for n in pool if n.id != node.id]
    out = [pair for pair in out if pair[1] <= max_distance_m]
    out.sort(key=lambda pair: pair[1])
    return out[:k]


def upwind_score(node: Node, other: Node, wind_from_deg: float) -> float:
    """How directly `other` lies upwind of `node`, in [0, 1].

    1.0 means the air arriving at node has just passed over other. 0.0 means
    other is downwind and can contribute nothing. Computed as cos^2 of the
    angle between the node->other bearing and the incoming wind direction,
    which is smooth enough for a regression feature and zero on the flank.
    """
    if node.id == other.id:
        return 0.0
    bearing = bearing_deg(node, other)
    delta = math.radians(bearing - wind_from_deg)
    return max(0.0, math.cos(delta)) ** 2


def adjacency(nodes: Sequence[Node], wind_from_deg: float, k: int = 4,
              max_distance_m: float = 900.0) -> dict[str, list[tuple[str, float]]]:
    """Directed, wind-weighted adjacency for one hour.

    Edge weight is distance-decayed upwind influence, so the graph is a
    different object every hour. That is the point: a static k-nearest graph
    encodes the street layout, not the transport.
    """
    pool = list(nodes)
    out: dict[str, list[tuple[str, float]]] = {}
    for node in pool:
        candidates = []
        for other, dist in nearest_neighbours(node, pool, k=k * 3,
                                              max_distance_m=max_distance_m):
            score = upwind_score(node, other, wind_from_deg)
            if score <= 1e-9:
                continue
            decay = 1.0 / (1.0 + dist / max_distance_m)
            candidates.append((other.id, float(score * decay)))
        candidates.sort(key=lambda item: -item[1])
        out[node.id] = candidates[:k]
    return out


def graph_density(graph: dict[str, list[tuple[str, float]]]) -> dict[str, float]:
    if not graph:
        return {"nodes": 0, "mean_edges": 0.0, "isolated_fraction": 1.0}
    edges = [len(v) for v in graph.values()]
    return {
        "nodes": len(graph),
        "mean_edges": round(float(np.mean(edges)), 2),
        "isolated_fraction": round(float(np.mean([e == 0 for e in edges])), 3),
    }
