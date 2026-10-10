"""Typed-ish access to the YAML files under configs/.

The loader validates the handful of invariants the rest of the code relies on
and refuses to start otherwise. Silently defaulting a missing threshold is how
a monitoring system ends up reporting an all-clear it never measured.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigError

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _read(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise ConfigError(f"missing config file {path}")
    with path.open("r", encoding="utf-8") as fh:
        try:
            doc = yaml.safe_load(fh)
        except yaml.YAMLError as exc:  # pragma: no cover - exercised by hand
            raise ConfigError(f"{path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return doc


def _require(doc: Mapping[str, Any], dotted: str, where: str) -> Any:
    node: Any = doc
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise ConfigError(f"{where}: missing required key {dotted!r}")
        node = node[part]
    return node


@dataclasses.dataclass(frozen=True)
class Site:
    """Everything city-specific. Swapping this is the whole adaptation story."""

    id: str
    label: str
    latitude: float
    longitude: float
    timezone: str
    extent_km: float
    spacing_m: float
    instrumented_nodes: int
    holdout_pattern: str
    rotation_deg: float
    naaqs: Mapping[str, float]
    who_2021: Mapping[str, float]
    coarse_action_ugm3: float
    cpcb_station: str
    cpcb_city: str
    cpcb_state: str
    node_report_seconds: int
    forecast_hours: tuple[int, ...]

    @property
    def cell_count_per_side(self) -> int:
        return int(round(self.extent_km * 1000 / self.spacing_m)) + 1

    @property
    def candidate_sites(self) -> int:
        return self.cell_count_per_side ** 2


def load_site(name: str = "rajkot") -> Site:
    doc = _read(f"{name}.yaml")
    where = f"{name}.yaml"

    lat = float(_require(doc, "site.latitude", where))
    lon = float(_require(doc, "site.longitude", where))
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise ConfigError(f"{where}: site coordinates out of range")

    extent_km = float(_require(doc, "pilot.extent_km", where))
    spacing_m = float(_require(doc, "pilot.spacing_m", where))
    if extent_km <= 0 or spacing_m <= 0:
        raise ConfigError(f"{where}: pilot extent and spacing must be positive")
    if spacing_m > 1000:
        # Above 1 km the grid is coarser than the 2.8x site-to-site spread it
        # exists to resolve, so the configuration is wrong even though every
        # individual number is plausible.
        raise ConfigError(
            f"{where}: spacing_m {spacing_m} is coarser than the gradient "
            "the grid is meant to resolve"
        )

    instrumented = int(_require(doc, "pilot.instrumented_nodes", where))
    candidates = int(round(extent_km * 1000 / spacing_m)) + 1
    if instrumented > candidates ** 2:
        raise ConfigError(
            f"{where}: asked for {instrumented} nodes but the lattice only "
            f"has {candidates ** 2} sites"
        )

    cadence = _require(doc, "cadence.node_report_seconds", where)
    if int(cadence) < 60:
        raise ConfigError(f"{where}: node_report_seconds below 60 is not a "
                          "cadence this prototype models")

    return Site(
        id=str(_require(doc, "site.id", where)),
        label=str(_require(doc, "site.label", where)),
        latitude=lat,
        longitude=lon,
        timezone=str(_require(doc, "site.timezone", where)),
        extent_km=extent_km,
        spacing_m=spacing_m,
        instrumented_nodes=instrumented,
        holdout_pattern=str(doc["pilot"].get("holdout_pattern", "none")),
        rotation_deg=float(doc["pilot"].get("rotation_deg", 0.0)),
        naaqs=dict(_require(doc, "thresholds.naaqs", where)),
        who_2021=dict(doc["thresholds"].get("who_2021", {})),
        coarse_action_ugm3=float(_require(doc, "thresholds.coarse_action_ugm3", where)),
        cpcb_station=str(_require(doc, "reference.cpcb.station", where)),
        cpcb_city=str(_require(doc, "reference.cpcb.city", where)),
        cpcb_state=str(_require(doc, "reference.cpcb.state", where)),
        node_report_seconds=int(cadence),
        forecast_hours=tuple(int(h) for h in _require(doc, "cadence.forecast_hours", where)),
    )


@dataclasses.dataclass(frozen=True)
class Signature:
    key: str
    label: str
    prior_weight: float
    actionable_locally: bool
    features: Mapping[str, tuple[str, float]]
    channels_simulated: tuple[str, ...]
    corroborates: str


def load_signatures() -> tuple[dict[str, Signature], int]:
    doc = _read("signatures.yaml")
    where = "signatures.yaml"
    out: dict[str, Signature] = {}
    total = 0.0
    for key, body in doc["sources"].items():
        feats = {
            name: (spec["direction"], float(spec["strength"]))
            for name, spec in body["features"].items()
        }
        weight = float(body["prior_weight"])
        total += weight
        out[key] = Signature(
            key=key,
            label=str(body["label"]),
            prior_weight=weight,
            actionable_locally=bool(body["actionable_locally"]),
            features=feats,
            channels_simulated=tuple(body.get("channels_simulated", ())),
            corroborates=str(body.get("corroborates", "")),
        )
    if not out:
        raise ConfigError(f"{where}: no sources defined")
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(
            f"{where}: prior weights sum to {total:.4f}, not 1.0 - the "
            "classifier would silently renormalise a mistake"
        )
    window = int(_require(doc, "baseline.window_hours", where))
    if window < 48:
        raise ConfigError(f"{where}: baseline window shorter than 48 h cannot "
                          "cover a diurnal cycle")
    return out, window


@dataclasses.dataclass(frozen=True)
class Lever:
    key: str
    label: str
    effect_pct: float | None
    half_life_min: float | None
    reentrainment_risk: str | None
    source: str
    owner: str
    unit_cost_inr: float

    @property
    def has_measured_effect(self) -> bool:
        return self.effect_pct is not None and self.half_life_min is not None

    def dose_interval_min(self, safety: float = 0.75) -> float | None:
        """Re-apply at a fraction of the half-life so coverage never dips.

        Returns None for levers with no measured decay, which the engine treats
        as "issue once and wait for a human", not "poll every hour".
        """
        if self.half_life_min is None:
            return None
        if not 0 < safety <= 1:
            raise ConfigError("safety must be in (0, 1]")
        return self.half_life_min * safety


@dataclasses.dataclass(frozen=True)
class Gates:
    min_source_posterior: float
    persistence_windows: int
    persistence_of: int
    uncertainty_must_exclude_threshold: bool
    min_dwell_minutes: float
    hysteresis_pct: float
    require_human_signoff: bool


@dataclasses.dataclass(frozen=True)
class ActionTable:
    levers: Mapping[str, Lever]
    gates: Gates
    policy: Mapping[str, tuple[str, ...]]
    refusals: Mapping[str, str]

    def levers_for(self, source: str) -> tuple[Lever, ...]:
        keys = self.policy.get(source, ())
        return tuple(self.levers[k] for k in keys if k in self.levers)


def load_action_table() -> ActionTable:
    doc = _read("action_table.yaml")
    where = "action_table.yaml"

    levers: dict[str, Lever] = {}
    for key, body in doc["levers"].items():
        eff = body.get("effect_pct")
        hl = body.get("half_life_min")
        levers[key] = Lever(
            key=key,
            label=str(body["label"]),
            effect_pct=None if eff is None else float(eff),
            half_life_min=None if hl is None else float(hl),
            reentrainment_risk=body.get("reentrainment_risk"),
            source=str(body.get("source", "")),
            owner=str(_require(body, "owner", f"{where}:levers.{key}")),
            unit_cost_inr=float(body.get("unit_cost_inr", 0)),
        )

    g = _require(doc, "gates", where)
    gates = Gates(
        min_source_posterior=float(g["min_source_posterior"]),
        persistence_windows=int(g["persistence_windows"]),
        persistence_of=int(g["persistence_of"]),
        uncertainty_must_exclude_threshold=bool(g["uncertainty_must_exclude_threshold"]),
        min_dwell_minutes=float(g["min_dwell_minutes"]),
        hysteresis_pct=float(g["hysteresis_pct"]),
        require_human_signoff=bool(g["require_human_signoff"]),
    )
    if gates.persistence_windows > gates.persistence_of:
        raise ConfigError(f"{where}: persistence k ({gates.persistence_windows}) "
                          "cannot exceed n")
    if not 0 < gates.min_source_posterior <= 1:
        raise ConfigError(f"{where}: min_source_posterior must be in (0, 1]")
    if gates.hysteresis_pct < 0:
        raise ConfigError(f"{where}: hysteresis_pct cannot be negative")

    policy = {
        src: tuple(keys)
        for src, keys in doc.get("policy", {}).items()
    }
    for src, keys in policy.items():
        for lever in keys:
            if lever not in levers:
                raise ConfigError(f"{where}: policy {src} references unknown "
                                  f"lever {lever}")

    return ActionTable(
        levers=levers,
        gates=gates,
        policy=policy,
        refusals=dict(doc.get("refusals", {})),
    )
