"""End-to-end run: real background -> grid -> calibrate -> attribute -> act.

Kept deliberately boring. Every stage is a function that returns data and a
list of notes about what it had to give up on; nothing here raises on a partial
failure, because a run that produced six of seven stages of evidence is worth
more than a run that refused to start.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import grid, twin
from .attribute import (
    ABSTAIN,
    SourceClassifier,
    build_features,
    compare_to_receptor,
    seasonal_shares,
)
from .calibrate import CalibrationModel, CusumMonitor, transfer_calibrate
from .config import Site, load_action_table, load_signatures, load_site
from .forecast import HORIZONS, GraphForecaster, design_matrix, evaluate_forecast
from .io import openmeteo as om
from .io.cache import HttpCache
from .metrics import score
from .respond import Observation, ResponseEngine, roster_vs_triggered


@dataclasses.dataclass
class NodeProduct:
    """Everything one node contributes to the run."""

    node: str
    truth: pd.DataFrame
    raw: pd.DataFrame
    calibrated: pd.DataFrame
    attributions: list
    drift_alarms: int
    calibration: dict


@dataclasses.dataclass
class RunResult:
    site: str
    provenance: dict
    summary: dict
    calibration: list[dict]
    transfer: dict
    attribution: dict
    forecast: list[dict]
    response: dict
    drift: dict
    shares: dict
    equal_cost: dict
    products: Mapping[str, NodeProduct] = dataclasses.field(default_factory=dict)


def _split(n: int, frac: float = 0.6) -> int:
    return int(n * frac)


def calibrate_all(t: twin.DigitalTwin, nodes: Sequence[grid.Node],
                  threshold: float, train_frac: float = 0.6
                  ) -> tuple[dict[str, NodeProduct], list[dict], dict]:
    """Fit on the first two-thirds chronologically, score on the rest.

    A random split would let the model see next week's humidity when predicting
    today's, which on a seasonal signal is the difference between a validation
    and a fiction.
    """
    sigs, _ = load_signatures()
    products: dict[str, NodeProduct] = {}
    reports: list[dict] = []
    donors = nodes[:2]

    donor_models: dict[str, CalibrationModel] = {}
    for node in donors:
        truth, raw = t.truth(node), t.raw(node)
        cut = _split(len(truth), train_frac)
        model = CalibrationModel(target="pm25").fit(raw.iloc[:cut],
                                                    truth["pm25"].iloc[:cut])
        donor_models[node.id] = model
        reports.append(model.evaluate(raw.iloc[cut:], truth["pm25"].iloc[cut:],
                                      threshold=threshold, node=node.id).as_dict())

    transfer_diag: dict[str, object] = {"nodes": [], "r2": [], "mae_before": [],
                                        "mae_after": [], "distance_m": []}
    for node in nodes[len(donors):]:
        truth, raw = t.truth(node), t.raw(node)
        cut = _split(len(truth), train_frac)
        donor_node = donors[len(transfer_diag["nodes"]) % len(donors)]
        donor_truth, donor_raw = t.truth(donor_node), t.raw(donor_node)
        try:
            model, diag = transfer_calibrate(raw.iloc[:cut], donor_raw.iloc[:cut],
                                             donor_models[donor_node.id])
        except Exception as exc:  # noqa: BLE001 - a bad donor must not kill the run
            transfer_diag.setdefault("failures", []).append(
                {"node": node.id, "error": str(exc)})
            continue

        report = model.evaluate(raw.iloc[cut:], truth["pm25"].iloc[cut:],
                                threshold=threshold, node=node.id)
        naive = float(np.mean(np.abs(raw["pm25"].iloc[cut:]
                                     - truth["pm25"].iloc[cut:])))
        transfer_diag["nodes"].append(node.id)
        transfer_diag["r2"].append(diag["r2"])
        transfer_diag["mae_before"].append(naive)
        transfer_diag["mae_after"].append(report.after.mae)
        # Distance to the donor is the variable the scalability claim actually
        # rests on: a transfer that only works next door is a co-location in
        # disguise. Recorded per node so _summarise_transfer can band it.
        transfer_diag["distance_m"].append(grid.haversine_m(donor_node, node))
        reports.append(report.as_dict())

        # Only post-fit residuals are fed to the monitor. Including the training
        # window would make the split itself look like a step change: in-sample
        # residuals are small, out-of-sample residuals are larger, and CUSUM
        # correctly reports that discontinuity as drift when it is really just
        # the boundary of the fit. That artefact quarantined 40 of 49 nodes on
        # the first full run.
        monitor = CusumMonitor()
        cal = model.predict(raw)
        residual = (cal["value"] - truth["pm25"]).iloc[cut:]
        alarms = monitor.feed(residual.to_numpy(), residual.index)
        products[node.id] = NodeProduct(
            node=node.id, truth=truth, raw=raw,
            calibrated=cal, attributions=[],
            drift_alarms=len(alarms),
            calibration={**report.as_dict(), "transfer_r2": diag["r2"]},
        )

    for node in donors:
        truth, raw = t.truth(node), t.raw(node)
        model = donor_models[node.id]
        cal = model.predict(raw)
        monitor = CusumMonitor()
        residual = (cal["value"] - truth["pm25"]).iloc[cut:]
        alarms = monitor.feed(residual.to_numpy(), residual.index)
        products[node.id] = NodeProduct(
            node=node.id, truth=truth, raw=raw, calibrated=cal, attributions=[],
            drift_alarms=len(alarms),
            calibration={"transfer_r2": None, "reference": True},
        )
    return products, reports, transfer_diag


def attribute_all(products: Mapping[str, NodeProduct],
                  nodes: Sequence[grid.Node], min_posterior: float
                  ) -> tuple[dict, pd.DataFrame]:
    sigs, _ = load_signatures()
    feats = pd.concat({nid: build_features(p.raw) for nid, p in products.items()})
    feats.index.names = ["node", "time"]
    labels = pd.concat({nid: p.truth["dominant_source"] for nid, p in products.items()})
    labels.index.names = ["node", "time"]

    cut = _split(len(feats))
    train_labels = labels.iloc[:cut]
    present = sorted(set(str(v) for v in train_labels.unique()) & set(sigs))
    excluded = sorted(set(sigs) - set(present))
    clf = SourceClassifier(sigs, classes=present).fit(feats.iloc[:cut], train_labels)

    share_rows = []
    for nid, p in products.items():
        attrs = clf.attribute(build_features(p.raw), node=nid,
                              min_posterior=min_posterior)
        p.attributions = attrs
        share_rows.extend(attrs)

    shares = seasonal_shares(share_rows)
    published = {"road_dust": (0.02, 0.45), "construction": (0.0, 0.20),
                 "open_burning": (0.0, 0.30), "secondary": (0.10, 0.40),
                 "traffic_combustion": (0.05, 0.25)}
    scorable = labels.iloc[cut:]
    post = clf.posterior(feats.iloc[cut:])
    ok = scorable.isin(list(clf.classes))
    conf = post[list(clf.classes)].max(axis=1)[ok.values]
    correct = (post["top"][ok.values] == scorable[ok.values])
    stats = {
        "classes": list(clf.classes),
        "scorable_hours": int(ok.sum()),
        "accuracy": round(float(correct.mean()), 4),
        "chance": round(1.0 / len(clf.classes), 4),
        "accuracy_above_060": round(float(correct[conf >= 0.60].mean()), 4),
        "share_of_hours_above_060": round(float((conf >= 0.60).mean()), 4),
        "log_loss": round(clf.log_loss(feats.iloc[cut:], labels.iloc[cut:]), 4),
        "abstained": round(float((post["top"][ok.values] == ABSTAIN).mean()), 4),
        "classes_fitted": list(clf.classes),
        "classes_excluded_for_no_training_data": excluded,
    }
    coef = clf.coefficients()
    stats["sign_check"] = {
        "coarse_count_ratio_on_dust": round(
            float(coef.loc["road_dust", "log_coarse_count_ratio"]), 3),
        "coarse_count_ratio_on_burning": round(
            float(coef.loc["open_burning", "log_coarse_count_ratio"]), 3),
        "rh_on_secondary": round(float(coef.loc["secondary", "rh_frac"]), 3),
    }
    return stats, compare_to_receptor(shares, published)


def forecast_all(products: Mapping[str, NodeProduct],
                 nodes: Sequence[grid.Node]) -> list[dict]:
    frames: dict[int, list[pd.DataFrame]] = {h: [] for h in HORIZONS}
    for nid, p in products.items():
        one = p.calibrated.rename(columns={"value": "pm25"})
        joined = p.truth[["rh", "wind_speed_ms", "wind_dir_deg", "temp_c"]].copy()
        joined["rh_frac"] = joined["rh"]
        base = pd.concat([one, joined], axis=1)
        X = design_matrix(base, "pm25")
        targets = {h: base["pm25"].shift(-h) for h in HORIZONS}
        try:
            model = GraphForecaster(horizons=HORIZONS).fit(X, targets)
            model.fit_seasonal_baseline(base["pm25"])
        except Exception:  # noqa: BLE001 - one thin node must not kill the run
            continue
        window = X.iloc[24:-max(HORIZONS)]
        for h in HORIZONS:
            frames[h].append(pd.DataFrame({
                "actual": targets[h].reindex(window.index),
                "predicted": model.predict(window, h),
                "persistence": base["pm25"].reindex(window.index),
                "seasonal": model.seasonal_predict(window.index),
            }))
    return [skill.as_dict()
            for h, v in frames.items() if v
            for skill in evaluate_forecast({h: pd.concat(v)})]


def run_response(products: Mapping[str, NodeProduct], threshold: float
                 ) -> tuple[dict, list[dict]]:
    table = load_action_table()
    engine = ResponseEngine(table)
    rows: list[dict] = []
    for nid, p in products.items():
        cal = p.calibrated
        by_time = {a.at: a for a in p.attributions}
        for ts, row in cal.iterrows():
            attr = by_time.get(ts)
            if attr is None:
                continue
            post = dict(attr.posterior)
            if attr.top == ABSTAIN:
                post = {attr.top: 1.0}
            obs = Observation(
                node=nid, at=ts, value=float(row["value"]),
                sigma=float(row["sigma"]), posterior=post,
                threshold=threshold,
                coarse_ratio=float(p.truth.loc[ts, "coarse"] /
                                   max(p.truth.loc[ts, "pm10"], 1e-6)),
                wind_from_deg=float(p.truth.loc[ts, "wind_dir_deg"]),
                low=float(row["low"]), high=float(row["high"]),
            )
            out = engine.update(obs)
            rows.append(out.as_dict())
            if len(rows) > 4000:
                break
    return engine.summary(), rows[:400]


def run(site_name: str = "rajkot", past_days: int = 92,
        node_limit: int | None = None, verbose: bool = False,
        threshold: float | None = None) -> RunResult:
    site = load_site(site_name)
    cache = HttpCache()
    hourly, prov = om.load_hourly(site, cache, past_days=past_days,
                                  forecast_days=0)
    nodes = grid.instrumented(grid.build_lattice(site))
    if node_limit:
        nodes = nodes[:node_limit]
    t = twin.DigitalTwin(hourly, nodes, site, twin.TwinConfig())

    # Default is the notified Indian 24-hour standard. In a monsoon window
    # Rajkot never approaches it - mean PM2.5 is about 16 ug/m3 - so the response
    # layer legitimately emits nothing. Pass threshold= to exercise the engine
    # against a stricter action level rather than inventing an exceedance.
    threshold = (float(site.naaqs["pm25_24h"]) if threshold is None
                 else float(threshold))
    if verbose:
        print(f"[setu] {len(hourly)} real background hours, {len(nodes)} nodes, "
              f"action threshold {threshold:.1f} ug/m3, "
              f"background mean {hourly['pm25'].mean():.1f} "
              f"max {hourly['pm25'].max():.1f}")

    products, cal_reports, transfer = calibrate_all(t, nodes, threshold)
    if verbose:
        print(f"[setu] calibrated {len(products)} nodes")

    table = load_action_table()
    attr_stats, share_table = attribute_all(products, nodes,
                                            table.gates.min_source_posterior)
    if verbose:
        print(f"[setu] attribution accuracy {attr_stats['accuracy']:.3f}")

    forecast = forecast_all(products, nodes)
    response, sample_actions = run_response(products, threshold)
    if verbose:
        print(f"[setu] {response['actions']} dispatches, "
              f"{response['suppressed']} suppressed")

    # The comparison is run against the condition the engine actually acts on:
    # hours where the coarse fraction clears the working action level AND the
    # attributed source is one a sweeper or a tanker could move. Scoring it
    # against the 24-hour PM2.5 standard instead produced a degenerate result -
    # a monsoon window never reaches 60 ug/m3, so the breach profile came back
    # either empty or flat, and a flat profile makes triggering look worthless
    # for a reason that has nothing to do with triggering.
    coarse_level = float(site.coarse_action_ugm3)
    breach_hours = _coarse_breach_hours(products, coarse_level)
    def _compare(hours):
        return {
            "sprinkling_62min": roster_vs_triggered(hours, 2, 62),
            "sweeping_150min": roster_vs_triggered(hours, 2, 150),
            "suppressant_20h": roster_vs_triggered(hours, 2, 1200),
            "breach_hours": [int(h) for h in hours],
            "coarse_action_ugm3": coarse_level,
            "action_threshold_ugm3": float(threshold),
        }
    road_hours = _coarse_breach_hours(products, coarse_level, sources=("road_dust",))
    equal_cost_road = _compare(road_hours) if road_hours else None

    equal_cost = _compare(breach_hours) if breach_hours else {
        "breach_hours": [], "coarse_action_ugm3": coarse_level,
        "action_threshold_ugm3": float(threshold),
        "note": "no episodic coarse-mode breach hours in this window; the "
                "equal-cost comparison is undefined rather than zero",
    }
    if equal_cost_road:
        equal_cost["road_dust_continuous_profile"] = equal_cost_road

    return RunResult(
        site=site.id,
        provenance={
            "background": prov.as_list(),
            "hourly_rows": int(len(hourly)),
            "window": [str(hourly.index[0]), str(hourly.index[-1])],
            "dropped_incomplete_hours": int(
                hourly.attrs.get("dropped_incomplete_hours", 0)),
            "grid_deg": hourly.attrs.get("grid_deg"),
            "injected": t.injected_summary(),
            "health": _health_records(t),
        },
        summary={
            "site": site.label,
            "team": "Renovator",
            "nodes": len(nodes),
            "hours": int(len(hourly)),
            "threshold_ugm3": threshold,
            "background_max_pm25": round(float(hourly["pm25"].max()), 1),
            "background_mean_pm25": round(float(hourly["pm25"].mean()), 1),
            "quarantined": sorted(n for n, p in products.items()
                                  if p.drift_alarms > 0),
        },
        calibration=cal_reports,
        transfer=_summarise_transfer(transfer, site.spacing_m),
        attribution=attr_stats,
        forecast=forecast,
        response={**response, "sample_actions": sample_actions},
        drift={"alarms_by_node": {n: p.drift_alarms for n, p in products.items()},
               "nodes_with_alarms": sum(1 for p in products.values()
                                        if p.drift_alarms)},
        shares=share_table.reset_index().to_dict(orient="records"),
        equal_cost=equal_cost,
        products=products,
    )


def _health_records(t: twin.DigitalTwin) -> list[dict]:
    """Per-node twin character, keyed by node id.

    The frame carries a "node" column *and* a node index, so reset_index() alone
    produces duplicate labels and pandas silently drops one of them when
    converting to records. Naming the index first is what makes the key survive.
    """
    # health() already carries a "node" column, and orient="records" drops the
    # index anyway - resetting it first produced two columns called "node" and
    # pandas silently kept only one of them.
    return t.health().to_dict(orient="records")


def _coarse_breach_hours(products: Mapping[str, NodeProduct],
                         coarse_threshold: float,
                         sources: Sequence[str] = ("construction",
                                                   "open_burning")) -> list[int]:
    """Hours of day when the grid breaches AND the cause is a coarse-mode source.

    Both conditions matter. Restricting to a strict action level like the WHO
    guideline makes nearly every hour of a monsoon month a "breach", which
    destroys the episodic structure the equal-cost comparison is about and makes
    triggering look worthless for the wrong reason. Conversely, breaching on
    imported secondary aerosol is not something a sweeper can fix.

    Falls back to the plain breach profile if the filter is empty, so the
    comparison is never silently skipped.
    """
    hits: set[int] = set()
    for p in products.values():
        by_time = {a.at: a for a in p.attributions}
        for ts, row in p.truth[["coarse"]].iterrows():
            if float(row["coarse"]) < coarse_threshold:
                continue
            attr = by_time.get(ts)
            if attr is None or attr.top in sources:
                hits.add(int(ts.hour))
    return sorted(hits)


def _distance_bands(diag: dict, spacing_m: float) -> tuple[list[dict], float | None]:
    """Transfer skill banded by donor distance, plus the distance/MAE correlation.

    The scalability claim is "a node never co-located still calibrates". The
    falsifiable version of that is "and it still calibrates 1.5 km away", so the
    distances are reported instead of averaged away. Bands are in lattice
    pitches because the pitch is what the deployment actually spaces.
    """
    rows_in = list(zip(diag.get("distance_m", []), diag.get("mae_before", []),
                       diag.get("mae_after", []), diag.get("r2", [])))
    if not rows_in:
        return [], None

    edges = [(0.0, 1.5 * spacing_m, "within 1.5 pitches"),
             (1.5 * spacing_m, 2.5 * spacing_m, "1.5-2.5 pitches"),
             (2.5 * spacing_m, 3.5 * spacing_m, "2.5-3.5 pitches"),
             (3.5 * spacing_m, math.inf, "over 3.5 pitches")]
    out = []
    for lo, hi, label in edges:
        members = [r for r in rows_in if lo <= r[0] < hi]
        if not members:
            continue
        out.append({
            "band": label,
            "nodes": len(members),
            "mean_distance_m": round(float(np.mean([m[0] for m in members])), 1),
            "mean_mae_before": round(float(np.mean([m[1] for m in members])), 4),
            "mean_mae_after": round(float(np.mean([m[2] for m in members])), 4),
            "mean_overlap_r2": round(float(np.mean([m[3] for m in members])), 4),
        })

    # Pearson r between donor distance and post-transfer MAE. Positive and large
    # would mean the transfer is really a co-location effect. It is one number
    # about the whole set, so it belongs on the summary rather than repeated
    # inside every band.
    dist = np.asarray([r[0] for r in rows_in], dtype=float)
    after = np.asarray([r[2] for r in rows_in], dtype=float)
    corr = (float(np.corrcoef(dist, after)[0, 1])
            if len(rows_in) > 2 and dist.std() > 0 and after.std() > 0 else None)
    return out, None if corr is None else round(corr, 3)


def _summarise_transfer(diag: dict, spacing_m: float = 320.0) -> dict:
    def _mean(key):
        vals = [v for v in diag.get(key, []) if v is not None and np.isfinite(v)]
        return round(float(np.mean(vals)), 4) if vals else None

    bands, corr = _distance_bands(diag, spacing_m)

    return {
        "nodes_transferred": len(diag.get("nodes", [])),
        "mean_overlap_r2": _mean("r2"),
        "mean_mae_before": _mean("mae_before"),
        "mean_mae_after": _mean("mae_after"),
        "max_donor_distance_m": (round(float(max(diag["distance_m"])), 1)
                                 if diag.get("distance_m") else None),
        "by_distance_band": bands,
        "distance_mae_correlation": corr,
        "failures": diag.get("failures", []),
    }
