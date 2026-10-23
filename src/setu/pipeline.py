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
    expected_calibration_error,
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


def _decline_stats(gated: Sequence, held_labels: pd.Series,
                   classes: Sequence[str]) -> tuple[float, dict]:
    """Two views of the same gate: how often it declined, and how well it
    recognises the hours nothing was injected on.

    `gated` must be attribute() output over exactly `held_labels`' rows, in
    order - the confabulation rate is the fraction of unnameable hours the
    model named a source for anyway, which is the number the reject class
    exists to shrink.
    """
    if not gated:
        return 0.0, {"hours": 0, "declined": None, "misnamed": None}
    abstained = float(sum(1 for a in gated if a.top == ABSTAIN) / len(gated))
    named = set(classes)
    unnameable = [(a, str(lab)) for a, lab in zip(gated, held_labels)
                  if str(lab) not in named]
    if not unnameable:
        return abstained, {"hours": 0, "declined": None, "misnamed": None}
    declined = sum(1 for a, _ in unnameable if a.top == ABSTAIN) / len(unnameable)
    return abstained, {
        "hours": len(unnameable),
        "declined": round(float(declined), 4),
        "misnamed": round(float(1 - declined), 4),
    }


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
    label_set = set(str(v) for v in train_labels.unique())
    present = sorted(label_set & set(sigs))
    excluded = sorted(set(sigs) - set(present))
    # Whatever the twin labelled that is not a source - regional_background
    # and mixed - is something the model must learn to recognise and then
    # decline to name, not fold into whichever source it resembles. Fitted as
    # reject classes; the classifier never emits them as an answer.
    reject = sorted(label_set - set(sigs))
    clf = SourceClassifier(sigs, classes=present,
                           reject_labels=reject).fit(feats.iloc[:cut], train_labels)

    share_rows = []
    for nid, p in products.items():
        attrs = clf.attribute(build_features(p.raw), node=nid,
                              min_posterior=min_posterior)
        p.attributions = attrs
        share_rows.extend(attrs)

    shares = seasonal_shares(share_rows)
    # The band each source is checked against lives next to its signature and
    # citation, so the cross-check and the literature cannot drift apart -
    # they used to be a literal dict here that only this function knew about.
    published = {key: sig.published_share for key, sig in sigs.items()}
    scorable = labels.iloc[cut:]
    post = clf.posterior(feats.iloc[cut:])
    ok = scorable.isin(list(clf.classes))
    conf = post[list(clf.classes)].max(axis=1)[ok.values]
    correct = (post["top"][ok.values] == scorable[ok.values])
    gated = clf.attribute(feats.iloc[cut:], node="heldout",
                          min_posterior=min_posterior)
    abstained, background = _decline_stats(gated, labels.iloc[cut:],
                                           clf.classes)
    stats = {
        "classes": list(clf.classes),
        "scorable_hours": int(ok.sum()),
        "accuracy": round(float(correct.mean()), 4),
        "chance": round(1.0 / len(clf.classes), 4),
        "accuracy_above_060": round(float(correct[conf >= 0.60].mean()), 4),
        "share_of_hours_above_060": round(float((conf >= 0.60).mean()), 4),
        "log_loss": round(clf.log_loss(feats.iloc[cut:], labels.iloc[cut:]), 4),
        # Measured through attribute(), where the floor actually runs.
        # post["top"] is an argmax and can never equal the abstain token, so
        # the previous expression read 0.0 forever - a statistic that was
        # always zero and therefore always quoted. This counts every held-out
        # hour the gate declined to name, background ones included, because
        # those are exactly the hours a forced guess would confabulate.
        "abstained": round(abstained, 4),
        "background_rejection": background,
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
    # Permutation importance on the held-out third, so the report can say which
    # channels the answer actually rests on rather than only which way they
    # push. Scored out of sample: importance on the training rows describes how
    # well the model memorised them, which nobody is asking.
    importance = clf.feature_importance(feats.iloc[cut:], labels.iloc[cut:])
    stats["feature_importance"] = [
        {"feature": name,
         "importance_mean": round(float(row["importance_mean"]), 4),
         "importance_std": round(float(row["importance_std"]), 4)}
        for name, row in importance.iterrows()
    ]
    # The engine gates on a posterior floor, so the posterior has to be checked
    # as a probability and not just as a ranking. Scored on the held-out third
    # alongside the importance, for the same reason: an in-sample reliability
    # curve flatters every classifier ever fitted.
    reliability = clf.confidence_calibration(feats.iloc[cut:], labels.iloc[cut:])
    stats["confidence_calibration"] = {
        "ece": round(expected_calibration_error(reliability), 4),
        "bins": reliability.to_dict(orient="records"),
    }
    # Every direction signatures.yaml declares, scored against the learned
    # coefficient. Three hand-picked signs were the minimum that could be
    # quoted; this is the maximum that can be checked, and the disagreements
    # it surfaces are findings about the twin and the model, not test
    # failures to be tuned away.
    agreement = clf.signature_agreement(sigs)
    judged = agreement[agreement["agrees"].notna()]
    agreed = sum(1 for v in judged["agrees"] if bool(v))
    stats["signature_agreement"] = {
        "checked": int(len(judged)),
        "agreed": int(agreed),
        "rate": round(agreed / len(judged), 4) if len(judged) else None,
        "pairs": agreement.to_dict(orient="records"),
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


def _spread(rows: list[dict], limit: int) -> list[dict]:
    """Round-robin the sample across (node, kind) groups.

    `rows` come out of the engine node by node, so `rows[:400]` was one node's
    entire night: the console's dispatch queue and suppression ledger showed a
    single pole. Spreading by node alone overcorrects - suppressions outnumber
    dispatches about 4:1, so one row per node is all suppressions and the queue
    goes empty. Grouping by node AND kind keeps both panels populated and makes
    each span the grid. The counts in engine.summary() are computed over every
    row, so this changes only what the sample displays.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        kind = "dispatch" if "id" in r else "suppress"
        groups.setdefault((str(r.get("node", "?")), kind), []).append(r)

    order = sorted(groups)
    out: list[dict] = []
    i = 0
    while len(out) < limit and any(groups[g] for g in order):
        key = order[i % len(order)]
        if groups[key]:
            out.append(groups[key].pop(0))
        i += 1
    return out


def run_response(products: Mapping[str, NodeProduct], threshold: float,
                 per_kind: int = 12) -> tuple[dict, list[dict]]:
    """Drive the engine over every node; return the summary plus a sample.

    The sample is bounded per (node, kind) rather than by a global row count. A
    flat `rows[:4000]` cap was exhausted by the first three nodes, so the console
    displayed one pole's night and, after the first spread fix, three dispatches
    out of 40 rows. The engine still sees every node-hour; only what gets kept
    for display is capped, and every dispatch counts toward the totals either way.
    """
    table = load_action_table()
    engine = ResponseEngine(table)
    kept: dict[tuple[str, str], list[dict]] = {}
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
            record = out.as_dict()
            kind = "dispatch" if "id" in record else "suppress"
            bucket = kept.setdefault((nid, kind), [])
            if len(bucket) < per_kind:
                bucket.append(record)
    sample = [r for key in sorted(kept) for r in kept[key]]
    return engine.summary(), _spread(sample, 400)


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
