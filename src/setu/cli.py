"""Command line entry point.

`python -m setu check` is the one to try first: it runs the whole pipeline on a
short window and prints the headline numbers, so a reviewer can see whether the
repo works before deciding whether to read it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_action_table, load_signatures, load_site
from .errors import SetuError
from .grid import build_lattice, holdout, instrumented
from .io.cache import DEFAULT_CACHE_DIR, HttpCache
from .io import cpcb as cpcb_mod
from .pipeline import run as run_pipeline
from .report import RESULTS_DIR, evidence_pack, to_json, to_markdown


def _cmd_fetch(args: argparse.Namespace) -> int:
    from .io import openmeteo as om

    site = load_site(args.site)
    cache = HttpCache()
    df, prov = om.load_hourly(site, cache, past_days=args.days, forecast_days=0)
    print(f"{len(df)} hourly rows for {site.label}")
    print(f"window {df.index[0]} -> {df.index[-1]}")
    print(f"dropped {df.attrs.get('dropped_incomplete_hours', 0)} rows with no met")
    for rec in prov.as_list():
        print(f"  {rec['label']}: {'cache' if rec['from_cache'] else 'live'}")
    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out)
        print(f"wrote {out}")
    return 0


def _cmd_grid(args: argparse.Namespace) -> int:
    site = load_site(args.site)
    nodes = build_lattice(site)
    print(f"{len(nodes)} lattice sites: {len(instrumented(nodes))} instrumented, "
          f"{len(holdout(nodes))} held out")
    for node in nodes[: args.show]:
        print(f"  {node.id}  {node.lat:.5f},{node.longitude:.5f}  "
              f"x={node.x_m:7.1f} y={node.y_m:7.1f}  {node.role}")
    return 0


def _cmd_probe_upstream(args: argparse.Namespace) -> int:
    site = load_site(args.site)
    report = cpcb_mod.reachability_report(HttpCache(), site=site)
    print(json.dumps(report, indent=2))
    return 0 if report["reachable"] else 1


def _cmd_run(args: argparse.Namespace) -> int:
    result = run_pipeline(args.site, past_days=args.days,
                          node_limit=args.nodes, verbose=True,
                          threshold=args.threshold)
    json_path = to_json(result)
    md_path = to_markdown(result)
    # sample_actions mixes dispatches with suppressions; only a dispatch has an
    # evidence pack. Suppressions are in validation.json under the ledger.
    dispatches = [a for a in result.response.get("sample_actions", [])
                  if "id" in a]
    for action in dispatches[: args.evidence]:
        evidence_pack(action, RESULTS_DIR)
    print()
    print(f"validation json -> {json_path}")
    print(f"validation md   -> {md_path}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    result = run_pipeline(args.site, past_days=args.days,
                          node_limit=args.nodes or 8, verbose=False,
                          threshold=args.threshold)
    a = result.attribution
    c = result.calibration
    r = result.response

    def mean(rows, dotted):
        vals = []
        node = rows
        for row in node:
            cur = row
            for key in dotted.split("."):
                cur = cur.get(key) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"site            {result.summary['site']}")
    print(f"threshold       {result.summary['threshold_ugm3']:.1f} ug/m3 "
          f"(background mean {result.summary['background_mean_pm25']}, "
          f"max {result.summary['background_max_pm25']})")
    print(f"background      {result.summary['hours']} real hours, "
          f"{result.summary['nodes']} nodes")
    print(f"calibration     MAE {mean(c, 'before.mae'):.2f} -> "
          f"{mean(c, 'after.mae'):.2f} ug/m3, "
          f"R^2 {mean(c, 'before.r2'):.2f} -> {mean(c, 'after.r2'):.2f}")
    print(f"interval cover  {mean(c, 'interval_coverage'):.3f} (nominal 0.80)")
    print(f"transfer        {result.transfer['nodes_transferred']} nodes, "
          f"MAE {result.transfer['mean_mae_before']:.2f} -> "
          f"{result.transfer['mean_mae_after']:.2f}")
    print(f"attribution     accuracy {a['accuracy']:.3f} "
          f"(chance {a['chance']:.2f}), log loss {a['log_loss']:.3f}")
    print(f"                above 0.60 posterior: {a['accuracy_above_060']:.3f} "
          f"on {a['share_of_hours_above_060']:.1%} of hours")
    for row in sorted(result.forecast, key=lambda x: x["horizon_h"]):
        print(f"forecast {row['horizon_h']:>2}h      RMSE {row['model_rmse']:.2f} "
              f"vs persistence {row['persistence_rmse']:.2f} "
              f"(skill {row['skill_vs_persistence']:+.3f})")
    print(f"response        {r['actions']} dispatches, {r['suppressed']} suppressed")
    print(f"drift           {result.drift['nodes_with_alarms']} nodes alarmed")
    # equal_cost mixes per-lever rows with bookkeeping keys (breach_hours,
    # coarse_action_ugm3, action_threshold_ugm3, note), so select the rows the
    # same way report.py does instead of assuming every value is a lever.
    gains = [f"{k}: {v['gain']:+.2f}"
             for k, v in result.equal_cost.items()
             if isinstance(v, dict) and "gain" in v]
    if gains:
        print("equal cost      " + ", ".join(gains))
    else:
        print(f"equal cost      {result.equal_cost.get('note', 'undefined in this window')}")
    failing = [row["source"] for row in result.shares if not row["within"]]
    if failing:
        print(f"share cross-check FAILS for {', '.join(failing)} - see "
              f"docs/LIMITATIONS.md, this is a known open item")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.server import build_app

    uvicorn.run(build_app(args.site), host=args.host, port=args.port,
                log_level="info")
    return 0


def _cmd_bom(args: argparse.Namespace) -> int:
    table = load_action_table()
    print("Levers available to the response engine:\n")
    for key, lever in table.levers.items():
        hl = "no measured decay" if lever.half_life_min is None else \
            f"{lever.half_life_min:.0f} min half-life"
        dose = "one-shot" if lever.dose_interval_min() is None else \
            f"re-dose every {lever.dose_interval_min():.0f} min"
        print(f"  {key:22s} {lever.label}")
        print(f"  {'':22s} {hl}, {dose}, owner={lever.owner}")
        print(f"  {'':22s} {lever.source}\n")
    print("Gates:", ", ".join(f"{k}={v}" for k, v in table.gates.__dict__.items()))
    print("\nRefusals (things the engine will not be extended to do):")
    for key in table.refusals:
        print(f"  - {key}")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    site = load_site(args.site)
    sigs, window = load_signatures()
    print(json.dumps({
        "site": site.__dict__ | {"naaqs": dict(site.naaqs)},
        "lattice_sites": site.candidate_sites,
        "sources": {k: {"prior": v.prior_weight,
                        "actionable": v.actionable_locally}
                    for k, v in sigs.items()},
        "baseline_window_hours": window,
        "cache": HttpCache().stats(),
    }, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="setu",
        description="SETU - source-attributed, evidence-graded trigger for "
                    "urban air quality")
    parser.add_argument("--site", default="rajkot", help="config name under configs/")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="pull the real background and report it")
    p.add_argument("--days", type=int, default=92)
    p.add_argument("--csv", help="also write the joined frame to this path")
    p.set_defaults(func=_cmd_fetch)

    p = sub.add_parser("grid", help="show the node lattice")
    p.add_argument("--show", type=int, default=12)
    p.set_defaults(func=_cmd_grid)

    p = sub.add_parser("probe-upstream",
                       help="report which data sources are reachable")
    p.set_defaults(func=_cmd_probe_upstream)

    p = sub.add_parser("run", help="full pipeline, writes results/")
    p.add_argument("--days", type=int, default=92)
    p.add_argument("--nodes", type=int)
    p.add_argument("--threshold", type=float,
                   help="override the action level in ug/m3 (default: NAAQS 24h)")
    p.add_argument("--evidence", type=int, default=10,
                   help="how many dispatches to write evidence packs for")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("check", help="fast end-to-end smoke run with headline numbers")
    p.add_argument("--days", type=int, default=45)
    p.add_argument("--nodes", type=int, default=8)
    p.add_argument("--threshold", type=float, default=15.0,
                   help="action level in ug/m3. Defaults to the WHO 24-hour "
                        "guideline rather than the 60 ug/m3 NAAQS limit: in a "
                        "monsoon window Rajkot never approaches the latter, so "
                        "the response layer would legitimately do nothing")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("serve", help="run the operator API and dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090,
                   help="8090, not 3000: that port is taken by Grafana on the "
                        "development machine")
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("bom", help="print the lever table and trigger gates")
    p.set_defaults(func=_cmd_bom)

    p = sub.add_parser("config", help="dump resolved configuration")
    p.set_defaults(func=_cmd_config)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SetuError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
