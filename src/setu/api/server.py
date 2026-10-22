"""FastAPI service for the operator console.

Endpoints are read-only. The one thing that must never happen from a browser is
dispatching a crew, which is why `requires_signoff` is carried all the way from
the config through to the payload rather than being asserted in a document.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import load_action_table, load_site
from ..grid import build_lattice, holdout, instrumented
from ..report import RESULTS_DIR, calibration_summary, standard_for

# server.py sits at <repo>/src/setu/api/server.py, so the repository root is
# four levels up - parents[2] resolves to src/ and the dashboard silently 404s.
DASHBOARD_DIR = Path(__file__).resolve().parents[3] / "dashboard"


class NoStoreStaticFiles(StaticFiles):
    """StaticFiles in the installed Starlette takes no `headers=` argument, so
    the cache directive is applied per response instead."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-store"
        return response


def _load_results(site: str) -> dict:
    path = RESULTS_DIR / "validation.json"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="no results yet - run `python -m setu run` first")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise HTTPException(status_code=500,
                            detail=f"results are corrupt: {exc}") from exc


def build_app(site_name: str = "rajkot") -> FastAPI:
    app = FastAPI(
        title="SETU operator console",
        version="0.1.0",
        description="Source-attributed, evidence-graded trigger for urban air "
                    "quality research and decision-support system. Rajkot, Gujarat.",
    )
    site = load_site(site_name)
    table = load_action_table()
    nodes = build_lattice(site)

    # ---------------------------------------------------------------- pages
    @app.get("/", include_in_schema=False)
    def index():
        target = DASHBOARD_DIR / "index.html"
        if not target.exists():
            raise HTTPException(status_code=404, detail="dashboard missing")
        return FileResponse(target, headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------ metadata
    @app.get("/api/site")
    def get_site():
        return {
            "id": site.id, "label": site.label,
            "latitude": site.latitude, "longitude": site.longitude,
            "naaqs": dict(site.naaqs), "who_2021": dict(site.who_2021),
            "coarse_action_ugm3": site.coarse_action_ugm3,
            "node_report_seconds": site.node_report_seconds,
            "sites": site.candidate_sites,
            "instrumented": len(instrumented(nodes)),
            "held_out": len(holdout(nodes)),
        }

    @app.get("/api/gates")
    def get_gates():
        return {
            "gates": table.gates.__dict__,
            "policy": {k: list(v) for k, v in table.policy.items()},
            "refusals": sorted(table.refusals),
            "levers": {
                k: {"label": v.label, "effect_pct": v.effect_pct,
                    "half_life_min": v.half_life_min, "owner": v.owner,
                    "unit_cost_inr": v.unit_cost_inr, "source": v.source}
                for k, v in table.levers.items()},
        }

    # ------------------------------------------------------------- grid view
    @app.get("/api/nodes")
    def get_nodes():
        results = _load_results(site_name)
        health = {row["node"]: row for row in results["provenance"]["health"]}
        cal = {row["node"]: row for row in results["calibration"]
               if "node" in row}
        alarms = results["drift"]["alarms_by_node"]
        out = []
        for node in nodes:
            h = health.get(node.id, {})
            out.append({
                "id": node.id, "lat": node.lat, "lon": node.longitude,
                "x_m": node.x_m, "y_m": node.y_m, "role": node.role,
                "road_factor": round(float(h.get("road_factor", 0)), 3),
                "near_construction": bool(h.get("has_construction", False)),
                "burning_prone": bool(h.get("burning_prone", False)),
                "dead": bool(h.get("dead", False)),
                "obstructed": bool(h.get("obstructed", False)),
                "drift_fraction": round(float(h.get("drift_fraction", 0)), 3),
                "drift_alarms": int(alarms.get(node.id, 0)),
                "quarantined": int(alarms.get(node.id, 0)) > 0,
                "calibration_mae": (cal.get(node.id, {})
                                        .get("after", {}).get("mae")),
                "interval_coverage": cal.get(node.id, {}).get("interval_coverage"),
            })
        return {"nodes": out, "provenance": "digital-twin"}

    # ------------------------------------------------------------- dispatches
    @app.get("/api/actions")
    def get_actions(limit: int = 60):
        results = _load_results(site_name)
        rows = [a for a in results.get("sample_actions", []) if "id" in a]
        return {
            "actions": rows[:max(1, min(limit, 200))],
            "ledger": [a for a in results.get("sample_actions", [])
                       if "id" not in a][:200],
            "totals": {k: v for k, v in results["response"].items()
                       if k not in {"sample_actions"}},
        }

    # ------------------------------------------------------------- evidence
    @app.get("/api/actions/{action_id}")
    def get_action(action_id: str):
        safe = "".join(c for c in action_id if c.isalnum())
        path = RESULTS_DIR / "evidence" / f"{safe}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="no such evidence pack")
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------- headline
    @app.get("/api/validation")
    def get_validation():
        results = _load_results(site_name)
        return {
            "summary": results["summary"],
            # The panel is titled with run-wide claims, so it gets run-wide
            # numbers. `example` is one node's row, labelled as such, for the
            # drill-down; it is not what the headline row quotes.
            "calibration": {
                "mean": calibration_summary(results["calibration"]),
                "example": results["calibration"][0] if results["calibration"] else None,
            },
            "action_standard": standard_for(results["summary"]["threshold_ugm3"]),
            "transfer": results["transfer"],
            "attribution": results["attribution"],
            "forecast": results["forecast"],
            "shares": results["shares"],
            "equal_cost": results["equal_cost"],
            "provenance": {
                "background": results["provenance"]["background"],
                "injected": results["provenance"]["injected"],
                "grid_deg": results["provenance"]["grid_deg"],
            },
        }

    @app.get("/api/health")
    def health():
        return {"ok": True, "results_present":
                (RESULTS_DIR / "validation.json").exists()}

    if DASHBOARD_DIR.exists():
        # no-store, deliberately. The console is edited between local development iterations, and a
        # browser holding an old app.js against a changed API shape renders
        # "n/a" under correct-looking headings - which reads as a broken pipeline
        # rather than a stale cache. Nothing here is large enough to merit
        # caching on a localhost operator console.
        app.mount("/static",
                  NoStoreStaticFiles(directory=str(DASHBOARD_DIR)),
                  name="static")
    return app
