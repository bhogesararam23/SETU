"""FastAPI service for the operator console.

Endpoints are read-only. The one thing that must never happen from a browser is
dispatching a crew, which is why `requires_signoff` is carried all the way from
the config through to the payload rather than being asserted in a document.

Two design rules shape the API:

*The payload is evidence, not a dashboard model.* Every number here is read
from `results/validation.json`, `results/evidence/`, or the config, and carries
the window and the fingerprint it came from. Nothing is recomputed per request
except views over that payload (ordering, grouping, filtering), so what the
console shows cannot drift from what the report printed.

*Failure is a rendered state, not a stack trace.* A missing run, a corrupt
payload, and a bad query each return a documented status with a `code` and a
`hint`, because the console has a panel for each of them and none of them should
look like an empty city.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException, Path as PathParam, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..config import load_action_table, load_site
from ..grid import build_lattice, holdout, instrumented
from ..report import RESULTS_DIR, calibration_summary, standard_for
from . import models

# server.py sits at <repo>/src/setu/api/server.py, so the repository root is
# four levels up - parents[2] resolves to src/ and the dashboard silently 404s.
DASHBOARD_DIR = Path(__file__).resolve().parents[3] / "dashboard"

#: How old a committed run may be before the console says so. This is the age of
#: the results file, not of the air: there is no live feed, and a console that
#: shows August's window in October without a date on it is a console that lies.
STALE_AFTER_HOURS = 24.0 * 7

#: A gate refuses for a reason; the console should not have to carry its own
#: copy of the prose. One source of truth, served with the ledger.
REASON_WHY = {
    "not_armed": "the value is below the arm level, so there is nothing to act on",
    "persistence": "one odd window is not an episode; k of n must agree",
    "uncertainty": "the calibrated interval still includes the action level, so "
                   "acting would be a guess",
    "low_posterior": "no source clears the confidence floor",
    "refused_by_evidence": "the attributed source has no local lever that could "
                           "move it",
    "dwell": "already dispatched for this zone inside the minimum dwell",
    "no_lever": "policy resolved to nothing actionable",
    "unknown_source": "no policy entry for the named source",
}

#: Which gate owns each reason, for payloads written before the engine started
#: stamping the gate number itself.
GATE_OF = {
    "unknown_source": 1, "low_posterior": 1, "refused_by_evidence": 1,
    "no_lever": 1, "persistence": 2, "uncertainty": 3, "dwell": 4,
    "not_armed": 5,
}
GATE_NAMES = {1: "source", 2: "persistence", 3: "uncertainty", 4: "dwell",
              5: "hysteresis"}

TAGS = [
    {"name": "run", "description": "What the committed run read, and how old it is."},
    {"name": "grid", "description": "The lattice, its health, and the site config."},
    {"name": "queue", "description": "Dispatches in the order an officer should work them."},
    {"name": "ledger", "description": "What the gates refused, grouped by gate, with retry clocks."},
    {"name": "forecast", "description": "Skill against persistence, by horizon and by fold."},
    {"name": "attribution", "description": "Source posteriors: importance, calibration, signatures."},
    {"name": "evidence", "description": "One folder-ready record per dispatch."},
    {"name": "validation", "description": "The run-wide report numbers, including the failures."},
]


class ApiError(HTTPException):
    """An HTTP error the console can render, not just log."""

    def __init__(self, status_code: int, code: str, detail: str,
                 hint: str | None = None) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.code = code
        self.hint = hint


class NoStoreStaticFiles(StaticFiles):
    """StaticFiles in the installed Starlette takes no `headers=` argument, so
    the cache directive is applied per response instead."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-store"
        return response


#: Parsed payloads, keyed by path, invalidated on mtime or size change. The
#: validation payload is ~100 kB and every panel asks for a slice of it; parsing
#: it once per panel is a localhost tax with no upside.
_RESULTS_CACHE: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}


def _results_path() -> Path:
    return RESULTS_DIR / "validation.json"


def _load_results(site: str) -> dict[str, Any]:
    path = _results_path()
    try:
        stat = path.stat()
    except FileNotFoundError:
        raise ApiError(
            503, "results_missing",
            "no results yet - run `python -m setu run` first",
            "The console reads a completed run; it does not compute one.") from None
    key = (stat.st_mtime_ns, stat.st_size)
    cached = _RESULTS_CACHE.get(str(path))
    if cached and cached[0] == key:
        return cached[1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ApiError(
            500, "results_corrupt", f"results are corrupt: {exc}",
            "Re-run `python -m setu run`; the committed payload is written "
            "atomically by that command.") from exc
    if not isinstance(payload, dict) or "summary" not in payload:
        raise ApiError(
            500, "results_shape",
            "results/validation.json is not a run payload",
            "The file exists but has no `summary`; it may be a partial write.")
    _RESULTS_CACHE[str(path)] = (key, payload)
    return payload


def _age_hours(path: Path) -> float | None:
    try:
        stamp = path.stat().st_mtime
    except FileNotFoundError:
        return None
    return round((datetime.now(timezone.utc).timestamp() - stamp) / 3600.0, 3)


def _iso_from_mtime(path: Path) -> str | None:
    try:
        stamp = path.stat().st_mtime
    except FileNotFoundError:
        return None
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat(timespec="seconds")


def _offline() -> bool:
    return os.environ.get("SETU_OFFLINE", "0") not in {"", "0", "false", "False"}


def _run_status(site: str) -> models.RunStatus:
    path = _results_path()
    present = path.exists()
    age = _age_hours(path)
    status: Literal["ok", "stale", "missing", "corrupt"] = "ok"
    notes: list[str] = []
    payload: dict[str, Any] = {}
    if not present:
        status = "missing"
        notes.append("No completed run in results/. The console has nothing to read.")
    else:
        try:
            payload = _load_results(site)
        except ApiError as exc:
            status = "corrupt"
            notes.append(str(exc.detail))
        else:
            if age is not None and age > STALE_AFTER_HOURS:
                status = "stale"
                notes.append(
                    f"This run is {age:.0f} hours old. It describes the window "
                    "below, not the air now.")
    provenance = payload.get("provenance") or {}
    summary = payload.get("summary") or {}
    quality = provenance.get("quality") or {}
    if _offline():
        notes.append("SETU_OFFLINE is set: this run read the committed cache "
                     "and made no network request.")
    if quality.get("ok") is False:
        notes.append("The data-quality check failed for this window; see "
                     "/api/quality rather than reading the metrics as clean.")
    if provenance.get("stale_cache_responses"):
        notes.append(f"{provenance['stale_cache_responses']} cached upstream "
                     "responses were past their TTL.")
    evidence = RESULTS_DIR / "evidence"
    packs = len(list(evidence.glob("*.json"))) if evidence.exists() else 0
    return models.RunStatus(
        results_present=present, status=status,
        generated_at=_iso_from_mtime(path), age_hours=age,
        stale_after_hours=STALE_AFTER_HOURS, stale=status == "stale",
        site=summary.get("site"), fingerprint=provenance.get("fingerprint"),
        window=provenance.get("window"), hours=summary.get("hours"),
        nodes=summary.get("nodes"),
        threshold_ugm3=summary.get("threshold_ugm3"),
        offline=_offline(), evidence_packs=packs, notes=notes)


def _queue_items(results: dict[str, Any]) -> tuple[list[dict[str, Any]], int, bool]:
    """The ranked queue, and how many dispatches it is a head of.

    Falls back to the sample for payloads written before the run kept the head
    of the ranking: those rows have no `rank`, and they are re-sorted here so an
    older payload is still served in priority order rather than arrival order.
    """
    response = results.get("response") or {}
    queue = response.get("queue") or {}
    if queue.get("items"):
        return list(queue["items"]), int(queue.get("total", len(queue["items"]))), \
            bool(queue.get("capacity_modelled", False))
    rows = [dict(row) for row in results.get("sample_actions", []) if "id" in row]
    rows.sort(key=lambda r: (-float(r.get("priority") or 0.0),
                             str(r.get("dispatched_at") or ""), str(r.get("node"))))
    for rank, row in enumerate(rows, 1):
        row.setdefault("rank", rank)
    return rows, len(rows), False


def _ledger(results: dict[str, Any]) -> models.LedgerResponse:
    response = results.get("response") or {}
    grouped = response.get("ledger")
    gates: list[models.LedgerGate] = []
    if grouped:
        for group in grouped:
            reasons = [
                models.LedgerReason(
                    reason=row["reason"], count=int(row["count"]),
                    retryable=bool(row.get("retryable")),
                    retry_within_minutes=row.get("retry_within_minutes"),
                    last_at=row.get("last_at"),
                    last_retry_at=row.get("last_retry_at"),
                    explanation=REASON_WHY.get(row["reason"], ""))
                for row in group.get("reasons", [])]
            gates.append(models.LedgerGate(
                gate=int(group["gate"]), name=group["name"],
                description=group["description"], total=int(group["total"]),
                retryable=int(group.get("retryable", 0)), reasons=reasons))
    else:
        buckets: dict[int, dict[str, int]] = {}
        for reason, count in (response.get("suppressed_by_reason") or {}).items():
            gate = GATE_OF.get(reason, 0)
            buckets.setdefault(gate, {})[reason] = int(count)
        for gate in sorted(buckets):
            rows = sorted(buckets[gate].items(), key=lambda kv: (-kv[1], kv[0]))
            gates.append(models.LedgerGate(
                gate=gate, name=GATE_NAMES.get(gate, "ungated"),
                description="", total=sum(buckets[gate].values()),
                retryable=0,
                reasons=[models.LedgerReason(
                    reason=r, count=c, retryable=False,
                    retry_within_minutes=None, last_at=None, last_retry_at=None,
                    explanation=REASON_WHY.get(r, "")) for r, c in rows]))
    return models.LedgerResponse(
        gates=gates,
        total=int(response.get("suppressed") or 0),
        reschedulable=int(response.get("reschedulable") or 0),
        note=("Grouped by the gate in the ordered chain, so what each gate cost "
              "is a sum rather than a reading exercise. A retry time is the "
              "gate's own clock, not a forecast of clean air."))


def _forecast(results: dict[str, Any]) -> models.ForecastResponse:
    rows = sorted(results.get("forecast") or [], key=lambda r: r["horizon_h"])
    horizons = [models.ForecastHorizon(**row) for row in rows]
    # The band's nominal level is a property of the fit, not of the payload, so
    # it is stated here rather than inferred from a coverage number.
    nominal_level = 0.80
    if not horizons:
        return models.ForecastResponse(
            horizons=[], diagnostics=None, nominal_coverage=nominal_level,
            beats_persistence_pooled=False,
            claim="No forecast rows in this run.")
    pooled = [h.skill_vs_persistence for h in horizons
              if h.skill_vs_persistence is not None]
    beats = bool(pooled) and all(v > 0 for v in pooled)
    losses = sum(1 for v in pooled if v <= 0)
    parts: list[str] = []
    if beats:
        parts.append("Pooled skill against persistence is positive at every "
                     "horizon in this window.")
    else:
        parts.append(f"Pooled skill is at or below zero at {losses} of "
                     f"{len(pooled)} horizons: across this window the model does "
                     "not beat persistence, and the panel says so rather than "
                     "quoting the best fold.")
    covers = [h.interval_coverage for h in horizons
              if h.interval_coverage is not None]
    if covers:
        mean = sum(covers) / len(covers)
        verdict = "narrower than advertised" if mean < nominal_level - 0.02 else \
                  "at or above nominal" if mean >= nominal_level else \
                  "close to nominal"
        parts.append(f"Interval coverage averages {mean:.3f} against a nominal "
                     f"{nominal_level:.2f}, so the band is {verdict}.")
    return models.ForecastResponse(
        horizons=horizons,
        diagnostics=next((h.get("diagnostics") for h in rows if h.get("diagnostics")),
                         None),
        nominal_coverage=nominal_level,
        beats_persistence_pooled=beats,
        claim=" ".join(parts))


def _quality(results: dict[str, Any]) -> models.QualityResponse:
    provenance = results.get("provenance") or {}
    quality = provenance.get("quality")
    notes: list[str] = []
    if quality:
        if quality.get("largest_gap_hours"):
            notes.append(f"Largest gap in the background series: "
                         f"{quality['largest_gap_hours']} hours.")
        if quality.get("duplicate_timestamps"):
            notes.append(f"{quality['duplicate_timestamps']} duplicate timestamps "
                         "were collapsed.")
        for problem in quality.get("problems") or []:
            notes.append(str(problem))
    if provenance.get("dropped_incomplete_hours"):
        notes.append(f"{provenance['dropped_incomplete_hours']} hours were dropped "
                     "as incomplete rather than forward-filled.")
    if _offline():
        notes.append("This run read the committed cache (SETU_OFFLINE set).")
    return models.QualityResponse(
        quality=quality, ok=(quality or {}).get("ok"),
        background=provenance.get("background") or [],
        window=provenance.get("window"),
        hourly_rows=provenance.get("hourly_rows"),
        fingerprint=provenance.get("fingerprint"),
        cache=provenance.get("cache"),
        stale_cache_responses=provenance.get("stale_cache_responses"),
        sources=provenance.get("sources"),
        injected=provenance.get("injected"),
        dropped_incomplete_hours=provenance.get("dropped_incomplete_hours"),
        offline=_offline(), notes=notes)


def _evidence_index() -> models.EvidenceIndex:
    directory = RESULTS_DIR / "evidence"
    if not directory.exists():
        return models.EvidenceIndex(
            packs=[], count=0,
            note="No evidence packs yet. `python -m setu run` writes one per "
                 "dispatch in the queue head and the displayed sample.")
    packs: list[models.EvidenceItem] = []
    skipped = 0
    for path in sorted(directory.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            skipped += 1
            continue
        if not isinstance(row, dict) or "id" not in row:
            skipped += 1
            continue
        packs.append(models.EvidenceItem(**{k: row.get(k) for k in (
            "id", "node", "lever", "source", "owner", "dispatched_at",
            "trigger_value", "priority", "requires_signoff")}))
    packs.sort(key=lambda p: (p.dispatched_at or "", p.id), reverse=True)
    note = ("One JSON record per dispatch, written by the run that produced it. "
            "It is a record of a simulated trigger, not a notice served.")
    if skipped:
        note += f" {skipped} unreadable pack(s) were skipped."
    return models.EvidenceIndex(packs=packs[:200], count=len(packs), note=note)


def _evidence_pack(action_id: str) -> dict[str, Any]:
    safe = "".join(c for c in action_id if c.isalnum())
    if not safe:
        raise ApiError(404, "evidence_not_found", "no such evidence pack",
                       "An evidence id is alphanumeric, e.g. A00001.")
    path = RESULTS_DIR / "evidence" / f"{safe}.json"
    if not path.exists():
        raise ApiError(
            404, "evidence_not_found", "no such evidence pack",
            "Packs exist only for dispatches this run published; the queue "
            "endpoint shows which ids have one.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ApiError(500, "evidence_corrupt",
                       f"evidence pack {safe} is unreadable: {exc}") from exc


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(int(value), high))


def build_app(site_name: str = "rajkot") -> FastAPI:
    site = load_site(site_name)
    table = load_action_table()
    nodes = build_lattice(site)

    app = FastAPI(
        title="SETU operator console",
        version="0.2.0",
        description=(
            "Read-only API behind the SETU operator console. Every number is "
            "read from a completed `python -m setu run`, carries the window and "
            "input fingerprint it came from, and is served with the parts that "
            "failed. Concentrations come from a digital twin seeded with real "
            "hourly CAMS air quality and IFS meteorology; the grid itself is "
            "simulated. There is no dispatch endpoint, by design: acting on a "
            "row requires a human sign-off outside this service."),
        openapi_tags=TAGS,
    )

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": exc.code, "hint": exc.hint})

    # Registered on Starlette's class, not FastAPI's: a request to an unknown
    # route raises the Starlette one, and a handler bound to the subclass leaves
    # exactly that case returning a bare `{"detail": "Not Found"}`.
    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        code = "not_found" if exc.status_code == 404 else "http_error"
        return JSONResponse(status_code=exc.status_code,
                            content={"detail": detail, "code": code, "hint": None})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors(), "code": "invalid_request",
                     "hint": "Check the query parameters against /openapi.json."})

    @app.middleware("http")
    async def _headers(request: Request, call_next: Callable):
        response = await call_next(request)
        path = request.url.path
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy", "geolocation=(), camera=(), microphone=()")
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        if path in {"/docs", "/redoc"}:
            # The generated docs load Swagger/ReDoc from a CDN and bootstrap with
            # an inline script; a strict policy here would leave an empty page.
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline' "
                "https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' "
                "https://cdn.jsdelivr.net; img-src 'self' data: "
                "https://fastapi.tiangolo.com; connect-src 'self'")
        else:
            # The console is vanilla JS and local SVG, so nothing needs to be
            # fetched from, or executed from, anywhere but this origin.
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
                "object-src 'none'; base-uri 'none'; form-action 'none'; "
                "frame-ancestors 'none'")
        if path.startswith("/api/") or path == "/":
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    # ---------------------------------------------------------------- pages
    @app.get("/", include_in_schema=False)
    def index():
        target = DASHBOARD_DIR / "index.html"
        if not target.exists():
            raise ApiError(404, "dashboard_missing",
                           "dashboard missing", "The dashboard/ directory is "
                           "part of the repository, not of the run output.")
        return FileResponse(target, headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------ metadata
    @app.get("/api/run", response_model=models.RunStatus, tags=["run"],
             summary="The run this console is showing, and how old it is",
             responses={503: {"model": models.ErrorBody,
                              "description": "No completed run on disk."}})
    def get_run():
        return _run_status(site_name)

    @app.get("/api/health", response_model=models.HealthResponse, tags=["run"],
             summary="Liveness plus whether a payload is readable")
    def health():
        status = _run_status(site_name)
        return models.HealthResponse(
            ok=status.results_present and status.status in {"ok", "stale"},
            results_present=status.results_present, status=status.status)

    @app.get("/api/site", response_model=models.SiteInfo, tags=["grid"],
             summary="City, standards, and the lattice split")
    def get_site():
        return models.SiteInfo(
            id=site.id, label=site.label, latitude=site.latitude,
            longitude=site.longitude,
            naaqs={k: float(v) for k, v in site.naaqs.items()},
            who_2021={k: float(v) for k, v in site.who_2021.items()},
            coarse_action_ugm3=float(site.coarse_action_ugm3),
            node_report_seconds=int(site.node_report_seconds),
            sites=int(site.candidate_sites),
            instrumented=len(instrumented(nodes)),
            held_out=len(holdout(nodes)))

    @app.get("/api/gates", response_model=models.GatesResponse, tags=["grid"],
             summary="The gates, the policy, and the do-not-build list")
    def get_gates():
        return models.GatesResponse(
            gates=models.Gates(**table.gates.__dict__),
            policy={k: list(v) for k, v in table.policy.items()},
            refusals=sorted(table.refusals),
            levers={k: models.LeverInfo(
                label=v.label, effect_pct=v.effect_pct,
                half_life_min=v.half_life_min, owner=v.owner,
                unit_cost_inr=v.unit_cost_inr, source=v.source)
                for k, v in table.levers.items()})

    # ------------------------------------------------------------- grid view
    @app.get("/api/nodes", response_model=models.NodesResponse, tags=["grid"],
             summary="Every lattice node, with health and calibration",
             responses={503: {"model": models.ErrorBody,
                              "description": "No completed run on disk."}})
    def get_nodes():
        results = _load_results(site_name)
        health_rows = {row["node"]: row
                       for row in (results.get("provenance") or {}).get("health", [])}
        cal = {row["node"]: row for row in results.get("calibration", [])
               if "node" in row}
        alarms = (results.get("drift") or {}).get("alarms_by_node", {})
        out = []
        for node in nodes:
            h = health_rows.get(node.id, {})
            out.append(models.NodeInfo(
                id=node.id, lat=node.lat, lon=node.longitude,
                x_m=node.x_m, y_m=node.y_m, role=node.role,
                road_factor=round(float(h.get("road_factor", 0)), 3),
                near_construction=bool(h.get("has_construction", False)),
                burning_prone=bool(h.get("burning_prone", False)),
                dead=bool(h.get("dead", False)),
                obstructed=bool(h.get("obstructed", False)),
                drift_fraction=round(float(h.get("drift_fraction", 0)), 3),
                drift_alarms=int(alarms.get(node.id, 0)),
                quarantined=int(alarms.get(node.id, 0)) > 0,
                calibration_mae=(cal.get(node.id, {}).get("after", {}) or {}).get("mae"),
                interval_coverage=cal.get(node.id, {}).get("interval_coverage")))
        return models.NodesResponse(nodes=out, provenance="digital-twin")

    # ------------------------------------------------------------- queue
    @app.get("/api/queue", response_model=models.QueueResponse, tags=["queue"],
             summary="Dispatches ordered by expected effect",
             responses={503: {"model": models.ErrorBody,
                              "description": "No completed run on disk."}})
    def get_queue(
        limit: int = Query(25, description="Rows to return, clamped to 1..200."),
        owner: str | None = Query(None, description="Filter by responsible desk."),
        lever: str | None = Query(None, description="Filter by lever key."),
        source: str | None = Query(None, description="Filter by attributed source."),
        min_priority: float | None = Query(
            None, ge=0.0,
            description="Drop rows whose expected effect is below this."),
        order: Literal["priority", "time"] = Query(
            "priority",
            description="`priority` is the working order; `time` is the audit order."),
    ):
        results = _load_results(site_name)
        items, total, capacity_modelled = _queue_items(results)
        rows = items
        if owner:
            rows = [r for r in rows if r.get("owner") == owner]
        if lever:
            rows = [r for r in rows if r.get("lever") == lever]
        if source:
            rows = [r for r in rows if r.get("source") == source]
        if min_priority is not None:
            rows = [r for r in rows
                    if float(r.get("priority") or 0.0) >= float(min_priority)]
        if order == "time":
            rows = sorted(rows, key=lambda r: (str(r.get("dispatched_at") or ""),
                                               -float(r.get("priority") or 0.0)))
        matched = len(rows)
        served = _clamp(limit, 1, 200)
        rows = rows[:served]
        response = results.get("response") or {}
        notes = [
            "Ordered by expected ug/m3 shaved per pass, not by arrival: two "
            "breaches in the same hour are not equally worth a truck.",
            "Read-only. Every row requires a human sign-off; there is no "
            "dispatch endpoint.",
        ]
        if not capacity_modelled:
            notes.append(
                "Fleet capacity is not modelled, so this is an ordering rather "
                "than a schedule; deferrals would need a crew count the "
                "prototype does not have.")
        return models.QueueResponse(
            total=total, matched=matched, returned=len(rows), limit=served,
            capacity_modelled=capacity_modelled,
            items=[models.QueueItem(**row) for row in rows],
            owners={k: int(v) for k, v in
                    (response.get("actions_by_owner") or {}).items()},
            levers={k: int(v) for k, v in
                    (response.get("actions_by_lever") or {}).items()},
            notes=notes)

    # ------------------------------------------------------------- ledger
    @app.get("/api/ledger", response_model=models.LedgerResponse, tags=["ledger"],
             summary="Suppressions grouped by gate, with retry clocks",
             responses={503: {"model": models.ErrorBody,
                              "description": "No completed run on disk."}})
    def get_ledger():
        return _ledger(_load_results(site_name))

    # ------------------------------------------------------------ dispatch
    @app.get("/api/actions", tags=["queue"],
             summary="Sample dispatches and refusals (compatibility view)",
             description="The 400-row spread sample published with the run, split "
                         "into dispatches and refusals. `/api/queue` and "
                         "`/api/ledger` are the aggregate views the console uses.")
    def get_actions(limit: int = Query(60)):
        results = _load_results(site_name)
        sample = results.get("sample_actions", [])
        rows = [a for a in sample if "id" in a]
        return {
            "actions": rows[: _clamp(limit, 1, 200)],
            "ledger": [a for a in sample if "id" not in a][:200],
            "totals": {k: v for k, v in (results.get("response") or {}).items()
                       if k != "sample_actions"},
        }

    # ------------------------------------------------------------- evidence
    @app.get("/api/evidence", response_model=models.EvidenceIndex,
             tags=["evidence"], summary="Evidence packs the run published")
    def list_evidence():
        return _evidence_index()

    @app.get("/api/evidence/{action_id}", response_model=models.EvidencePack,
             tags=["evidence"], summary="One dispatch, folder-ready",
             responses={404: {"model": models.ErrorBody,
                              "description": "No pack for that id."}})
    def get_evidence(
            action_id: str = PathParam(description="Dispatch id, e.g. A00001.")):
        return _evidence_pack(action_id)

    @app.get("/api/actions/{action_id}", response_model=models.EvidencePack,
             tags=["evidence"], include_in_schema=False,
             summary="Alias of /api/evidence/{action_id}")
    def get_action(action_id: str):
        return _evidence_pack(action_id)

    # ------------------------------------------------------------- headline
    @app.get("/api/validation", response_model=models.ValidationResponse,
             tags=["validation"],
             summary="Run-wide numbers, including the checks that failed",
             responses={503: {"model": models.ErrorBody,
                              "description": "No completed run on disk."}})
    def get_validation():
        results = _load_results(site_name)
        calibration = results.get("calibration") or []
        return models.ValidationResponse(
            summary=results["summary"],
            # The panel is titled with run-wide claims, so it gets run-wide
            # numbers. `example` is one node's row, labelled as such, for the
            # drill-down; it is not what the headline row quotes.
            calibration={
                "mean": calibration_summary(calibration),
                "example": calibration[0] if calibration else None,
            },
            action_standard=standard_for(results["summary"]["threshold_ugm3"]),
            transfer=results.get("transfer") or {},
            attribution=results.get("attribution") or {},
            forecast=results.get("forecast") or [],
            shares=results.get("shares") or [],
            equal_cost=results.get("equal_cost") or {},
            provenance={k: (results.get("provenance") or {}).get(k)
                        for k in ("background", "injected", "grid_deg")})

    # ---------------------------------------------------------------- panels
    @app.get("/api/forecast", response_model=models.ForecastResponse,
             tags=["forecast"], summary="Skill by horizon, and the folds behind it")
    def get_forecast():
        return _forecast(_load_results(site_name))

    @app.get("/api/attribution", response_model=models.AttributionResponse,
             tags=["attribution"],
             summary="Source attribution: accuracy, calibration, importance")
    def get_attribution():
        payload = _load_results(site_name).get("attribution") or {}
        if not payload:
            raise ApiError(503, "attribution_missing",
                           "this run has no attribution layer",
                           "Re-run the pipeline; the payload predates the "
                           "attribution stats.")
        return models.AttributionResponse(**payload)

    @app.get("/api/quality", response_model=models.QualityResponse, tags=["run"],
             summary="The data contract for the window this run read")
    def get_quality():
        return _quality(_load_results(site_name))

    if DASHBOARD_DIR.exists():
        # no-store, deliberately. The console is edited between local development
        # iterations, and a browser holding an old app.js against a changed API
        # shape renders "n/a" under correct-looking headings - which reads as a
        # broken pipeline rather than a stale cache. Nothing here is large enough
        # to merit caching on a localhost operator console.
        app.mount("/static",
                  NoStoreStaticFiles(directory=str(DASHBOARD_DIR)),
                  name="static")
    return app
