"""The operator console is a key user-facing interface, so its contract is tested.

These run against the committed results/validation.json rather than a mock, on
purpose: a mock would pass while the real payload kept changing shape under the
dashboard. If `setu run` writes a field the console reads, these tests notice.
"""

import json
import re
from pathlib import Path

import pytest

from setu.api import server
from setu.report import RESULTS_DIR

TestClient = pytest.importorskip("fastapi.testclient").TestClient


@pytest.fixture(scope="module")
def client():
    return TestClient(server.build_app("rajkot"))


@pytest.fixture(scope="module")
def committed():
    if not (RESULTS_DIR / "validation.json").exists():
        pytest.skip("no committed run output in this checkout")
    return json.loads((RESULTS_DIR / "validation.json").read_text(encoding="utf-8"))


def test_site_reports_the_lattice_split(client):
    body = client.get("/api/site").json()
    assert body["instrumented"] == 40
    assert body["held_out"] == 9
    assert body["naaqs"]["pm25_24h"] == 60          # notified CPCB value
    assert body["who_2021"]["pm25_24h"] == 15       # WHO 2021 guideline


def test_gates_expose_the_policy_that_refuses_secondary(client):
    body = client.get("/api/gates").json()
    g = body["gates"]
    assert g["min_source_posterior"] == 0.45
    assert g["require_human_signoff"] is True
    assert g["hysteresis_pct"] == 15
    # The design decision: advection is not a local lever.
    assert body["policy"]["secondary"] == []
    assert body["refusals"], "no do-not-build list served"
    assert all("source" in v and v["source"] for v in body["levers"].values()), \
        "a lever shipped without a citation"


def test_nodes_serve_the_provenance_banner_data(client):
    body = client.get("/api/nodes").json()
    assert body["provenance"] == "digital-twin"
    assert len(body["nodes"]) == 49          # full lattice, roles included
    roles = {n["role"] for n in body["nodes"]}
    assert roles == {"instrumented", "holdout"}
    for n in body["nodes"]:
        assert n["quarantined"] == (n["drift_alarms"] > 0)
    instrumented = [n for n in body["nodes"] if n["role"] == "instrumented"]
    assert any(n["calibration_mae"] is not None for n in instrumented)
    assert any(n["dead"] for n in body["nodes"]), "no dead nodes simulated at all"


def test_actions_split_dispatches_from_the_ledger(client, committed):
    body = client.get("/api/actions").json()
    totals = body["totals"]
    assert totals["actions"] == committed["response"]["actions"]
    assert totals["suppressed"] == committed["response"]["suppressed"]
    assert all("id" in a for a in body["actions"])
    assert all("id" not in a for a in body["ledger"])
    # Every refusal carries a reason; an unexplained refusal is a silent gate.
    assert all(a.get("reason") for a in body["ledger"])
    assert sum(committed["response"]["suppressed_by_reason"].values()) \
        == totals["suppressed"]


def test_action_limit_is_clamped(client):
    assert len(client.get("/api/actions?limit=3").json()["actions"]) <= 3
    assert len(client.get("/api/actions?limit=99999").json()["actions"]) <= 200


def test_evidence_pack_is_served_for_a_real_dispatch(client, committed):
    # to_json lifts sample_actions to the top level and strips it out of
    # response, so that is where the endpoint and this test both read it.
    ids = [a["id"] for a in committed.get("sample_actions", []) if "id" in a]
    assert ids, "the committed run dispatched nothing"
    # A pack is written for every row of the queue head, so that is the id an
    # officer can actually click. The sample also gets packs, but the queue is
    # the panel that links to them.
    queue = committed["response"]["queue"]["items"]
    first = queue[0]["id"]
    r = client.get(f"/api/actions/{first}")
    assert r.status_code == 200
    pack = r.json()
    assert pack["id"] == first
    assert "sigma" in json.dumps(pack).lower()

    index = client.get("/api/evidence").json()
    assert index["count"] >= len(queue)
    assert first in {p["id"] for p in index["packs"]}
    for row in index["packs"][:5]:
        assert client.get(f"/api/evidence/{row['id']}").status_code == 200


def test_unknown_or_traversing_action_id_is_a_404_not_a_crash(client):
    assert client.get("/api/actions/NOPE9999").status_code == 404
    # Sanitisation strips punctuation, so traversal cannot reach the filesystem.
    for probe in ["../../etc/passwd", "..%2F..%2Fetc", "A00001.json"]:
        assert client.get(f"/api/actions/{probe}").status_code in (404, 422)


def test_validation_endpoint_carries_the_distance_band_result(client, committed):
    body = client.get("/api/validation").json()
    t = body["transfer"]
    assert t["nodes_transferred"] == committed["transfer"]["nodes_transferred"]
    assert t["by_distance_band"], "transfer distance bands missing from the payload"
    assert sum(b["nodes"] for b in t["by_distance_band"]) == t["nodes_transferred"]
    assert t["max_donor_distance_m"] > 1000, "donors never reached across the grid"
    # The cross-check that fails is served as a failure, not hidden.
    assert any(not row["within"] for row in body["shares"])


def test_console_shows_run_wide_means_not_one_nodes_luck(client, committed):
    """The validation panel is titled "Validation", so it must carry the average
    over every node. It used to serve `calibration[:1]`, and node rajkot-00 scored
    14.70 -> 0.28 ug/m3 where the 40-node mean is 9.85 -> 1.82: the screen read
    much better than results/VALIDATION.md sitting next to it, under the same
    label. That is the kind of discrepancy a contributor encounters and then distrusts
    everything else."""
    rows = committed["calibration"]
    mean_mae_before = sum(r["before"]["mae"] for r in rows) / len(rows)
    mean_mae_after = sum(r["after"]["mae"] for r in rows) / len(rows)
    mean_cov = sum(r["interval_coverage"] for r in rows) / len(rows)

    body = client.get("/api/validation").json()
    c = body["calibration"]["mean"]
    assert c["nodes"] == len(rows)
    assert c["before_mae"] == pytest.approx(mean_mae_before, abs=1e-3)
    assert c["after_mae"] == pytest.approx(mean_mae_after, abs=1e-3)
    assert c["interval_coverage"] == pytest.approx(mean_cov, abs=1e-3)

    single = body["calibration"]["example"]
    assert single["after"]["mae"] != pytest.approx(mean_mae_after, abs=1e-3)

    # The headline is read from the report rather than hardcoded, so a retune
    # that moves the number updates both sides of this comparison at once -
    # the assertion is that the console and the report agree, not that they
    # agree with a constant frozen at some earlier run.
    md = (RESULTS_DIR / "VALIDATION.md").read_text(encoding="utf-8")
    headline = re.search(r"\|\s*MAE ug/m3\s*\|[^|]*\|\s*([0-9.]+)\s*\|", md)
    # No skip here. This test skipped itself when the pattern stopped matching,
    # and the pattern had stopped matching because the calibration table was
    # reformatted - so the guard tying the console to the report was silently
    # switched off and reported as a skip. A report missing the row this test
    # claims to read is a failure of the tie, not an environment problem.
    assert headline is not None, (
        "no MAE row in results/VALIDATION.md; the console/report tie cannot be "
        "checked and the report shape has moved without this test")
    assert abs(body["calibration"]["mean"]["after_mae"]
               - float(headline.group(1))) < 0.01, \
        "console drifted from the headline in results/VALIDATION.md"


def test_console_names_the_standard_the_run_actually_used(client, committed):
    body = client.get("/api/validation").json()
    level = float(committed["summary"]["threshold_ugm3"])
    expected = "WHO 2021 24-hour guideline" if level == 15 else \
               "India NAAQS 24-hour limit" if level == 60 else "operator-set action level"
    assert body["action_standard"] == expected
    js = client.get("/static/app.js").text
    assert "Threshold accuracy @ NAAQS" not in js, \
        "the panel hard-codes a statutory name the run did not use"


def test_dashboard_assets_are_not_cacheable(client):
    """A browser holding an old app.js renders last run's numbers under this
    run's labels - plausible-looking and wrong, which is the worst failure a
    console can have in a local environment."""
    for path in ("/", "/static/app.js", "/static/styles.css"):
        assert client.get(path).headers.get("cache-control") == "no-store"


def test_dashboard_html_is_served_and_its_assets_resolve(client):
    """The operator console is an important part of the project interface. `DASHBOARD_DIR` resolved one
    directory short once, which served a 404 at `/` and never mounted `/static` -
    a blank page, so both halves are checked."""
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="provenance-banner"' in r.text
    assert "digital twin" in r.text.lower()
    assert "/static/app.js" in r.text and "/static/styles.css" in r.text

    for asset in ("/static/app.js", "/static/styles.css"):
        a = client.get(asset)
        assert a.status_code == 200, f"{asset} not served"
        assert len(a.text) > 200, f"{asset} is a stub"


def test_dashboard_calls_only_endpoints_that_exist(client):
    """Every /api reference in the script, whether quoted or interpolated.

    The old version only scanned double-quoted literals, so a template literal
    could call any endpoint at all and still pass. Matching the reference itself
    and allowing a parameterised route to satisfy a prefix keeps the guard real
    without forcing the script to string-concatenate.
    """
    js = client.get("/static/app.js").text
    served = {route.path for route in client.app.routes}
    refs = set(re.findall(r"/api/[a-z_/]*", js))
    assert refs, "the dashboard calls no API at all"
    for ref in refs:
        ok = any(path == ref or (path.startswith(ref) and "{" in path)
                 for path in served)
        assert ok, f"dashboard calls {ref}, which no route serves"


def test_dashboard_script_only_touches_elements_the_page_has(client):
    """A rename in the markup leaves `$("id")` returning null, and the first
    property access throws away the whole render. That failure mode is silent in
    HTTP terms - the page still returns 200 - so it is checked here."""
    html = client.get("/").text
    js = client.get("/static/app.js").text
    ids = set(re.findall(r'id="([^"]+)"', html))
    used = set(re.findall(r'\$\("([^"]+)"\)', js))
    assert used, "the script looks up no elements by id"
    assert used <= ids, f"script queries missing ids: {sorted(used - ids)}"


def test_every_tab_has_a_panel_and_every_panel_has_a_tab(client):
    html = client.get("/").text
    tabs = set(re.findall(r'class="tab"[^>]*data-panel="([a-z]+)"', html))
    panels = set(re.findall(r'id="panel-([a-z]+)"', html))
    assert tabs == panels, f"tabs {tabs} do not match panels {panels}"
    assert len(tabs) >= 5, "the console lost a section"
    assert 'role="tablist"' in html


def test_the_console_loads_nothing_from_outside_this_origin(client):
    """The console runs on a municipal machine behind a firewall, and the CSP
    forbids off-origin loads. A CDN tag would be a blank page under it."""
    html = client.get("/").text
    assert "http://" not in html and "https://" not in html
    assert not re.search(r"<script(?![^>]*src=)", html), "inline script breaks the CSP"
    assert not re.search(r"\son[a-z]+=", html), "inline handler breaks the CSP"

    js = client.get("/static/app.js").text
    assert "innerHTML" in js
    assert "eval(" not in js and "new Function" not in js


def test_the_console_has_the_accessibility_scaffolding_it_claims(client):
    html = client.get("/").text
    assert 'href="#main"' in html, "no skip link"
    assert 'aria-live="polite"' in html
    assert 'aria-label="Console sections"' in html
    tables = re.findall(r"<table\b", html)
    captions = re.findall(r"<caption", html)
    assert len(tables) == len(captions), (
        f"{len(tables)} tables, {len(captions)} captions: a table without a "
        "caption is a wall of numbers to a screen reader")


def test_missing_results_is_a_503_with_an_actionable_message(client, tmp_path,
                                                             monkeypatch):
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    r = client.get("/api/nodes")
    assert r.status_code == 503
    assert "setu run" in r.json()["detail"]


def test_health_endpoint_states_what_is_present(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["results_present"] is True
    assert body["status"] in {"ok", "stale"}


# -- the queue --------------------------------------------------------------
def test_queue_is_ordered_by_priority_and_counts_the_whole_run(client, committed):
    body = client.get("/api/queue?limit=10").json()
    published = committed["response"]["queue"]
    assert body["total"] == published["total"] == committed["response"]["actions"]
    assert body["returned"] == 10
    assert body["capacity_modelled"] is False, (
        "a capacity this prototype does not have must not be implied")
    ranks = [row["rank"] for row in body["items"]]
    priorities = [row["priority"] for row in body["items"]]
    assert ranks == list(range(1, 11))
    assert priorities == sorted(priorities, reverse=True)
    assert body["items"][0]["priority"] == max(
        row["priority"] for row in published["items"])
    # A queue row is a dispatch, not a refusal.
    assert all(row["id"] for row in body["items"])
    assert all(row["requires_signoff"] for row in body["items"])


def test_queue_counts_every_dispatch_it_is_a_head_of(client):
    """`total` is the run, `matched` is the filter, `returned` is the page.
    Conflating them makes 8,047 dispatches look like a backlog of 25."""
    body = client.get("/api/queue?limit=5").json()
    assert body["total"] > body["returned"]
    assert body["matched"] >= body["returned"]
    owners = body["owners"]
    assert sum(owners.values()) == body["total"]


def test_queue_filters_narrow_without_losing_the_total(client):
    body = client.get("/api/queue?owner=traffic_police_its_operator").json()
    assert body["items"], "the filter matched nothing; the owner name drifted"
    assert all(row["owner"] == "traffic_police_its_operator" for row in body["items"])
    assert body["total"] == client.get("/api/queue").json()["total"]
    assert body["matched"] <= body["total"]

    lever = client.get("/api/queue?lever=vacuum_sweeping").json()
    assert all(row["lever"] == "vacuum_sweeping" for row in lever["items"])

    top = client.get("/api/queue?limit=1").json()["items"][0]["priority"]
    pruned = client.get(f"/api/queue?min_priority={top}").json()
    assert all(row["priority"] >= top for row in pruned["items"])


def test_queue_can_be_read_in_audit_order(client):
    worked = client.get("/api/queue?limit=10").json()["items"]
    audit = client.get("/api/queue?order=time&limit=10").json()["items"]
    stamps = [row["dispatched_at"] for row in audit]
    assert stamps == sorted(stamps), "audit order must be chronological"
    assert [r["id"] for r in audit] != [r["id"] for r in worked], \
        "the audit order is the same list as the working order"


# -- the ledger -------------------------------------------------------------
def test_ledger_totals_reconcile_with_the_run(client, committed):
    body = client.get("/api/ledger").json()
    assert body["total"] == committed["response"]["suppressed"]
    assert sum(g["total"] for g in body["gates"]) == body["total"]
    for gate in body["gates"]:
        assert gate["name"]
        assert sum(r["count"] for r in gate["reasons"]) == gate["total"]
        assert all(r["explanation"] for r in gate["reasons"]), \
            "a refusal with no explanation is a silent gate"
    assert [g["gate"] for g in body["gates"]] == sorted(g["gate"] for g in body["gates"])
    assert body["reschedulable"] == committed["response"]["reschedulable"]
    assert sum(g["retryable"] for g in body["gates"]) == body["reschedulable"]


def test_ledger_says_which_gates_have_a_clock_and_which_need_information(client):
    gates = {g["name"]: g for g in client.get("/api/ledger").json()["gates"]}
    dwell = next(r for r in gates["dwell"]["reasons"] if r["reason"] == "dwell")
    assert dwell["retryable"] is True
    assert 0 < dwell["retry_within_minutes"] <= 90
    assert dwell["last_retry_at"]

    source = next(r for r in gates["source"]["reasons"]
                  if r["reason"] == "refused_by_evidence")
    assert source["retryable"] is False
    assert source["retry_within_minutes"] is None
    assert source["last_retry_at"] is None


# -- forecast, attribution, data quality ------------------------------------
def test_forecast_serves_the_trajectory_and_the_folds(client, committed):
    body = client.get("/api/forecast").json()
    rows = body["horizons"]
    assert [r["horizon_h"] for r in rows] == [1, 3, 6]
    for row in rows:
        assert row["folds"], f"horizon {row['horizon_h']} served without its folds"
        assert len(row["folds"]) >= 2
    assert body["nominal_coverage"] == 0.80
    # The committed run does not beat persistence pooled; the claim must say so
    # rather than quote the fold that does.
    pooled = [r["skill_vs_persistence"] for r in rows]
    assert body["beats_persistence_pooled"] is (all(v > 0 for v in pooled))
    if not body["beats_persistence_pooled"]:
        assert "does not beat persistence" in body["claim"]
    assert "coverage" in body["claim"]
    assert body["diagnostics"]["nodes_total"] >= 1


def test_attribution_serves_importance_and_calibration(client, committed):
    body = client.get("/api/attribution").json()
    assert body["classes"] == committed["attribution"]["classes"]
    assert len(body["feature_importance"]) >= 5
    assert all({"feature", "importance_mean"} <= set(f)
               for f in body["feature_importance"])
    assert body["confidence_calibration"]["ece"] is not None
    assert body["sign_check"]["coarse_count_ratio_on_dust"] is not None


def test_quality_panel_matches_the_provenance_it_describes(client, committed):
    body = client.get("/api/quality").json()
    prov = committed["provenance"]
    assert body["fingerprint"] == prov["fingerprint"]
    assert body["window"] == prov["window"]
    assert body["background"], "a run with no background source is not traceable"
    assert body["offline"] is True, "the tests run offline against the snapshot"
    assert body["ok"] == prov["quality"]["ok"]


def test_run_status_dates_the_payload_instead_of_implying_it_is_live(client):
    body = client.get("/api/run").json()
    assert body["results_present"] is True
    assert body["status"] in {"ok", "stale"}
    assert body["generated_at"] and body["generated_at"].endswith("+00:00")
    assert body["age_hours"] >= 0
    assert body["stale_after_hours"] > 0
    assert body["stale"] is (body["status"] == "stale")
    assert body["evidence_packs"] == client.get("/api/evidence").json()["count"]
    assert body["offline"] is True


# -- errors -----------------------------------------------------------------
def test_unknown_evidence_is_a_documented_404(client):
    r = client.get("/api/evidence/NOPE9999")
    assert r.status_code == 404
    body = r.json()
    assert body["code"] == "evidence_not_found"
    assert body["hint"], "a 404 the console renders needs a next step"


def test_a_bad_query_is_a_422_with_a_code_not_a_traceback(client):
    for url in ("/api/queue?limit=abc", "/api/queue?order=sideways",
                "/api/queue?min_priority=-1"):
        r = client.get(url)
        assert r.status_code == 422, url
        assert r.json()["code"] == "invalid_request"


def test_an_unknown_route_still_gets_the_error_shape(client):
    r = client.get("/api/nope")
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"


def test_a_corrupt_payload_is_reported_as_corrupt(client, tmp_path, monkeypatch):
    (tmp_path / "validation.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    r = client.get("/api/run")
    assert r.status_code == 200
    assert r.json()["status"] == "corrupt"
    assert r.json()["notes"]
    assert client.get("/api/validation").status_code == 500
    assert client.get("/api/health").json()["ok"] is False


# -- headers and docs -------------------------------------------------------
def test_the_console_is_served_with_a_restrictive_csp(client):
    for path in ("/", "/api/queue"):
        headers = client.get(path).headers
        csp = headers["content-security-policy"]
        assert "default-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "script-src 'self'" in csp and "unsafe-eval" not in csp
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"
    assert "cdn.jsdelivr.net" not in client.get("/").headers[
        "content-security-policy"], \
        "the dashboard load must not be loosened for the docs page"


def test_the_generated_docs_are_served_and_can_load_their_assets(client):
    r = client.get("/docs")
    assert r.status_code == 200
    assert "cdn.jsdelivr.net" in r.headers["content-security-policy"]
    schema = client.get("/openapi.json").json()
    for path in ("/api/queue", "/api/ledger", "/api/forecast",
                 "/api/attribution", "/api/quality", "/api/run",
                 "/api/evidence"):
        assert path in schema["paths"], f"{path} missing from the API docs"
    assert "QueueResponse" in schema["components"]["schemas"]
    assert "LedgerGate" in schema["components"]["schemas"]
