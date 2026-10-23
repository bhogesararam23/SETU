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
    first = ids[0]
    r = client.get(f"/api/actions/{first}")
    assert r.status_code == 200
    pack = r.json()
    assert pack["id"] == first
    assert "sigma" in json.dumps(pack).lower()


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
    headline = re.search(r"\| PM2\.5 MAE \| [^|]+ \| \*\*([0-9.]+)", md)
    if headline is None:
        pytest.skip("no MAE headline in the committed report")
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
    js = client.get("/static/app.js").text
    served = {route.path for route in client.app.routes}
    for path in set(re.findall(r'"/api/[a-z_/{}]*"', js)):
        path = path.strip('"').replace("{action}", "")
        if "{" in path:
            continue
        assert path in served or path.startswith("/api/actions/"), f"dashboard calls {path}"


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
