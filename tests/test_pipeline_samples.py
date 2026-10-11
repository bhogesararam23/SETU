"""The committed sample is what the console displays, so it must span the grid.

`run_response` walks nodes in order, which meant `rows[:400]` was one node's
entire night. The operator console then showed a dispatch queue and a suppression
ledger from a single pole - a display artefact that read as a limitation of the
system.
"""

import pytest

from setu.pipeline import _spread


def rows_for(node, n, kind="action"):
    out = []
    for i in range(n):
        r = {"node": node, "at": f"2026-07-{i + 1:02d}T00:00:00"}
        if kind == "action":
            r["id"] = f"{node}-{i}"
            r["lever"] = "water_sprinkling"
        else:
            r["reason"] = "not_armed"
        out.append(r)
    return out


def test_spread_interleaves_nodes_instead_of_exhausting_one():
    rows = rows_for("rajkot-00", 5) + rows_for("rajkot-01", 5)
    out = _spread(rows, 6)
    nodes = [r["node"] for r in out]
    assert nodes == ["rajkot-00", "rajkot-01"] * 3


def test_spread_keeps_time_order_within_a_node():
    rows = rows_for("a", 4) + rows_for("b", 4)
    out = _spread(rows, 8)
    for node in ("a", "b"):
        stamps = [r["at"] for r in out if r["node"] == node]
        assert stamps == sorted(stamps)


def test_limit_is_respected_and_never_exceeded():
    rows = rows_for("a", 10) + rows_for("b", 10) + rows_for("c", 10)
    assert len(_spread(rows, 7)) == 7
    assert len(_spread(rows, 999)) == 30


def test_uneven_nodes_do_not_produce_duplicates_or_gaps():
    rows = rows_for("a", 9) + rows_for("b", 1)
    out = _spread(rows, 10)
    assert len(out) == 10
    assert sorted(r["node"] for r in out) == sorted(r["node"] for r in rows)


def test_rows_without_a_node_key_survive_the_spread():
    """Suppressions carry a node, and anything else in the ledger should not be
    silently dropped by a display helper."""
    out = _spread([{"at": "x"}, {"node": "a", "at": "y"}], 5)
    assert len(out) == 2


def test_empty_sample_is_empty():
    assert _spread([], 100) == []


def test_spread_keeps_both_kinds_in_the_head_of_the_sample():
    """The obvious mistake: spreading by node alone. Suppressions outnumber
    dispatches ~4:1, so the first row of every node is a suppression and a
    node-spread leaves the dispatch queue empty. Both panels must survive the
    truncation that report.py applies."""
    rows = (rows_for("a", 30, kind="suppress") + rows_for("a", 8)
            + rows_for("b", 30, kind="suppress") + rows_for("b", 8))
    head = _spread(rows, 16)
    dispatches = [r for r in head if "id" in r]
    suppressions = [r for r in head if "id" not in r]
    assert dispatches, "dispatch queue would render empty"
    assert suppressions, "suppression ledger would render empty"
    assert {r["node"] for r in dispatches} == {"a", "b"}


def test_committed_sample_spans_the_grid():
    """Data check: the payload the console renders, not just the helper."""
    import json
    from pathlib import Path

    from setu.report import RESULTS_DIR

    path = RESULTS_DIR / "validation.json"
    if not path.exists():
        pytest.skip(f"no committed run output at {path}")
    sample = json.loads(path.read_text(encoding="utf-8")).get("sample_actions", [])
    head = sample[:40]                      # what report.to_json publishes
    nodes = {r.get("node") for r in head if r.get("node")}
    assert len(nodes) >= 5, f"console sample covers only {sorted(nodes)}"

    dispatch_nodes = {r["node"] for r in head if "id" in r}
    assert len(dispatch_nodes) >= 5, f"dispatch queue spans only {sorted(dispatch_nodes)}"
    ledger_nodes = {r["node"] for r in head if "id" not in r}
    assert len(ledger_nodes) >= 5, f"suppression ledger spans only {sorted(ledger_nodes)}"
