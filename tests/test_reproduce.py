"""The replay check guards every number in results/, so it gets tested too.

A guard that silently passes is worse than no guard: it converts "nobody checked"
into "this was checked". These tests make it fail on purpose.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from setu import report
from setu.reproduce import Field, _dig, compare, format_value, summarise


def base_payload() -> dict:
    return {
        "summary": {"site": "Rajkot, Gujarat", "nodes": 40, "hours": 1772,
                    "threshold_ugm3": 15.0},
        "provenance": {
            "fingerprint": "abc123",
            "background": [{"label": "CAMS"}],
            "window": ["2026-07-29 12:00:00", "2026-08-25 07:00:00"],
        },
        "calibration": [{"node": "rajkot-00", "after": {"mae": 3.20}},
                        {"node": "rajkot-01", "after": {"mae": 3.22}}],
        "attribution": {"accuracy": 0.6004},
        "forecast": [
            {"horizon_h": 1, "skill_vs_persistence": -0.0278},
            {"horizon_h": 3, "skill_vs_persistence": -0.092},
            {"horizon_h": 6, "skill_vs_persistence": -0.0162},
        ],
        "response": {
            "actions": 8047, "suppressed": 62833,
            "suppressed_by_gate": {"persistence": 3029, "source": 25717},
            "queue": {"total": 8047, "items": []},
            "sample_actions": [{"id": "A1"}],
        },
        "sample_actions": [{"id": "A1"}],
    }


def by_name(checks):
    return {check.name: check for check in checks}


def test_an_identical_payload_reproduces():
    payload = base_payload()
    checks = compare(payload, copy.deepcopy(payload))
    assert checks and all(check.ok for check in checks)
    assert "all" in summarise(checks)


def test_a_count_that_moved_is_a_failure_with_no_tolerance():
    """8,047 dispatches either reproduces or it does not. A tolerance on an
    integer is a tolerance on a bug."""
    fresh = base_payload()
    fresh["response"]["actions"] = 8046
    checks = by_name(compare(base_payload(), fresh))
    assert checks["dispatches"].ok is False
    assert checks["dispatches"].tolerance is None
    assert "1 of" in summarise(compare(base_payload(), fresh))


def test_a_gate_total_that_moved_is_a_failure():
    fresh = base_payload()
    fresh["response"]["suppressed_by_gate"]["dwell"] = 5502
    checks = by_name(compare(base_payload(), fresh))
    assert checks["suppressions by gate"].ok is False
    assert checks["suppressions by gate"].detail == "moved"


def test_a_measured_quantity_inside_its_tolerance_passes_and_outside_fails():
    """A library upgrade may move the last decimal of an MAE. The allowance is
    small and stated per field rather than hidden in an approximate assert."""
    inside = base_payload()
    inside["calibration"] = [{"node": "a", "after": {"mae": 3.205}},
                             {"node": "b", "after": {"mae": 3.225}}]
    assert by_name(compare(base_payload(), inside))["calibration MAE (mean)"].ok is True

    outside = base_payload()
    outside["calibration"] = [{"node": "a", "after": {"mae": 3.60}},
                              {"node": "b", "after": {"mae": 3.80}}]
    assert by_name(compare(base_payload(), outside))["calibration MAE (mean)"].ok is False


def test_a_field_missing_from_the_fresh_run_is_a_failure_not_a_skip():
    """A payload that lost a key has also lost the guard that read it."""
    fresh = base_payload()
    del fresh["attribution"]["accuracy"]
    checks = by_name(compare(base_payload(), fresh))
    assert checks["attribution accuracy"].ok is False
    assert "missing" in checks["attribution accuracy"].detail


def test_a_field_the_committed_payload_predates_is_reported_as_skipped():
    """Older runs have no queue head. That is a comparison that cannot be made,
    and it says so rather than passing quietly."""
    committed = base_payload()
    del committed["response"]["queue"]
    checks = by_name(compare(committed, base_payload()))
    assert checks["queue head"].ok is True
    assert checks["queue head"].detail == "not in the committed payload"
    assert "skipped" in summarise(compare(committed, base_payload()))


def test_the_forecast_selector_reads_the_named_horizon():
    payload = base_payload()
    assert _dig(payload, "forecast[horizon_h=1].skill_vs_persistence") == -0.0278
    assert _dig(payload, "forecast[1].skill_vs_persistence") == -0.092
    with pytest.raises(KeyError):
        _dig(payload, "forecast[horizon_h=9].skill_vs_persistence")


def test_a_failure_names_the_field_rather_than_the_count():
    fresh = base_payload()
    fresh["forecast"][0]["skill_vs_persistence"] = 0.30
    text = summarise(compare(base_payload(), fresh))
    assert "forecast skill @1h" in text
    assert "all" not in text


def test_values_are_formatted_for_a_reader_not_a_debugger():
    assert format_value(None, None) == "n/a"
    assert format_value({"b": 1, "a": 2}, None) == '{"a": 2, "b": 1}'
    assert format_value(1772, None) == "1772"


def test_the_payload_compared_is_the_payload_written(tmp_path):
    """One builder for both: a field added to the file cannot be absent from the
    comparison that is supposed to guard it."""
    result = SimpleNamespace(
        summary={"hours": 1}, provenance={"fingerprint": "x"},
        calibration=[], transfer={}, attribution={}, forecast=[],
        response={"actions": 1, "sample_actions": [{"id": "A1"}]},
        drift={}, shares=[], equal_cost={})

    written = json.loads(report.to_json(result, tmp_path).read_text(encoding="utf-8"))
    built = json.loads(json.dumps(report.payload(result), default=str))

    assert written == built
    assert "sample_actions" not in written["response"]
    assert written["sample_actions"] == [{"id": "A1"}]


def test_the_cli_exposes_verify_and_defaults_to_the_committed_payload():
    from setu import cli

    args = cli.build_parser().parse_args(["verify"])
    assert args.func is cli._cmd_verify
    assert Path(args.against).name == "validation.json"
    assert args.threshold is None, (
        "verify must default to the level the committed run used, not to 60: "
        "replaying 15 ug/m3 numbers at the NAAQS limit compares two experiments")
    assert args.days == 92


def fake_result(actions: int = 8047, forecast_skill_1: float = -0.0278,
                forecast_skill_6: float = -0.0162, attribution_accuracy: float = 0.6004):
    """A RunResult stand-in that produces a payload-shaped dict.

    Each numeric field is parametrisable so a test that wants to break one
    comparison can do so without rewriting the fixture.
    """
    return SimpleNamespace(
        summary={"site": "Rajkot, Gujarat", "nodes": 40, "hours": 1772,
                 "threshold_ugm3": 15.0},
        provenance={"fingerprint": "abc123", "background": [{"label": "CAMS"}],
                    "window": ["2026-07-29 12:00:00", "2026-08-25 07:00:00"]},
        calibration=[{"node": "a", "after": {"mae": 3.20}},
                     {"node": "b", "after": {"mae": 3.22}}],
        transfer={}, attribution={"accuracy": attribution_accuracy},
        forecast=[{"horizon_h": 1, "skill_vs_persistence": forecast_skill_1},
                  {"horizon_h": 6, "skill_vs_persistence": forecast_skill_6}],
        response={"actions": actions, "suppressed": 62833,
                  "suppressed_by_gate": {"persistence": 3029, "source": 25717},
                  "queue": {"total": 8047}, "sample_actions": []},
        drift={}, shares=[], equal_cost={})

fake_result_matches_committed = fake_result(8047, -0.0278, -0.0162, 0.6004)


def test_verify_exits_non_zero_and_says_which_number_moved(tmp_path, monkeypatch,
                                                           capsys):
    """The exit code is the whole point: a check that reports a difference and
    exits 0 is a comment."""
    from setu import cli

    path = tmp_path / "validation.json"
    path.write_text(json.dumps(base_payload()), encoding="utf-8")
    monkeypatch.setattr(cli, "run_pipeline", lambda *a, **k: fake_result(8046))

    rc = cli._cmd_verify(SimpleNamespace(
        site="rajkot", days=92, nodes=None, threshold=None,
        against=str(path), verbose=False))

    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out
    assert "dispatches" in out


def test_verify_exits_zero_when_the_replay_matches(tmp_path, monkeypatch, capsys):
    """The exit code is the whole point: a check that reports a difference and
    exits 0 is a comment."""
    from setu import cli

    path = tmp_path / "validation.json"
    path.write_text(json.dumps(base_payload()), encoding="utf-8")
    monkeypatch.setattr(cli, "run_pipeline", lambda *a, **k: fake_result_matches_committed())

    rc = cli._cmd_verify(SimpleNamespace(
        site="rajkot", days=92, nodes=None, threshold=None,
        against=str(path), verbose=False))

    out = capsys.readouterr().out
    assert rc == 0
    assert "reproduced" in out
    assert "FAIL" not in out


def test_verify_reproduces_exact_counts_and_digest(tmp_path, monkeypatch, capsys):
    """An end-to-end guard on the single number a reader of results/ should
    be able to reproduce without ambiguity: the input fingerprint."""
    from setu import cli
    from setu.reproduce import load_committed

    path = tmp_path / "validation.json"
    path.write_text(json.dumps(base_payload()), encoding="utf-8")
    fresh = base_payload()
    monkeypatch.setattr(cli, "run_pipeline", lambda *a, **k: SimpleNamespace(**fresh))
    rc = cli._cmd_verify(SimpleNamespace(
        site="rajkot", days=92, nodes=None, threshold=None,
        against=str(path), verbose=False))
    assert rc == 0

    fresh_loaded = load_committed(path)
    assert compare(fresh_loaded, fresh)[0].ok is True


def test_the_cli_exposes_verify_and_defaults_to_the_committed_payload():
    from setu import cli

    args = cli.build_parser().parse_args(["verify"])
    assert args.func is cli._cmd_verify
    assert Path(args.against).name == "validation.json"
    assert args.threshold is None, (
        "verify must default to the level the committed run used, not to 60: "
        "replaying 15 ug/m3 numbers at the NAAQS limit compares two experiments")
    assert args.days == 92
