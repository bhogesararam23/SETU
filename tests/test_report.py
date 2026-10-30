"""The report is the human-readable validation artifact for users who do not run the code.

These tests exist because a generated report can state something factually
wrong about its own run - the wrong command name, or the wrong statutory
standard for the level it scored against - and nothing in the numerical
pipeline would notice.
"""

import argparse
import json
import re
from pathlib import Path

import pytest

from setu import report
from setu.cli import build_parser

REPORT_SOURCE = Path(report.__file__).read_text(encoding="utf-8")


def subcommands() -> set[str]:
    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("cli.py defines no subcommands")


@pytest.fixture(scope="module")
def committed_summary() -> dict:
    path = Path(report.RESULTS_DIR) / "validation.json"
    if not path.exists():
        pytest.skip("no committed run output in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def test_every_documented_command_actually_exists():
    """A report that tells the reader to run a command which is not implemented
    reads as copied boilerplate, so this is checked rather than eyeballed."""
    named = set(re.findall(r"python -m setu ([a-z][a-z-]+)", REPORT_SOURCE))
    assert named, "the report names no commands at all"
    assert named <= subcommands(), f"report documents {sorted(named - subcommands())}"


def test_headline_names_the_standard_the_run_scored_against():
    who = report.headline({"threshold_ugm3": report.WHO_PM25_24H_UGM3,
                           "site": "X", "nodes": 1, "hours": 1})
    assert "WHO 2021 24-hour guideline" in who
    assert "action level 15 ug/m3" in who

    naaqs = report.headline({"threshold_ugm3": report.NAAQS_PM25_24H_UGM3,
                             "site": "X", "nodes": 1, "hours": 1})
    assert "India NAAQS 24-hour limit" in naaqs

    # 15 ug/m3 is the WHO guideline. Calling it the NAAQS limit would be a
    # factual error, and the four-parameter Indian standard is 60.
    assert "NAAQS 24-hour limit)" not in who


def test_an_unrecognised_level_is_not_given_a_standard():
    other = report.headline({"threshold_ugm3": 30.0, "site": "X",
                             "nodes": 1, "hours": 1})
    assert "operator-set action level" in other


def test_threshold_accuracy_row_states_which_threshold(committed_summary):
    rows = dict((key, label) for label, key in report.calibration_rows(committed_summary))
    level = float(committed_summary["threshold_ugm3"])
    assert f"@ {level:.0f} ug/m3" in rows["threshold_accuracy"]


def test_evidence_packs_are_cleared_before_being_written(tmp_path):
    """A run that writes fewer packs than the run before it used to leave the
    difference behind: results/evidence/ then contains dispatches that
    validation.json does not mention, and a reader who opens one is citing an
    artefact of an older configuration."""
    from setu.cli import write_evidence

    (tmp_path / "evidence").mkdir(parents=True)
    stray = tmp_path / "evidence" / "A99999.json"
    stray.write_text('{"id": "A99999"}', encoding="utf-8")

    packs = write_evidence([{"id": "A00001"}, {"id": "A00002"}], tmp_path, limit=10)

    assert not stray.exists(), "stale pack survived the run"
    assert sorted(p.name for p in packs) == ["A00001.json", "A00002.json"]
    assert sorted(p.name for p in (tmp_path / "evidence").glob("*.json")) \
        == ["A00001.json", "A00002.json"]


def test_evidence_limit_is_honoured_and_negative_is_empty(tmp_path):
    from setu.cli import write_evidence

    (tmp_path / "evidence").mkdir(parents=True)
    three = [{"id": f"A0000{i}"} for i in (1, 2, 3)]

    assert len(write_evidence(three, tmp_path, limit=2)) == 2
    assert sorted(p.name for p in (tmp_path / "evidence").glob("*.json")) \
        == ["A00001.json", "A00002.json"], "the limit wrote more than it was given"

    assert write_evidence(three, tmp_path, limit=-1) == []
    assert list((tmp_path / "evidence").glob("*.json")) == [], \
        "a non-positive limit must leave no packs, not the previous run's"


def test_committed_evidence_matches_the_committed_dispatches(committed_summary):
    """Every published pack must be a dispatch the payload actually publishes.

    The queue head counts as published: packs are written for the rows the
    console links to, so the claim is against both lists, not just the sample.
    """
    path = Path(report.RESULTS_DIR) / "validation.json"
    if not path.exists():
        pytest.skip("no committed run output in this checkout")
    data = json.loads(path.read_text(encoding="utf-8"))
    claimed = {a["id"] for a in data.get("sample_actions", []) if "id" in a}
    claimed |= {a["id"] for a in data.get("response", {}).get("queue", {})
                .get("items", []) if "id" in a}
    published = {p.stem for p in (report.RESULTS_DIR / "evidence").glob("*.json")}
    assert published <= claimed, f"orphan packs: {sorted(published - claimed)}"
    assert published, "no packs published at all"


def test_the_published_report_matches_the_committed_run(committed_summary):
    """results/VALIDATION.md is generated, so its header has to agree with the
    numbers sitting beside it in results/validation.json. Regenerate with
    `python -m setu run` rather than editing the markdown by hand."""
    path = Path(report.RESULTS_DIR) / "VALIDATION.md"
    if not path.exists():
        pytest.skip("no committed run output in this checkout")
    text = path.read_text(encoding="utf-8")
    assert report.headline(committed_summary) in text
