"""The report is the only artifact a judge reads without running the code.

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


def test_the_published_report_matches_the_committed_run(committed_summary):
    """results/VALIDATION.md is generated, so its header has to agree with the
    numbers sitting beside it in results/validation.json. Regenerate with
    `python -m setu run` rather than editing the markdown by hand."""
    path = Path(report.RESULTS_DIR) / "VALIDATION.md"
    if not path.exists():
        pytest.skip("no committed run output in this checkout")
    text = path.read_text(encoding="utf-8")
    assert report.headline(committed_summary) in text
