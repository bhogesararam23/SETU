"""Does a fresh run land on the numbers that are committed?

The pipeline is seeded, reads a committed snapshot, and writes the figures the
README quotes. "It should reproduce" is a claim until something checks it, and a
CI job that merely re-runs the pipeline checks that nothing crashed - not that
the answer is the same.

So this module compares a fresh payload against `results/validation.json` and
names what moved. The distinctions it draws matter:

*Counts are exact.* 8,047 dispatches either reproduces or it does not, and a
tolerance on an integer is a tolerance on a bug.

*Measured quantities get a tolerance.* A library upgrade can move the last
decimal of an MAE without anything being wrong. The tolerance is stated per
field so the size of the allowance is visible rather than buried in an
`assertAlmostEqual`.

*A missing field is a failure, not a skip.* If the payload lost a key, the guard
that reads it has stopped guarding. The one exception is a field the committed
payload predates: that is a comparison that cannot be made, and it says so
instead of passing quietly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class Field:
    """One comparable quantity, and how much drift it is allowed."""

    name: str
    get: Callable[[dict], Any]
    #: Absolute tolerance for numbers. None means exact equality, which is what
    #: every count and every digest gets.
    tolerance: float | None = None


@dataclass(frozen=True)
class Check:
    name: str
    committed: Any
    fresh: Any
    ok: bool
    tolerance: float | None = None
    detail: str = ""


def _dig(payload: dict, path: str) -> Any:
    """Resolve `a.b[c=d].e` against a payload, or raise KeyError.

    The forecast rows are a list keyed by horizon, so a check has to be able to
    name one of them; a bare dotted path cannot.
    """
    node: Any = payload
    for part in path.split("."):
        if "[" in part:
            head, _, rest = part.partition("[")
            node = node[head]
            selector = rest.rstrip("]")
            if "=" in selector:
                key, value = selector.split("=", 1)
                # KeyError rather than StopIteration: a generator's exhaustion
                # escapes a `for` as an empty result and a `next` as a runtime
                # error, and neither reads like "this selector is not in the
                # payload".
                node = next((row for row in node
                             if str(row.get(key)) == value), None)
                if node is None:
                    raise KeyError(f"{path}: no row with {key}={value}")
            else:
                node = node[int(selector)]
        else:
            node = node[part]
    return node


def _mean(rows: Iterable[dict], *keys: str) -> float | None:
    values = []
    for row in rows:
        node: Any = row
        for key in keys:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, (int, float)):
            values.append(float(node))
    return sum(values) / len(values) if values else None


def _forecast_skill(payload: dict, horizon: int) -> Any:
    return _dig(payload, f"forecast[horizon_h={horizon}].skill_vs_persistence")


#: What a replay is checked against. Kept short on purpose: these are the
#: numbers the project quotes, so a change in any of them is a change to a claim.
FIELDS: tuple[Field, ...] = (
    Field("input fingerprint", lambda p: _dig(p, "provenance.fingerprint")),
    Field("window hours", lambda p: _dig(p, "summary.hours")),
    Field("nodes", lambda p: _dig(p, "summary.nodes")),
    Field("action level", lambda p: _dig(p, "summary.threshold_ugm3"), 1e-9),
    Field("dispatches", lambda p: _dig(p, "response.actions")),
    Field("suppressions", lambda p: _dig(p, "response.suppressed")),
    Field("suppressions by gate", lambda p: _dig(p, "response.suppressed_by_gate")),
    Field("queue head", lambda p: _dig(p, "response.queue.total")),
    Field("calibration MAE (mean)",
          lambda p: _mean(p.get("calibration") or [], "after", "mae"), 0.01),
    Field("attribution accuracy",
          lambda p: _dig(p, "attribution.accuracy"), 1e-4),
    Field("forecast skill @1h", lambda p: _forecast_skill(p, 1), 0.005),
    Field("forecast skill @6h", lambda p: _forecast_skill(p, 6), 0.005),
)


def _close(expected: Any, actual: Any, tolerance: float | None) -> bool:
    if tolerance is not None and isinstance(expected, (int, float)) \
            and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) <= tolerance
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected == actual
    return expected == actual


def compare(committed: dict, fresh: dict,
            fields: Sequence[Field] = FIELDS) -> list[Check]:
    """Every field, compared. Failures are the output, not an exception."""
    checks: list[Check] = []
    for field in fields:
        try:
            expected = field.get(committed)
        except (KeyError, TypeError, ValueError, StopIteration):
            checks.append(Check(field.name, None, None, True, field.tolerance,
                                "not in the committed payload"))
            continue
        try:
            actual = field.get(fresh)
        except (KeyError, TypeError, ValueError, StopIteration):
            checks.append(Check(field.name, expected, None, False,
                                field.tolerance, "missing from the fresh run"))
            continue
        ok = _close(expected, actual, field.tolerance)
        checks.append(Check(field.name, expected, actual, ok, field.tolerance,
                            "" if ok else "moved"))
    return checks


def load_committed(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarise(checks: Sequence[Check]) -> str:
    failed = [c for c in checks if not c.ok]
    skipped = [c for c in checks if c.ok and c.committed is None and c.fresh is None]
    if failed:
        return (f"{len(failed)} of {len(checks)} checks failed: "
                + ", ".join(c.name for c in failed))
    return (f"all {len(checks) - len(skipped)} comparable checks reproduced"
            + (f" ({len(skipped)} skipped as absent from the committed payload)"
               if skipped else ""))


def format_value(value: Any, tolerance: float | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}" if tolerance is None or tolerance < 1e-3 \
            else f"{value:.4f}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)
