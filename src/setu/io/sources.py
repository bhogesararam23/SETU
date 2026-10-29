"""Which upstreams are answering right now, and what that costs the run.

`probe_upstream` in the CLI could only ask about CPCB, and it did so by trying
the same four URLs the fetcher tries. That answers "can we reach the reference"
but not "is anything wrong upstream", which is the question an operator asks
when a run comes back thin: was the network down, or was the data bad?

This module probes every source the pipeline depends on, records latency and
status, and refuses to pretend in offline mode - an offline run did not check
anything, and saying "reachable: false" would blame the network for a flag we
set ourselves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..config import Site
from ..errors import UpstreamError
from . import cpcb, openmeteo
from .cache import HttpCache, offline_only


@dataclass(frozen=True)
class SourceProbe:
    name: str
    url: str
    reachable: bool
    status: int | None
    latency_ms: float | None
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "reachable": self.reachable,
            "status": self.status,
            "latency_ms": (round(self.latency_ms, 1)
                           if self.latency_ms is not None else None),
            "note": self.note,
        }


def _probe(name: str, url: str, cache: HttpCache, timeout: float) -> SourceProbe:
    started = time.monotonic()
    try:
        resp = cache.session.get(url, timeout=timeout)
        latency = (time.monotonic() - started) * 1000.0
    except Exception as exc:  # noqa: BLE001 - requests raises many types
        return SourceProbe(name, url.split("?")[0], False, None, None,
                           f"{type(exc).__name__}: {str(exc)[:120]}")
    body = (resp.text or "").lstrip()
    reachable = resp.status_code == 200 and bool(body)
    note = ("ok" if reachable else
            f"HTTP {resp.status_code}" if resp.status_code != 200 else "empty body")
    return SourceProbe(name, url.split("?")[0], reachable, resp.status_code,
                       latency, note)


def probe(site: Site, cache: HttpCache | None = None,
          timeout: float = 8.0) -> dict[str, Any]:
    """Ask each upstream one cheap question and report the answers.

    Both Open-Meteo endpoints are probed with a one-day request rather than a
    92-day one: reachability does not depend on the window, and a health check
    that pulls two months of data to answer a yes/no question is a health check
    that gets rate-limited.
    """
    cache = cache or HttpCache()
    if offline_only():
        return {
            "probed": False,
            "reason": "SETU_OFFLINE is set; no network request was made",
            "sources": [],
        }
    probes = [
        _probe("open-meteo-air-quality",
               openmeteo._url(openmeteo.AQ_BASE, site.latitude, site.longitude,
                              openmeteo.AQ_HOURLY, 1, 0, site.timezone),
               cache, timeout),
        _probe("open-meteo-forecast",
               openmeteo._url(openmeteo.MET_BASE, site.latitude, site.longitude,
                              openmeteo.MET_HOURLY, 1, 0, site.timezone),
               cache, timeout),
    ]
    report = cpcb.reachability_report(cache, site=site, timeout=timeout)
    for entry in report.get("probes", []):
        probes.append(SourceProbe(
            name="cpcb",
            url=str(entry.get("url")),
            reachable=entry.get("looks_like") == "csv",
            status=entry.get("status"),
            latency_ms=None,
            note=(entry.get("error") or f"returned {entry.get('looks_like')}"),
        ))
    return {
        "probed": True,
        "sources": [p.as_dict() for p in probes],
        "reachable": all(p.reachable for p in probes),
        "degraded": [p.name for p in probes if not p.reachable],
        "cpcb_consequence": report.get("consequence"),
    }


def require(name: str, report: dict[str, Any]) -> None:
    """Raise if a source a run genuinely needs did not answer.

    Only used by callers that have already decided they cannot proceed without
    the source; the pipeline itself degrades and reports instead.
    """
    if not report.get("probed"):
        raise UpstreamError(name, "cannot require a source in offline mode")
    missing = [s for s in report["sources"]
               if s["name"] == name and not s["reachable"]]
    if missing:
        raise UpstreamError(name, f"{missing[0]['note']}")
