"""CPCB CAAQMS adapter - the real calibration target, and an honest status.

The deck promises calibration against a reference-grade continuous station.
That is this module. It is written against CPCB's public CSV endpoints, which
need no key.

It does not work from the network this prototype was developed on: the host
refuses the connection outright and the mirror returns 404 for the report
paths. That is recorded, reported by `reachability_report()`, and surfaced in
the validation output rather than papered over - see docs/DATA_SOURCES.md.

The consequence for the numbers in this repository is narrow but real: the
"reference" the twin is calibrated against is CAMS reanalysis at ~10 km, not a
beta-gauge monitor. Every error metric in results/ is therefore an
algorithm-validation figure, not an instrument-compliance figure, and the
report says so on its face.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import pandas as pd

from ..config import Site
from ..errors import UpstreamError, UpstreamUnavailable
from .cache import HttpCache

APP_HOST = "https://app.cpcb.gov.in"
MIRROR_HOST = "https://airquality.cpcb.gov.in"

#: Tried in order. The first that answers wins.
HOURLY_CSV_PATHS = (
    "/CAAQM/CSVHourlyData.aspx",
    "/CAAQM2.0/CSVHourlyData.aspx",
)

STATION_LIST_PDF = f"{MIRROR_HOST}/ccr_docs/caaqms_list_all_india.pdf"

#: Columns CPCB emits, in the order the CSV arrives, normalised to snake_case.
CPCB_COLUMNS = [
    "station", "city", "state", "pollutant", "value", "unit",
    "aqi_subindex", "report_date", "report_hour", "source",
]

POLLUTANT_MAP = {
    "PM2.5": "pm25", "PM10": "pm10", "NO2": "no2", "NO": "no",
    "SO2": "so2", "CO": "co", "O3": "o3", "NH3": "nh3", "Benzene": "benzene",
}


@dataclass(frozen=True)
class StationRef:
    name: str
    city: str
    state: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.name}, {self.city}, {self.state}"


def hourly_url(base: str, path: str, ref: StationRef, day: date) -> str:
    return (
        f"{base}{path}?StationName={ref.name}&City={ref.city}&State={ref.state}"
        f"&Date={day.strftime('%d/%m/%Y')}&Format=DD/MM/YYYY"
    )


def _candidate_urls(ref: StationRef, day: date) -> Iterable[str]:
    for base in (APP_HOST, MIRROR_HOST):
        for path in HOURLY_CSV_PATHS:
            yield hourly_url(base, path, ref, day)


def _parse_csv(text: str) -> pd.DataFrame:
    """Pivot CPCB's long pollutant-per-row CSV into one column per species."""
    try:
        raw = pd.read_csv(io.StringIO(text))
    except Exception as exc:  # noqa: BLE001 - any parse failure is upstream's
        raise UpstreamError("cpcb", f"unparseable CSV: {exc}") from exc

    if raw.empty:
        raise UpstreamError("cpcb", "CSV had a header but no rows")

    renamed = {c: c.strip().lower().replace(" ", "_") for c in raw.columns}
    raw = raw.rename(columns=renamed)
    needed = {"station_location", "pollutant", "reporting_date", "reporting_hour"}
    present = set(raw.columns)
    if not needed.issubset(present):
        # Fall back to positional naming when the header text has drifted.
        if raw.shape[1] >= len(CPCB_COLUMNS):
            raw.columns = CPCB_COLUMNS + list(raw.columns[len(CPCB_COLUMNS):])
        else:
            raise UpstreamError(
                "cpcb", f"missing columns {sorted(needed - present)}, got {sorted(present)}"
            )

    raw["pollutant"] = raw["pollutant"].astype(str).str.strip().map(POLLUTANT_MAP)
    raw = raw.dropna(subset=["pollutant"])
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")

    stamp = pd.to_datetime(
        raw["reporting_date"].astype(str) + " " + raw["reporting_hour"].astype(str) + ":00",
        errors="coerce", dayfirst=True,
    )
    wide = (raw.assign(time=stamp)
               .dropna(subset=["time", "value"])
               .pivot_table(index="time", columns="pollutant", values="value",
                            aggfunc="mean"))
    wide = wide.rename_axis(None, axis=1).sort_index()
    if wide.empty:
        raise UpstreamError("cpcb", "no numeric readings after cleaning")
    return wide


def fetch_day(ref: StationRef, day: date, cache: HttpCache,
              timeout: float = 12.0) -> pd.DataFrame:
    """One station-day of real CPCB readings.

    Tries each host/path combination and reports the aggregate failure only if
    all of them fail, so the error message tells an operator which endpoint
    died rather than just that something did.
    """
    attempts: list[str] = []
    for url in _candidate_urls(ref, day):
        try:
            resp = cache.session.get(url, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - requests raises many types
            attempts.append(f"{url.split('?')[0]} -> {type(exc).__name__}")
            continue
        if resp.status_code != 200 or not resp.text.strip():
            attempts.append(f"{url.split('?')[0]} -> HTTP {resp.status_code}")
            continue
        if resp.text.lstrip().startswith("<"):
            attempts.append(f"{url.split('?')[0]} -> HTML, not CSV")
            continue
        frame = _parse_csv(resp.text)
        frame.attrs["station"] = str(ref)
        frame.attrs["source_url"] = url
        return frame

    raise UpstreamUnavailable(
        "cpcb",
        f"no endpoint served {ref} for {day.isoformat()}; tried: "
        + " | ".join(attempts),
    )


def fetch_window(site: Site, cache: HttpCache, days: int = 7,
                 end: date | None = None) -> pd.DataFrame:
    """Concatenate `days` of readings ending at `end` (default: yesterday).

    CPCB publishes with a lag, so today is deliberately not requested.
    """
    if days < 1:
        raise UpstreamError("cpcb", "days must be >= 1")
    end = end or (date.today() - timedelta(days=1))
    ref = StationRef(site.cpcb_station, site.cpcb_city, site.cpcb_state)
    frames = []
    for offset in range(days):
        day = end - timedelta(days=offset)
        try:
            frames.append(fetch_day(ref, day, cache))
        except UpstreamError as exc:
            # A single missing day is normal - stations go down. Keep going and
            # let the caller see the gap rather than losing the whole window.
            continue
    if not frames:
        raise UpstreamUnavailable("cpcb", f"no usable days in the {days}-day window")
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.attrs["station"] = str(ref)
    return out


def reachability_report(cache: HttpCache | None = None,
                        site: Site | None = None,
                        timeout: float = 10.0) -> dict[str, object]:
    """Probe every candidate endpoint and describe what happened.

    Called by the validation report so the reader sees the same evidence the
    developers saw, instead of taking a docstring's word for it.
    """
    cache = cache or HttpCache()
    site = site or load_default_site()
    ref = StationRef(site.cpcb_station, site.cpcb_city, site.cpcb_state)
    probes = []
    for url in _candidate_urls(ref, date.today() - timedelta(days=2)):
        entry: dict[str, object] = {"url": url.split("?")[0]}
        try:
            resp = cache.session.get(url, timeout=timeout)
            entry["status"] = resp.status_code
            body = (resp.text or "").lstrip()
            entry["looks_like"] = ("html" if body.startswith("<")
                                   else "csv" if "," in body[:200]
                                   else "empty" if not body else "other")
            entry["bytes"] = len(resp.content)
        except Exception as exc:  # noqa: BLE001
            entry["status"] = None
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        probes.append(entry)

    reachable = any(p.get("looks_like") == "csv" for p in probes)
    return {
        "station": str(ref),
        "reachable": reachable,
        "probes": probes,
        "consequence": (
            "Reference observations are available; CAMS is a backstop."
            if reachable else
            "CPCB unreachable from this network. Calibration runs against CAMS "
            "reanalysis at ~10 km, so the reported error metrics validate the "
            "algorithms and are NOT an instrument-compliance result."
        ),
    }


def load_default_site() -> Site:
    from ..config import load_site

    return load_site("rajkot")
