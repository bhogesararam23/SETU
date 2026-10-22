"""Open-Meteo fetchers: CAMS air-quality hindcast/forecast plus surface met.

Why this source carries the prototype:

  * No API key or account is required, avoiding a dependency on private service credentials.
  * Hourly, multi-species (PM2.5, PM10, CO, NO2, O3, SO2, mineral dust, AOD),
    which is enough to form a coarse/fine split and a combustion tracer.
  * It reaches 92 days back, which is long enough for a real hold-out
    validation rather than a handful of hand-picked hours.

What it is not: a reference monitor. The CAMS global ensemble is a model
analysis at roughly 0.1 degrees - about 10 km - so it cannot resolve a street,
and it is not the calibration target the design promises. The CPCB adapter in
`cpcb.py` is that target; until it is reachable, this module supplies the
background signal that the digital twin then distributes over the grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..config import Site
from ..errors import UpstreamError
from .cache import HttpCache

AQ_BASE = "https://air-quality-api.open-meteo.com/v1/air-quality"
MET_BASE = "https://api.open-meteo.com/v1/forecast"

#: Max lookback the free endpoint will serve in one request.
MAX_PAST_DAYS = 92

AQ_HOURLY = [
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
    "ozone", "sulphur_dioxide", "dust", "aerosol_optical_depth",
]
MET_HOURLY = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "surface_pressure", "boundary_layer_height",
]

# Open-Meteo variable name -> column name used everywhere downstream.
_RENAME = {
    "pm2_5": "pm25",
    "pm10": "pm10",
    "carbon_monoxide": "co",
    "nitrogen_dioxide": "no2",
    "ozone": "o3",
    "sulphur_dioxide": "so2",
    "dust": "dust",
    "aerosol_optical_depth": "aod",
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "rh",
    "wind_speed_10m": "wind_speed_ms",
    "wind_direction_10m": "wind_dir_deg",
    "surface_pressure": "pressure_hpa",
    "boundary_layer_height": "pblh_m",
}


@dataclass
class Provenance:
    """What we actually pulled, so a report can say where a number came from."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def add(self, label: str, resp) -> None:
        self.records.append({"label": label, **resp.as_dict()})

    def as_list(self) -> list[dict[str, Any]]:
        return list(self.records)


def _url(base: str, lat: float, lon: float, hourly: Iterable[str],
         past_days: int, forecast_days: int, timezone: str) -> str:
    return (
        f"{base}?latitude={lat:.4f}&longitude={lon:.4f}"
        f"&hourly={','.join(hourly)}"
        f"&past_days={past_days}&forecast_days={forecast_days}"
        f"&timezone={timezone}&wind_speed_unit=ms"
    )


def _frame(payload: dict[str, Any], columns: Iterable[str], source: str) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise UpstreamError(source, "response carried no hourly block")
    df = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for col in columns:
        if col not in hourly:
            raise UpstreamError(source, f"requested {col} but it is absent")
        df[col] = pd.to_numeric(pd.Series(hourly[col]), errors="coerce")
    df = df.rename(columns=_RENAME).set_index("time").sort_index()
    return df


def load_hourly(site: Site, cache: HttpCache | None = None,
                past_days: int = 92, forecast_days: int = 2
                ) -> tuple[pd.DataFrame, Provenance]:
    """Return an hourly frame for the site, joined across both endpoints.

    The index is local wall-clock time. Rows where the two endpoints disagree
    on coverage are inner-joined, so a missing met hour drops the air-quality
    hour rather than silently forward-filling wind onto a concentration.
    """
    if not 0 <= past_days <= MAX_PAST_DAYS:
        raise UpstreamError("open-meteo",
                            f"past_days={past_days} exceeds the {MAX_PAST_DAYS}-day limit")
    cache = cache or HttpCache()
    prov = Provenance()

    aq_resp = cache.get_json(
        _url(AQ_BASE, site.latitude, site.longitude, AQ_HOURLY,
             past_days, forecast_days, site.timezone),
        ttl_seconds=6 * 3600,
    )
    prov.add("open-meteo-air-quality", aq_resp)
    aq = _frame(aq_resp.payload, AQ_HOURLY, "open-meteo-air-quality")

    met_resp = cache.get_json(
        _url(MET_BASE, site.latitude, site.longitude, MET_HOURLY,
             past_days, forecast_days, site.timezone),
        ttl_seconds=6 * 3600,
    )
    prov.add("open-meteo-forecast", met_resp)
    met = _frame(met_resp.payload, MET_HOURLY, "open-meteo-forecast")

    df = aq.join(met, how="inner")

    # The two endpoints do not share a lookback: the air-quality archive reaches
    # 92 days back while the met model history is shorter, and Open-Meteo
    # returns the full time grid with nulls rather than truncating it. An inner
    # join therefore keeps hours that have no wind or humidity at all. Those
    # hours are dropped here and the count is carried in attrs, because the
    # attribution layer needs a real wind vector and forward-filling one would
    # manufacture exactly the transport signal we are trying to measure.
    required = ["pm25", "pm10", "temp_c", "rh", "wind_speed_ms", "wind_dir_deg"]
    complete = df[required].notna().all(axis=1)
    df.attrs["dropped_incomplete_hours"] = int((~complete).sum())
    df = df[complete]

    df = _derive(df)
    df.attrs["source"] = "open-meteo (CAMS global ensemble + IFS met)"
    # The CAMS global ensemble runs on a 0.1 degree grid, about 10 km. Recorded
    # so that nothing downstream can pretend this resolves a street.
    df.attrs["grid_deg"] = float(aq_resp.payload.get("cell_size") or 0.1)
    df.attrs["elevation_m"] = float(aq_resp.payload.get("elevation") or 0.0)
    df.attrs["model"] = str(aq_resp.payload.get("models", ""))
    return df, prov


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add the coarse-mode channels the whole argument rests on.

    coarse is clipped at zero because CAMS reports PM10 and PM2.5 from
    separately interpolated fields, and at low concentrations the fine value
    occasionally exceeds the coarse-inclusive one by a rounding amount.
    """
    out = df.copy()
    out["coarse"] = (out["pm10"] - out["pm25"]).clip(lower=0.0)
    total = out["pm10"].replace(0.0, np.nan)
    out["coarse_ratio"] = (out["coarse"] / total).clip(0.0, 1.0)
    out["fine_ratio"] = (out["pm25"] / total).clip(0.0, 1.0)
    out["rh_frac"] = (out["rh"] / 100.0).clip(0.0, 1.0)
    # Meridional and zonal wind components: cheaper to reason about than a
    # bearing, and the graph adjacency needs a vector not an angle.
    theta = np.deg2rad(out["wind_dir_deg"])
    out["wind_u"] = -out["wind_speed_ms"] * np.sin(theta)
    out["wind_v"] = -out["wind_speed_ms"] * np.cos(theta)
    out["hour"] = out.index.hour
    out["dow"] = out.index.dayofweek
    return out


def excess_over_baseline(df: pd.DataFrame, column: str,
                         window_hours: int = 336) -> pd.Series:
    """Rolling-median excess, which survives the diurnal cycle intact.

    A mean would be dragged by the morning and evening peaks; a median over a
    whole-week window tracks the site's own background instead.
    """
    if window_hours < 48:
        raise UpstreamError("open-meteo", "baseline window must cover a diurnal cycle")
    base = df[column].rolling(window_hours, min_periods=48, center=True).median()
    return df[column] - base


def summary(df: pd.DataFrame) -> dict[str, Any]:
    """One-paragraph description of a frame, used in the validation report."""
    hours = len(df)
    stats: dict[str, Any] = {
        "hours": int(hours),
        "start": str(df.index[0]) if hours else None,
        "end": str(df.index[-1]) if hours else None,
        "source": df.attrs.get("source"),
        "columns": sorted(df.columns.tolist()),
    }
    for col in ("pm25", "pm10", "coarse", "co", "no2", "o3", "dust",
                "wind_speed_ms", "rh", "temp_c"):
        if col in df:
            series = df[col].dropna()
            if len(series):
                stats[f"{col}_mean"] = round(float(series.mean()), 2)
                stats[f"{col}_p95"] = round(float(series.quantile(0.95)), 2)
                stats[f"{col}_max"] = round(float(series.max()), 2)
    return stats
