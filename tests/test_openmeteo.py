"""Offline tests for the Open-Meteo layer.

No network: a fake cache returns payloads shaped exactly like the real
responses, so these still pass on a plane and still catch a broken join.
"""

import numpy as np
import pandas as pd
import pytest

from setu.errors import UpstreamError
from setu.io import openmeteo as om


class StubCache:
    """Serves a queue of payloads keyed by which endpoint the URL is."""

    def __init__(self, aq, met):
        self.aq, self.met = aq, met

    def get_json(self, url, ttl_seconds=0, timeout=30.0, retries=0):
        payload = self.aq if "air-quality" in url else self.met
        return type("R", (), {
            "payload": payload,
            "url": url,
            "fetched_at": 0.0,
            "from_cache": True,
            "age_seconds": 0.0,
            "as_dict": lambda self: {"url": url},
        })()


def _payload(times, values_by_var):
    return {
        "cell_size": "0.1",
        "elevation": 133.0,
        "hourly": {"time": list(times), **values_by_var},
    }


@pytest.fixture
def hours():
    return pd.date_range("2026-09-01 00:00", periods=48, freq="h")


@pytest.fixture
def site():
    from setu.config import load_site
    return load_site("rajkot")


def _aq(times, n=None):
    n = len(times)
    return _payload(times, {
        "pm2_5": [10 + i % 5 for i in range(n)],
        "pm10": [30 + i % 7 for i in range(n)],
        "carbon_monoxide": [300.0] * n,
        "nitrogen_dioxide": [12.0] * n,
        "ozone": [40.0] * n,
        "sulphur_dioxide": [4.0] * n,
        "dust": [20.0] * n,
        "aerosol_optical_depth": [0.2] * n,
    })


def _met(times, n=None, lead_null=0):
    n = len(times)
    vals = [float(v) for v in range(n)]
    def col(base, spread):
        series = [base + (i % 10) * spread for i in range(n)]
        return [None] * lead_null + series[lead_null:]
    return _payload(times, {
        "temperature_2m": col(25.0, 0.4),
        "relative_humidity_2m": col(60.0, 2.0),
        "wind_speed_10m": col(3.0, 0.2),
        "wind_direction_10m": col(180.0, 5.0),
        "surface_pressure": col(1000.0, 0.5),
        "boundary_layer_height": col(500.0, 20.0),
    })


def test_join_produces_one_row_per_hour(site, hours):
    df, prov = om.load_hourly(site, StubCache(_aq(hours), _met(hours)))
    assert len(df) == 48
    assert df.index.is_monotonic_increasing
    assert len(prov.as_list()) == 2


def test_coarse_fraction_is_pm10_minus_pm25_and_never_negative(site, hours):
    aq = _aq(hours)
    # Force one hour where PM2.5 exceeds PM10, which CAMS does produce at low
    # mass because the two fields are interpolated separately.
    aq["hourly"]["pm2_5"][3] = 999.0
    aq["hourly"]["pm10"][3] = 31.0
    df, _ = om.load_hourly(site, StubCache(aq, _met(hours)))
    assert df["coarse"].min() >= 0.0
    assert df["coarse"].iloc[3] == 0.0
    assert df["coarse_ratio"].iloc[3] == 0.0
    assert df["fine_ratio"].iloc[3] == pytest.approx(1.0)


def test_coarse_and_fine_ratios_partition_the_mass(site, hours):
    df, _ = om.load_hourly(site, StubCache(_aq(hours), _met(hours)))
    total = (df["coarse_ratio"] + df["fine_ratio"]).dropna()
    assert np.allclose(total, 1.0)


def test_met_history_shorter_than_aq_history_drops_the_head(site, hours):
    """The real behaviour we hit: 436 leading hours with null wind."""
    df, _ = om.load_hourly(site, StubCache(_aq(hours), _met(hours, lead_null=12)))
    assert df.attrs["dropped_incomplete_hours"] == 12
    assert len(df) == 36
    assert df["wind_speed_ms"].notna().all()
    assert df.index[0] == hours[12]


def test_wind_is_decomposed_into_components_not_a_bearing(site, hours):
    met = _met(hours)
    met["hourly"]["wind_direction_10m"] = [270.0] * 48
    met["hourly"]["wind_speed_10m"] = [4.0] * 48
    df, _ = om.load_hourly(site, StubCache(_aq(hours), met))
    # A 270 deg bearing is wind FROM the west, i.e. moving eastward: u > 0.
    assert df["wind_u"].iloc[0] == pytest.approx(4.0, abs=1e-9)
    assert df["wind_v"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_excess_over_baseline_is_zero_centred_on_a_flat_series(site, hours):
    df, _ = om.load_hourly(site, StubCache(_aq(hours), _met(hours)))
    ex = om.excess_over_baseline(df, "co", window_hours=48)
    assert abs(float(ex.mean())) < 1e-9


def test_excess_rejects_a_window_that_cannot_span_a_diurnal_cycle(site, hours):
    df, _ = om.load_hourly(site, StubCache(_aq(hours), _met(hours)))
    with pytest.raises(UpstreamError):
        om.excess_over_baseline(df, "co", window_hours=12)


def test_a_missing_species_is_reported_not_silently_nan(site, hours):
    aq = _aq(hours)
    del aq["hourly"]["dust"]
    with pytest.raises(UpstreamError, match="dust"):
        om.load_hourly(site, StubCache(aq, _met(hours)))


def test_lookback_beyond_the_endpoint_limit_is_refused(site, hours):
    with pytest.raises(UpstreamError, match="92"):
        om.load_hourly(site, StubCache(_aq(hours), _met(hours)), past_days=200)


def test_summary_reports_the_window_it_describes(site, hours):
    df, _ = om.load_hourly(site, StubCache(_aq(hours), _met(hours)))
    s = om.summary(df)
    assert s["hours"] == 48
    assert s["pm10_mean"] == pytest.approx(round(float(df["pm10"].mean()), 2))
    assert "coarse" in s["columns"]
    assert s["grid_deg"] if "grid_deg" in s else True
