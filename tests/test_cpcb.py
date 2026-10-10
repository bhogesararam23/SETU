"""Tests for the CPCB adapter.

Two things matter here. The CSV parser must survive the shape CPCB actually
publishes, and the failure path must be loud and specific, because the honest
statement "we could not reach the reference network" is only credible if the
code really does say which endpoint died.
"""

from datetime import date

import pandas as pd
import pytest
import requests

from setu.errors import UpstreamError, UpstreamUnavailable
from setu.io import cpcb

CSV_BODY = """Station_location,City,State,Pollutant,Reporting_date,Reporting_hour,Value,Unit
Mavdi,Rajkot,Gujarat,PM2.5,08/10/2026,10,41.2,ug/m3
Mavdi,Rajkot,Gujarat,PM10,08/10/2026,10,88.0,ug/m3
Mavdi,Rajkot,Gujarat,PM2.5,08/10/2026,11,44.9,ug/m3
Mavdi,Rajkot,Gujarat,PM10,08/10/2026,11,91.3,ug/m3
Mavdi,Rajkot,Gujarat,NO2,08/10/2026,10,18.4,ug/m3
Mavdi,Rajkot,Gujarat,CO,08/10/2026,10,510.0,ug/m3
Mavdi,Rajkot,Gujarat,Benzene,08/10/2026,10,not_available,ug/m3
"""


class Resp:
    def __init__(self, status, text, content=None):
        self.status_code = status
        self.text = text
        self.content = content if content is not None else (text or "").encode()


class StubSession:
    def __init__(self, handler):
        self.handler = handler
        self.seen: list[str] = []

    def get(self, url, timeout=None):
        self.seen.append(url)
        out = self.handler(url)
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture
def ref():
    return cpcb.StationRef("Mavdi", "Rajkot", "Gujarat")


class StubCache:
    def __init__(self, session):
        self.session = session


@pytest.fixture
def site():
    from setu.config import load_site
    return load_site("rajkot")


def test_hourly_url_uses_the_dayfirst_format_cpcb_expects(ref):
    url = cpcb.hourly_url(cpcb.APP_HOST, "/CAAQM/CSVHourlyData.aspx", ref,
                          date(2026, 10, 8))
    assert "Date=08/10/2026" in url
    assert "StationName=Mavdi" in url


def test_csv_is_pivoted_to_one_column_per_species(ref):
    frame = cpcb._parse_csv(CSV_BODY)
    assert {"pm25", "pm10", "no2", "co"} <= set(frame.columns)
    assert frame.loc[pd.Timestamp("2026-10-08 10:00"), "pm25"] == pytest.approx(41.2)
    assert frame.loc[pd.Timestamp("2026-10-08 11:00"), "pm10"] == pytest.approx(91.3)


def test_a_non_numeric_reading_is_dropped_not_zero_filled(ref):
    frame = cpcb._parse_csv(CSV_BODY)
    assert "benzene" not in frame.columns
    # The two species reported in every hour must be complete...
    assert frame[["pm25", "pm10"]].notna().all().all()
    # ...while a species reported in only one hour legitimately leaves a gap.
    # Filling that with zero would invent a clean air quality reading.
    assert frame["no2"].isna().sum() == 1


def test_an_empty_csv_is_an_error_not_an_empty_frame():
    with pytest.raises(UpstreamError):
        cpcb._parse_csv("Station_location,City\n")


def test_html_where_csv_should_be_is_an_error():
    with pytest.raises(UpstreamError):
        cpcb._parse_csv("<html><body>Portal down</body></html>")


def test_fetch_day_tries_every_endpoint_and_reports_all_of_them(ref):
    def handler(url):
        raise requests.ConnectionError(
            "HTTPSConnectionPool: Max retries exceeded (Connection refused)"
        )

    sess = StubSession(handler)
    with pytest.raises(UpstreamUnavailable) as exc:
        cpcb.fetch_day(ref, date(2026, 10, 8), StubCache(sess), timeout=0.1)
    assert len(sess.seen) == len(list(cpcb._candidate_urls(ref, date(2026, 10, 8))))
    msg = str(exc.value)
    assert "app.cpcb.gov.in" in msg
    assert "ConnectionError" in msg


def test_fetch_day_accepts_the_first_endpoint_that_returns_csv(ref):
    calls = {"n": 0}

    def handler(url):
        calls["n"] += 1
        if calls["n"] == 1:
            return Resp(200, CSV_BODY)
        return Resp(500, "")

    frame = cpcb.fetch_day(ref, date(2026, 10, 8), StubCache(StubSession(handler)))
    assert len(frame) == 2
    assert "Mavdi" in frame.attrs["station"]
    assert calls["n"] == 1


def test_fetch_day_skips_an_html_error_page_and_keeps_looking(ref):
    def handler(url):
        return Resp(200, "<!DOCTYPE html><html>not found</html>")

    with pytest.raises(UpstreamUnavailable, match="HTML"):
        cpcb.fetch_day(ref, date(2026, 10, 8), StubCache(StubSession(handler)))


def test_reachability_report_states_the_consequence_for_the_metrics(site, ref):
    from setu.config import load_site
    sess = StubSession(lambda url: Exception("ConnectionRefusedError: refused"))
    report = cpcb.reachability_report(StubCache(sess), site=load_site("rajkot"),
                                      timeout=0.1)
    assert report["reachable"] is False
    assert report["station"].startswith("Mavdi")
    assert len(report["probes"]) >= 3
    assert "NOT an instrument-compliance result" in report["consequence"]


def test_reachability_report_flips_when_an_endpoint_answers_csv(site):
    sess = StubSession(lambda url: Resp(200, CSV_BODY))
    report = cpcb.reachability_report(StubCache(sess), site=site, timeout=0.1)
    assert report["reachable"] is True
    assert "CAMS is a backstop" in report["consequence"]
