"""Data quality and the CPCB adapter, tested against the format it receives.

The adapter has been unreachable from every network this project was developed
on, which is exactly the situation in which untested parsing rots. These tests
drive it with a payload in CPCB's published column layout, so the day the host
answers, the first failure will be a real one rather than a header that drifted
three years ago unseen.
"""

import pandas as pd
import pytest

from setu.errors import UpstreamError
from setu.io import cpcb
from setu.io.quality import FROZEN_LIMIT_HOURS, check_hourly, fingerprint

#: CPCB's CSV export, as published: one row per pollutant per hour, with the
#: station repeated on every line. Column order matters to the positional
#: fallback, so this is the order the endpoint emits.
CPCB_CSV = """Station Location,City,State,Pollutant,Value,Unit,AQI Subindex,Reporting Date,Reporting Hour,Prominent Pollutant
Rajkot - Raiya,Rajkot,Gujarat,PM2.5,45.0,ug/m3,44,01/10/2026,08:00,PM2.5
Rajkot - Raiya,Rajkot,Gujarat,PM10,98.0,ug/m3,51,01/10/2026,08:00,PM10
Rajkot - Raiya,Rajkot,Gujarat,NO2,22.0,ug/m3,18,01/10/2026,08:00,PM2.5
Rajkot - Raiya,Rajkot,Gujarat,PM2.5,52.0,ug/m3,49,01/10/2026,09:00,PM2.5
Rajkot - Raiya,Rajkot,Gujarat,PM10,110.0,ug/m3,58,01/10/2026,09:00,PM10
Rajkot - Raiya,Rajkot,Gujarat,PM2.5,not-a-number,ug/m3,,01/10/2026,10:00,PM2.5
"""


class _Resp:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code


class FakeSession:
    """Serves one scripted body per call, in order."""

    def __init__(self, bodies):
        self.bodies = list(bodies)
        self.calls = 0

    def get(self, url, timeout=None):
        self.calls += 1
        return self.bodies.pop(0) if self.bodies else _Resp("", 404)


class FakeCache:
    def __init__(self, bodies):
        self.session = FakeSession(bodies)


def _frame(rows=6, start="2026-10-01 00:00"):
    """A frame that varies in every channel, as a real one does.

    A constant channel would trip the frozen-sensor check, which is a property
    of this fixture rather than of the contract, so the values wander slightly.
    """
    idx = pd.date_range(start, periods=rows, freq="h")
    return pd.DataFrame({
        "pm25": [20.0 + 0.7 * i for i in range(rows)],
        "pm10": [40.0 + 1.3 * i for i in range(rows)],
        "temp_c": [30.0 + 0.2 * (i % 5) for i in range(rows)],
        "rh": [60.0 + 0.5 * (i % 7) for i in range(rows)],
        "wind_speed_ms": [3.0 + 0.1 * (i % 4) for i in range(rows)],
        "wind_dir_deg": [270.0 + 3.0 * i for i in range(rows)],
    }, index=idx)


# -- the quality contract ---------------------------------------------------
def test_a_clean_frame_passes_and_says_so():
    report = check_hourly(_frame(24))
    assert report.ok, report.problems
    assert report.coverage == 1.0
    assert report.missing_hours == 0
    assert report.largest_gap_hours == 0


def test_coverage_is_measured_against_the_span_not_the_row_count():
    """The frame that hides an outage between two tidy endpoints."""
    df = _frame(24).drop(pd.date_range("2026-10-01 06:00", periods=8, freq="h"))
    report = check_hourly(df)
    assert report.rows == 16
    assert report.expected_hours == 24
    assert report.coverage == pytest.approx(16 / 24, abs=1e-6)
    assert report.largest_gap_hours == 8
    assert not report.ok
    assert any("coverage" in p for p in report.problems)


def test_duplicate_timestamps_are_counted_not_silently_kept():
    df = pd.concat([_frame(6), _frame(6).iloc[[2]]])
    report = check_hourly(df)
    assert report.duplicate_timestamps == 1
    assert any("duplicate" in p for p in report.problems)


def test_a_unit_change_is_caught_as_out_of_range():
    df = _frame(12)
    df.loc[df.index[3], "pm25"] = 45_000.0      # ug/m3 reported as ng/m3
    df.loc[df.index[4], "rh"] = 130.0           # percent beyond 100
    report = check_hourly(df)
    assert report.out_of_range["pm25"] == 1
    assert report.out_of_range["rh"] == 1
    assert not report.ok


def test_a_frozen_sensor_is_not_hidden_by_completeness():
    """Complete and wrong is the case a coverage check cannot see."""
    df = _frame(24)
    df.loc[df.index[2:2 + FROZEN_LIMIT_HOURS + 2], "pm25"] = 41.0
    report = check_hourly(df)
    assert report.coverage == 1.0
    assert report.longest_frozen_run_hours["pm25"] >= FROZEN_LIMIT_HOURS + 2
    assert any("frozen" in p for p in report.problems)


def test_a_missing_required_channel_is_an_error_not_a_warning():
    with pytest.raises(ValueError, match="wind_dir_deg"):
        check_hourly(_frame(6).drop(columns=["wind_dir_deg"]))


def test_the_fingerprint_moves_only_when_the_readings_do():
    df = _frame(6)
    assert fingerprint(df) == fingerprint(df.copy())
    # A timestamp shift is a different dataset even if the values repeat.
    assert fingerprint(df) != fingerprint(df.set_index(
        df.index + pd.Timedelta(hours=1)))
    # ... but reordering columns is not.
    assert fingerprint(df) == fingerprint(df[df.columns[::-1]])
    changed = df.copy()
    changed.loc[changed.index[2], "pm25"] += 0.5
    assert fingerprint(df) != fingerprint(changed)


# -- the CPCB adapter -------------------------------------------------------
def test_the_long_format_csv_pivots_into_one_column_per_species():
    wide = cpcb._parse_csv(CPCB_CSV)
    assert set(wide.columns) == {"pm25", "pm10", "no2"}
    assert len(wide) == 2, "the unparseable value must be dropped, not raised"
    assert wide["pm25"].iloc[0] == 45.0
    assert wide.index[0] == pd.Timestamp("2026-10-01 08:00")


def test_an_html_error_page_is_not_parsed_as_csv():
    # An HTML error page has no pollutant column and no numeric readings; what
    # must not happen is that it comes back as a frame.
    with pytest.raises(UpstreamError):
        cpcb._parse_csv("<html><body>Service unavailable</body></html>")


def test_fetch_day_returns_the_parsed_frame_with_provenance():
    cache = FakeCache([_Resp(CPCB_CSV)])
    ref = cpcb.StationRef("Rajkot - Raiya", "Rajkot", "Gujarat")
    frame = cpcb.fetch_day(ref, pd.Timestamp("2026-10-01").date(), cache)
    assert len(frame) == 2
    assert "Rajkot - Raiya" in frame.attrs["station"]
    assert "CSVHourlyData" in frame.attrs["source_url"]


def test_fetch_day_reports_every_endpoint_it_tried():
    cache = FakeCache([_Resp("", 404), _Resp("<html>down</html>", 200),
                       _Resp("", 503), _Resp("", 500)])
    ref = cpcb.StationRef("Rajkot - Raiya", "Rajkot", "Gujarat")
    with pytest.raises(Exception) as exc:
        cpcb.fetch_day(ref, pd.Timestamp("2026-10-01").date(), cache)
    message = str(exc.value)
    assert "404" in message and "HTML" in message
    assert len(cpcb.HOURLY_CSV_PATHS) * 2 == 4, "both hosts, both paths"


def test_a_window_survives_a_station_going_down_for_a_day(monkeypatch):
    ref_days = []

    def fake_fetch(ref, day, cache, timeout=12.0):
        ref_days.append(day)
        if day.day == 2:
            raise UpstreamError("cpcb", "station down")
        return cpcb._parse_csv(CPCB_CSV).assign(day=day)

    monkeypatch.setattr(cpcb, "fetch_day", fake_fetch)

    class Site:
        cpcb_station = "Rajkot - Raiya"
        cpcb_city = "Rajkot"
        cpcb_state = "Gujarat"

    out = cpcb.fetch_window(Site(), FakeCache([]), days=3,
                            end=pd.Timestamp("2026-10-03").date())
    assert len(ref_days) == 3
    assert not out.empty
    assert out.index.is_monotonic_increasing


def test_asking_for_no_days_is_refused():
    class Site:
        cpcb_station = "x"
        cpcb_city = "y"
        cpcb_state = "z"

    with pytest.raises(UpstreamError, match="days"):
        cpcb.fetch_window(Site(), FakeCache([]), days=0)


def test_offline_mode_records_that_nothing_was_probed(monkeypatch):
    from setu.config import load_site
    from setu.io import sources

    monkeypatch.setenv("SETU_OFFLINE", "1")
    report = sources.probe(load_site("rajkot"), FakeCache([]))
    assert report["probed"] is False
    assert report["sources"] == []
    assert "SETU_OFFLINE" in report["reason"]
