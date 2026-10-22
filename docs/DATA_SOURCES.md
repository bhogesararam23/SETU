# Data sources

Everything here is keyless. That is a design constraint, not an accident: a
prototype that needs three registered API tokens is a prototype nobody can
re-run, and contributors should be able to reproduce the numbers on a restricted
network or in an offline environment.

## In use

### Open-Meteo Air Quality API — the background concentration field

```
https://air-quality-api.open-meteo.com/v1/air-quality
  ?latitude=..&longitude=..
  &hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide,dust,aerosol_optical_depth
  &past_days=92&forecast_days=2&timezone=Asia/Kolkata
```

CAMS global ensemble. Hourly, ~0.1° (≈10 km) grid, 92 days of lookback in a
single request — verified returning 2,256 hours for Rajkot. No key, no account.

Why it carries the prototype: it is the only keyless source that gives a
**multi-species** hourly series including mineral dust, which is what makes a
coarse/fine split and a combustion tracer possible at all.

What it is not: a reference monitor. See the caveat below.

### Open-Meteo Forecast API — the meteorological field

```
https://api.open-meteo.com/v1/forecast
  ?latitude=..&longitude=..
  &hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,boundary_layer_height
  &wind_speed_unit=ms&past_days=92&forecast_days=2
```

IFS. Wind at 10 m in m/s, which is what the adjacency graph needs.
`boundary_layer_height` is fetched because inversion is the mechanism behind a
winter episode, even though nothing currently consumes it.

### Static reference values, encoded in `configs/`

| Value | Source |
| --- | --- |
| NAAQS PM2.5 40/60, PM10 60/100 µg/m³ (annual/24-h) | CPCB notification B-29016/20/90/PCI-L, 18 Nov 2009 |
| WHO 2021 AQG PM2.5 5/15, PM10 15/45 | WHO global air quality guidelines, ISBN 978-92-4-003422-8 |
| CPCB AQI bands and sub-index interpolation | cpcb.gov.in National Air Quality Index |
| Lever effect sizes and half-lives | cited per-lever in `configs/action_table.yaml` |
| Source size signatures | cited per-source in `configs/signatures.yaml` |

## Attempted and unavailable

### CPCB CAAQMS — the intended reference

`src/setu/io/cpcb.py` is a real adapter against the documented public CSV
endpoints, needing no key:

```
https://app.cpcb.gov.in/CAAQM/CSVHourlyData.aspx?StationName=..&City=..&State=..&Date=DD/MM/YYYY&Format=DD/MM/YYYY
https://airquality.cpcb.gov.in/CAAQM/CSVHourlyData.aspx?...
```

From the development network `app.cpcb.gov.in` refuses the connection outright
and `airquality.cpcb.gov.in` returns 404 for every report path tried. The adapter
tries each candidate host and path and the aggregate error names each one, so
"unreachable" is evidence rather than an assertion. `python -m setu
probe-upstream` re-runs that probe and prints the result; the validation report
calls the same function so a reader sees what the developers saw.

**Consequence, stated where it matters:** the calibration target is CAMS
reanalysis at ~10 km, not a β-gauge monitor. Every error figure in `results/`
therefore validates the software against known ground truth and is **not** an
instrument-compliance result. A 10 km model also cannot be the reference for a
320 m grid — which is precisely the scale this project argues matters.

## Deliberately not used

| Source | Why not |
| --- | --- |
| OpenAQ v3 | requires an API key as of 2025 |
| WAQI / aqicn.org | requires a token; also converts to the US EPA AQI scale rather than CPCB's |
| IQAir | commercial key |
| NASA FIRMS fire detections | API key; open-burning corroboration currently comes from the CO and fine-mode signature instead |
| data.gov.in resource API | key required |
| CPCB annual "Compendium" PDFs | real but static; no hourly signal to validate against |

## Reproducibility

`src/setu/io/cache.py` writes every response to `data/cache/` with its URL and
fetch time, and serves it for six hours. `SETU_OFFLINE=1` forbids all network
egress, so the pipeline can be run on a machine without network access
against whatever was cached last. `SETU_CACHE_DIR` points the cache somewhere
else without a code change.

The daily-work cache is gitignored. **A snapshot of it is not**:
`data/fixtures/upstream/` holds the real Open-Meteo responses the Rajkot pilot
window used, verbatim, with their `.meta.json` provenance records. That is what
makes the committed `results/` reproducible offline:

```bash
SETU_CACHE_DIR=data/fixtures/upstream SETU_OFFLINE=1 python -m setu run --threshold 15
```

The CI build is written to use the same path, which is why it needs no secrets —
and why the end-to-end workflow can be reproduced offline. See
`data/fixtures/upstream/README.md` for the fetch dates, row counts and licence
attribution, and `docs/VALIDATION_PROTOCOL.md` §4 for what is currently running.

The tests that need upstream-shaped data use stubs rather than the snapshot, so
`pytest` passes with no network and no cache at all.

## Rate and etiquette

Two requests per site per six hours in normal operation. The User-Agent
identifies the project and links to this repository. Nothing here is scraped,
nothing is behind a login, and no terms are worked around.
