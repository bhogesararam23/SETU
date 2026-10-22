# Committed upstream snapshot

Real responses from Open-Meteo, saved verbatim, so that the pipeline can be run
with **no network at all**:

```bash
SETU_CACHE_DIR=data/fixtures/upstream SETU_OFFLINE=1 python -m setu check
```

`SETU_OFFLINE=1` makes the cache serve a copy past its TTL instead of refusing,
which is what a snapshot needs. This is the path the CI build is written to use,
for exactly this reason; see `docs/VALIDATION_PROTOCOL.md` §4 for which parts of
that build are live and which are still waiting on a push permission.

## What these files are

Filename-sha256-of-the-request-URL, as written by `src/setu/io/cache.py`. Each
`.json` is the response body; each `.meta.json` records the request URL and the
UTC fetch time, and is the provenance record the pipeline reports.

| Request | Window | Rows | Fetched (UTC) |
| --- | --- | --- | --- |
| CAMS air quality, Rajkot | `past_days=92` | 2,208 hourly | 2026-10-10 21:41 |
| IFS met (T, RH, wind, pressure, PBLH), Rajkot | `past_days=92` | 2,208 hourly | 2026-10-10 21:41 |
| Same two APIs | `past_days=60` | 1,440 | 2026-10-10 21:34 |
| Same two APIs | `past_days=45` | 1,080 | 2026-10-10 22:26 |
| Same two APIs | `past_days=30` | 720 | 2026-10-10 21:20 |

Four lookbacks are cached because the request URL contains `past_days`, and the
URL hash is the cache key - so `setu run --days 92` and `setu check --days 45`
are different requests to the cache, and CI needs whichever one it asks for.
Coordinates are the Rajkot site centre from `configs/rajkot.yaml`.

## Why a snapshot rather than a live call in CI

- **Reproducibility.** `results/VALIDATION.md` quotes numbers from a fixed
  background window. CI re-fetching live data would compare today's CAMS run
  against a committed report and fail for reasons that have nothing to do with
  the code.
- **Politeness.** The user-agent identifies this project; the etiquette note in
  `docs/DATA_SOURCES.md` says fetch at reasonable rates during development; CI and tests should use fixtures. A
  cached fixture fetches zero times.
- **Availability.** Open-Meteo is generous but it is not ours to depend on for a
  run that depends on a live network connection.

## What this does NOT make real

The snapshot is real model output, not real measurement. CAMS global ensemble is
a ~10 km analysis; it is background signal, not a street-level reference. The
distinction is stated in `docs/LIMITATIONS.md` and in the header of every
report, and committing these files does not change it.

Redistribution: Open-Meteo's API responses are CC-BY-4.0 as delivered, built on
CAMS (Copernicus Atmosphere Monitoring Service, ECMWF) and IFS data. Attribution
kept here as well as in `docs/DATA_SOURCES.md`.
