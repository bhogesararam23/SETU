# SETU

**S**ource-attributed, **E**vidence-graded **T**rigger for **U**rban air.
*setu* — Sanskrit सेतु, a bridge.

A prototype decision layer for urban air quality: it takes a hyperlocal sensor
grid, works out **which source** is causing a breach at **which street**, and
doses the municipality's existing levers to the measured decay curve of each one.

An open-source research project on urban air quality, anchored on Rajkot, Gujarat. SETU explores how source attribution, uncertainty-aware calibration, and measured intervention effects can support more local air-quality decisions.

> India does not have an air-data problem. It has an air-decision problem.

---

## The gap this targets

Three facts that do not currently fit together:

1. **Coverage.** India ran fewer than 600 continuous air-quality stations for
   ~1.44 billion people in 2024. The World Bank counts roughly one monitor per
   0.3 million people in high-income countries. 28 of the 130 NCAP cities have no
   continuous station at all.
2. **Aggregation.** A city AQI is one number. ARAI's receptor study measured
   winter PM2.5 across 20 Delhi-NCR sites at a mean of 168 ± 45 µg/m³ with a
   range of 92–254 — a **2.8× spread reported as a single value**.
3. **Response.** The Graded Response Action Plan is the only statutory mechanism
   that turns a reading into an action, it is legally confined to the NCR, and it
   triggers on the city average. Its measured effect has been characterised as
   *"limited and meteorology-dependent"*. The other 129 NCAP cities have nothing
   equivalent.

So action is triggered late, applied uniformly to every street and every resident,
and never evaluated.

## What SETU does

```
   real background                     per-node, per hour
   ┌──────────────┐   ┌───────────┐   ┌──────────────────────────────┐
   │ CAMS PM2.5,  │   │ 40-node   │   │ 1  hygroscopic correction    │
   │ PM10, CO,    ├──►│ lattice   ├──►│ 2  calibration to a reference│
   │ NO2, O3, dust│   │ @ 320 m   │   │ 3  drift quarantine (CUSUM)  │
   │ + IFS wind,  │   │ SPS30 +   │   └──────────────────────────────┘
   │ RH, T        │   │ PMS5003   │        ┌─────────────────────────┐
   └──────────────┘   └───────────┘        │ source posterior + σ    │
        ▲                                  │ road dust / construction│
        │                                  │ burning / secondary     │
   ┌────┴─────────┐   ┌────────────────┐   └────────────┬────────────┘
   │ re-measure   │   │ graded local   │                ▼
   │ the treated  │◄──│ response:      │   ┌─────────────────────────┐
   │ stretch at   │   │ dispatch +     │   │ wind-aware graph        │
   │ 45 min       │   │ owner + σ      │◄──│ nowcast, 1/3/6 h        │
   └──────────────┘   └────────────────┘   └─────────────────────────┘
        ▲                      │                    ▲
        └──────────────────────┴────────────────────┘
              every intervention is an experiment
```

Three layers, each independently testable:

- **Perceive** — a node built around the fraction nobody else measures. The
  SPS30 reports four mass channels (PM1/2.5/4/10), so coarse PM₂.₅–₁₀ — the
  road-dust channel — is *measured*, not inferred. Most low-cost units and every
  consumer app report only PM2.5 and are structurally blind to it.
- **Attribute** — calibration that carries an uncertainty a municipal officer can
  act on and defend, then a learned source classifier over size fractions, gases
  and wind. This is the novelty claim, and it is narrow: the state of the art
  (Kaur, Mangin & Kelly, *AMT* 19:1077, 2026) uses a fixed coarse/ultrafine
  threshold to *flag* dust and is explicitly not a learned classifier.
- **Act** — a graded local response table keyed to source, not to a city average,
  with each lever re-applied on a fraction of its **measured** half-life.

## The result that makes triggering worth building at all

Every lever has a published decay constant, and they are brutal:

| Lever | Measured effect | How long it lasts | Source |
| --- | --- | --- | --- |
| Water sprinkling, arterial | 37–53% ambient PM | **~62 min**, gone by 72 | *J. Environ. Sci.* 2023, doi:10.1016/j.jes.2021.12.005 |
| Sweeping + flushing | up to 24% PM10 | 2–3 h | AIRUSE Report 15, 2016 |
| CaCl₂ at 30 g/m² | 97.6% PM10 emission factor | clearly worse at 24 h | *J. Air Pollut. Health* 8(4), 2023 |

Delhi's measured road silt load runs 2.0–12.5 g/m² against a recommended ceiling
under 2 — a six-fold spread no city average can see. So the product is not
"sprinkle more". It is dose to the decay curve, which is only possible if you
know which street, which source, and when.

## Validation

`python -m setu run` produces `results/VALIDATION.md`. Current headline, 40 nodes
over 1,772 hours of real Rajkot background:

| | Raw sensor | Calibrated |
| --- | --- | --- |
| PM2.5 MAE | 9.85 µg/m³ | **1.82 µg/m³** (−81.5%) |
| PM2.5 R² | −1.56 | **0.76** |
| Bias | +7.65 | **−0.55** |
| Threshold accuracy @ 15 µg/m³ | 0.74 | **0.93** |

- **Transfer** — 38 nodes calibrated by pair-based transfer without ever seeing a
  reference: MAE 9.89 → 1.90 µg/m³, overlap R² 0.971. Scored by donor distance
  rather than averaged over it: MAE 1.13 within one lattice pitch, 2.06 beyond
  3.5 pitches (mean 1.8 km, out to 2.5 km), Pearson r between distance and error
  **+0.263**. Transfer is not co-location in disguise, and that is the number
  which would have shown it if it were.
- **Attribution** — 0.789 accuracy on 22,720 node-hours against 0.200 chance;
  0.820 above a 0.60 posterior, on 90.4% of hours. Coefficient signs match the
  published size signatures.
- **Forecast** — 1 h RMSE 1.54 vs persistence 2.36 over 69,680 node-hours.
- **Response** — over the full 40-node grid: 14,224 dispatches, 56,656 suppressed, every one carrying a logged gate reason. That is 4.8 dispatches per zone per day, which is a lot and is written up in `docs/LIMITATIONS.md` rather than smoothed over. The figure used to read 739/3,299, which was wrong: a display cap was silently stopping the engine after about three nodes.

**Read [docs/LIMITATIONS.md](docs/LIMITATIONS.md) before quoting any of them.**
In short: no hardware exists, so the sensors are a digital twin over real
background; CPCB was unreachable, so the reference is CAMS at ~10 km rather than
a β-gauge; the source-share cross-check fails for two sources for a stated
reason; and triggering earns **nothing** on a long-lived lever or a chronic
source, which is reported in the same table as the case where it does.

## Running it

No API keys, no accounts, no paid services.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=src    # or pip install -e .
                         # Windows cmd: set PYTHONPATH=src

pytest                       # 209 tests, no network needed
python -m setu check         # fast end-to-end run, headline numbers to stdout
python -m setu run           # full run, writes results/VALIDATION.md + .json
python -m setu serve         # operator console on http://127.0.0.1:8090
python -m setu probe-upstream  # which data sources are reachable from here
python -m setu bom           # lever table, gates, and the do-not-build list
```

To reproduce `results/` exactly as committed - no network, no keys, same numbers
the README quotes:

```bash
SETU_CACHE_DIR=data/fixtures/upstream SETU_OFFLINE=1 \
  python -m setu run --threshold 15
```

`--threshold 15` matters. It is the WHO 2021 24-hour guideline, and it is what
the headline dispatch counts were produced at. Left out, the run defaults to the
Indian NAAQS 24-hour limit of 60 µg/m³ - which Rajkot's monsoon and post-monsoon
window never reaches, so the response layer correctly does nothing at all. That
is a real finding about a clean season, and it is reported as such.

`data/fixtures/upstream/` is a committed snapshot of the real Open-Meteo
responses used by the reproducible validation run. It allows the full pipeline
to be re-run without credentials or network access.

The local operator console runs at `http://127.0.0.1:8090` by default.

## What is real and what is not

| | Real | Simulated |
| --- | --- | --- |
| Hourly PM2.5, PM10, CO, NO₂, O₃, SO₂, mineral dust, AOD | ✅ CAMS global ensemble via Open-Meteo, 92 days back | — |
| Temperature, RH, 10 m wind vector, PBL height | ✅ IFS via Open-Meteo | — |
| India NAAQS, WHO 2021 guidelines, CPCB AQI breakpoints | ✅ notified values | — |
| Lever effect sizes and half-lives | ✅ published, cited inline in `configs/action_table.yaml` | — |
| Sensor readings, size channels, drift, dead benches | — | ✅ digital twin, `src/setu/twin.py` |
| The 40-node lattice | — | ✅ synthetic, centred on the city |
| CPCB reference observations | adapter written | ❌ endpoint unreachable from the dev network |
| Hardware | BOM priced, firmware written | ❌ never built or flashed |

Everything simulated is marked `provenance: digital-twin` in the payloads.

## Layout

```
configs/       per-city files: site, source signatures, action table  ← citations live here
src/setu/
  io/          keyless fetchers + auditable HTTP cache
  grid.py      node placement, wind-aware directed graph
  twin.py      digital twin over the real background
  calibrate/   hygroscopic correction, regression, transfer, CUSUM quarantine
  attribute/   size-fraction source posterior with carried uncertainty
  forecast/    graph nowcast and 1/3/6 h horizon
  respond/     graded local response engine, dose schedule, effect scoring
  metrics.py   error + threshold-classification statistics
  pipeline.py  end-to-end orchestration
  report.py    VALIDATION.md / validation.json / evidence packs
  cli.py       argparse entry point
  api/         FastAPI operator service
dashboard/     operator console, no build step
firmware/      ESP32-S3 node, BOM, wiring — designed, not built
tests/         209 tests, all offline - HTTP is faked, so CI needs no network
docs/          architecture, data sources, validation protocol, limitations
results/       generated output, committed so the repo is readable without running it
data/fixtures/ committed snapshot of the real upstream responses (see its README)
```

Automated GitHub Actions CI is not currently configured on `main`. Before
relying on a code change, run the test suite and the offline end-to-end replay
locally; `docs/VALIDATION_PROTOCOL.md` documents the validation protocol and
its current evidence.

## Design decisions worth arguing about

Recorded with their reasoning in `docs/ARCHITECTURE.md`. The short version:

- **Refuse to act on secondary aerosol.** It is the single largest winter source
  in the Delhi receptor studies and 84% of it arrives from beyond 30 km. A system
  that dispatches a sweeper against advection spends the budget and forfeits the
  credibility, so `secondary` maps to an empty action list on purpose.
- **A do-not-build list next to the policy.** Smog towers (CAQM and NEERI advised
  the NGT they are ineffective; IIT Bombay withdrew from the Delhi pilot),
  blanket odd-even (measured to have cut nothing), default tree belts (can raise
  street-canyon PM by trapping).
- **Every suppressed trigger is logged.** A dispatcher who gets the same alert
  forty times in a night stops reading it.
- **Human sign-off on every notice.** The system produces an evidence pack with σ
  attached; it never produces a penalty.
- **Gradient-boosted trees, not a GNN**, despite the design arguing for a graph
  model — because ~58k node-hours is the wrong dataset size for that
  architecture, and saying so beats quietly overfitting something impressive.

## Maintainers and contributors

- **Ram Bhogesara** — project maintainer
- **Bhavik Pandit** — project contributor

SETU is developed as an open-source research project based in Gujarat, India.

## License

MIT.
