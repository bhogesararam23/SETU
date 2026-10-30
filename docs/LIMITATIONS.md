# Limitations

Written first, not last, because the honest boundary of this prototype is the
most useful thing in it.

## 1. No hardware has been built

The most important limitation. Nothing in this repository has measured real air
with a real low-cost sensor. There is no SPS30 on a bench, no flashed ESP32, no
co-location campaign.

What exists instead is a **digital twin**: real hourly CAMS air quality and
meteorology for Rajkot, distributed over a 40-node lattice, given point sources
with published size modes, and passed through the documented failure modes of an
optical particle counter — hygroscopic over-read, per-node gain and dark-count
error, gain drift, inlet obstruction, counting noise, dead optical bench.

That makes the calibration, attribution and response layers testable in the only
sense currently available: does the pipeline recover a signal whose ground truth
is known, and does it survive artefacts that were added on purpose. It does not
make any error figure here an instrument-compliance result.

**What would change if real sensors were attached:** sensor-to-sensor variability
would be larger than the seeded spread; drift would be non-monotone and
humidity-coupled rather than a clean ramp; the number bins would have counting
efficiency curves we have modelled as ideal. Expect the calibration MAE to worsen
and the attribution accuracy to fall, most likely below the 0.45 posterior floor
for the construction/road-dust pair, which are the two hardest to separate.

## 2. The reference is a model, not a monitor

CPCB's public endpoints are unreachable from the network this was developed on —
`app.cpcb.gov.in` refuses the connection and the mirror 404s every report path.
`src/setu/io/cpcb.py` is a real adapter against the documented CSV endpoints with
a `reachability_report()` that records what happened, but it has never returned a
row.

So the calibration target is CAMS reanalysis at ~0.1° (~10 km), not a β-gauge
station. Consequences:

- A 10 km model cannot be the reference for a 320 m grid. The spatial argument
  this project is built on is precisely the thing the current reference cannot
  resolve.
- Absolute concentrations carry CAMS bias, which for Indian winter episodes is
  reported in the literature at −17 µg/m³ by day three of a forecast.
- Every metric in `results/` validates the algorithms against the twin's own
  ground truth. That is a real test of the software and no test at all of the
  instrument.

The first field action is a 60–90 day co-location against one CPCB CAAQMS.
Nothing else in the plan is worth doing before that.

## 3. The share cross-check passes because the twin was tuned to pass it

An earlier run of this report failed the share cross-check for two sources:
road dust at 59.5% of attributed hours against a 2–45% published band, and
traffic combustion at 2.8% against 5–25%. The cause was the twin, not the
classifier: the generator injected a road-dust increment at any node whose
seeded road factor exceeded 1.0, which was four nodes in five, so `road_dust`
was simply the most frequent label in the record and no classifier trained on
it could have passed.

The fix, named here before it was done, was to tune the injection rates to a
published apportionment. That is now the default: road re-suspension clears its
threshold at about half the lattice, traffic exhaust adds enough at peak NO₂ to
win 14–24% of classifiable hours depending on the window, burning and
construction occur often enough to register, and secondary aerosol no longer
wins over half the record. `tests/test_twin.py` asserts the generator's own
label shares sit inside the published bands on the 30-day fixture, so a change
that quietly re-breaks the apportionment fails a test rather than a report
nobody reads.

What this does and does not prove:

- It proves the classifier can recover a literature-shaped apportionment from
  size fractions, gases and wind. That is a real test of discrimination, and
  the coefficient signs and permutation importances above are what stop it
  being circular.
- It does **not** make the shares a measurement of Rajkot. The bands are
  seasonal Delhi-NCR ranges applied to a monsoon window in Gujarat; the tuning
  matches the bands, not the city.
- The classifier still inflates road dust relative to the twin's own labels on
  some windows, because coarse-dominated background hours resemble it. The
  reject classes decline the hours with nothing injected; they do not separate
  background dust from re-suspended dust, and nothing short of measuring the
  road can.

## 4. Triggering earns nothing in two of three configurations

The equal-cost comparison is reported for both the favourable and the
unfavourable case:

| Basis | Sprinkling (62 min) | Sweeping (150 min) | CaCl₂ (~20 h) |
| --- | --- | --- | --- |
| Episodic sources (construction, burning) | **+0.12** | **+0.18** | 0.00 |
| Continuous road-proximate dust | **0.00** | 0.00 | 0.00 |

A lever that lasts a day does not care where you put it. A source that runs
around the clock cannot be fixed by scheduling at all — only by a longer-lived
lever or by removing the emission. So the value of SETU is conditional on a
short-lived lever acting on an episodic source, and the system reports a
zero-gain result rather than hiding it.

## 5. A calibration does not survive a season

On the 92-day run, most of the grid tripped its drift monitor within the
evaluation window. The calibration was fitted on the monsoon and scored on the
post-monsoon period, and the residual relationship moved with it.

That is a real operational finding, not a bug to be tuned away: recalibration is
periodic, not one-off. The cost stays manageable because only the co-location
step needs a vehicle — one reference visit refreshes every node transferred from
it, which is the practical argument for pair-based transfer over fitting each node
independently.

## 6. Interval coverage is under-nominal

The 10th–90th quantile band covers 0.658 of held-out observations against a 0.80
nominal. The response engine's third gate refuses to dispatch while the interval
straddles the threshold, so an under-covering interval means that gate is
slightly more permissive than advertised. Widening to 5/95 or applying a
conformal correction would fix it; it has not been done because the current
behaviour is already reported honestly.

## 7. Attribution is evaluated against the generator that made it

The classifier is trained and scored on labels produced by the twin's own
injection model. It is therefore validated as "can this recover the truth that
was put in", which is the right test for the software and the wrong test for the
science. The real question — do published size signatures actually separate
construction dust from road dust in what a ₹3,099 sensor reports — cannot be
answered without the co-location campaign in §2.

## 8. Adaptation to another city is a config change in code, not in practice

`configs/rajkot.yaml` and `configs/action_table.yaml` are the whole per-city
surface, and swapping them does produce a working run. What does not come free is
municipal integration: a signal-retiming action needs an ITS operator's API and a
legal basis, a site notice needs a register of construction sites to reference,
and a patrol route needs someone to receive it. The action table is portable; the
permissions are not.

## 9. The console is a research instrument, not a deployable service

Milestone 8 turned the operator dashboard into a decision-support console. What
it did **not** do is make it safe to expose, and the gap is listed here rather
than left for a reader to discover.

- **There is no authentication or authorization.** The service binds
  `127.0.0.1` by default, is read-only, holds no personal data, and has no
  endpoint that dispatches anything. That is a loopback research console, not a
  multi-user system. Before it is reachable from a municipal network it needs a
  reverse proxy that authenticates and authorizes the desk reading it, and the
  `owner` field is a printed desk name rather than an access-control list.
- **There is no real-time feed, and the console does not pretend there is.** It
  renders a completed `python -m setu run`. The header shows the run's file time,
  `/api/run` reports its age, and past seven days the console marks the payload
  stale. Live monitoring is not implemented because there is no hardware feeding
  it; a poller over a static file would be theatre.
- **Fleet capacity is not modelled.** The queue is an ordering by expected
  ug/m3 shaved per pass, not a schedule. `dispatch_plan` can defer rows against a
  capacity, but the prototype has no crew count, so the queue payload reports
  `capacity_modelled: false` and the console says so instead of inventing a
  deferral time.
- **The queue is a page, not the whole backlog.** The run keeps the 200 highest
  priority dispatches out of 8,047, and the filters apply inside that page. The
  displayed totals distinguish the page from the run for exactly this reason.
- **Evidence packs cover the queue head (60), not every dispatch.** Writing
  8,047 JSON files into a git repository is a poor trade; the rows without a pack
  say `no pack` rather than pretending otherwise.
- The console renders numbers; it never recomputes them. Skill, coverage,
  calibration and attribution are read from the run payload and are subject to
  every limitation above.

## 10. Small things a reviewer will notice

- `client.setInsecure()` in the firmware. Cert pinning is a real TODO.
- `readBme()` and `readWind()` are stubs.
- The forecast is gradient-boosted trees, not the graph neural network the design
  argues for. At 40 nodes and 92 days (~58k node-hours) a GNN is the right
  architecture for the deployed system and the wrong one for this dataset; the
  literature results quoted in the write-up come from dense reference networks and
  are not comparable to the skill figures here.
- **The forecast does not beat persistence across the window.** Scored
  walk-forward, pooled skill is -0.028 / -0.092 / -0.016 at 1/3/6 h, and only 3 of
  40 nodes win outright at one hour. Only the fold nearest its own training data
  is positive (+0.167 / +0.282 / +0.300). Read per fold that says the model is
  worth using exactly as long as it is refitted on current data, which is what the
  deployment does; read pooled it says this is not yet a forecast to dispatch on
  without persistence alongside it. An earlier version of this project quoted
  +0.33/+0.57/+0.68, which was in-sample.
- **The forecast band under-covers out of sample.** Split conformal at a nominal
  0.80 measures 0.658-0.729, because the band is calibrated on the slice before a
  regime shift that the calibration slice does not see. The response engine's
  uncertainty gate is tuned to that band, so it is stricter in practice than its
  nominal level suggests.
- **Intervention effects have not been measured on anything.** The framework is
  built - before/after against an untreated-node control, and it refuses a window
  shorter than its reporting cadence - but every sample returns
  `insufficient_data` on this data: the twin simulates no intervention, and hourly
  cadence cannot resolve a 62-minute lever. A half-life claimed from this data
  would be invented, so none is claimed.
- **The equal-cost comparison rests on a footprint model**, not on measured
  downwind decay: one pass is credited with `ceil(half-life / 60)` hours of
  coverage, uniformly over the day. The comparison is a first-order argument, and
  the suppressant row exists to show where it stops being informative.
- The `regional_background` and `mixed` labels the twin emits are deliberately
  excluded from classifier accuracy, since the model has no class for them. That
  raises the reported accuracy and is disclosed here rather than buried.
- Node coordinates are a synthetic lattice centred on the city, not surveyed
  pole positions.
