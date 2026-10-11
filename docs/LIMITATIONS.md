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

## 3. The share cross-check fails, and the cause is known

SETU reports road dust at 59.5% of attributed hours where receptor studies put
dust at 23–45% of PM10, and traffic combustion at 2.8% against a 5–25% range.

The cause is twin fidelity, not classifier discrimination: the twin injects a
road-dust increment at any node whose seeded road factor exceeds 1.0, which is
roughly half the lattice, so `road_dust` is simply the most frequent labelled
source in the synthetic record. Comparing its share to a city apportionment is
comparing the twin's parameterisation to a receptor model.

Within-class discrimination is measured separately and holds (0.789 against 0.200
chance, rising to 0.820 above a 0.60 posterior, with coefficient signs matching
the published physics). Absolute shares cannot be validated until the twin's
injection rates are tuned to a published apportionment — the fix is to draw
event frequencies from the ARAI/CPCB and CAQM seasonal shares rather than from
flat seeded probabilities.

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

## 9. Small things a reviewer will notice

- `client.setInsecure()` in the firmware. Cert pinning is a real TODO.
- `readBme()` and `readWind()` are stubs.
- The forecast is gradient-boosted trees, not the graph neural network the design
  argues for. At 40 nodes and 92 days (~58k node-hours) a GNN is the right
  architecture for the deployed system and the wrong one for this dataset; the
  literature results quoted in the deck come from dense reference networks and
  are not comparable to the skill figures here.
- The `regional_background` and `mixed` labels the twin emits are deliberately
  excluded from classifier accuracy, since the model has no class for them. That
  raises the reported accuracy and is disclosed here rather than buried.
- Node coordinates are a synthetic lattice centred on the city, not surveyed
  pole positions.
