# Architecture

## Layering

Each layer is a pure function of the layer below plus a config file, so each can
be tested against a published number without running the others.

```
configs/*.yaml ──► config.py (validates, refuses silent defaults)
                          │
io/openmeteo ──► hourly frame ──► grid.py ──► twin.py ──► calibrate/ ──► attribute/
 (real CAMS+IFS)                             (digital     (raw→ref +     (posterior
                                              twin)        σ, transfer,    + σ)
                                                           CUSUM)               │
io/cpcb ──► reference (unreachable)                                            ▼
                                                            respond/engine.py
                                                            (gates → dispatch
                                                             → re-measure)
```

## Decisions and their reasons

### Why the coarse fraction is a hardware decision, not a software one

PM10 − PM2.5 is the road-dust channel. A sensor that reports only PM2.5 cannot
see the target source at all, and that is what most low-cost units and every
consumer app do. The SPS30 is chosen because it is the only common low-cost
device with four mass channels (PM1/2.5/4/10), so the coarse mode is measured
rather than inferred, and the difference is taken on-device from one sample
volume — computed in the cloud it inherits the reporting jitter of both channels.

### Why the response is keyed to source rather than to concentration

Because the same concentration from different sources needs opposite actions.
Secondary inorganic aerosol was 32% of winter PM2.5 at the Delhi supersite and
84% of it arrives from beyond 30 km (AQuaIM/DPCC, 2023). Dispatching a sweeper
against advection spends the budget and forfeits the credibility. So
`secondary` maps to an empty action list, deliberately, and the engine reports
`refused_by_evidence` rather than inventing a lever.

### Why uncertainty is carried end to end instead of computed on request

The response engine's third gate refuses to fire while the interval straddles the
threshold. That is the only reason a municipal officer can act on a low-cost
number and defend the act — and it directly answers CPCB's stated objection that
low-cost accuracy, linearity, reliability and long-term performance are "not yet
fully established". Two quantile regressors are fitted alongside the median so σ
is a property of the prediction, not a post-hoc confidence score.

Coverage is currently 0.658 against a 0.80 nominal, so the gate is slightly more
permissive than advertised. Tracked in `docs/LIMITATIONS.md`.

### Why the drift monitor is centred on its in-control mean

A node that is stably 1.5 µg/m³ high has a calibration offset, not drift.
Accumulating a CUSUM on the raw residual quarantines it eventually no matter how
well behaved it is — which is what happened on the first full run, where 40 of 49
healthy nodes were pulled out of the grid. Centring on the warm-up mean makes the
monitor detect *change*, which is the thing that actually invalidates a
calibration.

### Why gradient-boosted trees and not a graph neural network

The design argument is a graph model, and the literature supports it (Chen et al.,
*Sci. Rep.* 14, 2024: RMSE 19.11, R² 0.84 at 7 h, 8.6% better than an LSTM). But
~58k node-hours is the wrong dataset size for that architecture, and the published
figures come from dense reference networks on clean inputs. A boosted-tree
regressor per horizon with wind-weighted neighbour features is auditable, trains
in seconds, and can be honestly compared to a persistence baseline. The GNN is
the right thing to try once there are months of real node data — not before.

### Why the twin labels by what was injected, not by the resulting ratio

If the ground-truth label were derived from a band on the coarse/fine ratio, the
classifier would be graded against a restatement of its own input feature. So the
twin tracks each injected increment separately and the label is the argmax of
what was actually added. Hours where nothing was injected are labelled
`regional_background`, and hours where two sources are within 55/45 of each other
are labelled `mixed` — the classifier is allowed to be told it has no business
guessing.

### Why transfer calibration exists at all

Reference stations cost roughly ₹7 lakh to buy and ₹15 lakh a year to run. Fitting
each node independently means coverage scales with reference-station budgets,
which is the bottleneck the project exists to get around. Pair-based transfer lets
one co-located donor teach 38 neighbours: MAE 9.89 → 1.90 µg/m³ on the current
run.

The first implementation mapped channel-to-channel and reused the donor's model.
That is wrong — it assumes both nodes breathe the same mixture, and it made
things *worse* (MAE 5.24 → 10.17). The fix was to learn raw-channels → donor's
calibrated output instead.

That failure was invisible in a summary MAE, which is why transfer is now
reported banded by donor distance rather than averaged: the mechanism only has to
survive *distance* to be worth anything, and one number cannot show either the
survival or the collapse. Current run: 1.13 µg/m³ within one pitch, 2.06 µg/m³
beyond 3.5 pitches (mean 1.79 km), r = +0.263. Skill decays with distance and
does not become co-location; both halves are in the same table.

### Why the do-not-build list is in the config, next to the policy

Because a well-meaning contributor will otherwise add a smog tower in a pull
request. The published reasons are in `configs/action_table.yaml` under
`refusals:` where they will be read at the moment someone is deciding what to
add, not in a document nobody opens.

### Why suppression is logged as carefully as dispatch

A dispatcher who receives the same alert forty times in a night stops reading it.
That is how pilot monitoring systems die. So every non-action carries a reason
code, the ledger is on the dashboard, and the validation report tabulates it —
which lets the gates be audited for what they cost as well as what they prevented.

### Why the report leads with what it cannot support

A report that opens with its own strengths is read as marketing. `report.py`
emits a "what this run can and cannot support" table before any accuracy number,
and the equal-cost section reports the configuration where triggering gains
nothing alongside the one where it gains something.

## Failure posture

| Failure | Detection | Behaviour |
| --- | --- | --- |
| Dead optical bench | reports plausible zeros → health flag, not a value check | node quarantined, stays visible on the dashboard |
| Inlet obstruction | reads low vs upwind neighbours with elevated variance | flagged, down-weighted |
| Gain drift | CUSUM on centred residual | quarantined, recalibration scheduled |
| Upstream unreachable | `probe-upstream`, cache with staleness flag | falls back to cache; `SETU_OFFLINE=1` forbids egress entirely |
| Missing met history | leading hours dropped and counted | never forward-filled onto a concentration |
| Class absent from training | loader refuses, or caller narrows the class set | exclusion reported, priors renormalised |
| Config that would mislead | loader raises (priors not summing to 1, grid coarser than the gradient, k > n) | refuses to start |

The common rule: a missing value is never replaced by a plausible one.
