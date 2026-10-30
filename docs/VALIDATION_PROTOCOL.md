# Validation protocol

How every number in `results/VALIDATION.md` was produced, and what would have to
happen for it to be believed.

The rule applied throughout: **a claim is only as good as the test that could
have failed it.** Where a layer could not be given a test that could fail, it is
said so rather than decorated.

---

## 1. Reference hierarchy

Ranked by what they license. The prototype sits at level 2 and says so.

| Level | Reference | What it licenses | Status here |
| --- | --- | --- | --- |
| 0 | CEN/TR approved β-gauge or TEOM at the node | Instrument-compliance claims, regulatory substitution, "meets NAAQS" | ❌ not available |
| 1 | CPCB CAAQMS hourly reference, co-located 30 days | Calibration transfer, drift quotas, threshold accuracy at the statutory limit | ❌ adapter written, endpoint unreachable from the development network (`setu probe-upstream`) |
| 2 | CAMS global ensemble via Open-Meteo (~0.1°, ≈10 km) | Background signal with known hourly structure; a defensible *reference field* for a synthetic grid | ✅ used |
| 3 | Digital twin over level 2 | Whether the maths recovers a signal it was given | ✅ used, and the only thing the headline error figures measure |

Level 3 is why the report header says the error figures are **not**
instrument-compliance results. A reviewer is entitled to ask "calibrated against
what?" and the answer is in the report, not in a footnote.

---

## 2. The claim table

Each row is a claim in the write-up or README, the test that could have falsified
it, the baseline it is scored against, and the pass rule agreed before running.

### Calibration

| | |
| --- | --- |
| **Claim** | A learned map from raw low-cost channels to a reference recovers most of the error of an uncorrected sensor. |
| **Test** | Fit on the first 60% chronologically, score on the last 40%, per node, against twin truth. |
| **Baseline** | The node's own raw readings (identity map). |
| **Pass rule** | MAE must fall, and the fall must land inside the published range for the method family; otherwise the split or the features are wrong. |
| **Published anchor** | Bush et al., *AMT* 15:3261, 2022 — 37–94% MAE reduction for co-located low-cost calibration, R² up to 0.91. |
| **Result** | 10.41 → 3.21 µg/m³, −69.1%, inside the range. R² −1.06 → 0.59. |
| **Could have failed** | Yes. Scoring the window the model was fitted on produces an in-sample number that means nothing; the 60/40 chronological split is what forces 0.76. A leak of exactly that kind was found and fixed in the attribution features (a centred rolling median was reading the future), and `tests/test_attribute.py` now asserts NaNs appear only in the leading warm-up. |

### Uncertainty

| | |
| --- | --- |
| **Claim** | The interval attached to each reading is honest enough to gate a dispatch on. |
| **Test** | Two extra quantile regressions at the 10th and 90th percentile; count the share of held-out truths inside the band. Nominal coverage 0.80. |
| **Pass rule** | Coverage within ~0.05 of nominal. |
| **Result** | **0.658** — under-nominal, so this claim is currently **failed and reported**, not tuned. σ is derived from the 10–90 range via Z = 2.5631 and inherits the same shortfall. |
| **Consequence** | The uncertainty gate still refuses to dispatch when the band straddles the action level, but it refuses ~14% less often than it should. `docs/LIMITATIONS.md` §6. |

### Calibration transfer

| | |
| --- | --- |
| **Claim** | A node that never sees a reference can be calibrated from a donor that did — this is what makes coverage affordable. |
| **Test** | Two donors fitted on their own truth. The other 38 nodes are mapped from *their* raw channels onto the *donor's calibrated output* (never channel-to-channel, which silently assumes two identical pollutant mixtures). |
| **Baseline** | Naive raw error at the receiving node. |
| **Pass rule** | Post-transfer MAE below the raw MAE, plus a distance check: skill must not collapse with donor distance, or the "transfer" is co-location wearing a longer hat. |
| **Result** | 9.89 → 1.90 µg/m³, overlap R² 0.971. Banded by donor distance: 1.13 µg/m³ within 1.5 pitches (2 nodes), 1.08 at 1.5–2.5 (4), 1.99 at 2.5–3.5 (5), **2.06 beyond 3.5 pitches** (27 nodes, mean 1.79 km, furthest 2.50 km). Pearson r = **+0.263**. Skill does decay with distance — nearly doubling from the nearest to the furthest band — but nowhere near enough to make the mechanism co-location. Both facts are in the same table. |
| **Could have failed** | Yes, and it did: the first formulation made transfer *worse* than raw (5.24 → 10.17). That failure is what produced the raw-channels-to-donor-output design. |

### Attribution

| | |
| --- | --- |
| **Claim** | Size-fraction and gas features separate the dominant local source at a street. |
| **Test** | 5-class posterior over 22,720 node-hours, scored against the twin's injected-mode argmax. Chance level 0.200. |
| **Baselines** | Chance (0.200); prior-only (the published share vector); a fixed coarse/ultrafine threshold rule, which is what the current literature actually does. |
| **Pass rule** | Accuracy materially above chance **and** coefficient signs matching the published size signatures. Both are required: above-chance accuracy with anti-physics signs would be a shortcut, not a classifier. |
| **Result** | 0.600 accuracy against 0.20 chance; 0.838 above a 0.60 posterior on 66.8% of hours; log loss 1.7935; signs match (coarse count ratio +1.257 on dust, −0.803 on burning). |
| **Honest caveat** | The labels come from the generator (§7). This is a proof that the features carry source information, not that they identify real-world sources. |

### Forecast

| | |
| --- | --- |
| **Claim** | A refitted short-horizon model beats persistence in the hours it was fitted near, and beats a diurnal climatology throughout. |
| **Test** | Per-horizon models at 1/3/6 h, scored walk-forward in three folds. Each fold trains on everything before its score slice and calibrates its band on the slice just before that, so no scored hour is an hour the model saw. Features are own lags (0,1,2,3,6,12,24 h), cyclical time, met, and the wind-decayed upwind mean over the adjacency graph; the model fits the h-hour increment rather than the level. |
| **Baseline** | Persistence (today's value, unchanged) and an hour-of-week climatology. |
| **Pass rule** | Skill above zero in the fold nearest its own training data, and above the climatology in every fold. A model fitted once at the start of the window and never refitted losing at every horizon is a finding about recalibration, not a failed run — it is why the report prints the decay curve. |
| **Published anchor** | Chen et al., *Sci. Rep.* 2024: RMSE 19.11, R² 0.84 at 7 h — on real data and a much dirtier city. |
| **Result** | Pooled skill **-0.028 / -0.092 / -0.016** at 1/3/6 h: across the window as a whole the model does not beat persistence, and the report leads with that. Fold 0 scores **+0.167 / +0.282 / +0.300**; folds 1 and 2 are negative. Only 3 of 40 nodes beat persistence at 1 h. Against the climatology the model wins everywhere (+0.65 at 1 h pooled). Band coverage 0.658-0.729 against a nominal 0.80. |
| **Honest caveat** | An earlier version of this section quoted +0.33/+0.57/+0.68. Those were in-sample: one split, fitted and scored on the same regime. The number is not small, it was wrong, and the per-fold table is what replaced it. |

### Response

| | |
| --- | --- |
| **Claim** | Triggering beats a fixed roster at equal cost, when the lever is short-lived and the source is episodic. |
| **Test** | Both arms get the same passes per day. Roster spaces them from a 05:00 shift start; trigger places them on hours the grid actually breaches. Coverage compared per lever using its measured half-life. |
| **Baselines** | The roster, at equal cost. Scored on genuinely episodic breach hours (construction, open burning) **and** on the continuous road-dust case. |
| **Pass rule** | Positive gain on short-lived levers, ~zero on a 20-hour suppressant. A positive gain on the suppressant would mean the comparison is rigged. |
| **Result** | +0.12 and +0.18 on the 62-min and 150-min levers, **0.00** on the ~20 h suppressant, **+0.00** on chronic road dust. The suppressant saturates - it needs 1 pass of the 2 it is given, so that row is a saturated lever and not a draw. Over the full grid: 8,047 dispatches and 62,833 suppressions, each carrying the gate from the ordered chain that refused it (source 25,717, hysteresis 26,586, dwell 5,502, persistence 3,029, uncertainty 1,999) and, where a clock will clear it, the time it could next pass. |
| **Effect measurement** | `measure_effect` refuses to score an intervention without native-cadence data, and takes an untreated-node control so a sea breeze that clears the grid is not scored as a successful sweep. Hourly input is unscorable and the test asserts that. On the twin, 40 sampled dispatches return `insufficient_data` 40 times, which is the correct null: the twin simulates no intervention, and a positive effect here would mean the framework was reading weather. |

### Drift

| | |
| --- | --- |
| **Claim** | A node that decays can notice its own decay. |
| **Test** | Two-sided CUSUM on post-fit residuals only (allowance 0.5σ, decision interval 10σ, 336 h warm-up to estimate σ, 168 h cooldown). σ from the warm-up window alone, centred on the in-control mean. |
| **Pass rule** | Detect an injected step; alarm *rate* on clean noise near zero — asserted as a rate, not as zero, because a control chart with no false alarms is a control chart that isn't looking. |
| **Result** | 40 of 40 nodes alarmed inside the evaluation window. This is reported as a finding: a calibration fitted on the monsoon does not survive into post-monsoon, so recalibration is periodic, not one-off. Only 2 donors were fitted, so one reference visit refreshes every node transferred from it. |
| **Could have failed** | Yes. The first version quarantined 40 of 49 nodes because it accumulated on raw residuals — stable per-node offsets and the season boundary looked like drift. Centring on the in-control mean fixed the false part; the season part was real and is left in. |

---

## 3. Experimental hygiene rules

These are the rules the code enforces, not aspirations.

**Chronological splits only.** Every fit uses `train_frac=0.6` on the time axis.
A random split lets the model see next week's humidity when predicting today's;
on a seasonal signal that is the difference between validation and fiction.

**Trailing features only.** The rolling median excess used by the classifier is
trailing, never centred. A centred window leaks the future into training — that
leak was present, cost a few points of fabricated accuracy, and is now pinned by
a test asserting NaNs appear only in the leading warm-up window.

**Post-fit residuals only into the drift monitor.** Feeding it the training
window makes the split boundary itself look like a step change.

**Fixed seeds.** `random_state=20261010` on every estimator. A run that cannot
be reproduced is a run that cannot be audited.

**Refusal is a return value.** `ResponseEngine.update()` returns an `Action` or a
`Suppressed`, never `None`. Suppressions are counted by reason and the counts
must sum to the total — CI asserts this.

**Provenance on every payload.** Twin-derived fields carry
`provenance: digital-twin`. The dashboard shows a permanent banner saying so.

**Abstain instead of guessing.** The classifier returns `ambiguous` when no
posterior clears the margin; the engine will not act on a source it is not
confident about (`min_source_posterior: 0.45`).

**Config refuses silent defaults.** Priors that do not sum to 1.0, spacing over
1,000 m, or a reference count larger than the node count all raise rather than
guess.

---

## 4. Gates the build enforces

The build runs offline against `data/fixtures/upstream` — no keys, no accounts.
`regression-gates` fails it if:

1. calibrated MAE is not below raw MAE, or exceeds 5.0 µg/m³;
2. attribution accuracy is not at least 0.2 above chance;
3. transfer MAE is not below its own raw MAE;
4. the engine dispatches nothing (engine unwired) or suppresses nothing (gates unwired);
5. suppression reasons do not sum to the suppression count;
6. any forecast horizon loses to persistence;
7. the committed report header disagrees with the committed JSON;
8. a replay from the snapshot moves any committed number outside its stated
   tolerance (`python -m setu verify`).

The gates assert *relationships*, not exact numbers: BLAS and scikit-learn
releases move the third decimal, and a build that fails on 3.21 → 3.22 teaches
nobody anything. The replay gate is the exception where it can be: counts and
the input fingerprint must match exactly, and measured quantities carry a small
per-field tolerance rather than a blanket one.

**Status of the workflow itself.** The workflow is active on `main` and on
`milestone-*/**` branches; the three jobs above run on every push and on pull
requests. The replay job no longer just re-runs the pipeline - it compares the
fresh payload against `results/validation.json` and fails on a moved count, a
moved digest, or a metric outside its tolerance. An earlier version of this
section said the workflow could not be pushed for want of the `workflow` scope;
that was true when it was written and is not true now.

---

## 5. Reproduction

Exact command that produced the committed `results/`:

```bash
PYTHONPATH=src SETU_CACHE_DIR=data/fixtures/upstream SETU_OFFLINE=1 \
  python -m setu run --threshold 15
```

`--threshold 15` is the WHO 2021 24-hour guideline. Without it the run defaults
to the Indian NAAQS 24-hour limit of 60 µg/m³, which Rajkot's monsoon and
post-monsoon window never reaches, and the response layer correctly produces
zero dispatches. That is a real result about a clean season, not an implementation failure —
but it is not the configuration the headline numbers come from, so it is named.

`setu check` (45 days × 8 nodes) runs in about 25 s from the snapshot; the full
40-node run takes a few minutes. Neither touches the network.

The same replay is a check rather than a claim:

```bash
PYTHONPATH=src SETU_CACHE_DIR=data/fixtures/upstream SETU_OFFLINE=1 \
  python -m setu verify --days 92
```

`setu verify` reads the action level out of the committed payload (rather than
defaulting to 60 µg/m³ and comparing two different experiments), re-runs the
pipeline, and puts the fresh payload beside the committed one field by field:
the input fingerprint, the window and node count, dispatches, suppressions and
their gate totals, the queue head, mean calibration MAE, attribution accuracy,
and forecast skill at 1 h and 6 h. Counts and the digest are exact; measured
quantities have a stated tolerance, because a library upgrade moves the last
decimal without anything being wrong. A field the committed payload predates is
reported as skipped rather than silently passed.

The current committed output reproduces on all 12 comparable checks: 1,772
hours, 40 nodes, MAE 10.41 → 3.21, attribution 0.600, 8,047 dispatches and
62,833 suppressions. The counts reported here moved twice during development and
both moves were real: a `len(rows) > 4000` guard meant to bound the *displayed*
sample was sitting inside the engine loop, so the run scored roughly three nodes
and reported them as forty; and the sample spread that replaced it changed what
the console showed without changing the totals. That is why the guard on this
section is a comparison and not a remembered figure.

---

## 6. What would change the verdict

Stated as the observations that would break each claim, so that a reviewer can
see we would notice.

| If this were observed | Then this claim dies |
| --- | --- |
| A co-located CPCB reference shows calibrated MAE staying above ~15 µg/m³ | The twin's error model is too mild; the headline calibration result is an artefact of the generator |
| Transfer skill decays sharply with donor distance (r ≳ +0.6) | "Calibrate once, transfer everywhere" is really co-location; the cost model collapses |
| Coefficient signs stop matching the published size signatures | The classifier found a shortcut, not physics |
| Interval coverage stays near 0.66 after recalibration | Uncertainty cannot gate a municipal action; the engine must fall back to persistence + dwell only |
| Triggering shows positive gain on the ~20 h suppressant | The equal-cost comparison is rigged |
| Real sensor drift alarms at a rate far above the twin's | `TwinConfig.drift_fraction` is wrong and quarantine quotas are fiction |

---

## 7. Negative and inconclusive register

Reported because they are the part a reviewer trusts most.

1. **Share cross-check fails** for `road_dust` and `traffic_combustion` against
   published receptor ranges. Cause is known: the twin's injection rates were
   never tuned to a published apportionment. Until they are, source *shares* are
   not a validated quantity — only the *ranking* and the coefficient signs are.
2. **Interval coverage 0.658 vs 0.80 nominal.**
3. **Season transfer fails**: 40/40 drift alarms; a four-month-old calibration
   does not survive a monsoon-to-post-monsoon change.
4. **Equal-cost gain is 0.00** for long-lived levers and for chronic road dust —
   triggering earns nothing there, and that is in the same table as the wins.
5. **No hardware.** Firmware is written and never compiled or flashed;
   `readBme()` and `readWind()` are documented stubs; `client.setInsecure()` is
   an unresolved TODO.
6. **Evaluated against our own generator** for attribution — see §2.
7. **The engine dispatches 2.7 times per zone per day.** 8,047 dispatches over 40 zones and 74 days. The dwell gate is the only thing limiting re-dosing, and a ward sanitation supervisor receiving three work orders a day for the same stretch of road is a plausible failure mode that this prototype cannot test without a real roster. At 15 µg/m³ the trigger is being asked to act on a level most of the window exceeds, which is why milestone 5's finding matters: without a forecast that beats persistence, the queue is reacting to now.
8. **`instrumented_nodes: 40` is an engineering choice, not a permit.** A 320 m
   lattice over 4 km² is what the BOM and the deployment story support; no
   municipal permission has been sought.

---

## 8. What a reader should not infer from this document

- That SETU measures anything. It decides. Every concentration in the prototype
  comes from a model field or a synthetic node.
- That the accuracy numbers are instrument performance. They are the recovery of
  a known synthetic signal, which is a statement about the maths.
- That beating persistence on a twin means beating persistence in Rajkot.
- That the literature figures quoted here were measured by us. Each carries its
  own citation, and `docs/DATA_SOURCES.md` lists what we tried and could not get.
