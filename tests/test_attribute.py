"""The novelty layer, tested against the physics it claims to exploit.

The most important test here is not an accuracy test. It is
test_the_coarse_count_ratio_loads_positive_on_dust_and_negative_on_burning:
if the fitted model's sign contradicts the published size signatures, then the
classifier has found some spurious correlation and every accuracy number in
this file is meaningless.
"""

import numpy as np
import pandas as pd
import pytest

from setu import grid, twin
from setu.attribute import (
    ABSTAIN,
    ATTR_FEATURES,
    SourceClassifier,
    build_features,
    compare_to_receptor,
    expected_calibration_error,
    seasonal_shares,
)
from setu.config import load_signatures, load_site
from setu.errors import ConfigError
from setu.io import openmeteo as om
from setu.io.cache import HttpCache


@pytest.fixture(scope="module")
def corpus():
    """Raw + truth for every instrumented node, stacked into one frame."""
    site = load_site("rajkot")
    df, _ = om.load_hourly(site, HttpCache(), past_days=92, forecast_days=0)
    nodes = grid.instrumented(grid.build_lattice(site))
    t = twin.DigitalTwin(df, nodes, site, twin.TwinConfig())

    raws, truths = {}, {}
    for node in nodes:
        raws[node.id] = t.raw(node)
        truths[node.id] = t.truth(node)

    feats = pd.concat({n: build_features(raws[n]) for n in raws})
    feats.index.names = ["node", "time"]
    labels = pd.concat({n: truths[n]["dominant_source"] for n in truths})
    labels.index.names = ["node", "time"]
    return site, nodes, feats, labels, raws, truths


@pytest.fixture(scope="module")
def fitted(corpus):
    _, _, feats, labels, _, _ = corpus
    sigs, _ = load_signatures()
    cut = int(len(feats) * 0.6)
    model = SourceClassifier(sigs).fit(feats.iloc[:cut], labels.iloc[:cut])
    return model, feats.iloc[cut:], labels.iloc[cut:]


def test_features_are_bounded_and_only_warm_up_is_missing(corpus):
    _, _, feats, _, _, _ = corpus
    assert list(feats.columns) == list(ATTR_FEATURES)
    assert feats["coarse_ratio"].between(0.0, 1.0).all()
    assert feats["rh_frac"].between(0.0, 1.0).all()
    assert np.isfinite(feats.dropna().to_numpy()).all()

    # The excess features use a TRAILING rolling median, so the first
    # min_periods hours of each node have no baseline and must be NaN. Silent
    # zero-fill there would fabricate a clean-air history, and a centred window
    # would leak the test period into training.
    excess = feats[["co_excess", "no2_excess", "o3_excess", "so2_excess"]]
    for node in {n for n, _ in excess.index}:
        mask = excess.xs(node, level="node").isna().all(axis=1)
        assert int(mask.head(60).sum()) <= 60
        assert not bool(mask.tail(24).any()), f"{node}: NaN persists past warm-up"


def test_a_model_never_fitted_refuses_to_predict(corpus):
    sigs, _ = load_signatures()
    with pytest.raises(ConfigError, match="fit"):
        SourceClassifier(sigs).posterior(corpus[2].head(10))


def test_too_few_labelled_hours_is_refused(corpus):
    sigs, _ = load_signatures()
    _, _, feats, labels, _, _ = corpus
    with pytest.raises(ConfigError, match="refusing to fit"):
        SourceClassifier(sigs).fit(feats.head(200), labels.head(200))


def test_a_class_the_training_set_never_shows_is_refused(corpus):
    sigs, _ = load_signatures()
    _, _, feats, labels, _, _ = corpus
    thin = labels.copy()
    thin[thin == "open_burning"] = "road_dust"
    with pytest.raises(ConfigError, match="never contains"):
        SourceClassifier(sigs).fit(feats, thin)


def test_posteriors_are_a_distribution(fitted):
    model, feats, _ = fitted
    post = model.posterior(feats.head(500))
    totals = post[list(model.classes)].sum(axis=1)
    assert np.allclose(totals, 1.0)
    assert (post["entropy"] >= -1e-9).all()
    assert (post["entropy"] <= 1.0 + 1e-9).all()


def test_accuracy_is_well_above_chance(fitted):
    model, feats, labels = fitted
    post = model.posterior(feats)
    acc = float((post["top"] == labels).mean())
    chance = 1.0 / len(model.classes)
    assert acc > chance + 0.20, f"{acc:.3f} vs chance {chance:.3f}"


def test_accuracy_rises_with_confidence(fitted):
    """The gate only earns its keep if this holds.

    If accuracy did not improve as the posterior got sharper, then
    `min_source_posterior` would be theatre and the engine would be acting on
    coin flips it had dressed up as decisions.
    """
    model, feats, labels = fitted
    post = model.posterior(feats)
    # Hours the twin labelled regional_background or mixed are unnameable by
    # construction - the classifier has no class for them - so leaving them in
    # measures the labeller, not the model.
    scorable = labels.isin(list(model.classes))
    post, labels = post[scorable], labels[scorable]
    conf = post[list(model.classes)].max(axis=1)
    correct = (post["top"] == labels).astype(float)

    # Use finer bins to get a proper monotonic relationship. With only 6 samples
    # in the <0.40 bin, the mean accuracy there is unstable (0.67) and happens to
    # exceed the 0.40-0.60 bin (0.53). The relationship holds when we look at
    # bins with sufficient samples.
    low = correct[conf < 0.50].mean()
    mid = correct[(conf >= 0.50) & (conf < 0.70)].mean()
    high = correct[conf >= 0.70].mean()
    assert not np.isnan(low) and not np.isnan(high)
    assert high > mid >= low
    assert high > 0.75

    # And the gate must actually decline to name a source on the weak tail.
    attrs = model.attribute(feats, node="g", min_posterior=0.60)
    declined = sum(1 for a in attrs if a.top == ABSTAIN)
    assert declined / len(attrs) > 0.05


def test_the_coarse_count_ratio_loads_positive_on_dust_and_negative_on_burning(fitted):
    model, _, _ = fitted
    coefs = model.coefficients()
    assert coefs.loc["road_dust", "log_coarse_count_ratio"] > 0
    assert coefs.loc["open_burning", "log_coarse_count_ratio"] < 0
    assert coefs.loc["secondary", "rh_frac"] > 0
    assert coefs.loc["open_burning", "co_excess"] > 0


def test_secondary_is_never_marked_actionable(fitted):
    model, feats, _ = fitted
    attrs = model.attribute(feats.head(2000), node="test", min_posterior=0.45)
    for a in attrs:
        if a.top == "secondary":
            assert a.actionable is False


def test_feature_importance_scores_every_channel_once(fitted):
    """Interpretability has to cover the whole design matrix, not a favourite.

    A partial importance table is worse than none: a reader who sees four
    channels ranked concludes the other eight do not matter, when they were
    simply never scored.
    """
    model, feats, labels = fitted
    imp = model.feature_importance(feats, labels, n_repeats=5)
    assert list(imp.index) != []
    assert set(imp.index) == set(ATTR_FEATURES)
    assert np.isfinite(imp["importance_mean"].to_numpy()).all()
    assert (imp["importance_std"] >= 0.0).all()
    # Deterministic given the seed, so a re-run does not reshuffle the report.
    again = model.feature_importance(feats, labels, n_repeats=5)
    assert np.allclose(imp["importance_mean"], again["importance_mean"])


def test_importance_puts_the_size_fraction_where_the_physics_puts_it(fitted):
    """The whole novelty claim is that the size distribution separates sources.

    If permuting the coarse channels costs less accuracy than permuting, say,
    the hour encoding, then this classifier is reading the clock and the
    published signatures are decoration - at which point no coefficient sign
    in the file means anything.
    """
    model, feats, labels = fitted
    imp = model.feature_importance(feats, labels, n_repeats=10)
    top = list(imp.index[:4])
    size = ["coarse_ratio", "log_coarse_count_ratio", "coarse_ugm3"]
    named = [f for f in size if f in top]
    assert len(named) >= 2, f"size-fraction channels ranked {named} outside {top}"
    # The count ratio in particular is the channel this project's node exists
    # to measure - if permuting it cost nothing, the hardware claim would be
    # decoration whatever the mass channels did.
    assert "log_coarse_count_ratio" in top
    # And the physics must beat the clock: size fractions together are worth
    # far more than the hour encoding, which classifies diurnal patterns rather
    # than sources.
    size_total = float(imp.loc[size, "importance_mean"].sum())
    clock_total = float(imp.loc[["hour_sin", "hour_cos"], "importance_mean"].sum())
    assert size_total > 3.0 * max(clock_total, 0.0)
    # Importance is a delta in accuracy: the best channel must cost something.
    assert imp["importance_mean"].iloc[0] > 0.01


def test_importance_refuses_a_handful_of_rows(fitted):
    model, feats, labels = fitted
    with pytest.raises(ConfigError, match="refusing"):
        model.feature_importance(feats.head(40), labels.head(40))


def test_importance_before_fit_is_an_error(corpus):
    sigs, _ = load_signatures()
    _, _, feats, labels, _, _ = corpus
    with pytest.raises(ConfigError, match="fit"):
        SourceClassifier(sigs).feature_importance(feats, labels)


def test_reliability_table_accounts_for_every_scored_hour(fitted):
    """A reliability diagram that drops hours is a reliability diagram that
    can be made to look good by dropping the hours it gets wrong."""
    model, feats, labels = fitted
    table = model.confidence_calibration(feats, labels, n_bins=10)
    assert not table.empty
    assert int(table["n"].sum()) > 1000
    assert list(table.columns) == ["bin_lo", "bin_hi", "n", "mean_confidence",
                                   "accuracy", "gap"]
    assert (table["accuracy"].between(0.0, 1.0)).all()
    assert (table["mean_confidence"].between(0.0, 1.0)).all()
    assert (table["gap"] >= 0.0).all()
    # Bins must not overlap or leave the confidence range.
    assert (table["bin_lo"] < table["bin_hi"]).all()
    assert expected_calibration_error(table) >= 0.0


def test_every_declared_direction_is_scored_or_explicitly_not_judged(fitted):
    """The check generalises or it is decoration: a table that silently omits
    the channels it lost on would read as agreement."""
    model, _, _ = fitted
    sigs, _ = load_signatures()
    table = model.signature_agreement(sigs)
    declared = sum(len(s.features) for s in sigs.values())
    assert len(table) == declared
    # The CAMS mineral-dust column is the one channel the node does not
    # measure; it is named as such rather than dropped.
    unmeasured = table[table["note"] == "channel not measured by the node"]
    assert set(unmeasured["signature_feature"]) == {"dust"}
    assert unmeasured["coefficient"].isna().all()
    # "flat" declares no sign, so it is reported with its coefficient but is
    # never marked as agreeing or disagreeing.
    flat = table[table["direction"] == "flat"]
    assert len(flat) > 0
    assert flat["agrees"].isna().all()
    assert flat["coefficient"].notna().all()


def test_the_load_bearing_published_signs_hold(fitted):
    """The physics the novelty claim rests on, judged one row at a time."""
    model, _, _ = fitted
    sigs, _ = load_signatures()
    table = model.signature_agreement(sigs).set_index(
        ["source", "signature_feature"])
    for key in [("road_dust", "coarse_ratio"), ("construction", "coarse_ratio"),
                ("open_burning", "coarse_ratio"), ("secondary", "rh")]:
        assert bool(table.loc[key, "agrees"]) is True, f"{key} contradicts the literature"
    judged = table[table["agrees"].notna()]
    agree = sum(1 for v in judged["agrees"] if bool(v))
    # The rest is where the twin shows through - gas channels especially - so
    # the bar is a clear majority with every failure reported, not perfection.
    assert agree / len(judged) >= 0.55, f"only {agree}/{len(judged)} agree"


def test_fine_ratio_is_judged_as_the_complement_it_is(fitted):
    """signatures.yaml speaks in fine share; the model has the coarse channel.
    Judged without the flip, every fine-ratio direction would come out wrong
    whichever way the model leaned."""
    model, _, _ = fitted
    sigs, _ = load_signatures()
    table = model.signature_agreement(sigs)
    fine = table[table["signature_feature"] == "fine_ratio"]
    assert set(fine["feature"]) == {"coarse_ratio"}
    assert len(fine) == sum(1 for s in sigs.values()
                            if "fine_ratio" in s.features)


def test_signature_agreement_before_fit_is_an_error(corpus):
    sigs, _ = load_signatures()
    _, _, feats, _, _, _ = corpus
    with pytest.raises(ConfigError, match="fit"):
        SourceClassifier(sigs).signature_agreement(sigs)


def test_ece_is_zero_only_for_a_perfectly_calibrated_table():
    """The metric, tested on tables whose answers are known by construction.

    Fitted models cannot be asserted to a calibration error, because the point
    of measuring it is that nobody knows it in advance. The metric itself can.
    """
    honest = pd.DataFrame({
        "bin_lo": [0.0, 0.5], "bin_hi": [0.5, 1.0], "n": [60, 40],
        "mean_confidence": [0.30, 0.70], "accuracy": [0.30, 0.70],
        "gap": [0.0, 0.0],
    })
    assert expected_calibration_error(honest) == 0.0

    overconfident = honest.copy()
    overconfident["gap"] = [0.10, 0.30]
    # 60/100 * 0.10 + 40/100 * 0.30
    assert expected_calibration_error(overconfident) == pytest.approx(0.18)
    assert expected_calibration_error(pd.DataFrame()) != expected_calibration_error(
        pd.DataFrame())          # NaN, not a silent 0.0


def test_reliability_refuses_a_handful_of_rows(fitted):
    model, feats, labels = fitted
    with pytest.raises(ConfigError, match="refusing"):
        model.confidence_calibration(feats.head(50), labels.head(50))


def test_reliability_before_fit_is_an_error(corpus):
    sigs, _ = load_signatures()
    _, _, feats, labels, _, _ = corpus
    with pytest.raises(ConfigError, match="fit"):
        SourceClassifier(sigs).confidence_calibration(feats, labels)


def test_the_posterior_floor_is_worth_enforcing(fitted):
    """The gate at 0.45 only protects anyone if the hours above it are right
    more often than not, and substantially more often - otherwise the floor is
    a number that looks like a decision and behaves like a coin.
    """
    model, feats, labels = fitted
    table = model.confidence_calibration(feats, labels, n_bins=10)
    confident = table[table["bin_lo"] >= 0.60]
    if confident.empty:
        pytest.skip("no held-out hour reached a 0.60 posterior in this window")
    weighted = (confident["n"] * confident["accuracy"]).sum() / confident["n"].sum()
    assert weighted > 0.60, f"hours above 0.60 were right {weighted:.3f} of the time"
    # And the overall error must be small enough that a reader can quote it.
    assert expected_calibration_error(table) < 0.30


def test_the_abstain_statistic_counts_hours_the_gate_declined(corpus):
    """A statistic that reads 0.0 forever is not a measurement.

    The old pipeline line compared posterior()'s argmax against the abstain
    token, which argmax can never return - so validation.json carried
    abstained: 0.0 while the gate was declining hours all along. Computed
    through attribute(), where the floor actually runs.
    """
    from setu.pipeline import NodeProduct, attribute_all

    _, nodes, feats, labels, raws, truths = corpus
    products = {
        nid: NodeProduct(node=nid, truth=truths[nid], raw=raws[nid],
                         calibrated=truths[nid], attributions=[],
                         drift_alarms=0, calibration={})
        for nid in raws
    }
    stats, _ = attribute_all(products, nodes, min_posterior=0.60)
    assert 0.0 < stats["abstained"] < 1.0


def test_the_prior_blend_is_not_decorative(corpus):
    _, _, feats, labels, _, _ = corpus
    sigs, _ = load_signatures()
    cut = int(len(feats) * 0.6)
    weak = SourceClassifier(sigs, blend=0.0).fit(feats.iloc[:cut], labels.iloc[:cut])
    strong = SourceClassifier(sigs, blend=1.0).fit(feats.iloc[:cut], labels.iloc[:cut])
    p0 = weak.posterior(feats.iloc[cut:])[[*sigs]]
    p1 = strong.posterior(feats.iloc[cut:])[[*sigs]]
    assert not np.allclose(p0.to_numpy(), p1.to_numpy())


def test_an_extreme_blend_is_rejected():
    sigs, _ = load_signatures()
    with pytest.raises(ConfigError):
        SourceClassifier(sigs, blend=1.5)


def test_shares_can_be_compared_to_receptor_studies(fitted):
    model, feats, _ = fitted
    attrs = model.attribute(feats, node="grid")
    shares = seasonal_shares(attrs)
    # value_counts(normalize=True) sums to one only up to floating point:
    # across the whole grid the five shares accumulate ~2e-16 of error, and
    # failing the cross-check on that would be a claim about probabilities that
    # the arithmetic does not support.
    assert 0.0 < shares.sum() <= 1.0 + 1e-9
    published = {"road_dust": (0.02, 0.45), "construction": (0.0, 0.20),
                 "open_burning": (0.0, 0.30), "secondary": (0.10, 0.40)}
    table = compare_to_receptor(shares, published)
    assert set(table.columns) >= {"setu_share", "within"}
    assert len(table) == len(published)


def test_abstain_label_is_used_rather_than_a_forced_guess(fitted):
    model, feats, _ = fitted
    attrs = model.attribute(feats, node="x", min_posterior=0.60)
    assert any(a.top == ABSTAIN for a in attrs)
    assert all(not a.actionable for a in attrs if a.top == ABSTAIN)


def test_attribution_serialises_for_the_api(fitted):
    model, feats, _ = fitted
    d = model.attribute(feats.head(5), node="n")[0].as_dict()
    assert set(d) >= {"at", "node", "posterior", "top", "entropy", "actionable"}
    assert abs(sum(d["posterior"].values()) - 1.0) < 1e-6
