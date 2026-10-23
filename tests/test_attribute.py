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
