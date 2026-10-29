"""Forecast skill against the baselines it has to beat."""

import numpy as np
import pandas as pd
import pytest

from setu import grid, twin
from setu.config import load_site
from setu.errors import CalibrationError
from setu.forecast import (
    HORIZONS,
    LAGS,
    GraphForecaster,
    cyclical,
    design_matrix,
    evaluate_forecast,
    hourly_adjacency,
    lag_frame,
    upwind_mean,
    walk_forward_slices,
)
from setu.io import openmeteo as om
from setu.io.cache import HttpCache
from setu.pipeline import NodeProduct, bridge_met_gaps, forecast_all


@pytest.fixture(scope="module")
def node_series():
    site = load_site("rajkot")
    df, _ = om.load_hourly(site, HttpCache(), past_days=60, forecast_days=0)
    nodes = grid.instrumented(grid.build_lattice(site))[:6]
    t = twin.DigitalTwin(df, grid.instrumented(grid.build_lattice(site)), site)
    return site, nodes, {n.id: t.truth(n) for n in nodes}


def test_lags_shift_by_the_right_number_of_hours(node_series):
    _, _, series = node_series
    one = next(iter(series.values()))
    lags = lag_frame(one, "pm25", lags=(1, 2, 24))
    assert lags["lag_1"].iloc[5] == one["pm25"].iloc[4]
    assert lags["lag_24"].iloc[30] == one["pm25"].iloc[6]
    assert lags["lag_1"].isna().sum() == 1


def test_cyclical_hour_encoding_is_continuous_across_midnight():
    idx = pd.date_range("2026-09-30 23:00", periods=3, freq="h")
    df = pd.DataFrame(index=idx)
    c = cyclical(df)
    # 23:00 and 00:00 must be adjacent on the circle, not 23 hours apart.
    assert abs(c["hour_sin"].iloc[0] - c["hour_sin"].iloc[1]) < 0.3
    assert c["hour_cos"].iloc[1] > c["hour_cos"].iloc[0]


def test_upwind_mean_is_zero_when_nothing_is_upwind():
    site = load_site("rajkot")
    nodes = grid.instrumented(grid.build_lattice(site))[:5]
    idx = pd.date_range("2026-09-01", periods=10, freq="h")
    readings = {n.id: pd.Series(1.0, index=idx) for n in nodes}
    empty_graph = {n.id: [] for n in nodes}
    out = upwind_mean(nodes[0], readings, empty_graph)
    assert (out == 0.0).all()
    assert len(out) == len(idx)


def test_upwind_mean_uses_the_neighbours_the_wind_comes_from():
    site = load_site("rajkot")
    nodes = grid.instrumented(grid.build_lattice(site))
    idx = pd.date_range("2026-09-01", periods=5, freq="h")
    readings = {n.id: pd.Series(float(i), index=idx)
                for i, n in enumerate(nodes)}
    graph = {nodes[0].id: [(nodes[1].id, 1.0), (nodes[2].id, 1.0)]}
    out = upwind_mean(nodes[0], readings, graph)
    assert out.iloc[0] == pytest.approx(1.5)     # mean of the 1.0 and 2.0 series


def test_design_matrix_has_no_infinities(node_series):
    _, _, series = node_series
    one = next(iter(series.values()))
    X = design_matrix(one, "pm25")
    assert np.isfinite(X.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()).all()
    assert f"lag_{LAGS[0]}" in X.columns
    # Humidity must arrive as a fraction whichever frame shape fed it in.
    assert X["rh_frac"].between(0.0, 1.0).all()


def test_a_missing_met_channel_is_reported_not_imputed():
    from setu.errors import CalibrationError
    idx = pd.date_range("2026-09-01", periods=5, freq="h")
    bare = pd.DataFrame({"pm25": [1.0] * 5, "pm10": [2.0] * 5}, index=idx)
    with pytest.raises(CalibrationError, match="rh_frac"):
        design_matrix(bare, "pm25")


def test_forecaster_refuses_an_underfed_horizon(node_series):
    _, _, series = node_series
    one = next(iter(series.values()))
    X = design_matrix(one, "pm25")
    y = {1: one["pm25"].shift(-1).iloc[:50]}
    with pytest.raises(CalibrationError, match="usable rows"):
        GraphForecaster(horizons=(1,)).fit(X.iloc[:50], y)


def test_predicting_an_unfitted_horizon_is_an_error(node_series):
    _, _, series = node_series
    one = next(iter(series.values()))
    X = design_matrix(one, "pm25")
    y = {1: one["pm25"].shift(-1)}
    model = GraphForecaster(horizons=(1,)).fit(X.iloc[:800], {1: y[1].iloc[:800]})
    with pytest.raises(CalibrationError, match="no model"):
        model.predict(X, 24)


def test_evaluate_forecast_rejects_an_incomplete_frame():
    idx = pd.date_range("2026-09-01", periods=60, freq="h")
    bad = pd.DataFrame({"actual": np.arange(60.0), "predicted": np.arange(60.0)},
                       index=idx)
    with pytest.raises(CalibrationError, match="missing columns"):
        evaluate_forecast({1: bad})


def test_the_baseline_the_model_actually_beats_is_the_climatology(node_series):
    """Scored out of sample, the win is against the hour-of-week mean.

    The earlier version of this test fitted and scored on the same rows and
    asserted skill > 0.20 against persistence. It passed and it was measuring
    the wrong thing: an in-sample fit on the monsoon, quoted as the claim the
    forecast earns its place on. Walk-forward on the same window puts skill
    against persistence below zero at every horizon, because the held-out
    slices are a different regime and persistence is exactly what a different
    regime rewards.

    What survives out of sample is the diurnal climatology: a forecast that
    cannot beat "this hour last week" is a table, not a forecast. That is the
    comparison asserted here, and the pooled persistence number is asserted to
    be present rather than assumed positive.
    """
    _, nodes, series = node_series
    per_horizon: dict[int, list[pd.DataFrame]] = {h: [] for h in HORIZONS}
    for nid, one in series.items():
        X = design_matrix(one, "pm25")
        targets = {h: one["pm25"].shift(-h) for h in HORIZONS}
        for train, calib, score in walk_forward_slices(len(X)):
            model = GraphForecaster(horizons=HORIZONS).fit(
                X.iloc[train], {h: s.iloc[train] for h, s in targets.items()})
            model.calibrate_intervals(
                X.iloc[calib], {h: s.iloc[calib] for h, s in targets.items()})
            model.fit_seasonal_baseline(one["pm25"].iloc[train])
            window = X.iloc[score]
            for h in HORIZONS:
                lo, hi = model.predict_interval(window, h)
                frame = pd.DataFrame({
                    "actual": targets[h].reindex(window.index),
                    "predicted": model.predict(window, h),
                    "persistence": one["pm25"].reindex(window.index),
                    "seasonal": model.seasonal_predict(window.index),
                    "interval_lo": lo,
                    "interval_hi": hi,
                })
                frame["node"] = nid
                per_horizon[h].append(frame)

    rows = [evaluate_forecast({h: pd.concat(v)})[0] for h, v in per_horizon.items()]
    by_h = sorted(rows, key=lambda r: r.horizon_h)

    assert by_h[0].horizon_h == 1
    assert by_h[0].skill_vs_diurnal > 0.30
    assert all(r.skill_vs_diurnal > 0 for r in by_h)
    # Absolute error must grow with horizon, and so must the baseline it is
    # measured against - that is a property of the data, not of the model, and
    # the report's decay curve is what says how much survives.
    assert by_h[0].model_rmse < by_h[-1].model_rmse
    assert by_h[-1].persistence_rmse > by_h[0].persistence_rmse
    # Pooled skill against persistence is reported, not claimed: it is negative
    # on this window and the report says so in as many words.
    assert all(np.isfinite(r.skill_vs_persistence) for r in by_h)
    assert all(r.interval_coverage is not None for r in by_h)


def test_a_chained_path_is_hourly_and_carries_the_last_observation(node_series):
    """Multi-step ahead: 1h predictions chained, not three models stitched.

    The engine acts on 1/3/6 h horizons, but it also has to know when a breach
    starts, and only the 1h model can be iterated because it is the only one
    whose input is a single observed hour. The path must therefore begin at the
    hour after the last observation, be hourly throughout, and stay physical.
    """
    _, _, series = node_series
    one = next(iter(series.values()))
    met_cols = ["rh", "wind_speed_ms", "wind_dir_deg", "temp_c"]
    X = design_matrix(one, "pm25")
    targets = {h: one["pm25"].shift(-h) for h in HORIZONS}
    model = GraphForecaster(horizons=HORIZONS).fit(X.iloc[:800], {
        h: s.iloc[:800] for h, s in targets.items()})

    path = model.forecast_path(one["pm25"].iloc[:800], one[met_cols].iloc[700:], steps=6)

    assert len(path) == 6
    assert path.index[0] == one.index[800]
    assert (path.index.to_series().diff().dropna()
            == pd.Timedelta(hours=1)).all()
    assert np.isfinite(path.to_numpy()).all()
    assert (path >= 0.0).all()

    # A model that fits increments has no-change as its floor, so a history
    # that has been flat cannot produce a path that climbs: anything else
    # would mean the chain was re-reading a stale lag as fresh weather.
    flat = pd.Series(20.0, index=one.index[:800])
    flat_model = GraphForecaster(horizons=HORIZONS).fit(
        design_matrix(one.assign(pm25=flat), "pm25").iloc[:800],
        {h: flat.shift(-h) for h in HORIZONS})
    flat_path = flat_model.forecast_path(flat, one[met_cols].iloc[700:], steps=4)
    assert (flat_path - 20.0).abs().max() < 5.0


def test_a_path_without_weather_says_so(node_series):
    """A path cannot invent the weather it is a forecast of."""
    _, _, series = node_series
    one = next(iter(series.values()))
    met_cols = ["rh", "wind_speed_ms", "wind_dir_deg", "temp_c"]
    X = design_matrix(one, "pm25")
    model = GraphForecaster(horizons=HORIZONS).fit(X.iloc[:800], {
        h: one["pm25"].shift(-h).iloc[:800] for h in HORIZONS})

    with pytest.raises(CalibrationError, match="meteorology"):
        model.forecast_path(one["pm25"].iloc[:400], one[met_cols].iloc[300:340],
                            steps=6)
    with pytest.raises(CalibrationError, match="history"):
        model.forecast_path(one["pm25"].iloc[:10], one[met_cols].iloc[:400], steps=2)
    with pytest.raises(CalibrationError, match="at least 1"):
        model.forecast_path(one["pm25"].iloc[:800], one[met_cols].iloc[700:], steps=0)


def test_hourly_adjacency_is_cached_per_hour(node_series):
    site, nodes, _ = node_series
    wind = pd.Series([270.0, 90.0, 180.0],
                     index=pd.date_range("2026-09-01", periods=3, freq="h"))
    graphs = hourly_adjacency(nodes[:4], wind)
    assert len(graphs) == 3
    assert graphs[wind.index[0]] != graphs[wind.index[1]]


def test_walk_forward_slices_are_ordered_disjoint_and_expanding():
    """The scored hours must be hours the model never saw.

    Walk-forward answers the question a deployment asks: as data arrives and
    the model refits, what is it worth on the next slice? Each fold's score
    slice must lie entirely after its calibration slice, which lies after its
    training slice, and the training slice always starts at row 0 - that is
    what makes it expanding rather than rolling.
    """
    folds = walk_forward_slices(2000, folds=3)
    assert len(folds) == 3
    for train, calib, score in folds:
        assert train.start == 0, "training slice must expand from the window start"
        # Half-open slices: sharing a boundary is disjoint, overlapping is not.
        assert train.stop <= calib.start, "fit and calibration overlap"
        assert calib.stop <= score.start, "calibration and score overlap"
        assert score.stop is not None
    # Score slices march forward in time and never repeat an hour.
    scores = [(f[2].start, f[2].stop) for f in folds]
    assert scores == sorted(scores), "score slices do not advance"
    assert len({r for s in scores for r in range(*s)}) == sum(
        stop - start for start, stop in scores), "score slices overlap"


def test_a_window_too_thin_to_split_says_so():
    with pytest.raises(CalibrationError, match="no walk-forward fold"):
        walk_forward_slices(300, folds=3)


def test_walk_forward_drops_folds_that_could_not_be_fitted():
    """The first fold of a thin window has no history to train on. It is
    dropped, not scored on a model that barely fit - the optimistic number
    lives exactly there."""
    folds = walk_forward_slices(1000, folds=3, min_train=450)
    assert folds, "a 1000-row window should yield at least one fold"
    for train, calib, score in folds:
        assert train.stop - train.start >= 450, "training floor not honoured"
        assert score.start >= train.stop


def test_the_nearest_fold_survives_the_training_floor():
    """Regression: at the run's real window (1772 rows, 3 folds) the old
    generator placed the first score slice at n/4 = 443 rows, under the 450
    training floor, so the guard dropped the fold nearest the training data -
    the one a deployment sees first - and every node silently lost it. The
    score region now starts where the floor permits, and all three folds
    come back for every window the fixtures actually use.
    """
    for n in (1772, 1440, 1080, 720):
        folds = walk_forward_slices(n, folds=3)
        assert len(folds) == 3, f"n={n} yielded {len(folds)} folds"
        # Nearest fold first: its score slice begins right after the floor.
        first = folds[0]
        assert first[0].stop >= 450
        assert first[2].start >= first[0].stop


def test_intervals_cover_at_the_nominal_rate(node_series):
    """A band that covers nothing is not uncertainty, and a band that covers
    everything is not information. The nominal rate is 80%, and measured
    coverage on held-out hours has to land near it - wide tolerance because
    this is one window of one node set, but a band at 40% or 99.5% fails."""
    _, _, series = node_series
    one = next(iter(series.values()))
    X = design_matrix(one, "pm25")
    targets = {h: one["pm25"].shift(-h) for h in HORIZONS}
    train, calib, score = walk_forward_slices(len(X), folds=3)[-1]
    model = GraphForecaster(horizons=HORIZONS)
    model.fit(X.iloc[train], {h: s.iloc[train] for h, s in targets.items()})
    model.calibrate_intervals(X.iloc[calib],
                              {h: s.iloc[calib] for h, s in targets.items()})
    window = X.iloc[score]
    for h in HORIZONS:
        lo, hi = model.predict_interval(window, h)
        actual = targets[h].reindex(window.index)
        scored = actual.notna()
        coverage = ((actual[scored] >= lo[scored])
                    & (actual[scored] <= hi[scored])).mean()
        assert 0.55 <= coverage <= 0.98, f"{h}h band covers {coverage:.3f}"
        assert (lo <= hi).all(), "inverted band"
        assert (lo >= 0).all(), "negative lower bound on a concentration"


def test_a_wider_horizon_gets_a_wider_band(node_series):
    """Six hours out is less certain than one hour out. A band that does not
    grow with horizon is measuring the model, not the future."""
    _, _, series = node_series
    one = next(iter(series.values()))
    X = design_matrix(one, "pm25")
    targets = {h: one["pm25"].shift(-h) for h in HORIZONS}
    train, calib, _ = walk_forward_slices(len(X), folds=3)[-1]
    model = GraphForecaster(horizons=HORIZONS)
    model.fit(X.iloc[train], {h: s.iloc[train] for h, s in targets.items()})
    model.calibrate_intervals(X.iloc[calib],
                              {h: s.iloc[calib] for h, s in targets.items()})
    assert model._quantiles[1] < model._quantiles[6]


def test_intervals_must_be_calibrated_before_they_are_predicted(node_series):
    _, _, series = node_series
    one = next(iter(series.values()))
    X = design_matrix(one, "pm25")
    y = {1: one["pm25"].shift(-1)}
    model = GraphForecaster(horizons=(1,)).fit(X.iloc[:800], {1: y[1].iloc[:800]})
    with pytest.raises(CalibrationError, match="calibrate"):
        model.predict_interval(X, 1)


def test_evaluating_without_intervals_still_scores(node_series):
    """The interval columns are optional: a frame without them scores skill
    and reports no coverage, rather than failing or inventing a band."""
    idx = pd.date_range("2026-09-01", periods=80, freq="h")
    base = np.arange(80.0)
    frame = pd.DataFrame({"actual": base, "predicted": base + 0.5,
                          "persistence": base, "seasonal": base + 1.0}, index=idx)
    rows = evaluate_forecast({1: frame})
    assert len(rows) == 1
    assert rows[0].interval_coverage is None
    assert rows[0].interval_width is None


def test_forecast_all_scores_walk_forward_and_carries_the_decay_curve(node_series):
    """The payload contract: pooled row per horizon, a per-fold decay curve
    under it, and diagnostics that count what was skipped. The report reads
    exactly these keys, and a regression here shows up as a silently missing
    section rather than an error."""
    from setu.pipeline import NodeProduct, forecast_all

    site, nodes, series = node_series
    products = {
        # The real pipeline's calibrated frame is one `value` column; truth
        # carries the meteorology. Faking both from one frame would duplicate
        # columns the design matrix reads by name.
        nid: NodeProduct(
            node=nid, truth=one, raw=one,
            calibrated=pd.DataFrame({"value": one["pm25"]}, index=one.index),
            attributions=[], drift_alarms=0, calibration={})
        for nid, one in series.items()
    }
    rows = forecast_all(products, nodes[:len(series)])
    assert rows, "no horizon scored"
    by_h = {r["horizon_h"]: r for r in rows}
    assert set(by_h) == set(HORIZONS)
    for row in rows:
        assert row["n"] > 0
        assert row["interval_coverage"] is not None, "band not calibrated"
        folds = row["folds"]
        assert folds, "no per-fold decay curve in the payload"
        assert [f["fold"] for f in folds] == sorted(f["fold"] for f in folds)
        for f in folds:
            assert f["n"] > 0 and f["horizon_h"] == row["horizon_h"]
        assert row["nodes_compared"] == len(series)
        assert 0 <= row["nodes_beating_persistence"] <= row["nodes_compared"]
        assert np.isfinite(row["bias"])
    diag = rows[0]["diagnostics"]
    assert diag["nodes_scored"] + diag["nodes_skipped"] == diag["nodes_total"]
    assert diag["folds_scored"] + diag["folds_skipped"] >= diag["nodes_scored"]
    # Warm-up and missing weather are separate losses, and on this grid the
    # warm-up is the larger one - reporting them as one number made 960 rows of
    # lag window look like a meteorology outage.
    assert diag["rows_warmup"] >= diag["warmup_hours"] * len(series)
    assert diag["rows_missing_met"] >= 0
    assert diag["met_hours_bridged"] >= 0


def test_only_short_weather_gaps_are_bridged():
    """A two-hour outage is bridged; a nine-hour one is weather, not a gap."""
    # hours 0-2 observed, 3-4 missing (short gap), 5-7 observed,
    # 8-12 missing (long outage), 13 observed.
    idx = pd.date_range("2026-09-01", periods=14, freq="h")
    df = pd.DataFrame({
        "rh_frac": [0.5] * 3 + [np.nan] * 2 + [0.6] * 3 + [np.nan] * 5 + [0.7],
        "wind_speed_ms": [3.0] * 3 + [np.nan] * 2 + [4.0] * 3 + [np.nan] * 5 + [6.0],
        "wind_dir_deg": 270.0,
        "temp_c": 30.0,
    }, index=idx)

    out, bridged = bridge_met_gaps(df, limit=3)

    # Two consecutive missing hours: filled from the hour before them.
    assert out["rh_frac"].iloc[3] == pytest.approx(0.5)
    assert out["rh_frac"].iloc[4] == pytest.approx(0.5)
    assert out["wind_speed_ms"].iloc[4] == pytest.approx(3.0)
    # Five consecutive: the first three fill, the last two stay missing rather
    # than carrying a wind from five hours ago into a wind-aware model.
    assert out["wind_speed_ms"].iloc[8:11].notna().all()
    assert out["wind_speed_ms"].iloc[8] == pytest.approx(4.0)
    assert out["rh_frac"].iloc[11:13].isna().all()
    assert bridged == 5
    # Nothing that was observed may be rewritten.
    observed = [0, 1, 2, 5, 6, 7, 13]
    assert (out["rh_frac"].iloc[observed]
            == df["rh_frac"].iloc[observed]).all()
    assert (out["wind_speed_ms"].iloc[observed]
            == df["wind_speed_ms"].iloc[observed]).all()


def test_every_fold_scores_hours_the_model_never_trained_on(node_series):
    """Regression guard for the pipeline's own split: no fold may score an
    hour that appears in its own training slice. The bug this guards against
    is the one the whole walk-forward change exists to remove - a model
    graded on its own homework."""
    _, _, series = node_series
    one = next(iter(series.values()))
    X = design_matrix(one, "pm25")
    for train, calib, score in walk_forward_slices(len(X), folds=3):
        trained = set(X.index[train]) | set(X.index[calib])
        scored = set(X.index[score])
        assert not trained & scored, \
            "scored window reaches back into the fit or calibration slice"
