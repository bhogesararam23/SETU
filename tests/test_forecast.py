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
)
from setu.io import openmeteo as om
from setu.io.cache import HttpCache


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


def test_beats_persistence_at_one_hour(node_series):
    """The claim the forecast earns its place on.

    If it cannot beat "the value will be what it is now", the response engine
    should run on observations and the model should stay in shadow mode.
    """
    _, nodes, series = node_series
    per_horizon: dict[int, list[pd.DataFrame]] = {h: [] for h in HORIZONS}
    for nid, one in series.items():
        X = design_matrix(one, "pm25")
        targets = {h: one["pm25"].shift(-h) for h in HORIZONS}
        model = GraphForecaster(horizons=HORIZONS).fit(X, targets)
        model.fit_seasonal_baseline(one["pm25"])
        base = X.iloc[24:-7]
        for h in HORIZONS:
            frame = pd.DataFrame({
                "actual": targets[h].reindex(base.index),
                "predicted": model.predict(base, h),
                "persistence": one["pm25"].reindex(base.index),
                "seasonal": model.seasonal_predict(base.index),
            })
            frame["node"] = nid
            per_horizon[h].append(frame)

    rows = [evaluate_forecast({h: pd.concat(v)}, )[0]
            for h, v in ((h, per_horizon[h]) for h in HORIZONS)]

    one_h = next(r for r in rows if r.horizon_h == 1)
    assert one_h.skill_vs_persistence > 0.20
    assert one_h.skill_vs_diurnal > 0.0
    # Absolute error must grow with horizon. Relative skill against persistence
    # does the opposite - it rises - because the persistence baseline degrades
    # far faster than the model does. Asserting that skill decays would encode
    # a misconception, and this is exactly the number a reviewer is likely to
    # read as "too good at 6 hours than at 1 hour, suspicious".
    by_h = sorted(rows, key=lambda r: r.horizon_h)
    assert by_h[0].model_rmse < by_h[-1].model_rmse
    assert by_h[-1].persistence_rmse > by_h[0].persistence_rmse
    assert all(r.skill_vs_persistence > 0 for r in by_h)


def test_hourly_adjacency_is_cached_per_hour(node_series):
    site, nodes, _ = node_series
    wind = pd.Series([270.0, 90.0, 180.0],
                     index=pd.date_range("2026-09-01", periods=3, freq="h"))
    graphs = hourly_adjacency(nodes[:4], wind)
    assert len(graphs) == 3
    assert graphs[wind.index[0]] != graphs[wind.index[1]]
