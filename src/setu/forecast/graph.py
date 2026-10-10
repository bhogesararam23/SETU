"""Wind-aware short-horizon forecasting on the node graph.

Why a graph and not a per-node time series: pollution moves. A node that is
about to be overplumed can be warned by the node the plume is currently over,
and that warning is worth roughly the travel time between them - which at a
320 m grid and 3 m/s wind is under three minutes, but at 2 km and 6 m/s it is
eleven, and eleven minutes is longer than the response has.

Reported skill in the literature, for calibration of expectations rather than
for copying: Chen et al., Sci. Rep. 14 (2024), doi:10.1038/s41598-024-55060-2
reaches RMSE 19.11 and R^2 0.84 at 7 hours on three Beijing stations, 8.6%
better than an LSTM and 25.8% better than CNN-LSTM. Those are dense reference
networks over clean inputs. This is 40 low-cost nodes over minutes on a
drifting stream, so the number that matters here is skill against persistence,
not R^2 against a paper.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..errors import CalibrationError
from ..grid import Node, adjacency

#: Lags used as own-history predictors, in hours.
LAGS: tuple[int, ...] = (1, 2, 3, 6, 12, 24)
#: Horizons the engine is allowed to act on.
HORIZONS: tuple[int, ...] = (1, 3, 6)

BASE_FEATURES: tuple[str, ...] = (
    "rh_frac", "wind_speed_ms", "wind_dir_deg", "temp_c", "hour_sin", "hour_cos",
    "dow",
)


def lag_frame(df: pd.DataFrame, column: str, lags: Sequence[int] = LAGS
              ) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for lag in lags:
        out[f"lag_{lag}"] = df[column].shift(lag)
    return out


def cyclical(df: pd.DataFrame) -> pd.DataFrame:
    hour = df.index.hour.to_numpy()
    out = pd.DataFrame(index=df.index)
    out["hour_sin"] = np.sin(2 * math.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * math.pi * hour / 24.0)
    out["dow"] = df.index.dayofweek.to_numpy()
    return out


def upwind_mean(target: Node, readings: Mapping[str, pd.Series],
                graph: Mapping[str, Sequence[tuple[str, float]]],
                weight_floor: float = 0.05) -> pd.Series:
    """Weighted mean of a channel over the nodes the wind is blowing from.

    Returns an all-zero series when the target has no upwind neighbour this
    hour, rather than NaN: a node on the leading edge of a front genuinely has
    no upwind information, and the forecaster should see that as a fact about
    the weather rather than as missing data to be imputed.
    """
    edges = [e for e in graph.get(target.id, []) if e[1] >= weight_floor]
    if not edges:
        return pd.Series(0.0, index=next(iter(readings.values())).index)
    stacked = pd.concat([readings[nid] * w for nid, w in edges], axis=1).sum(axis=1)
    norm = sum(w for _, w in edges)
    return (stacked / norm).fillna(0.0)


def _met_column(node_df: pd.DataFrame, name: str) -> pd.Series:
    """Read a meteorological channel from either frame shape.

    The background frame from openmeteo carries `rh_frac` already normalised;
    the twin's truth frame carries percent `rh`. Accepting only one of them
    would make the forecaster work on whichever path happened to be tested.
    """
    if name in node_df:
        return node_df[name].astype(float)
    if name == "rh_frac" and "rh" in node_df:
        return node_df["rh"].astype(float) / 100.0
    if name == "rh" and "rh_frac" in node_df:
        return node_df["rh_frac"].astype(float) * 100.0
    raise CalibrationError(f"no {name!r} channel and no equivalent to derive it from")


def design_matrix(node_df: pd.DataFrame, target: str,
                  upwind: pd.Series | None = None) -> pd.DataFrame:
    X = pd.concat([lag_frame(node_df, target), cyclical(node_df)], axis=1)
    for col in ("rh_frac", "wind_speed_ms", "wind_dir_deg", "temp_c"):
        X[col] = np.clip(_met_column(node_df, col), -1e6, 1e6)
    X["upwind"] = (upwind if upwind is not None
                   else pd.Series(0.0, index=node_df.index))
    return X.replace([np.inf, -np.inf], np.nan)


@dataclasses.dataclass(frozen=True)
class ForecastSkill:
    target: str
    horizon_h: int
    n: int
    model_rmse: float
    persistence_rmse: float
    diurnal_rmse: float
    skill_vs_persistence: float
    skill_vs_diurnal: float
    model_mae: float
    model_r2: float

    def as_dict(self) -> dict[str, object]:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


class GraphForecaster:
    """One gradient-boosted regressor per horizon.

    Deliberately not a deep network. At 40 nodes and 60 days there are about
    58k node-hours; a graph neural network would be the right architecture for
    the deployed system and the wrong one for a dataset this size, and saying
    so is better than quietly overfitting a pretty model.
    """

    def __init__(self, target: str = "pm25", horizons: Sequence[int] = HORIZONS,
                 random_state: int = 20261010):
        self.target = target
        self.horizons = tuple(horizons)
        self.random_state = random_state
        self._models: dict[int, HistGradientBoostingRegressor] = {}
        self._diurnal: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y_next: Mapping[int, pd.Series]) -> "GraphForecaster":
        if len(y_next) != len(self.horizons):
            raise CalibrationError("need one target series per horizon")
        for h in self.horizons:
            series = y_next[h]
            common = X.index.intersection(series.index)
            feats = X.loc[common].ffill()
            keep = series.loc[common].notna() & feats.notna().all(axis=1)
            if keep.sum() < 400:
                raise CalibrationError(
                    f"horizon {h}h has only {int(keep.sum())} usable rows")
            model = HistGradientBoostingRegressor(
                random_state=self.random_state, max_iter=250, learning_rate=0.07,
                max_depth=4, min_samples_leaf=32, l2_regularization=1.0)
            model.fit(feats[keep].to_numpy(), series.loc[common][keep].to_numpy())
            self._models[h] = model
        return self

    def predict(self, X: pd.DataFrame, horizon_h: int) -> pd.Series:
        if horizon_h not in self._models:
            raise CalibrationError(f"no model for {horizon_h}h")
        feats = X.ffill()
        vals = self._models[horizon_h].predict(feats.to_numpy())
        return pd.Series(np.clip(vals, 0.0, None), index=feats.index)

    def fit_seasonal_baseline(self, y: pd.Series) -> "GraphForecaster":
        """Mean by (hour, day-of-week) - the cheap forecast it must beat."""
        key = pd.MultiIndex.from_arrays([y.index.hour, y.index.dayofweek])
        table = pd.Series(y.to_numpy(), index=key).groupby(level=[0, 1]).mean()
        self._diurnal = table
        return self

    def seasonal_predict(self, index: pd.DatetimeIndex) -> pd.Series:
        if self._diurnal is None:
            raise CalibrationError("call fit_seasonal_baseline first")
        pairs = list(zip(index.hour, index.dayofweek))
        return pd.Series([self._diurnal.get(p, np.nan) for p in pairs], index=index,
                         dtype=float)


def evaluate_forecast(frames: Mapping[int, pd.DataFrame],
                     target: str = "pm25") -> list[ForecastSkill]:
    """Score each horizon against persistence and an hour-of-week climatology.

    Takes one DataFrame per horizon with columns actual, predicted, persistence
    and seasonal, already aligned by the caller. Taking four separately-indexed
    series instead would have hidden an alignment bug: stacking six nodes onto a
    shared timestamp index produces duplicate labels, and pandas then refuses to
    reindex - or worse, silently reindexes only the first occurrence.
    """
    from ..metrics import mae, r2, rmse

    required = {"actual", "predicted", "persistence", "seasonal"}
    out: list[ForecastSkill] = []
    for h in sorted(frames):
        df = frames[h]
        missing = required - set(df.columns)
        if missing:
            raise CalibrationError(f"horizon {h}h is missing columns {sorted(missing)}")
        clean = df.dropna()
        if len(clean) < 50:
            continue
        a, p = clean["actual"], clean["predicted"]
        b, se = clean["persistence"], clean["seasonal"]
        m_rmse, b_rmse, s_rmse = rmse(a, p), rmse(a, b), rmse(a, se)
        out.append(ForecastSkill(
            target=target, horizon_h=int(h), n=int(len(clean)),
            model_rmse=m_rmse, persistence_rmse=b_rmse, diurnal_rmse=s_rmse,
            skill_vs_persistence=(1 - m_rmse / b_rmse) if b_rmse > 0 else float("nan"),
            skill_vs_diurnal=(1 - m_rmse / s_rmse) if s_rmse > 0 else float("nan"),
            model_mae=mae(a, p), model_r2=r2(a, p),
        ))
    return out


def hourly_adjacency(nodes: Sequence[Node], wind_dir: pd.Series
                     ) -> dict[pd.Timestamp, dict[str, list[tuple[str, float]]]]:
    """Precompute the directed graph for every hour in the window.

    Cached because adjacency depends only on the wind, and recomputing 40x4
    pairwise bearings for every node and every horizon would dominate the run.
    """
    return {ts: adjacency(nodes, float(w)) for ts, w in wind_dir.items()}
