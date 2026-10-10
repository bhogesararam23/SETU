"""Raw-to-reference regression, and the uncertainty that travels with it.

The method follows Bush et al., Atmos. Meas. Tech. 15:3261 (2022),
doi:10.5194/amt-15-3261-2022, which mapped low-cost channels to a reference
with a tree ensemble and reported MAE reductions of 37-94% and validation R^2
up to 0.91. Gradient boosting is used here rather than a linear fit for the
same reason they give: the humidity response is not linear and the sensor
saturates.

The part that makes this usable by a municipal officer rather than only by a
data scientist is the interval. Two extra quantile regressors are fitted
alongside the median model, so every prediction carries a sigma, and the
response engine refuses to fire when that interval straddles the threshold it
is about to act on.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..errors import CalibrationError
from ..metrics import Scorecard, score

#: Channels the node emits that are legal to use as predictors. Nothing here
#: leaks the target: the mass channels are the raw, uncorrected readings.
FEATURES: tuple[str, ...] = (
    "pm1", "pm25", "pm4", "pm10",
    "n0.3", "n0.5", "n1.0", "n2.5", "n5.0", "n10.0",
    "rh", "temp_c", "wind_speed_ms",
)

#: Quantile interval used to build the reported uncertainty.
LOW_Q, HIGH_Q = 0.1, 0.9
#: A normal-approximation divisor converting a 10-90 range into ~1 sigma.
Z_90 = 2.5631


@dataclasses.dataclass(frozen=True)
class CalibrationReport:
    node: str
    target: str
    n_train: int
    n_eval: int
    features: tuple[str, ...]
    before: Scorecard
    after: Scorecard
    interval_coverage: float
    mean_sigma: float
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "node": self.node,
            "target": self.target,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "features_used": list(self.features),
            "before": self.before.as_dict(),
            "after": self.after.as_dict(),
            "interval_coverage": round(self.interval_coverage, 4),
            "mean_sigma": round(self.mean_sigma, 3),
            "mae_reduction_pct": round(
                100.0 * (1 - self.after.mae / self.before.mae)
                if self.before.mae > 0 else 0.0, 1),
            "notes": self.notes,
        }


def _design(raw: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    missing = [f for f in features if f not in raw.columns]
    if missing:
        raise CalibrationError(f"raw frame is missing features {missing}")
    return raw.loc[:, list(features)].astype(float)


class CalibrationModel:
    """Median plus two quantile gradient-boosted regressors."""

    def __init__(self, target: str = "pm25", features: Sequence[str] = FEATURES,
                 random_state: int = 20261010, max_iter: int = 300):
        if target not in FEATURES + ("pm25", "pm10", "coarse"):
            raise CalibrationError(f"unusable calibration target {target!r}")
        self.target = target
        # The target itself can never be a predictor.
        self.features = tuple(f for f in features if f != target)
        self.random_state = random_state
        self.max_iter = max_iter
        self._median: HistGradientBoostingRegressor | None = None
        self._lo: HistGradientBoostingRegressor | None = None
        self._hi: HistGradientBoostingRegressor | None = None
        self.n_train = 0

    # -- fitting -----------------------------------------------------------
    def fit(self, raw: pd.DataFrame, reference: pd.Series) -> "CalibrationModel":
        X = _design(raw, self.features)
        y = pd.Series(reference, index=raw.index).reindex(raw.index).astype(float)
        keep = y.notna() & np.isfinite(X.to_numpy()).all(axis=1)
        X, y = X[keep], y[keep]
        if len(y) < 240:
            raise CalibrationError(
                f"only {len(y)} usable co-located hours; refusing to fit a "
                "300-tree ensemble on less than 240"
            )

        def make(alpha: float | None = None) -> HistGradientBoostingRegressor:
            kwargs = dict(random_state=self.random_state, max_iter=self.max_iter,
                          learning_rate=0.06, max_depth=4, min_samples_leaf=24,
                          l2_regularization=1.0)
            if alpha is not None:
                kwargs.update(loss="quantile", alpha=alpha)
            return HistGradientBoostingRegressor(**kwargs)

        self._median = make()
        self._lo = make(LOW_Q)
        self._hi = make(HIGH_Q)
        for model in (self._median, self._lo, self._hi):
            model.fit(X.to_numpy(), y.to_numpy())
        self.n_train = int(len(y))
        return self

    @property
    def is_fitted(self) -> bool:
        return self._median is not None

    # -- prediction --------------------------------------------------------
    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Returns columns value, low, high, sigma aligned to raw's index."""
        if not self.is_fitted:
            raise CalibrationError("call fit() first")
        X = _design(raw, self.features).to_numpy()
        med = self._median.predict(X)
        lo = self._lo.predict(X)
        hi = self._hi.predict(X)
        # Quantile crossing is possible on a small sample; enforcing the order
        # is a modelling assumption, not a measurement, so it is done here
        # where it is visible rather than buried in the engine.
        lo = np.minimum(lo, med)
        hi = np.maximum(hi, med)
        out = pd.DataFrame({
            "value": np.clip(med, 0.0, None),
            "low": np.clip(lo, 0.0, None),
            "high": np.clip(hi, 0.0, None),
        }, index=raw.index)
        out["sigma"] = np.maximum((out["high"] - out["low"]) / (2.0 * Z_90), 0.5)
        return out

    # -- evaluation --------------------------------------------------------
    def evaluate(self, raw: pd.DataFrame, reference: pd.Series,
                 threshold: float | None = None, node: str = "?") -> CalibrationReport:
        ref = pd.Series(reference, index=raw.index).reindex(raw.index).astype(float)
        keep = ref.notna()
        raw, ref = raw[keep], ref[keep]
        pred = self.predict(raw)

        before = score(ref, raw[self.target].astype(float), threshold=threshold)
        after = score(ref, pred["value"], threshold=threshold)
        inside = ((ref >= pred["low"]) & (ref <= pred["high"])).mean()
        return CalibrationReport(
            node=node, target=self.target, n_train=self.n_train, n_eval=int(len(ref)),
            features=self.features, before=before, after=after,
            interval_coverage=float(inside), mean_sigma=float(pred["sigma"].mean()),
        )


def transfer_calibrate(target_raw: pd.DataFrame, donor_raw: pd.DataFrame,
                       donor: CalibrationModel, overlap_hours: int = 168
                       ) -> tuple[CalibrationModel, dict[str, float]]:
    """Pair-based transfer: calibrate a node that was never co-located.

    A donor node that *is* calibrated is placed near the target for a short
    overlap window; a linear map is fitted between their raw readings over that
    window; thereafter the target's raw readings are pushed through the map and
    then through the donor's model. This is the approach in Kumar et al. (2015)
    and it is what lets coverage scale without a reference at every site.

    Returns the usable model for the target and the transfer diagnostics. A
    poor overlap fit is reported rather than hidden - the caller decides whether
    to trust the node.
    """
    common = target_raw.index.intersection(donor_raw.index)[:overlap_hours]
    if len(common) < 48:
        raise CalibrationError(
            f"only {len(common)} overlap hours; transfer needs at least 48"
        )
    src = target_raw.loc[common, donor.target].astype(float).to_numpy()
    dst = donor_raw.loc[common, donor.target].astype(float).to_numpy()
    if np.std(src) < 1e-9:
        raise CalibrationError("target raw channel is constant over the overlap")

    slope, intercept = np.polyfit(src, dst, 1)
    fitted = intercept + slope * src
    ss_res = float(np.sum((dst - fitted) ** 2))
    ss_tot = float(np.sum((dst - np.mean(dst)) ** 2))
    diag = {
        "slope": float(slope),
        "intercept": float(intercept),
        "overlap_hours": int(len(common)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }

    class _Transferred:
        """Wraps a donor model with the linear map applied first."""

        def __init__(self, inner: CalibrationModel, a: float, b: float):
            self.inner, self.a, self.b = inner, a, b
            self.target = inner.target
            self.features = inner.features
            self.n_train = inner.n_train
            self.is_fitted = True

        def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
            shifted = raw.copy()
            shifted[self.target] = self.b + self.a * raw[self.target].astype(float)
            return self.inner.predict(shifted)

        def evaluate(self, raw, reference, threshold=None, node="?"):
            shifted = raw.copy()
            shifted[self.target] = self.b + self.a * raw[self.target].astype(float)
            return self.inner.evaluate(shifted, reference, threshold, node)

    return _Transferred(donor, float(slope), float(intercept)), diag  # type: ignore[return-value]
