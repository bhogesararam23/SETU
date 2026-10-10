"""Source attribution from the particulate size distribution.

This is the layer the novelty claim rests on, so the claim is stated narrowly
and the honesty is structural rather than rhetorical.

What exists in the literature: Kaur, Mangin and Kelly, Atmos. Meas. Tech.
19:1077 (2026) use the coarse-to-ultrafine mass ratio from a Plantower
PMS5003/6003 with humidity to *flag* dust-dominated PM, cutting bias from
59.8 +- 24.3% and lifting R^2 from 0.290 to 0.653 - with fixed thresholds and
non-linear regression, explicitly not a learned classifier.

What this module adds: a learned multinomial classifier over low-cost size
fractions, gases and meteorology, whose output is a posterior rather than a
label, and whose published size signatures enter as a prior that the model is
required to respect rather than as prose in a design document.

The prior blend is the part worth reading. Setting PRIOR_BLEND = 0.0 recovers a
pure data-driven classifier; setting it to 1.0 ignores the data entirely. It is
not a hidden knob: the validation report prints the value used, and
test_prior_blend_changes_the_result_printably fails if it turns out to be
decorative.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from ..config import Signature
from ..errors import ConfigError

#: Derived feature names. Kept explicit so a reader can audit what the model
#: is actually allowed to see.
ATTR_FEATURES: tuple[str, ...] = (
    "coarse_ratio",
    "log_coarse_count_ratio",
    "coarse_ugm3",
    "co_excess",
    "no2_excess",
    "o3_excess",
    "so2_excess",
    "rh_frac",
    "wind_speed_ms",
    "upwind_coarse_ratio",
    "hour_sin",
    "hour_cos",
)

#: Classes the classifier may name. "ambiguous" is not a class - it is what the
#: gate returns when no class clears it.
ABSTAIN = "ambiguous"

#: How strongly the published prior is allowed to pull. 0 = ignore it.
PRIOR_BLEND = 0.5


def _rolling_median_excess(series: pd.Series, window: int = 168) -> pd.Series:
    """Excess over a TRAILING rolling median.

    A centred window is the obvious choice and is wrong: it lets the baseline at
    09:00 know what the concentration will be at 15:00, so a model fitted on the
    first two-thirds is scored against features that already contain the test
    period. Trailing keeps every feature computable at the moment it is reported.
    """
    base = series.rolling(window, min_periods=48).median()
    return (series - base).clip(lower=0.0)


def build_features(raw: pd.DataFrame,
                   upwind_coarse_ratio: pd.Series | None = None) -> pd.DataFrame:
    """Turn a node's raw channels into the attribution design matrix.

    `upwind_coarse_ratio` is supplied by the caller because it is a *network*
    quantity: a node cannot tell regional transport from local emissions on its
    own, which is the entire reason SETU is a grid rather than a sensor.
    """
    f = pd.DataFrame(index=raw.index)
    pm10 = raw["pm10"].astype(float).clip(lower=1e-6)
    f["coarse_ratio"] = ((pm10 - raw["pm25"].astype(float)) / pm10).clip(0.0, 1.0)
    coarse_bins = raw["n2.5"].astype(float) + raw["n5.0"].astype(float) + raw["n10.0"].astype(float)
    fine_bins = raw["n0.3"].astype(float) + raw["n0.5"].astype(float) + 1.0
    f["log_coarse_count_ratio"] = np.log1p(coarse_bins / fine_bins)
    f["coarse_ugm3"] = (pm10 - raw["pm25"].astype(float)).clip(lower=0.0)
    f["co_excess"] = _rolling_median_excess(raw["co"].astype(float))
    f["no2_excess"] = _rolling_median_excess(raw["no2"].astype(float))
    f["o3_excess"] = _rolling_median_excess(raw["o3"].astype(float))
    f["so2_excess"] = _rolling_median_excess(raw["so2"].astype(float))
    f["rh_frac"] = np.clip(raw["rh"].astype(float) / 100.0, 0.0, 1.0)
    f["wind_speed_ms"] = raw["wind_speed_ms"].astype(float)
    if upwind_coarse_ratio is None:
        f["upwind_coarse_ratio"] = 0.0
    else:
        f["upwind_coarse_ratio"] = upwind_coarse_ratio.reindex(raw.index).fillna(0.0)
    hour = raw.index.hour.to_numpy()
    f["hour_sin"] = np.sin(2 * math.pi * hour / 24.0)
    f["hour_cos"] = np.cos(2 * math.pi * hour / 24.0)
    return f.loc[:, list(ATTR_FEATURES)]


@dataclasses.dataclass(frozen=True)
class Attribution:
    at: pd.Timestamp
    node: str
    posterior: Mapping[str, float]
    top: str
    entropy: float
    actionable: bool
    margin: float

    def as_dict(self) -> dict[str, object]:
        return {
            "at": str(self.at),
            "node": self.node,
            "posterior": {k: round(v, 4) for k, v in self.posterior.items()},
            "top": self.top,
            "margin": round(self.margin, 4),
            "entropy": round(self.entropy, 4),
            "actionable": self.actionable,
        }


class SourceClassifier:
    def __init__(self, signatures: Mapping[str, Signature],
                 blend: float = PRIOR_BLEND, random_state: int = 20261010,
                 c_param: float = 1.0,
                 classes: Sequence[str] | None = None):
        """`classes` restricts the fitted label set.

        A short run over few nodes may never contain a rare source - a
        twelve-day construction campaign does not appear in a 45-day window at
        all if none of the sampled nodes sit near a site. Refusing to fit is
        right for a validation run and wrong for a demo, so the caller can
        narrow the class set instead; the priors are renormalised over what
        remains and the exclusion is reported rather than hidden.
        """
        if not 0.0 <= blend <= 1.0:
            raise ConfigError("blend must be in [0, 1]")
        available = set(signatures)
        chosen = tuple(sorted(classes)) if classes is not None else tuple(sorted(available))
        missing = set(chosen) - available
        if missing:
            raise ConfigError(f"unknown sources requested: {sorted(missing)}")
        self.classes = chosen
        self.excluded = tuple(sorted(available - set(chosen)))
        raw = np.array([signatures[c].prior_weight for c in self.classes])
        total = float(raw.sum())
        if total <= 0:
            raise ConfigError("requested classes carry zero prior mass")
        self.prior = raw / total
        self.actionable = np.array([signatures[c].actionable_locally
                                    for c in self.classes])
        self.blend = blend
        self.random_state = random_state
        self.c_param = c_param
        self._model: LogisticRegression | None = None
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None

    # -- fit ---------------------------------------------------------------
    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "SourceClassifier":
        X, y = self._align(features, labels)
        if len(y) < 500:
            raise ConfigError(f"only {len(y)} labelled hours; refusing to fit")
        missing = set(self.classes) - set(np.unique(y))
        if missing:
            raise ConfigError(f"training set never contains {sorted(missing)}")
        self._mu = X.mean(axis=0)
        self._sd = np.where(X.std(axis=0) < 1e-9, 1.0, X.std(axis=0))
        # scikit-learn removed `multi_class` in 1.9 - multinomial is now the
        # only behaviour for lbfgs with more than two classes. Fall back for
        # older installs so a clone works either way.
        common = dict(C=self.c_param, max_iter=2000, class_weight="balanced",
                      random_state=self.random_state)
        try:
            self._model = LogisticRegression(**common)
        except TypeError:  # pragma: no cover - depends on installed version
            self._model = LogisticRegression(multi_class="multinomial", **common)
        self._model.fit((X - self._mu) / self._sd, y)
        return self

    def _align(self, features: pd.DataFrame, labels: pd.Series
               ) -> tuple[np.ndarray, np.ndarray]:
        common = features.index.intersection(labels.index)
        f = features.loc[common, list(ATTR_FEATURES)].astype(float)
        lab = labels.loc[common].astype(str)
        keep = f.notna().all(axis=1) & lab.isin(self.classes)
        return f[keep].to_numpy(), lab[keep].to_numpy()

    # -- predict -----------------------------------------------------------
    def posterior(self, features: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise ConfigError("call fit() first")
        X = features.loc[:, list(ATTR_FEATURES)].astype(float).to_numpy()
        Z = (np.nan_to_num(X, nan=0.0) - self._mu) / self._sd
        p = self._model.predict_proba(Z)

        # Prior odds adjustment. blend=0 leaves the learned posterior alone;
        # blend=1 discards it. The exponent is applied to the ratio of prior to
        # uniform so a class the literature says is rare cannot be conjured by
        # one unusual humidity reading.
        if self.blend > 0:
            uniform = 1.0 / len(self.classes)
            tilt = (self.prior / uniform) ** self.blend
            p = p * tilt
            p = p / p.sum(axis=1, keepdims=True)

        out = pd.DataFrame(p, index=features.index, columns=list(self.classes))
        out["entropy"] = -(out * np.log(out.clip(lower=1e-12))).sum(axis=1) / math.log(
            len(self.classes))
        top = out[list(self.classes)].idxmax(axis=1)
        ordered = np.sort(out[list(self.classes)].to_numpy(), axis=1)
        out["top"] = top
        out["margin"] = ordered[:, -1] - ordered[:, -2]
        return out

    def attribute(self, features: pd.DataFrame, node: str,
                  min_posterior: float = 0.45) -> list[Attribution]:
        post = self.posterior(features)
        out: list[Attribution] = []
        for at, row in post.iterrows():
            top = str(row["top"])
            conf = float(row[top])
            is_actionable = bool(self.actionable[self.classes.index(top)])
            clear = conf >= min_posterior
            out.append(Attribution(
                at=at, node=node,
                posterior={c: float(row[c]) for c in self.classes},
                top=top if clear else ABSTAIN,
                entropy=float(row["entropy"]),
                actionable=bool(clear and is_actionable),
                margin=float(row["margin"]),
            ))
        return out

    # -- diagnostics -------------------------------------------------------
    def coefficients(self) -> pd.DataFrame:
        """Per-class standardised coefficients, so the model is inspectable.

        If the learned signs contradict the published signatures, that is a
        finding worth seeing rather than a detail buried in a pickle.
        """
        if self._model is None:
            raise ConfigError("call fit() first")
        return pd.DataFrame(self._model.coef_, index=self.classes,
                            columns=list(ATTR_FEATURES))

    def log_loss(self, features: pd.DataFrame, labels: pd.Series) -> float:
        X, y = self._align(features, labels)
        p = self.posterior(pd.DataFrame(X, columns=list(ATTR_FEATURES),
                                        index=features.index[: len(X)]))
        return float(log_loss(y, p[list(self.classes)].to_numpy(),
                              labels=list(self.classes)))


def seasonal_shares(attributions: Sequence[Attribution]) -> pd.Series:
    """Time share per source, for comparison against receptor studies.

    This is the cross-check the deck promised: if SETU says construction is
    40% of the year where chemical mass balance says 0-15%, the classifier is
    wrong and the report has to say so.
    """
    labels = [a.top for a in attributions if a.top != ABSTAIN]
    if not labels:
        return pd.Series(dtype=float)
    counts = pd.Series(labels).value_counts(normalize=True)
    return counts.rename("share")


def compare_to_receptor(shares: pd.Series, published: Mapping[str, tuple[float, float]]
                        ) -> pd.DataFrame:
    """Flag every source whose SETU share falls outside the published range."""
    rows = []
    for source, (lo, hi) in published.items():
        observed = float(shares.get(source, 0.0))
        rows.append({
            "source": source,
            "setu_share": round(observed, 4),
            "published_low": lo,
            "published_high": hi,
            "within": bool(lo <= observed <= hi),
        })
    return pd.DataFrame(rows).set_index("source")
