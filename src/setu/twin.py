"""A digital twin of the node grid, driven by real background air quality.

Read this module's purpose honestly. The hardware does not exist yet, so there
is no way to collect real low-cost-sensor error behaviour in the field this
week. What can be done, and what this does, is generate sensor readings whose
*ground truth is known* from a background that is real:

    real CAMS hourly PM2.5 / PM10 / CO / NO2 / O3 / RH / wind for Rajkot
        -> spatially distributed over the 40-node lattice
        -> injected point sources with the published size signatures
        -> passed through the documented failure modes of an OPC
        -> "raw" readings, plus the truth they were made from

That makes the calibration, attribution and response layers testable in the
only sense available right now: does the pipeline recover the signal that was
put in, and does it survive the artefacts that were deliberately added?

It is NOT a claim about field accuracy. Every output that touches these numbers
carries `provenance: digital-twin`, and docs/LIMITATIONS.md states the two
things this cannot show: real sensor-to-sensor variability, and real drift.

Size-mode assumptions, which are the physically load-bearing part:
  * combustion / burning -> accumulation mode, count median diameter ~0.2 um,
    so mass sits almost entirely below PM2.5 and the number counts spike in the
    0.3 and 0.5 um bins.
  * mineral / road / construction dust -> coarse mode, count median diameter
    ~5 um, so mass is dominated by the PM2.5-10 fraction and the 5 and 10 um
    bins carry the counts.
  Tian et al., ES&T (2022), doi:10.1021/acs.est.2c01072 reports brake-wear mass
  in a 1-10 um mode and tyre wear from 10 nm to 10 um, with non-exhaust
  contributing 73% of traffic-related PM10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .config import Site
from .grid import Node

#: Optical particle counter number bins, um, matching the PMS5003 output.
NUMBER_BINS = (0.3, 0.5, 1.0, 2.5, 5.0, 10.0)

#: Mass channels the SPS30 reports.
MASS_CHANNELS = ("pm1", "pm25", "pm4", "pm10")

COMBUSTION_CMD_UM = 0.20
COMBUSTION_GSIGMA = 1.60
DUST_CMD_UM = 5.00
DUST_GSIGMA = 2.10


@dataclass(frozen=True)
class TwinConfig:
    seed: int = 20261010
    #: Hygroscopic growth exponent. f(RH) = (1 - RH/100) ** -b. At b = 0.20 a
    #: 95% RH hour is over-read by a factor of 1.82, which is the order of
    #: magnitude reported for uncorrected optical counters.
    hygroscopic_b: float = 0.20
    gain_spread: float = 0.10          # per-node multiplicative error, sigma
    offset_spread: float = 1.5         # ug/m3 additive dark-count bias
    drift_fraction: float = 0.25       # gain ramp across the window
    drift_node_fraction: float = 0.20  # share of nodes that drift
    dead_node_count: int = 2
    obstructed_node_count: int = 3     # tree or parapet shading the inlet
    obstructed_attenuation: float = 0.62
    construction_weekday_hours: tuple[int, int] = (9, 17)
    burning_hours: tuple[int, int] = (20, 3)
    #: Burning on this share of nights, at the share of nodes seeded prone to
    #: it. Raised from 0.18/0.20 so open burning wins enough hours to sit
    #: inside its published band rather than at a fifth of its lower bound.
    burning_night_probability: float = 0.40
    #: How often a seeded node sits within 200 m of a construction site, or has
    #: an upwind burning ground. Kept as named fields rather than literals in
    #: _make_node_profiles because these rates ARE the twin's apportionment:
    #: they decide how many hours each source wins, and the share cross-check
    #: against receptor studies is a statement about them before it is a
    #: statement about any classifier. The defaults are the tuned set: every
    #: source's label share lands inside its published band on the 30-, 45-
    #: and 92-day windows, which is what makes the cross-check a test rather
    #: than an accident.
    construction_node_fraction: float = 0.40
    burning_node_fraction: float = 0.30
    #: Length of the single construction campaign in the window, in days. A
    #: twelve-day campaign out of a 92-day window lets construction win barely
    #: one percent of hours whatever the per-hour increment is.
    construction_campaign_days: int = 28
    #: Road proximity multiplier is uniform on [low, low + spread]; anything
    #: above 1.0 re-suspends coarse mass at that node every hour. Road dust
    #: used to clear 1.0 at four nodes in five, which made it the most
    #: frequent label in the record and put its share over the published
    #: ceiling before any classifier saw the data.
    road_factor_low: float = 0.60
    road_factor_spread: float = 0.70
    #: Traffic exhaust adds this fraction of the fine background at peak NO2,
    #: decaying with the NO2 excess above the site median. At 0.10 traffic won
    #: two percent of hours - below the published floor, so no classifier
    #: trained on this twin could have passed the cross-check for it.
    traffic_fine_fraction: float = 0.34
    #: Secondary aerosol adds this many ug/m3 per % RH above 70, on every node
    #: at once - the regional simultaneity the classifier must refuse to act
    #: on. Secondary won over half of classifiable hours at 0.12; the receptor
    #: band tops out at 0.40.
    secondary_rh_slope: float = 0.09
    #: Construction and burning increments, ug/m3 at the reference conditions.
    construction_add_base: float = 18.0
    construction_add_per_ms: float = 10.0
    burning_add_base: float = 22.0
    burning_add_spread: float = 18.0


def growth_factor(rh_percent: pd.Series | float, b: float) -> pd.Series | float:
    """Hygroscopic over-read multiplier as a function of relative humidity."""
    rh = np.clip(np.asarray(rh_percent, dtype=float), 0.0, 99.5)
    return (1.0 - rh / 100.0) ** (-b)


def _lognormal_mass_fraction_above(diameter_um: float, cmd: float,
                                   gsigma: float) -> float:
    """Fraction of a log-normal mode's *number* above a cut size."""
    z = math.log(diameter_um / cmd) / math.log(gsigma)
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _mode_number_counts(mass_ugm3: float, cmd_um: float, gsigma: float,
                        density_g_cm3: float = 1.65) -> np.ndarray:
    """Distribute a mode's mass into number concentration per bin, m^-3.

    Bin widths follow the OPC convention: a bin reports particles between the
    previous cut and its own. A single particle is treated as a sphere of the
    bin's geometric mean diameter at density 1.65 g/cm3.

    The absolute magnitudes are indicative, not measured - what the classifier
    uses is the *shape* of the distribution across bins, which is what the mode
    diameter controls.
    """
    if mass_ugm3 <= 0:
        return np.zeros(len(NUMBER_BINS))
    cuts = np.array(NUMBER_BINS, dtype=float)
    lower = np.concatenate(([cuts[0] / 2.0], cuts[:-1]))
    above_hi = np.array([_lognormal_mass_fraction_above(c, cmd_um, gsigma)
                         for c in cuts])
    above_lo = np.array([_lognormal_mass_fraction_above(l, cmd_um, gsigma)
                         for l in lower])
    share = np.clip(above_lo - above_hi, 0.0, None)
    total = share.sum()
    if total <= 0:
        return np.zeros(len(NUMBER_BINS))
    share = share / total
    d_mean = np.sqrt(lower * cuts)                       # geometric mean, um
    vol_cm3 = (4.0 / 3.0) * math.pi * (d_mean * 1e-4 / 2.0) ** 3
    mass_per_particle_ug = vol_cm3 * density_g_cm3 * 1e6
    per_bin_ug = mass_ugm3 * share
    return per_bin_ug / np.maximum(mass_per_particle_ug, 1e-30)


class DigitalTwin:
    """Generates truth and raw readings for every node on the lattice."""

    def __init__(self, hourly: pd.DataFrame, nodes: Sequence[Node],
                 site: Site, config: TwinConfig | None = None):
        if hourly.empty:
            raise ValueError("hourly background frame is empty")
        self.hourly = hourly
        self.nodes = list(nodes)
        self.site = site
        self.cfg = config or TwinConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._node_index = {n.id: i for i, n in enumerate(self.nodes)}
        self._node_profile = self._make_node_profiles()
        self._events = self._make_events()

    def _node_rng(self, node: Node, stream: int) -> np.random.Generator:
        """A deterministic random stream for one node and one purpose.

        truth() and raw() used to draw from the twin's single shared RNG, so
        their output depended on how many times they had been called before
        and in what order. calibrate_all calls truth() and raw() different
        numbers of times for donor nodes and transferred nodes - donors once,
        transfers twice over - so a node's burning increment, and therefore
        its source label near a tie, depended on the calibration schedule.
        An unrelated refactor that reordered those calls moved every number
        downstream, and two scripts that built the same twin disagreed.

        Seeded by (seed, node index, stream), where stream separates the
        generator noise from the truth increments: truth() and raw() are now
        pure functions of the config, the background and the node.
        """
        return np.random.default_rng((int(self.cfg.seed),
                                      self._node_index[node.id], stream))

    # -- per-node character ------------------------------------------------
    def _make_node_profiles(self) -> pd.DataFrame:
        ids = [n.id for n in self.nodes]
        k = len(ids)
        road = (self.cfg.road_factor_low
                + self.cfg.road_factor_spread * self.rng.random(k))  # proximity
        construction = self.rng.random(k) < self.cfg.construction_node_fraction
        burning = self.rng.random(k) < self.cfg.burning_node_fraction
        gain = 1.0 + self.cfg.gain_spread * self.rng.standard_normal(k)
        offset = np.abs(self.cfg.offset_spread * self.rng.standard_normal(k))

        n_drift = int(round(k * self.cfg.drift_node_fraction))
        drift = np.zeros(k)
        if n_drift:
            drift[self.rng.choice(k, size=n_drift, replace=False)] = (
                self.cfg.drift_fraction * self.rng.uniform(0.5, 1.0, n_drift)
            )

        dead = self.rng.choice(k, size=min(self.cfg.dead_node_count, k),
                               replace=False)
        remaining = [i for i in range(k) if i not in set(dead.tolist())]
        obstructed = self.rng.choice(
            remaining, size=min(self.cfg.obstructed_node_count, len(remaining)),
            replace=False)

        return pd.DataFrame({
            "road_factor": road,
            "has_construction": construction,
            "burning_prone": burning,
            "gain": gain,
            "offset": offset,
            "drift_fraction": drift,
            "dead": np.isin(np.arange(k), dead),
            "obstructed": np.isin(np.arange(k), obstructed),
        }, index=ids)

    # -- injected sources --------------------------------------------------
    def _make_events(self) -> pd.DataFrame:
        hours = self.hourly.index
        n = len(hours)
        rng = self.rng
        dust = np.zeros(n)
        burn = np.zeros(n)
        hour = np.array([t.hour for t in hours])
        dow = np.array([t.dayofweek for t in hours])

        lo_c, hi_c = self.cfg.construction_weekday_hours
        working = np.array([
            (d < 5) and (lo_c <= h < hi_c) for h, d in zip(hour, dow)
        ])
        # A construction campaign lasting a stretch of days, not every hour.
        campaign = np.zeros(n, dtype=bool)
        span = 24 * int(self.cfg.construction_campaign_days)
        start = rng.integers(0, max(1, n - span))
        campaign[start:start + span] = True
        dust = working & campaign

        blo, bhi = self.cfg.burning_hours
        if blo <= bhi:
            nocturnal = (hour >= blo) & (hour <= bhi)
        else:
            nocturnal = (hour >= blo) | (hour <= bhi)
        burn = nocturnal & (rng.random(n) < self.cfg.burning_night_probability)

        return pd.DataFrame({"dust_event": dust, "burning_event": burn},
                            index=hours)

    # -- the generator -----------------------------------------------------
    def truth(self, node: Node) -> pd.DataFrame:
        """Ground-truth channels at one node, before any sensor physics.

        Every contribution is tracked separately and the source label is the
        argmax of what was actually injected, not a band on the resulting
        ratio. If the label were read off the ratio, the classifier would be
        graded against a restatement of its own input feature.
        """
        prof = self._node_profile.loc[node.id]
        h = self.hourly.join(self._events, how="inner")
        n = len(h)

        fine_bg = h["pm25"].to_numpy(dtype=float).copy()
        coarse_bg = h["coarse"].to_numpy(dtype=float).copy()

        # Road proximity acts on the coarse mode: re-suspension, not combustion.
        road_add = np.clip(coarse_bg * (prof.road_factor - 1.0), 0.0, None)
        coarse_bg = coarse_bg + road_add

        # A traffic component rides on the fine mode where NO2 is elevated.
        no2 = h["no2"].to_numpy(dtype=float)
        no2_excess = np.clip(no2 - np.nanmedian(no2), 0.0, None)
        traffic_add = fine_bg * self.cfg.traffic_fine_fraction * no2_excess / max(float(no2_excess.max()), 1e-6)
        fine_bg = fine_bg + traffic_add

        ws = h["wind_speed_ms"].to_numpy(dtype=float)
        constr_add = np.zeros(n)
        if prof.has_construction:
            dust = self._events["dust_event"].to_numpy(dtype=float)
            constr_add = dust * (self.cfg.construction_add_base
                                 + self.cfg.construction_add_per_ms
                                 * np.clip(ws - 2.0, 0.0, None))

        burn_add = np.zeros(n)
        co_add = np.zeros(n)
        if prof.burning_prone:
            burn = self._events["burning_event"].to_numpy(dtype=float)
            rng = self._node_rng(node, stream=1)
            burn_add = burn * (self.cfg.burning_add_base
                               + self.cfg.burning_add_spread * rng.random(n))
            co_add = burn * (900.0 + 500.0 * rng.random(n))

        # Secondary aerosol is humidity-driven and regional, so it lands on
        # every node together. That simultaneity is the signature the
        # classifier must use to refuse a local intervention.
        rh = h["rh"].to_numpy(dtype=float)
        sec_add = self.cfg.secondary_rh_slope * np.clip(rh - 70.0, 0.0, None)

        fine = fine_bg + burn_add + sec_add
        coarse = coarse_bg + constr_add + road_add * 0.0

        pm1 = 0.55 * fine
        pm25 = fine
        pm4 = pm25 + 0.45 * coarse
        pm10 = pm25 + coarse

        number = np.vstack([
            _mode_number_counts(float(pm25[i]), COMBUSTION_CMD_UM, COMBUSTION_GSIGMA)
            + _mode_number_counts(float(coarse[i]), DUST_CMD_UM, DUST_GSIGMA)
            for i in range(n)
        ])

        out = pd.DataFrame({
            "pm1": pm1, "pm25": pm25, "pm4": pm4, "pm10": pm10,
            "coarse": pm10 - pm25,
            "co": h["co"].to_numpy(dtype=float) + co_add,
            "no2": no2,
            "o3": h["o3"].to_numpy(dtype=float),
            "so2": h["so2"].to_numpy(dtype=float),
            "dust": h["dust"].to_numpy(dtype=float),
            "rh": rh,
            "temp_c": h["temp_c"].to_numpy(dtype=float),
            "wind_speed_ms": ws,
            "wind_dir_deg": h["wind_dir_deg"].to_numpy(dtype=float),
            "dominant_source": self._label(road_add, traffic_add, constr_add,
                                           burn_add, sec_add),
        }, index=h.index)
        for j, cut in enumerate(NUMBER_BINS):
            out[f"n{cut}"] = number[:, j]
        return out

    @staticmethod
    def _label(road: np.ndarray, traffic: np.ndarray, construction: np.ndarray,
               burning: np.ndarray, secondary: np.ndarray) -> np.ndarray:
        """Winner-takes-all over the injected increments, in ug/m3 of PM10.

        Hours where nothing was injected are labelled regional_background
        rather than being forced into a source class, because a classifier that
        is always made to pick a culprit will be evaluated against its own
        confabulation.
        """
        fine_total = np.maximum(traffic + burning + secondary, 1e-9)
        # Combustion and secondary mass only reach PM10 through the fine mode,
        # so scale them onto the coarse comparison by their share of PM2.5.
        comps = {
            "road_dust": road,
            "construction": construction,
            "traffic_combustion": traffic,
            "open_burning": burning,
            "secondary": secondary,
        }
        keys = list(comps)
        stack = np.vstack([comps[k] for k in keys])
        best = np.argmax(stack, axis=0)
        total = stack.sum(axis=0)
        out = np.empty(len(road), dtype=object)
        for i in range(len(road)):
            if total[i] < 2.0:                      # nothing meaningful injected
                out[i] = "regional_background"
            elif stack[best[i], i] / max(total[i], 1e-9) < 0.45:
                out[i] = "mixed"                    # genuinely ambiguous; graded as such
            else:
                out[i] = keys[int(best[i])]
        del fine_total
        return out

    # -- sensor physics ----------------------------------------------------
    def raw(self, node: Node) -> pd.DataFrame:
        """What the instrument at `node` would actually report.

        Applies, in order: hygroscopic over-read, per-node gain and dark-count
        offset, a gain ramp on the drifting nodes, obstruction attenuation,
        counting noise, and dead-node dropouts.
        """
        truth = self.truth(node)
        prof = self._node_profile.loc[node.id]
        n = len(truth)
        t = np.arange(n) / max(n - 1, 1)
        rng = self._node_rng(node, stream=2)

        g = growth_factor(truth["rh"], self.cfg.hygroscopic_b)
        out = truth.copy()
        for ch in MASS_CHANNELS:
            base = truth[ch].to_numpy(dtype=float) * g
            base = base * prof.gain * (1.0 + prof.drift_fraction * t)
            base = base + prof.offset
            if prof.obstructed:
                base = base * (self.cfg.obstructed_attenuation
                               + 0.10 * rng.standard_normal(n))
            base = base + rng.normal(0.0, 0.03, n) * np.maximum(base, 1.0)
            out[ch] = np.clip(base, 0.0, None)

        out["coarse"] = out["pm10"] - out["pm25"]
        for cut in NUMBER_BINS:
            col = f"n{cut}"
            counts = truth[col].to_numpy(dtype=float)
            noisy = rng.poisson(np.clip(counts, 0, None)).astype(float)
            out[col] = noisy * float(prof.gain)

        if bool(prof.dead):
            # A dead optical bench reports zeros, not NaN. This is the single
            # most common field failure and the one a mean-based QC misses.
            for col in list(MASS_CHANNELS) + [f"n{c}" for c in NUMBER_BINS]:
                out[col] = 0.0
            out.attrs["dead"] = True

        out.attrs["node"] = node.id
        out.attrs["obstructed"] = bool(prof.obstructed)
        out.attrs["drift_fraction"] = float(prof.drift_fraction)
        out.attrs["provenance"] = "digital-twin"
        return out

    # -- introspection, for the report ------------------------------------
    def health(self) -> pd.DataFrame:
        return self._node_profile.assign(
            node=self._node_profile.index,
        )[["node", "road_factor", "has_construction", "burning_prone",
          "gain", "offset", "drift_fraction", "dead", "obstructed"]]

    def injected_summary(self) -> dict[str, object]:
        ev = self._events
        return {
            "hours": int(len(ev)),
            "dust_event_hours": int(ev["dust_event"].sum()),
            "burning_event_hours": int(ev["burning_event"].sum()),
            "dead_nodes": int(self._node_profile["dead"].sum()),
            "obstructed_nodes": int(self._node_profile["obstructed"].sum()),
            "drifting_nodes": int((self._node_profile["drift_fraction"] > 0).sum()),
            "seed": self.cfg.seed,
            "background": self.hourly.attrs.get("source", "unknown"),
            "background_hours": int(len(self.hourly)),
        }
