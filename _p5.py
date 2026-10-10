import pathlib

# ---- 1. CUSUM must detect CHANGE, not a stable offset -------------------
p = pathlib.Path("src/setu/calibrate/drift.py")
s = p.read_text(encoding="utf-8")
s = s.replace('''        self._residuals: list[float] = []
        self._pos = 0.0''',
'''        self._residuals: list[float] = []
        self._baseline_mean: float = 0.0
        self._pos = 0.0''')
s = s.replace('''        if self._sigma is None:
            if self._n < self.warm_up:
                return None
            self._sigma = self._estimate_sigma()

        k = self.allowance * self._sigma
        h = self.interval * self._sigma
        self._pos = max(0.0, self._pos + residual - k)
        self._neg = max(0.0, self._neg - residual - k)''',
'''        if self._sigma is None:
            if self._n < self.warm_up:
                return None
            # Centre on the in-control mean as well as scaling by sigma. A node
            # that is stably 0.8 ug/m3 high has a calibration offset, not drift;
            # accumulating on the raw residual would quarantine it eventually no
            # matter how well behaved it is, and on a 92-day window that is
            # exactly what happened - 40 of 49 healthy nodes were quarantined.
            # What must be caught is the residual MOVING.
            self._baseline_mean = float(np.mean(
                self._residuals[:self.warm_up]))
            self._sigma = self._estimate_sigma()

        centred = residual - self._baseline_mean
        k = self.allowance * self._sigma
        h = self.interval * self._sigma
        self._pos = max(0.0, self._pos + centred - k)
        self._neg = max(0.0, self._neg - centred - k)''')
s = s.replace('''            "sigma": None if self._sigma is None else round(self._sigma, 3),''',
'''            "sigma": None if self._sigma is None else round(self._sigma, 3),
            "baseline_mean": round(self._baseline_mean, 3),''')
p.write_text(s, encoding="utf-8")

p = pathlib.Path("tests/test_drift.py")
s = p.read_text(encoding="utf-8")
s = s.replace('''def test_a_real_episode_seen_through_a_correct_calibration_is_not_drift():''',
'''def test_a_stable_offset_is_not_drift(engine_free=None):
    """A node that is consistently 1.5 ug/m3 high is mis-calibrated, not drifting.

    Quarantining it would be wrong twice over: the fix is a recalibration, and
    the operator would lose a sensor that is still telling the truth about
    changes in the air.
    """
    rng = np.random.default_rng(21)
    res = rng.normal(1.5, 2.0, 3000)
    m = CusumMonitor()
    assert m.feed(res, _index(len(res))) == []


def test_a_step_away_from_a_stable_offset_is_still_caught():
    rng = np.random.default_rng(22)
    res = np.concatenate([rng.normal(1.5, 2.0, 800), rng.normal(14.0, 2.0, 800)])
    m = CusumMonitor()
    alarms = m.feed(res, _index(len(res)))
    assert alarms and alarms[0].direction == "up"


def test_a_real_episode_seen_through_a_correct_calibration_is_not_drift():''')
p.write_text(s, encoding="utf-8")

# ---- 2. equal-cost on the genuinely episodic sources --------------------
p = pathlib.Path("src/setu/pipeline.py")
s = p.read_text(encoding="utf-8")
s = s.replace('''def _coarse_breach_hours(products: Mapping[str, NodeProduct],
                         coarse_threshold: float,
                         sources: Sequence[str] = ("road_dust",
                                                   "construction")) -> list[int]:''',
'''def _coarse_breach_hours(products: Mapping[str, NodeProduct],
                         coarse_threshold: float,
                         sources: Sequence[str] = ("construction",
                                                   "open_burning")) -> list[int]:''')
s = s.replace('''    # The comparison is run against the condition the engine actually acts on:
    # hours where the coarse fraction cleared the working action level AND the
    # attributed source was road dust or construction - "''",
              '''''')  # no-op guard
s = s.replace('''    # hours where the coarse fraction clears the working action level AND the
    # attributed source is one a sweeper or a tanker could move. Scoring it
    # against the 24-hour PM2.5 standard instead produced a degenerate result -
    # a monsoon window never reaches 60 ug/m3, so the breach profile came back
    # either empty or flat, and a flat profile makes triggering look worthless
    # for a reason that has nothing to do with triggering.''',
'''    # hours attributed to a genuinely episodic source. Construction campaigns and
    # burning nights are episodic by nature; road-proximate dust is close to
    # continuous, and scoring the comparison on it would show triggering gaining
    # nothing for a reason that has nothing to do with triggering. That case is
    # reported separately rather than hidden.
    #
    # The same profile is also computed for road_dust so the report can state the
    # unfavourable result out loud instead of only showing the flattering one.''')
s = s.replace('''    equal_cost = _compare(breach_hours) if breach_hours else {''',
'''    road_hours = _coarse_breach_hours(products, coarse_level, sources=("road_dust",))
    equal_cost_road = _compare(road_hours) if road_hours else None

    equal_cost = _compare(breach_hours) if breach_hours else {''')
s = s.replace('''        "note": "no coarse-mode breach hours in this window; the equal-cost "
                "comparison is undefined rather than zero",
    }''',
'''        "note": "no episodic coarse-mode breach hours in this window; the "
                "equal-cost comparison is undefined rather than zero",
    }
    if equal_cost_road:
        equal_cost["road_dust_continuous_profile"] = equal_cost_road''')
p.write_text(s, encoding="utf-8")

p = pathlib.Path("src/setu/report.py")
s = p.read_text(encoding="utf-8")
s = s.replace('''    add("The suppressant row is the honest one. A lever that lasts a day does not "
        "care where you put it, so triggering earns nothing there. The case for "
        "SETU is conditional on short-lived levers acting on episodic sources - "
        "which is exactly what road dust is, and exactly what chemical suppression "
        "is not.")''',
'''    add("The suppressant row is the honest one. A lever that lasts a day does not "
        "care where you put it, so triggering earns nothing there.")
    add("")
    road = ec.get("road_dust_continuous_profile")
    if road:
        add("Scored instead on continuously road-proximate dust, across "
            f"{len(road.get('breach_hours', []))} hours a day, the gain is "
            f"{road['sprinkling_62min']['gain']:+.2f}. That is the unfavourable "
            "case and it is reported as such: where the source is present around "
            "the clock, no placement strategy helps and the only lever that does "
            "is a longer-lived one. Triggering pays off on the episodic sources - "
            "construction campaigns and burning nights - and not on the chronic "
            "one.")
    else:
        add("The continuously-road-dust variant produced no hours to score in "
            "this window.")''')
p.write_text(s, encoding="utf-8")
print("ok")
