"""Response schemas for the operator API.

The console is a decision-support surface, so the payloads are part of the
contract rather than an implementation detail: a field that changes name takes
a panel with it, and a number that loses its provenance becomes a claim.

Two rules run through this file.

*Run data is passed through, not reshaped.* Models that wrap payloads produced
by `setu run` allow extra keys (`extra="allow"`). Dropping a field here would
make the API quietly poorer than the report sitting next to it, which is exactly
the kind of divergence nobody notices until a reader compares the two.

*An absent number stays absent.* Every metric that the pipeline can legitimately
fail to produce is `float | None`, never 0.0. A zero that means "not measured"
reads on a dashboard as a perfect score.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SiteInfo(BaseModel):
    """The city, the lattice, and the standards the run is judged against."""

    id: str
    label: str
    latitude: float
    longitude: float
    naaqs: dict[str, float] = Field(
        description="Notified Indian 24-hour standards, in ug/m3.")
    who_2021: dict[str, float] = Field(
        description="WHO 2021 guideline values, in ug/m3.")
    coarse_action_ugm3: float = Field(
        description="Coarse-mode level at which a dust action is worth "
                    "considering; stricter than the 24-hour standard because "
                    "it is an hourly decision.")
    node_report_seconds: int
    sites: int = Field(description="Candidate lattice cells, instrumented and held out.")
    instrumented: int
    held_out: int


class Gates(BaseModel):
    model_config = ConfigDict(extra="allow")

    min_source_posterior: float
    persistence_windows: int
    persistence_of: int
    uncertainty_must_exclude_threshold: bool
    min_dwell_minutes: float
    hysteresis_pct: float
    require_human_signoff: bool


class LeverInfo(BaseModel):
    label: str
    effect_pct: float | None = Field(
        description="Published effect size. Null means the literature has no "
                    "number to hold the lever to, not that the effect is zero.")
    half_life_min: float | None
    owner: str
    unit_cost_inr: float | None
    source: str = Field(description="Citation. A lever without one is not shipped.")


class GatesResponse(BaseModel):
    gates: Gates
    policy: dict[str, list[str]] = Field(
        description="Source -> levers. An empty list is a design decision: "
                    "regional aerosol has no local lever.")
    refusals: list[str]
    levers: dict[str, LeverInfo]


class NodeInfo(BaseModel):
    id: str
    lat: float
    lon: float
    x_m: float
    y_m: float
    role: Literal["instrumented", "holdout"]
    road_factor: float
    near_construction: bool
    burning_prone: bool
    dead: bool
    obstructed: bool
    drift_fraction: float
    drift_alarms: int
    quarantined: bool
    calibration_mae: float | None
    interval_coverage: float | None


class NodesResponse(BaseModel):
    nodes: list[NodeInfo]
    provenance: Literal["digital-twin", "reference"] = Field(
        description="Where these concentrations came from. The lattice is "
                    "simulated; only the hourly background is measured.")


class QueueItem(BaseModel):
    """One dispatch, as the engine ranked it."""

    model_config = ConfigDict(extra="allow")

    rank: int
    id: str
    node: str
    source: str
    lever: str
    owner: str
    priority: float = Field(
        description="Expected ug/m3 shaved per pass: how far the trigger clears "
                    "the level, times the lever's published effect, times the "
                    "posterior. The queue is ordered on this, not on arrival.")
    trigger_value: float
    trigger_sigma: float
    threshold: float
    dispatched_at: str
    recheck_at: str | None
    requires_signoff: bool = Field(
        description="True means this row cannot be actioned from a browser. "
                    "The API is read-only and there is no dispatch endpoint.")
    evidence: dict[str, Any] = Field(default_factory=dict)


class QueueResponse(BaseModel):
    total: int = Field(description="Dispatches in the whole run, before any filter.")
    matched: int = Field(description="Dispatches matching the current filters.")
    returned: int
    limit: int
    capacity_modelled: bool = Field(
        description="False: the prototype has no crew count, so this is an "
                    "ordering rather than a schedule.")
    items: list[QueueItem]
    owners: dict[str, int]
    levers: dict[str, int]
    notes: list[str]


class LedgerReason(BaseModel):
    reason: str
    count: int
    retryable: bool = Field(
        description="True when a clock can clear this refusal. False means only "
                    "new information can.")
    retry_within_minutes: float | None = Field(
        description="The delay the gate set, median over the run. Null for a "
                    "gate no clock will re-open.")
    last_at: str | None
    last_retry_at: str | None
    explanation: str


class LedgerGate(BaseModel):
    gate: int
    name: str
    description: str
    total: int
    retryable: int
    reasons: list[LedgerReason]


class LedgerResponse(BaseModel):
    gates: list[LedgerGate]
    total: int
    reschedulable: int = Field(
        description="Refusals a clock can clear, as opposed to an archive.")
    note: str


class ForecastHorizon(BaseModel):
    model_config = ConfigDict(extra="allow")

    horizon_h: int
    n: int
    model_rmse: float | None
    persistence_rmse: float | None
    diurnal_rmse: float | None
    skill_vs_persistence: float | None = Field(
        description="1 - model_rmse / persistence_rmse. Positive means the model "
                    "beats persistence; zero or below means it does not.")
    skill_vs_diurnal: float | None
    model_mae: float | None
    model_r2: float | None
    interval_coverage: float | None = Field(
        description="Share of outcomes inside the conformal band. Nominal is 0.80.")
    interval_width: float | None
    bias: float | None
    nodes_beating_persistence: int | None
    nodes_compared: int | None
    folds: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Walk-forward refits. The pooled row above can hide a fold "
                    "that wins and two that lose.")


class ForecastResponse(BaseModel):
    horizons: list[ForecastHorizon]
    diagnostics: dict[str, Any] | None
    nominal_coverage: float
    beats_persistence_pooled: bool
    claim: str = Field(description="What these numbers do and do not support.")


class AttributionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    classes: list[str]
    scorable_hours: int
    accuracy: float | None
    chance: float | None
    accuracy_above_060: float | None
    share_of_hours_above_060: float | None
    log_loss: float | None
    abstained: float | None
    background_rejection: dict[str, Any] | None
    feature_importance: list[dict[str, Any]]
    confidence_calibration: dict[str, Any] | None
    signature_agreement: dict[str, Any] | None
    sign_check: dict[str, Any] | None


class QualityResponse(BaseModel):
    """The data contract for the window the run actually read."""

    model_config = ConfigDict(extra="allow")

    quality: dict[str, Any] | None
    ok: bool | None
    background: list[dict[str, Any]]
    window: list[str] | None
    hourly_rows: int | None
    fingerprint: str | None
    cache: dict[str, Any] | None
    stale_cache_responses: int | None
    sources: dict[str, Any] | None
    injected: dict[str, Any] | None
    dropped_incomplete_hours: int | None
    offline: bool
    notes: list[str]


class RunStatus(BaseModel):
    """What the console header needs to be honest about the payload."""

    results_present: bool
    status: Literal["ok", "stale", "missing", "corrupt"]
    generated_at: str | None = Field(
        description="File time of results/validation.json. There is no live feed "
                    "behind this console, so this is the age of the run.")
    age_hours: float | None
    stale_after_hours: float
    stale: bool
    site: str | None
    fingerprint: str | None
    window: list[str] | None
    hours: int | None
    nodes: int | None
    threshold_ugm3: float | None
    offline: bool
    evidence_packs: int
    notes: list[str]


class HealthResponse(BaseModel):
    ok: bool
    results_present: bool
    status: Literal["ok", "stale", "missing", "corrupt"]


class EvidenceItem(BaseModel):
    """An index row. Everything but the id is optional: an older pack that
    predates a field should still be listed, not rejected."""

    model_config = ConfigDict(extra="allow")

    id: str
    node: str | None = None
    lever: str | None = None
    source: str | None = None
    owner: str | None = None
    dispatched_at: str | None = None
    trigger_value: float | None = None
    priority: float | None = None
    requires_signoff: bool | None = None


class EvidenceIndex(BaseModel):
    packs: list[EvidenceItem]
    count: int
    note: str


class EvidencePack(BaseModel):
    """A folder-ready record of one dispatch, for the enforcement officer."""

    model_config = ConfigDict(extra="allow")

    id: str
    node: str
    source: str
    lever: str
    owner: str
    dispatched_at: str
    recheck_at: str | None
    trigger_value: float
    trigger_sigma: float
    threshold: float
    evidence: dict[str, Any]
    requires_signoff: bool
    priority: float


class CalibrationMean(BaseModel):
    model_config = ConfigDict(extra="allow")

    nodes: int
    before_mae: float | None
    after_mae: float | None
    interval_coverage: float | None


class ValidationResponse(BaseModel):
    summary: dict[str, Any]
    calibration: dict[str, Any]
    action_standard: str
    transfer: dict[str, Any]
    attribution: dict[str, Any]
    forecast: list[dict[str, Any]]
    shares: list[dict[str, Any]]
    equal_cost: dict[str, Any]
    provenance: dict[str, Any]


class ErrorBody(BaseModel):
    detail: str
    code: str
    hint: str | None = None
