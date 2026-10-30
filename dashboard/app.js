/* SETU operator console.
 *
 * Plain JavaScript and no build step keep local setup simple and make the
 * interface easy to inspect alongside the API and validation outputs. There is
 * no framework here on purpose: the console has to be readable next to the
 * report it renders, and a build pipeline would be a second thing to trust.
 *
 * The screen never computes a number the API did not serve. Ordering, grouping
 * and filtering happen here; skill, coverage, counts and posteriors do not.
 */

"use strict";

const SOURCE_LABEL = {
  road_dust: "road dust / non-exhaust traffic",
  construction: "construction and demolition",
  open_burning: "open waste or biomass burning",
  secondary: "secondary aerosol (regional)",
  traffic_combustion: "traffic exhaust",
  regional_background: "regional background",
  mixed: "ambiguous",
  ambiguous: "abstained",
};

const FOLD_COLOURS = ["#ff7a45", "#8fa3bb", "#6f8cff", "#c9a227"];

const $ = (id) => document.getElementById(id);

const state = {
  site: null,
  gates: null,
  run: null,
  nodes: null,
  validation: null,
  queue: null,
  ledger: null,
  forecast: null,
  attribution: null,
  quality: null,
  evidence: null,
  packs: new Map(),
  filters: { owner: "", lever: "", min_priority: "", order: "priority" },
  failed: {},
};

/* ------------------------------------------------------------------ utils */

class ApiFailure extends Error {
  constructor(status, code, message, hint) {
    super(message);
    this.status = status;
    this.code = code;
    this.hint = hint || null;
  }
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function fmt(n, dp = 1) {
  const v = Number(n);
  return (n === null || n === undefined || Number.isNaN(v))
    ? "n/a" : v.toFixed(dp);
}

function signed(n, dp = 3) {
  const v = Number(n);
  if (n === null || n === undefined || Number.isNaN(v)) return "n/a";
  return `${v > 0 ? "+" : ""}${v.toFixed(dp)}`;
}

function stamp(ts, { date = true } = {}) {
  if (!ts) return "n/a";
  const text = String(ts).replace("T", " ");
  return date ? text.slice(5, 16) : text.slice(11, 16);
}

function pretty(key) {
  if (!key) return "n/a";
  const text = String(key).replace(/_/g, " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function leverLabel(key) {
  const row = state.gates?.levers?.[key];
  return row?.label || pretty(key);
}

function sourceLabel(key) {
  return SOURCE_LABEL[key] || pretty(key);
}

function ownerLabel(key) {
  return pretty(key);
}

function minutesLabel(m) {
  const v = Number(m);
  if (!Number.isFinite(v)) return "n/a";
  if (v < 90) return `+${Math.round(v)} min`;
  if (v < 60 * 36) return `+${(v / 60).toFixed(1)} h`;
  return `+${(v / 1440).toFixed(1)} d`;
}

async function getJSON(url) {
  let res;
  try {
    res = await fetch(url, {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
  } catch (err) {
    throw new ApiFailure(
      0, "network", `the service did not answer ${url}`,
      "Start it with `python -m setu serve`, then reload.");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = Array.isArray(body.detail)
      ? "the request was rejected by the API"
      : (body.detail || `${res.status} ${url}`);
    throw new ApiFailure(res.status, body.code || "http_error", detail,
                         body.hint || null);
  }
  return res.json();
}

function failureBlock(err, key) {
  const code = err && err.code ? err.code : "error";
  const message = err && err.message ? err.message : "something went wrong";
  const hint = err && err.hint ? err.hint : "";
  return `<div class="fail-block" role="alert">
    <p class="fail-head"><span class="badge fail">${esc(code)}</span> ${esc(message)}</p>
    ${hint ? `<p class="fail-hint">${esc(hint)}</p>` : ""}
    <button class="btn ghost" data-retry="${esc(key)}">Retry</button>
  </div>`;
}

/* ---------------------------------------------------------------- loading */

const LOADERS = {
  site: () => getJSON("/api/site"),
  gates: () => getJSON("/api/gates"),
  run: () => getJSON("/api/run"),
  nodes: () => getJSON("/api/nodes"),
  validation: () => getJSON("/api/validation"),
  ledger: () => getJSON("/api/ledger"),
  forecast: () => getJSON("/api/forecast"),
  attribution: () => getJSON("/api/attribution"),
  quality: () => getJSON("/api/quality"),
  evidence: () => getJSON("/api/evidence"),
  queue: () => getJSON(queueURL()),
};

function queueURL() {
  const params = new URLSearchParams({ limit: "200" });
  const f = state.filters;
  if (f.owner) params.set("owner", f.owner);
  if (f.lever) params.set("lever", f.lever);
  if (f.min_priority !== "") params.set("min_priority", f.min_priority);
  if (f.order === "time") params.set("order", "time");
  return `/api/queue?${params.toString()}`;
}

const RENDERERS = {
  site: renderSite,
  gates: renderGates,
  run: renderRun,
  nodes: renderLattice,
  validation: renderValidation,
  ledger: renderLedger,
  forecast: renderForecast,
  attribution: () => renderLimits(),
  quality: renderQuality,
  evidence: renderEvidence,
  queue: renderQueue,
};

async function attempt(key) {
  const host = retryHost(key);
  if (host && key !== "site" && key !== "run" && key !== "gates") {
    host.setAttribute("aria-busy", "true");
  }
  try {
    const data = await LOADERS[key]();
    state[key] = data;
    delete state.failed[key];
    RENDERERS[key](data);
  } catch (err) {
    state.failed[key] = err;
    showFailure(key, err);
  } finally {
    if (host) host.removeAttribute("aria-busy");
  }
}

function retryHost(key) {
  return {
    nodes: $("lattice")?.closest(".panel"),
    validation: $("validation")?.closest(".panel"),
    queue: $("queue")?.closest(".panel"),
    ledger: $("ledger-gates")?.closest(".panel"),
    forecast: $("forecast")?.closest(".panel"),
    evidence: $("evidence")?.closest(".panel"),
    quality: $("quality-facts")?.closest(".panel"),
    run: $("status-strip"),
  }[key] || null;
}

function showFailure(key, err) {
  const block = failureBlock(err, key);
  if (key === "run") {
    $("run-chip").textContent = "run unavailable";
    $("run-chip").className = "chip bad";
    const strip = $("status-strip");
    strip.hidden = false;
    strip.innerHTML = block;
    return;
  }
  if (key === "site") {
    $("site-label").textContent = "site unavailable";
    return;
  }
  if (key === "nodes") {
    $("lattice").innerHTML = "";
    $("lattice-foot").innerHTML = block;
    return;
  }
  if (key === "validation") {
    $("validation").querySelector("tbody").innerHTML =
      `<tr><td>${block}</td></tr>`;
    return;
  }
  if (key === "queue") {
    $("queue").querySelector("tbody").innerHTML = "";
    $("queue-empty").hidden = false;
    $("queue-empty").innerHTML = block;
    $("queue-count").textContent = "unavailable";
    return;
  }
  if (key === "ledger") {
    $("ledger-gates").innerHTML = block;
    return;
  }
  if (key === "forecast") {
    $("forecast").querySelector("tbody").innerHTML = `<tr><td colspan="8">${block}</td></tr>`;
    $("forecast-claim").textContent = "";
    return;
  }
  if (key === "evidence") {
    $("evidence").querySelector("tbody").innerHTML = "";
    $("evidence-empty").hidden = false;
    $("evidence-empty").innerHTML = block;
    return;
  }
  if (key === "quality") {
    $("quality-facts").innerHTML = block;
    return;
  }
  // No panel of its own: surface it in the status strip rather than swallowing
  // it, because a panel that silently drops a layer is the failure this console
  // exists to avoid.
  const strip = $("status-strip");
  strip.hidden = false;
  strip.innerHTML = `${strip.innerHTML}${block}`;
}

/* ------------------------------------------------------------- tab shell */

const tabs = Array.from(document.querySelectorAll(".tab"));
let activePanel = "overview";

function showPanel(name, { focus = false } = {}) {
  const known = tabs.some((t) => t.dataset.panel === name);
  if (!known) name = "overview";
  activePanel = name;
  for (const tab of tabs) {
    const on = tab.dataset.panel === name;
    tab.setAttribute("aria-selected", on ? "true" : "false");
    tab.tabIndex = on ? 0 : -1;
    const panel = $(`panel-${tab.dataset.panel}`);
    if (panel) panel.hidden = !on;
    if (on && focus) tab.focus();
  }
  if (history.replaceState) {
    history.replaceState(null, "", `#${name}`);
  }
  // Charts are laid out in viewBox units, so opening a hidden panel needs no
  // resize pass; the tables only need to exist.
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => showPanel(tab.dataset.panel));
  tab.addEventListener("keydown", (event) => {
    const index = tabs.indexOf(tab);
    let next = null;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabs.length - 1;
    if (next === null) return;
    event.preventDefault();
    showPanel(tabs[next].dataset.panel, { focus: true });
  });
});

window.addEventListener("hashchange", () => {
  showPanel((location.hash || "#overview").slice(1));
});

/* ------------------------------------------------------------------- site */

function renderSite(site) {
  $("site-label").textContent = site.label;
  $("site-label").title =
    `${site.instrumented} instrumented nodes, ${site.held_out} held out, ` +
    `${site.sites} lattice cells`;
  if (state.queue) renderTiles();
}

function renderGates(gates) {
  if (state.queue) {
    renderTiles();
    renderQueue(state.queue);
  }
}

/* -------------------------------------------------------------------- run */

function renderRun(run) {
  const chip = $("run-chip");
  const label = run.generated_at
    ? `run ${stamp(run.generated_at)}Z`
    : "no completed run";
  chip.textContent = run.stale ? `${label} · stale` : label;
  chip.className = `chip ${run.status === "ok" ? "" : run.status === "stale" ? "warn" : "bad"}`.trim();
  chip.title = [
    run.window ? `window ${run.window[0].slice(0, 10)} → ${run.window[1].slice(0, 10)}` : "",
    run.fingerprint ? `input fingerprint ${run.fingerprint}` : "",
    run.age_hours !== null ? `${run.age_hours.toFixed(1)} h old` : "",
  ].filter(Boolean).join(" · ");

  const strip = $("status-strip");
  const notes = run.notes || [];
  if (!notes.length) {
    strip.hidden = true;
    strip.innerHTML = "";
  } else {
    strip.hidden = false;
    strip.className = `strip ${run.status === "ok" ? "warn-strip" : "bad-strip"}`;
    strip.innerHTML = notes.map((n) => `<span>${esc(n)}</span>`).join("");
  }
}

/* ------------------------------------------------------------------ tiles */

function renderTiles() {
  const queue = state.queue;
  const site = state.site;
  const gates = state.gates;
  if (!queue) return;

  const list = queue.items || [];
  if (!list.length) {
    $("t-what").textContent = "Nothing to dispatch";
    $("t-what-note").textContent =
      "In this window the grid stayed under the action level, or every breach " +
      "was refused by a gate. Both are real answers, not a stalled system. The " +
      "Gates tab shows what the chain stopped.";
    $("t-where").textContent = site ? site.label : "—";
    $("t-where-note").textContent = state.run?.window
      ? `window ${state.run.window[0].slice(0, 10)} → ${state.run.window[1].slice(0, 10)}`
      : "";
    $("t-source").textContent = "—";
    $("t-source-note").textContent = "";
    $("t-action").textContent = "—";
    $("t-action-note").textContent = "";
    renderTriage();
    return;
  }

  const top = list[0];
  const ev = top.evidence || {};
  const floor = gates?.gates?.min_source_posterior;
  const conf = (ev.posterior || {})[top.source];

  $("t-what").textContent = `${fmt(top.trigger_value)} µg/m³`;
  $("t-what-note").textContent =
    `± ${fmt(top.trigger_sigma)} · action level ${fmt(top.threshold)} · ` +
    `interval low ${fmt(top.trigger_value - top.trigger_sigma)}`;

  $("t-where").textContent = top.node;
  $("t-where-note").textContent =
    `rank 1 of ${queue.total.toLocaleString()} · ${ownerLabel(top.owner)}`;

  $("t-source").textContent = sourceLabel(top.source);
  $("t-source-note").textContent =
    `posterior ${fmt(conf, 2)} · floor ${fmt(floor, 2)}` +
    (top.source === "secondary" ? " · no local lever exists" : "");

  $("t-action").textContent = leverLabel(top.lever);
  $("t-action-note").textContent =
    `half-life ${fmt(ev.lever_half_life_min, 0)} min · recheck ` +
    `${stamp(top.recheck_at, { date: false })} · ` +
    `${top.requires_signoff ? "awaiting human sign-off" : "auto"}`;

  renderTriage();
}

function renderTriage() {
  const queue = state.queue;
  const ledger = state.ledger;
  const host = $("triage");
  if (!queue) return;
  const peak = state.validation?.summary;
  const rows = [
    ["Dispatches", queue.total.toLocaleString(),
      "ordered by expected effect, not by arrival"],
    ["Matching this filter", queue.matched.toLocaleString(),
      `${queue.returned} shown · fleet capacity not modelled`],
    ["Refused by a gate", (ledger?.total ?? 0).toLocaleString(),
      `${(ledger?.reschedulable ?? 0).toLocaleString()} clear on a clock`],
    ["Window", peak ? `${peak.hours} h · ${peak.nodes} nodes` : "n/a",
      peak ? `action level ${fmt(peak.threshold_ugm3)} µg/m³` : ""],
  ];
  host.innerHTML = rows.map(([label, value, note]) => `
    <div class="stat">
      <span class="stat-label">${esc(label)}</span>
      <span class="stat-value">${esc(value)}</span>
      <span class="stat-note">${esc(note)}</span>
    </div>`).join("");
  $("triage-label").textContent = `${queue.total.toLocaleString()} dispatches`;
}

/* ---------------------------------------------------------------- lattice */

function renderLattice(payload) {
  const svg = $("lattice");
  const nodes = payload.nodes;
  const span = 2000;                       // 2 km lattice
  const scale = 380 / span;
  const cell = 320 * scale;

  const x = (n) => (n.x_m + span / 2) * scale + 10;
  const y = (n) => 390 - (n.y_m + span / 2) * scale;

  let out = "";
  for (let gx = 0; gx < span; gx += 320) {
    for (let gy = 0; gy < span; gy += 320) {
      out += `<rect class="cell" x="${(gx * scale + 10).toFixed(1)}" ` +
             `y="${(390 - (gy + 320) * scale).toFixed(1)}" ` +
             `width="${cell.toFixed(1)}" height="${cell.toFixed(1)}" rx="2"/>`;
    }
  }
  for (const n of nodes) {
    let cls = "ok";
    if (n.role === "holdout") cls = "out";
    else if (n.quarantined || n.dead) cls = "bad";
    else if (n.drift_fraction > 0 || n.obstructed) cls = "warn";
    const notes = [
      `${n.id} · ${n.role}`,
      n.dead ? "optical bench dead" : "",
      n.obstructed ? "inlet obstructed" : "",
      n.drift_fraction ? `drifting (+${(n.drift_fraction * 100).toFixed(0)}%)` : "",
      n.quarantined ? "QUARANTINED" : "",
      n.calibration_mae != null ? `cal MAE ${n.calibration_mae.toFixed(2)}` : "",
      n.interval_coverage != null ? `coverage ${n.interval_coverage.toFixed(2)}` : "",
    ].filter(Boolean).join(" · ");
    out += `<circle class="node ${cls}" cx="${x(n).toFixed(1)}" cy="${y(n).toFixed(1)}" ` +
           `r="${n.role === "holdout" ? 4 : 6}"><title>${esc(notes)}</title></circle>`;
  }
  svg.innerHTML = out;

  const q = nodes.filter((n) => n.quarantined).length;
  const d = nodes.filter((n) => n.dead).length;
  const o = nodes.filter((n) => n.obstructed).length;
  const inst = nodes.filter((n) => n.role !== "holdout").length;
  $("lattice-foot").textContent =
    `${inst} instrumented, ${nodes.length - inst} held out · ` +
    `${d} dead benches, ${o} obstructed inlets, ${q} quarantined by the drift ` +
    "monitor. Held-out nodes never train the calibration and are not drawn as reporting.";
}

/* ------------------------------------------------------------- validation */

function renderValidation(v) {
  const c = (v.calibration || {}).mean || {};
  const before = {};
  const after = {};
  for (const k of ["mae", "rmse", "nmae", "r2", "bias", "threshold_accuracy"]) {
    before[k] = c[`before_${k}`];
    after[k] = c[`after_${k}`];
  }
  const standard = v.action_standard || "action level";
  const att = v.attribution || {};
  const f1 = (v.forecast || []).find((r) => r.horizon_h === 1) || {};
  const t = v.transfer || {};
  const level = fmt((v.summary || {}).threshold_ugm3, 0);

  const rows = [
    ["Calibration MAE", `${fmt(before.mae, 2)} → ${fmt(after.mae, 2)} µg/m³`,
      after.mae != null && before.mae != null && after.mae < before.mae],
    ["Calibration R²", `${fmt(before.r2, 2)} → ${fmt(after.r2, 2)}`,
      after.r2 != null && after.r2 > 0.5],
    [`Threshold accuracy @ ${level} µg/m³ (${standard})`,
      `${fmt(before.threshold_accuracy, 3)} → ${fmt(after.threshold_accuracy, 3)}`,
      (after.threshold_accuracy ?? 0) > 0.9],
    ["Interval coverage (mean)", `${fmt(c.interval_coverage, 3)} (nominal 0.80)`,
      (c.interval_coverage ?? 0) >= 0.7],
    ["Transfer, never co-located",
      t.nodes_transferred
        ? `MAE ${fmt(t.mean_mae_before, 2)} → ${fmt(t.mean_mae_after, 2)} µg/m³`
        : "n/a",
      (t.mean_mae_after ?? 1e9) < (t.mean_mae_before ?? 0)],
    ["Attribution accuracy",
      `${fmt(att.accuracy, 3)} vs ${fmt(att.chance, 2)} chance`,
      (att.accuracy ?? 0) > (att.chance ?? 1) + 0.2],
    ["Above 0.60 posterior",
      `${fmt(att.accuracy_above_060, 3)} on ${fmt((att.share_of_hours_above_060 ?? 0) * 100, 0)}% of hours`,
      (att.accuracy_above_060 ?? 0) > (att.accuracy ?? 0)],
    ["1-hour forecast skill",
      `RMSE ${fmt(f1.model_rmse, 2)} vs ${fmt(f1.persistence_rmse, 2)} (${signed(f1.skill_vs_persistence)})`,
      (f1.skill_vs_persistence ?? 0) > 0],
  ];

  $("validation").querySelector("tbody").innerHTML = rows.map(([k, val, ok]) =>
    `<tr><th scope="row">${esc(k)}</th>` +
    `<td class="num ${ok ? "pass" : "fail"}">${esc(val)}</td></tr>`).join("");

  const failing = (v.shares || []).filter((s) => !s.within).map((s) => beautifulSource(s.source));
  $("shares-foot").innerHTML = failing.length
    ? `Share cross-check against receptor studies <span class="fail">fails</span> for ` +
      `${esc(failing.join(", "))}. The twin's injection rates are not tuned to a ` +
      "published apportionment, so absolute shares measure the twin rather than the " +
      "city; within-class discrimination is the row above. Tracked as an open item."
    : "Share cross-check agrees with published receptor ranges.";

  const w = v.summary || {};
  $("window-label").textContent =
    `${w.hours ?? "?"} real hours · ${w.nodes ?? "?"} nodes · action level ` +
    `${level} µg/m³ (${esc(standard)})`;

  renderEqualCost(v.equal_cost || {});
  renderLimits();
}

function beautifulSource(key) {
  return sourceLabel(key);
}

function renderEqualCost(ec) {
  const host = $("equal-cost");
  const labels = {
    sprinkling_62min: "Water sprinkling · 62 min",
    sweeping_150min: "Vacuum sweeping · 150 min",
    suppressant_20h: "CaCl₂ suppressant · ~20 h",
  };
  const rows = Object.entries(ec)
    .filter(([, row]) => row && typeof row === "object" && "roster_coverage" in row);
  if (!rows.length) {
    host.innerHTML = `<p class="empty">No episodic coarse-mode breach hours in this ` +
      "window, so the equal-cost comparison is undefined rather than zero.</p>";
    return;
  }
  host.innerHTML = rows.map(([key, row]) => {
    const gain = Number(row.gain) || 0;
    const saturated = row.saturated
      ? ` · saturated at ${row.passes_used} of ${row.passes_per_day} passes`
      : "";
    return `<div class="bar-row">
      <span class="bar-label">${esc(labels[key] || key)}<span class="sub">${esc(saturated)}</span></span>
      <span class="bar-track">
        <span class="bar-fill roster" style="width:${(row.roster_coverage * 100).toFixed(0)}%"></span>
        <span class="bar-fill" style="width:${(row.triggered_coverage * 100).toFixed(0)}%;opacity:.55"></span>
      </span>
      <span class="bar-gain ${gain > 0.05 ? "pos" : "zero"}">${gain > 0.05 ? "+" + gain.toFixed(2) : "no gain"}</span>
    </div>`;
  }).join("") +
  `<p class="foot">Grey bar: fixed roster coverage of breach hours. ` +
  `Bright: same number of passes, placed by the trigger. Both arms cost the same.</p>`;
}

/* ------------------------------------------------------------------ queue */

function populateFilters() {
  const owners = Object.keys(state.queue?.owners || {});
  const ownerSel = $("queue-owner");
  const current = ownerSel.value;
  ownerSel.innerHTML = `<option value="">All desks</option>` +
    owners.map((k) => `<option value="${esc(k)}">${esc(ownerLabel(k))} ` +
      `(${state.queue.owners[k]})</option>`).join("");
  ownerSel.value = owners.includes(current) ? current : "";

  const levers = Object.keys(state.gates?.levers || state.queue?.levers || {});
  const leverSel = $("queue-lever");
  const currentLever = leverSel.value;
  leverSel.innerHTML = `<option value="">All levers</option>` +
    levers.map((k) => `<option value="${esc(k)}">${esc(leverLabel(k))}</option>`).join("");
  leverSel.value = levers.includes(currentLever) ? currentLever : "";
}

function renderQueue(queue) {
  populateFilters();
  // The tiles and the triage strip are derived from the queue, so the queue is
  // what finishes them. They used to be painted only when `site` and `gates`
  // happened to resolve afterwards, which left four skeletons on a loaded page.
  renderTiles();
  const tbody = $("queue").querySelector("tbody");
  const empty = $("queue-empty");
  const rows = queue.items || [];
  const maxPriority = Math.max(1e-9, ...rows.map((r) => Number(r.priority) || 0));

  $("queue-count").textContent =
    `${queue.matched.toLocaleString()} matching · ${queue.total.toLocaleString()} in the run`;
  $("queue-foot").innerHTML =
    `Showing ${rows.length} of ${queue.matched.toLocaleString()} matching ` +
    `dispatches; the run produced ${queue.total.toLocaleString()} in total, and ` +
    "this API serves the highest-priority page of them. " +
    (queue.notes || []).map(esc).join(" ") +
    ` <a href="/docs#/queue" target="_blank" rel="noopener">API</a>`;

  if (!rows.length) {
    tbody.innerHTML = "";
    empty.hidden = false;
    const filtered = queue.matched === 0 && queue.total > 0;
    empty.innerHTML = filtered
      ? `<strong>No dispatches match these filters.</strong>
         <p>The queue holds the highest-priority page of ${queue.total.toLocaleString()}
         dispatches, so a narrow filter can fall outside it. Clear the filters to
         see the head of the queue.</p>`
      : `<strong>Nothing to dispatch in this run.</strong>
         <p>That is a result, not a fault: the grid stayed under the action level,
         or every breach was refused by a gate. The Gates tab shows which.</p>`;
    return;
  }
  empty.hidden = true;
  tbody.innerHTML = rows.map((row) => {
    const pack = state.packs.get(row.id);
    const width = Math.max(6, (Number(row.priority) / maxPriority) * 100);
    const top = Object.entries(row.evidence?.posterior || {})
      .sort((a, b) => b[1] - a[1])[0] || [];
    return `<tr>
      <td class="num">${row.rank}</td>
      <td><span class="mono">${esc(row.node)}</span></td>
      <td>${esc(leverLabel(row.lever))}
        <span class="sub">${esc(sourceLabel(row.source))}${
          top.length ? ` · ${esc(sourceLabel(top[0]))} ${fmt(top[1], 2)}` : ""}</span></td>
      <td>${esc(ownerLabel(row.owner))}</td>
      <td class="prio">
        <span class="prio-track"><span class="prio-fill" style="width:${width.toFixed(0)}%"></span></span>
        <span class="num">${fmt(row.priority, 3)}</span>
      </td>
      <td class="num">${fmt(row.trigger_value)} <span class="sub">± ${fmt(row.trigger_sigma)}</span></td>
      <td class="mono">${esc(stamp(row.dispatched_at))}
        <span class="sub">recheck ${esc(stamp(row.recheck_at, { date: false }))}</span></td>
      <td>${row.requires_signoff
        ? `<span class="badge warn">sign-off</span>`
        : `<span class="badge">auto</span>`}</td>
      <td>${pack
        ? `<button class="btn small" data-pack="${esc(row.id)}">Open</button>`
        : `<span class="muted" title="Packs are written for the head of the queue, not for every dispatch">no pack</span>`}</td>
    </tr>`;
  }).join("");
}

/* ----------------------------------------------------------------- ledger */

function renderLedger(ledger) {
  const host = $("ledger-gates");
  $("ledger-count").textContent =
    `${ledger.total.toLocaleString()} suppressed · ` +
    `${ledger.reschedulable.toLocaleString()} clear on a clock`;
  $("ledger-lede").innerHTML =
    `${esc(ledger.note)} A dispatcher who receives the same alert forty times a ` +
    "night stops reading it, so every refusal is counted here rather than only " +
    "the ones that fired.";

  if (!ledger.gates.length) {
    host.innerHTML = `<p class="empty">No suppressions in this run.</p>`;
    return;
  }

  host.innerHTML = ledger.gates.map((gate) => {
    const share = ledger.total ? (gate.total / ledger.total) * 100 : 0;
    const rows = gate.reasons.map((r) => `
      <tr>
        <th scope="row" class="mono">${esc(r.reason)}</th>
        <td class="num">${r.count.toLocaleString()}</td>
        <td>${r.retryable
          ? `<span class="badge ok">${esc(minutesLabel(r.retry_within_minutes))}</span>`
          : `<span class="badge">new information</span>`}</td>
        <td class="mono">${r.last_retry_at ? esc(stamp(r.last_retry_at)) : "—"}</td>
        <td class="why">${esc(r.explanation)}</td>
      </tr>`).join("");
    return `<section class="gate">
      <header class="gate-head">
        <span class="gate-no">${gate.gate}</span>
        <div class="gate-title">
          <strong>${esc(gate.name)}</strong>
          <span class="muted">${esc(gate.description)}</span>
        </div>
        <div class="gate-sum">
          <span class="num">${gate.total.toLocaleString()}</span>
          <span class="sub">${share.toFixed(1)}% of refusals · ` +
          `${gate.retryable.toLocaleString()} on a clock</span>
        </div>
      </header>
      <div class="scroll-x">
        <table class="tbl">
          <caption class="visually-hidden">Reasons the ${esc(gate.name)} gate refused</caption>
          <thead><tr>
            <th scope="col">Reason</th>
            <th scope="col" class="num">Count</th>
            <th scope="col">Next opportunity</th>
            <th scope="col">Latest window</th>
            <th scope="col">Why the gate exists</th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </section>`;
  }).join("");

  $("ledger-foot").innerHTML =
    `The delay is the gate's own clock — one window, or the dwell expiry — not a ` +
    `prediction of clean air. "New information" means no amount of waiting ` +
    "re-opens that gate: the source either is or is not attributable.";
}

/* --------------------------------------------------------------- forecast */

const chartPad = { l: 46, r: 16, t: 16, b: 34 };

function drawAxes(svg, W, H, yMin, yMax, labels, fmtY) {
  const { l, r, t, b } = chartPad;
  const x0 = l, x1 = W - r, y0 = t, y1 = H - b;
  const y = (v) => y0 + ((yMax - v) / (yMax - yMin)) * (y1 - y0);
  let out = "";
  for (let i = 0; i <= 4; i += 1) {
    const v = yMin + ((yMax - yMin) * i) / 4;
    const yy = y(v).toFixed(1);
    out += `<line class="grid" x1="${x0}" y1="${yy}" x2="${x1}" y2="${yy}"/>`;
    out += `<text class="axis" x="${x0 - 6}" y="${(Number(yy) + 3).toFixed(1)}" ` +
           `text-anchor="end">${fmtY(v)}</text>`;
  }
  labels.forEach((label, i) => {
    const xx = labels.length === 1 ? (x0 + x1) / 2
      : x0 + (i * (x1 - x0)) / (labels.length - 1);
    out += `<text class="axis" x="${xx.toFixed(1)}" y="${y1 + 18}" text-anchor="middle">${esc(label)}</text>`;
  });
  return out;
}

function xAt(index, count, W) {
  const { l, r } = chartPad;
  if (count === 1) return (l + (W - r)) / 2;
  return l + (index * (W - r - l)) / (count - 1);
}

function yAt(value, yMin, yMax, H) {
  const { t, b } = chartPad;
  return t + ((yMax - value) / (yMax - yMin)) * (H - b - t);
}

function drawForecast(forecast) {
  const rows = forecast.horizons || [];
  const svg = $("forecast-skill");
  const W = 560;
  const H = 240;
  if (!rows.length) {
    svg.innerHTML = "";
    $("forecast-claim").textContent = "No forecast rows in this run.";
    return;
  }
  const labels = rows.map((r) => `${r.horizon_h} h`);
  const folds = new Map();
  for (const row of rows) {
    for (const fold of row.folds || []) {
      const key = fold.fold ?? 0;
      if (!folds.has(key)) folds.set(key, []);
      folds.get(key).push({ x: row.horizon_h, y: fold.skill_vs_persistence });
    }
  }
  const all = [
    ...rows.map((r) => r.skill_vs_persistence),
    ...[...folds.values()].flat().map((p) => p.y),
  ].filter((v) => Number.isFinite(Number(v))).map(Number);
  const reach = Math.max(0.1, ...all.map((v) => Math.abs(v))) * 1.18;
  const yMax = reach;
  const yMin = -reach;

  let out = drawAxes(svg, W, H, yMin, yMax, labels, (v) => v.toFixed(2));
  out += `<line class="zero" x1="${chartPad.l}" y1="${yAt(0, yMin, yMax, H).toFixed(1)}" ` +
         `x2="${W - chartPad.r}" y2="${yAt(0, yMin, yMax, H).toFixed(1)}"/>`;

  const series = [];
  series.push({
    label: "pooled", colour: "#2ed3b7", width: 2.6,
    points: rows.map((r, i) => ({ x: xAt(i, rows.length, W),
                                  y: yAt(Number(r.skill_vs_persistence) || 0, yMin, yMax, H),
                                  value: r.skill_vs_persistence,
                                  at: `${r.horizon_h}h pooled` })),
  });
  [...folds.entries()].sort((a, b) => a[0] - b[0]).forEach(([key, pts], i) => {
    series.push({
      label: `fold ${key}`, colour: FOLD_COLOURS[i % FOLD_COLOURS.length],
      width: 1.6, dash: "5 4",
      points: pts.map((p) => {
        const index = rows.findIndex((r) => r.horizon_h === p.x);
        return { x: xAt(index, rows.length, W),
                 y: yAt(Number(p.y), yMin, yMax, H),
                 value: p.y, at: `${p.x}h fold ${key}` };
      }),
    });
  });

  for (const line of series) {
    out += `<polyline fill="none" stroke="${line.colour}" stroke-width="${line.width}" ` +
           `${line.dash ? `stroke-dasharray="${line.dash}"` : ""} ` +
           `points="${line.points.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ")}"/>`;
    for (const p of line.points) {
      out += `<circle class="pt" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="3" ` +
             `fill="${line.colour}"><title>${esc(p.at)}: ${signed(p.value)}</title></circle>`;
    }
  }
  svg.innerHTML = out;

  $("forecast-skill-caption").innerHTML = series.map((line) =>
    `<span class="lg"><i style="background:${line.colour}"></i>${esc(line.label)}</span>`).join("") +
    `<span class="muted">positive = beats persistence; the dashed folds are refits scored ` +
    "further from their training data.</span>";

  $("forecast-claim").textContent = forecast.claim;
  $("forecast-claim").className = forecast.beats_persistence_pooled ? "lede" : "lede warn";
}

function drawCoverage(forecast) {
  const rows = forecast.horizons || [];
  const svg = $("forecast-coverage");
  const W = 480;
  const H = 200;
  const nominal = forecast.nominal_coverage ?? 0.8;
  if (!rows.length) {
    svg.innerHTML = "";
    return;
  }
  const yMin = 0;
  const yMax = 1;
  let out = drawAxes(svg, W, H, yMin, yMax,
                     rows.map((r) => `${r.horizon_h} h`), (v) => v.toFixed(2));
  const { l, r, t, b } = chartPad;
  const band = (W - l - r) / rows.length;
  rows.forEach((row, i) => {
    const value = Number(row.interval_coverage);
    if (!Number.isFinite(value)) return;
    const x = l + i * band + band * 0.22;
    const width = band * 0.56;
    const yv = yAt(value, yMin, yMax, H);
    const good = value >= nominal - 0.02;
    out += `<rect x="${x.toFixed(1)}" y="${yv.toFixed(1)}" width="${width.toFixed(1)}" ` +
           `height="${(H - b - yv).toFixed(1)}" rx="3" fill="${good ? "#2ed3b7" : "#ff7a45"}" ` +
           `opacity=".8"><title>${row.horizon_h}h coverage ${value.toFixed(3)}</title></rect>`;
    out += `<text class="axis" x="${(x + width / 2).toFixed(1)}" y="${(yv - 5).toFixed(1)}" ` +
           `text-anchor="middle">${value.toFixed(3)}</text>`;
  });
  const yNom = yAt(nominal, yMin, yMax, H);
  out += `<line class="nominal" x1="${l}" y1="${yNom.toFixed(1)}" x2="${W - r}" y2="${yNom.toFixed(1)}"/>`;
  out += `<text class="axis" x="${W - r}" y="${(yNom - 5).toFixed(1)}" text-anchor="end">nominal ${nominal.toFixed(2)}</text>`;
  svg.innerHTML = out;
  $("coverage-label").textContent =
    `nominal ${nominal.toFixed(2)} · ${rows.map((row) => fmt(row.interval_coverage, 3)).join(" / ")}`;
}

function renderForecast(forecast) {
  drawForecast(forecast);
  drawCoverage(forecast);

  $("forecast").querySelector("tbody").innerHTML = (forecast.horizons || []).map((row) => {
    const skill = Number(row.skill_vs_persistence);
    const cls = Number.isFinite(skill) && skill > 0 ? "pass" : "fail";
    return `<tr>
      <th scope="row">${row.horizon_h} h</th>
      <td class="num">${fmt(row.model_rmse, 2)}</td>
      <td class="num">${fmt(row.persistence_rmse, 2)}</td>
      <td class="num ${cls}">${signed(row.skill_vs_persistence)}</td>
      <td class="num">${fmt(row.interval_coverage, 3)}</td>
      <td class="num">${fmt(row.interval_width, 2)}</td>
      <td class="num">${signed(row.bias, 2)}</td>
      <td class="num">${row.nodes_beating_persistence ?? "n/a"}${
        row.nodes_compared != null ? ` / ${row.nodes_compared}` : ""}</td>
    </tr>`;
  }).join("");

  const foldRows = [];
  for (const row of forecast.horizons || []) {
    for (const fold of row.folds || []) {
      foldRows.push({ ...fold, horizon_h: row.horizon_h });
    }
  }
  foldRows.sort((a, b) => (a.fold ?? 0) - (b.fold ?? 0) || a.horizon_h - b.horizon_h);
  $("folds").querySelector("tbody").innerHTML = foldRows.map((fold) => {
    const skill = Number(fold.skill_vs_persistence);
    return `<tr>
      <td>${fold.fold ?? "—"}</td>
      <td>${fold.horizon_h} h</td>
      <td class="num ${skill > 0 ? "pass" : "fail"}">${signed(fold.skill_vs_persistence)}</td>
      <td class="num">${fmt(fold.interval_coverage, 3)}</td>
      <td class="num">${fold.n ?? "—"}</td>
    </tr>`;
  }).join("") || `<tr><td colspan="5" class="muted">No fold rows published.</td></tr>`;

  const d = forecast.diagnostics;
  $("diagnostics").textContent = d
    ? `${d.nodes_scored} of ${d.nodes_total} nodes scored across ${d.folds_scored} ` +
      `walk-forward fits; ${d.nodes_skipped} nodes and ${d.folds_skipped} folds were ` +
      `too thin to score. ${d.rows_warmup.toLocaleString()} node-hours went to the ` +
      `${d.warmup_hours}-hour lag warm-up, ${d.rows_missing_met.toLocaleString()} had ` +
      `no usable meteorology, and ${d.met_hours_bridged.toLocaleString()} short weather ` +
      "gaps were bridged (longer ones left missing)."
    : "";
  renderLimits();
}

/* --------------------------------------------------------------- evidence */

function renderEvidence(index) {
  state.packs = new Map((index.packs || []).map((p) => [p.id, p]));
  const tbody = $("evidence").querySelector("tbody");
  $("evidence-count").textContent =
    `${index.count} pack${index.count === 1 ? "" : "s"} · one per dispatch in the queue head`;
  // The queue rendered before this in the parallel boot, so give it the pack
  // buttons it was missing rather than leaving "no pack" next to a real one.
  if (state.queue) renderQueue(state.queue);
  if (!index.packs?.length) {
    tbody.innerHTML = "";
    $("evidence-empty").hidden = false;
    $("evidence-empty").innerHTML =
      `<strong>No evidence packs yet.</strong><p>${esc(index.note)}</p>`;
    return;
  }
  $("evidence-empty").hidden = true;
  tbody.innerHTML = index.packs.map((pack) => `
    <tr>
      <th scope="row" class="mono">${esc(pack.id)}</th>
      <td class="mono">${esc(pack.node)}</td>
      <td>${esc(leverLabel(pack.lever))}</td>
      <td class="num">${fmt(pack.priority, 3)}</td>
      <td class="mono">${esc(stamp(pack.dispatched_at))}</td>
      <td>${esc(sourceLabel(pack.source))}</td>
      <td><button class="btn small" data-pack="${esc(pack.id)}">Open</button></td>
    </tr>`).join("");
}

async function openPack(id) {
  const dialog = $("pack-dialog");
  const body = $("pack-body");
  $("pack-title").textContent = `Evidence pack ${id}`;
  body.innerHTML = `<span class="skeleton sk-block"></span>`;
  if (!dialog.open) {
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
  }

  let pack;
  try {
    pack = await getJSON(`/api/evidence/${encodeURIComponent(id)}`);
  } catch (err) {
    body.innerHTML = failureBlock(err, `pack:${id}`);
    return;
  }
  state.packs.set(id, pack);

  const ev = pack.evidence || {};
  const posterior = Object.entries(ev.posterior || {}).sort((a, b) => b[1] - a[1]);
  const maxPost = Math.max(1e-9, ...posterior.map(([, v]) => v));
  const low = Number(pack.trigger_value) - Number(pack.trigger_sigma);
  const high = Number(pack.trigger_value) + Number(pack.trigger_sigma);
  const scale = Math.max(high, Number(pack.threshold)) * 1.15 || 1;

  body.innerHTML = `
    <div class="pack-grid">
      <div>
        <h3>${esc(sourceLabel(pack.source))} at <span class="mono">${esc(pack.node)}</span></h3>
        <p class="sub">${esc(leverLabel(pack.lever))} · ${esc(ownerLabel(pack.owner))} ·
          dispatched ${esc(pack.dispatched_at)} · recheck ${esc(pack.recheck_at || "n/a")}</p>
        <div class="chart-hold">
          <div class="band">
            <span class="band-threshold" style="left:${(Number(pack.threshold) / scale * 100).toFixed(1)}%"
                  title="action level ${fmt(pack.threshold)}"></span>
            <span class="band-range" style="left:${(low / scale * 100).toFixed(1)}%;
                  width:${((high - low) / scale * 100).toFixed(1)}%"></span>
            <span class="band-point" style="left:${(Number(pack.trigger_value) / scale * 100).toFixed(1)}%"></span>
          </div>
          <p class="foot">Trigger ${fmt(pack.trigger_value)} ± ${fmt(pack.trigger_sigma)} µg/m³
            (${fmt(low)}–${fmt(high)}) against an action level of ${fmt(pack.threshold)}.
            The dot is the reading; the bar is the calibrated interval; the line is the level.</p>
        </div>
        <h3>Source posterior</h3>
        <div class="posterior">
          ${posterior.map(([key, value]) => `
            <div class="post-row">
              <span>${esc(sourceLabel(key))}</span>
              <span class="post-track"><span class="post-fill" style="width:${(value / maxPost * 100).toFixed(0)}%"></span></span>
              <span class="num">${fmt(value, 2)}</span>
            </div>`).join("")}
        </div>
      </div>
      <div>
        <h3>What the trigger used</h3>
        <dl class="facts">
          <dt>Priority</dt><dd>${fmt(pack.priority, 4)} µg/m³ per pass</dd>
          <dt>Coarse fraction</dt><dd>${fmt(ev.coarse_ratio, 3)}</dd>
          <dt>Wind from</dt><dd>${fmt(ev.wind_from_deg, 0)}°</dd>
          <dt>Lever effect</dt><dd>${fmt(ev.lever_effect_pct, 0)}%</dd>
          <dt>Half-life</dt><dd>${fmt(ev.lever_half_life_min, 0)} min</dd>
          <dt>Posterior floor</dt><dd>${fmt(ev.gate_min_posterior, 2)}</dd>
          <dt>Provenance</dt><dd>${esc(ev.provenance || "unknown")}</dd>
        </dl>
        <p class="foot"><strong>Citation.</strong> ${esc(ev.lever_source || "none recorded")}</p>
        <p class="foot">This record describes a <strong>simulated</strong> trigger from the
          digital twin. It is a template for what an enforcement officer would be given,
          not a notice served on anyone.</p>
        ${pack.requires_signoff
          ? `<p class="signoff">Human sign-off required. This console cannot dispatch a
             crew; acting on this row is a decision taken outside the service.</p>`
          : ""}
        <p class="foot"><a href="/api/evidence/${encodeURIComponent(pack.id)}"
          target="_blank" rel="noopener">Download this pack as JSON</a></p>
      </div>
    </div>`;
}

function closePack() {
  const dialog = $("pack-dialog");
  if (typeof dialog.close === "function") dialog.close();
  else dialog.removeAttribute("open");
}

/* ------------------------------------------------------------------- data */

function renderQuality(quality) {
  const q = quality.quality || {};
  const facts = [
    ["Window", quality.window ? `${quality.window[0]} → ${quality.window[1]}` : "n/a"],
    ["Hourly rows", quality.hourly_rows != null ? quality.hourly_rows.toLocaleString() : "n/a"],
    ["Coverage of claimed span", fmt(q.coverage, 3)],
    ["Missing hours", q.missing_hours != null ? q.missing_hours.toLocaleString() : "n/a"],
    ["Largest gap", q.largest_gap_hours != null ? `${q.largest_gap_hours} h` : "n/a"],
    ["Duplicate timestamps", q.duplicate_timestamps ?? "n/a"],
    ["Out-of-range readings", q.out_of_range ? Object.keys(q.out_of_range).length : "n/a"],
    ["Longest frozen run", q.longest_frozen_run_hours
      ? Object.entries(q.longest_frozen_run_hours).map(([k, v]) => `${k} ${v}h`).join(", ")
      : "n/a"],
    ["Dropped incomplete hours", quality.dropped_incomplete_hours ?? "n/a"],
    ["Input fingerprint", quality.fingerprint || "n/a"],
    ["Quality check", quality.ok === true ? "passed" : quality.ok === false ? "FAILED" : "not run"],
  ];
  $("quality-facts").innerHTML = facts.map(([k, v]) =>
    `<dt>${esc(k)}</dt><dd class="${k === "Quality check" && quality.ok === false ? "fail" : ""}">${esc(v)}</dd>`).join("");
  $("quality-notes").textContent = (quality.notes || []).join(" ");

  $("provenance-sources").innerHTML = (quality.background || []).map((src) => `
    <div class="source">
      <a href="${esc(src.url)}" target="_blank" rel="noopener">${esc(src.label)}</a>
      <span class="sub mono">${esc(src.fetched_at_iso)} ·
        ${src.from_cache ? "cache" : "network"}${src.stale ? " · STALE" : ""}
        · ${fmt(src.age_seconds / 3600, 1)} h old</span>
    </div>`).join("");

  const cache = quality.cache || {};
  const cacheFacts = [
    ["Entries", cache.entries ?? "n/a"],
    ["Size", cache.bytes != null ? `${(cache.bytes / 1024).toFixed(0)} kB` : "n/a"],
    ["Oldest", cache.oldest_age_hours != null ? `${fmt(cache.oldest_age_hours, 1)} h` : "n/a"],
    ["Newest", cache.newest_age_hours != null ? `${fmt(cache.newest_age_hours, 1)} h` : "n/a"],
    ["Mode", quality.offline ? "offline snapshot" : "network with cache"],
    ["Upstream probe", quality.sources?.probed
      ? (quality.sources.reachable ? "reachable" : "degraded")
      : "not probed"],
  ];
  $("cache-facts").innerHTML = cacheFacts.map(([k, v]) =>
    `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");

  renderLimits();
}

function renderLimits() {
  const v = state.validation;
  const f = state.forecast;
  const items = [];
  items.push("Concentrations on this screen are simulated by a digital twin. The " +
    "hourly background is real CAMS reanalysis; the grid is not, and the hardware " +
    "does not exist yet.");
  if (f?.claim) items.push(f.claim);
  const failing = (v?.shares || []).filter((s) => !s.within);
  if (failing.length) {
    items.push(`Absolute source shares do not match published receptor apportionment for ` +
      `${failing.map((s) => sourceLabel(s.source)).join(", ")}. The twin's injection ` +
      "rates are not tuned to a published split, so the shares measure the twin.");
  }
  const effects = v?.response?.effects;
  if (effects) {
    const verdicts = Object.entries(effects.verdicts || {})
      .map(([k, n]) => `${k.replace(/_/g, " ")} (${n})`).join(", ") || "none";
    items.push(`Intervention effect measurement returned ${verdicts} on ` +
      `${effects.attempted} sampled dispatches at ${effects.cadence} cadence. The twin ` +
      "contains no intervention dynamics, so a null is the correct answer here.");
  }
  if (v?.attribution?.abstained != null) {
    items.push(`${fmt(v.attribution.abstained * 100, 1)}% of hours were abstained on. ` +
      "The classifier is allowed to say it cannot tell.");
  }
  if (v?.calibration?.mean?.interval_coverage != null) {
    items.push(`Interval coverage is ${fmt(v.calibration.mean.interval_coverage, 3)} ` +
      "against a nominal 0.80, so the uncertainty gate is slightly more permissive " +
      "than advertised.");
  }
  items.push("Calibration is against CAMS reanalysis at ~10 km, not a reference-grade " +
    "CPCB monitor, because the reference station was unreachable from the development " +
    "network. Every error figure validates the algorithms, not an instrument.");
  items.push("There is no real-time feed behind this console. It reads a completed run, " +
    "and the header says how old that run is.");

  $("limits").innerHTML = items.map((text) => `<li>${esc(text)}</li>`).join("");
}

/* --------------------------------------------------------------- controls */

function debounce(fn, ms) {
  let handle = null;
  return (...args) => {
    clearTimeout(handle);
    handle = setTimeout(() => fn(...args), ms);
  };
}

const reloadQueue = debounce(() => attempt("queue"), 250);

$("queue-owner").addEventListener("change", (event) => {
  state.filters.owner = event.target.value;
  reloadQueue();
});
$("queue-lever").addEventListener("change", (event) => {
  state.filters.lever = event.target.value;
  reloadQueue();
});
$("queue-min").addEventListener("input", (event) => {
  state.filters.min_priority = event.target.value.trim();
  reloadQueue();
});
$("queue-reset").addEventListener("click", () => {
  state.filters = { owner: "", lever: "", min_priority: "", order: "priority" };
  $("queue-owner").value = "";
  $("queue-lever").value = "";
  $("queue-min").value = "";
  setOrder("priority");
  attempt("queue");
});

function setOrder(order) {
  state.filters.order = order;
  $("order-priority").classList.toggle("on", order === "priority");
  $("order-priority").setAttribute("aria-pressed", order === "priority" ? "true" : "false");
  $("order-time").classList.toggle("on", order === "time");
  $("order-time").setAttribute("aria-pressed", order === "time" ? "true" : "false");
}
$("order-priority").addEventListener("click", () => { setOrder("priority"); reloadQueue(); });
$("order-time").addEventListener("click", () => { setOrder("time"); reloadQueue(); });

$("pack-close").addEventListener("click", closePack);
$("pack-dialog").addEventListener("click", (event) => {
  if (event.target === $("pack-dialog")) closePack();
});

document.addEventListener("click", (event) => {
  const pack = event.target.closest("[data-pack]");
  if (pack) {
    openPack(pack.dataset.pack);
    return;
  }
  const retry = event.target.closest("[data-retry]");
  if (retry) {
    const key = retry.dataset.retry;
    if (key.startsWith("pack:")) openPack(key.slice(5));
    else attempt(key);
  }
});

/* ------------------------------------------------------------------ boot */

function tick() {
  const now = new Date();
  $("clock").textContent = now.toLocaleTimeString();
  $("clock").title = "Local browser time. The run window is in the header.";
}

async function boot() {
  showPanel((location.hash || "#overview").slice(1));
  tick();
  setInterval(tick, 1000);

  // Independent loads: one failing panel must not blank the rest of the
  // console. `attempt` renders each panel as its payload arrives.
  await Promise.allSettled(Object.keys(LOADERS).map((key) => attempt(key)));

  if (Object.keys(state.failed).length) {
    const strip = $("status-strip");
    strip.hidden = false;
    strip.className = "strip bad-strip";
    if (!strip.innerHTML) {
      strip.innerHTML = `<span>Some panels could not load: ` +
        `${esc(Object.keys(state.failed).join(", "))}.</span>`;
    }
  }
}

boot();
