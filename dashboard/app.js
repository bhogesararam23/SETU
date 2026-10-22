/* SETU operator console.
 *
 * Plain JavaScript and no build step keep local setup simple and make the
 * interface easy to inspect alongside the API and validation outputs.
 */

const GATE_WHY = {
  not_armed: "value is below the arm level, so there is nothing to act on",
  persistence: "one odd window is not an episode; k of n must agree",
  uncertainty: "the interval still includes clean air - acting would be a guess",
  low_posterior: "no source clears the confidence floor",
  refused_by_evidence: "the attributed source has no local lever that could move it",
  dwell: "already dispatched for this zone inside the minimum dwell",
  no_lever: "policy resolved to nothing actionable",
  unknown_source: "no policy entry for the named source",
};

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

const $ = (id) => document.getElementById(id);

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${url}`);
  }
  return res.json();
}

function fmt(n, dp = 1) {
  return (n === null || n === undefined || Number.isNaN(Number(n)))
    ? "n/a" : Number(n).toFixed(dp);
}

function renderSite(site) {
  $("site-label").textContent = site.label;
  $("t-where-note").textContent =
    `${site.instrumented} instrumented nodes on a ` +
    `${(site.sites ** 0.5 - 1) * 320 / 1000} km grid, ` +
    `${site.held_out} held out from the fit`;
}

function renderTiles(actions, validation, gates) {
  const list = actions.actions || [];
  if (!list.length) {
    $("t-what").textContent = "Nothing to dispatch";
    $("t-what-note").textContent =
      "In this window the grid stayed under the action level. That is a real " +
      "answer, not a stalled system.";
    return;
  }
  const latest = list[0];
  const ev = latest.evidence || {};

  $("t-what").textContent = `${fmt(latest.trigger_value)} µg/m³`;
  $("t-what-note").textContent =
    `±${fmt(latest.trigger_sigma)} · threshold ${fmt(latest.threshold)} · ` +
    `interval low ${fmt((latest.trigger_value - latest.trigger_sigma))}`;

  $("t-where").textContent = latest.node;
  $("t-where-note").textContent = `owner: ${latest.owner}`;

  const conf = (ev.posterior || {})[latest.source];
  $("t-source").textContent = SOURCE_LABEL[latest.source] || latest.source;
  $("t-source-note").textContent =
    `posterior ${fmt(conf, 2)} · floor ${fmt(gates.gates.min_source_posterior, 2)}` +
    (latest.source === "secondary" ? " · no local lever exists" : "");

  $("t-action").textContent = latest.lever.replace(/_/g, " ");
  $("t-action-note").textContent =
    `half-life ${fmt(ev.lever_half_life_min, 0)} min · recheck ` +
    `${(latest.recheck_at || "n/a").slice(11, 16)}` +
    ` · ${latest.requires_signoff ? "awaiting human sign-off" : "auto"}`;
}

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
    const title =
      `${n.id} · ${n.role}` +
      `${n.dead ? " · optical bench dead" : ""}` +
      `${n.obstructed ? " · inlet obstructed" : ""}` +
      `${n.drift_fraction ? ` · drifting (+${(n.drift_fraction * 100).toFixed(0)}%)` : ""}` +
      `${n.quarantined ? " · QUARANTINED" : ""}` +
      `${n.calibration_mae != null ? ` · cal MAE ${n.calibration_mae.toFixed(2)}` : ""}`;
    out += `<circle class="node ${cls}" cx="${x(n).toFixed(1)}" cy="${y(n).toFixed(1)}" ` +
           `r="${n.role === 'holdout' ? 4 : 6}"><title>${title}</title></circle>`;
  }
  svg.innerHTML = out;

  const q = nodes.filter((n) => n.quarantined).length;
  const d = nodes.filter((n) => n.dead).length;
  const o = nodes.filter((n) => n.obstructed).length;
  $("lattice-foot").textContent =
    `${nodes.filter((n) => n.role !== "holdout").length} instrumented, ` +
    `${nodes.filter((n) => n.role === "holdout").length} held out · ` +
    `${d} dead benches, ${o} obstructed inlets, ${q} quarantined by the drift monitor`;
}

function renderActions(payload) {
  const ul = $("actions");
  const rows = (payload.actions || []).slice(0, 14);
  $("dispatch-count").textContent =
    `${payload.totals?.actions ?? 0} dispatches · ` +
    `${payload.totals?.suppressed ?? 0} suppressed`;
  if (!rows.length) {
    ul.innerHTML = `<li><span class="who">No dispatches in this window.</span></li>`;
    return;
  }
  ul.innerHTML = rows.map((a) => {
    const ev = a.evidence || {};
    const post = Object.entries(ev.posterior || {})
      .sort((p, q) => q[1] - p[1]).slice(0, 2)
      .map(([k, v]) => `${(SOURCE_LABEL[k] || k).split(" ")[0]} ${v.toFixed(2)}`)
      .join(", ");
    return `<li>
      <span class="head">${a.node} → ${a.lever.replace(/_/g, " ")}</span>
      <span class="when">${(a.dispatched_at || "").slice(5, 16).replace("T", " ")}</span>
      <span class="who">${a.owner} · ${fmt(a.trigger_value)} ± ${fmt(a.trigger_sigma)} µg/m³</span>
      <span class="conf">${a.requires_signoff ? "sign-off required" : "auto"}</span>
      <span class="post">${SOURCE_LABEL[a.source] || a.source} · ${post}</span>
    </li>`;
  }).join("");
}

function renderLedger(payload) {
  const totals = payload.totals || {};
  const counts = totals.suppressed_by_reason || {};
  const tbody = $("ledger").querySelector("tbody");
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    tbody.innerHTML = `<tr><td colspan="3" class="why">Nothing suppressed.</td></tr>`;
    return;
  }
  tbody.innerHTML = entries.map(([reason, count]) =>
    `<tr><td>${reason.replace(/_/g, " ")}</td>
         <td class="num">${count}</td>
         <td class="why">${GATE_WHY[reason] || ""}</td></tr>`).join("");
}

function renderValidation(v) {
  // v.calibration.mean is the average over every node in the run. Reading one
  // node's row here made the console disagree with results/VALIDATION.md under
  // an identical label, so the aggregate is now served deliberately.
  const c = (v.calibration || {}).mean || {};
  const before = {}, after = {};
  for (const k of ["mae", "rmse", "nmae", "r2", "bias", "threshold_accuracy"]) {
    before[k] = c["before_" + k];
    after[k] = c["after_" + k];
  }
  const standard = v.action_standard || "action level";
  const att = v.attribution || {};
  const f1 = (v.forecast || []).find((r) => r.horizon_h === 1) || {};
  const t = v.transfer || {};

  const rows = [
    ["Calibration MAE", `${fmt(before.mae, 2)} → ${fmt(after.mae, 2)} µg/m³`,
      (after.mae != null && before.mae != null && after.mae < before.mae)],
    ["Calibration R²", `${fmt(before.r2, 2)} → ${fmt(after.r2, 2)}`,
      (after.r2 != null && after.r2 > 0.5)],
    [`Threshold accuracy @ ${fmt((v.summary || {}).threshold_ugm3, 0)} µg/m³`,
      `${fmt(before.threshold_accuracy, 3)} → ${fmt(after.threshold_accuracy, 3)} (${standard})`,
      (after.threshold_accuracy ?? 0) > 0.9],
    ["Interval coverage (mean)", `${fmt(c.interval_coverage, 3)} (nominal 0.80)`,
      (c.interval_coverage ?? 0) >= 0.7],
    ["Transfer, never co-located",
      t.nodes_transferred ? `MAE ${fmt(t.mean_mae_before, 2)} → ${fmt(t.mean_mae_after, 2)} µg/m³` : "n/a",
      (t.mean_mae_after ?? 1e9) < (t.mean_mae_before ?? 0)],
    ["Attribution accuracy",
      `${fmt(att.accuracy, 3)} vs ${fmt(att.chance, 2)} chance`,
      (att.accuracy ?? 0) > (att.chance ?? 1) + 0.2],
    ["Above 0.60 posterior",
      `${fmt(att.accuracy_above_060, 3)} on ${fmt((att.share_of_hours_above_060 ?? 0) * 100, 0)}% of hours`,
      (att.accuracy_above_060 ?? 0) > (att.accuracy ?? 0)],
    ["1-hour forecast skill", `RMSE ${fmt(f1.model_rmse, 2)} vs ${fmt(f1.persistence_rmse, 2)} (${fmt(f1.skill_vs_persistence, 3)})`,
      (f1.skill_vs_persistence ?? 0) > 0],
  ];

  $("validation").querySelector("tbody").innerHTML = rows.map(([k, val, ok]) =>
    `<tr><td>${k}</td><td class="num ${ok ? "pass" : "fail"}">${val}</td></tr>`).join("");

  const failing = (v.shares || []).filter((s) => !s.within).map((s) => s.source);
  $("shares-foot").innerHTML = failing.length
    ? `Share cross-check against receptor studies <span class="fail">fails</span> for ` +
      `${failing.join(", ")}. The twin's injection rates are not tuned to a published ` +
      `apportionment, so absolute shares measure the twin rather than the city; ` +
      `within-class discrimination is the row above. Tracked as an open item.`
    : `Share cross-check agrees with published receptor ranges.`;

  const w = v.summary || {};
  $("window-label").textContent =
    `${w.hours ?? "?"} real hours · ${w.nodes ?? "?"} nodes · threshold ` +
    `${fmt(w.threshold_ugm3, 0)} µg/m³ (${standard})`;
}

function renderEqualCost(ec) {
  const host = $("equal-cost");
  const labels = {
    sprinkling_62min: "Water sprinkling · 62 min",
    sweeping_150min: "Vacuum sweeping · 150 min",
    suppressant_20h: "CaCl₂ suppressant · ~20 h",
  };
  host.innerHTML = Object.entries(ec)
    .filter(([, row]) => row && typeof row === "object" && "roster_coverage" in row)
    .map(([key, row]) => {
      const gain = Number(row.gain) || 0;
      return `<div class="bar-row">
        <span class="bar-label">${labels[key] || key}</span>
        <span class="bar-track">
          <span class="bar-fill roster" style="width:${(row.roster_coverage * 100).toFixed(0)}%"></span>
          <span class="bar-fill" style="width:${(row.triggered_coverage * 100).toFixed(0)}%;opacity:.35"></span>
        </span>
        <span class="bar-gain ${gain > 0.05 ? "pos" : "zero"}">${gain > 0.05 ? "+" + gain.toFixed(2) : "no gain"}</span>
      </div>`;
    }).join("") +
    `<p class="foot">Grey bar: fixed roster coverage of breach hours. ` +
    `Green: same number of passes, triggered. Both arms cost the same.</p>`;
}

function tick() {
  $("clock").textContent = new Date().toLocaleTimeString();
}

async function boot() {
  try {
    const [site, gates, nodes, actions, validation] = await Promise.all([
      getJSON("/api/site"), getJSON("/api/gates"), getJSON("/api/nodes"),
      getJSON("/api/actions?limit=40"), getJSON("/api/validation"),
    ]);
    renderSite(site);
    renderTiles(actions, validation, gates);
    renderLattice(nodes);
    renderActions(actions);
    renderLedger(actions);
    renderValidation(validation);
    renderEqualCost(validation.equal_cost || {});
  } catch (err) {
    $("provenance-banner").innerHTML =
      `<strong>Service unavailable.</strong> ${err.message}. ` +
      `Run <code>python -m setu run</code> first, then <code>python -m setu serve</code>.`;
  }
  tick();
  setInterval(tick, 1000);
}

boot();
