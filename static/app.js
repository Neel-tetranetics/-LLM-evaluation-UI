// LLM Evaluation UI. Plain JavaScript: talks to the JSON API in app.py.

let config = null;
let currentRunId = null;
let currentResults = [];
let refreshTimer = null;

// ---------- small helpers ----------

// Model answers are untrusted text, so everything shown goes through this
function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (char) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]
  ));
}

function showMessage(text, kind = "error") {
  const box = document.getElementById("message");
  box.textContent = text;
  box.className = `message ${kind}`;
  if (kind === "ok") setTimeout(() => box.classList.add("hidden"), 4000);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const isJson = (response.headers.get("content-type") || "").includes("json");
  const data = isJson ? await response.json() : await response.text();
  if (!response.ok) throw new Error(data.detail || data || `HTTP ${response.status}`);
  return data;
}

// Wraps a UI action so every failure is shown to the user, never ignored
function safely(action) {
  return async (...args) => {
    try {
      await action(...args);
    } catch (error) {
      showMessage(error.message);
    }
  };
}

function percent(score) {
  return `${(score * 100).toFixed(1)}%`;
}

function isRunning(status) {
  return !["done", "failed", "cancelled", "interrupted"].includes(status);
}

// ---------- page switching ----------

function showPage(pageId) {
  document.querySelectorAll(".page").forEach((page) => page.classList.toggle("hidden", page.id !== pageId));
  document.querySelectorAll("nav button").forEach((button) => button.classList.toggle("active", button.dataset.page === pageId));
  clearInterval(refreshTimer);
  if (pageId === "runs") { loadRuns(); refreshTimer = setInterval(safely(loadRuns), 3000); }
  if (pageId === "datasets") safely(loadDatasets)();
  if (pageId === "settings") safely(loadSettings)();
}

document.querySelectorAll("nav button").forEach((button) => {
  button.addEventListener("click", () => showPage(button.dataset.page));
});

// ---------- New run ----------

async function loadNewRunForm() {
  config = await api("/api/config");
  const form = document.getElementById("run-form");
  form.temperature.value = config.run.temperature;
  form.max_tokens.value = config.run.max_tokens;
  form.top_p.value = config.run.top_p ?? 1;
  form.top_k.value = config.run.top_k ?? -1;
  form.presence_penalty.value = config.run.presence_penalty ?? 0;
  // This function also re-runs after uploads/settings changes: never overwrite what the user typed
  if (!form.system_prompt.value) form.system_prompt.value = config.run.default_system_prompt || "";
  document.getElementById("quick-size").textContent = config.run.quick_size;

  const presetSelect = document.getElementById("preset-select");
  presetSelect.innerHTML = '<option value="">— paste your own command —</option>' +
    (config.test_model_presets || []).map((preset, index) => `<option value="${index}">${escapeHtml(preset.name)}</option>`).join("");

  document.getElementById("judge-select").innerHTML = config.judges.map((judge) =>
    `<option ${judge.name === config.default_judge ? "selected" : ""}>${escapeHtml(judge.name)}</option>`).join("");

  const datasets = await api("/api/datasets");
  const byCategory = {};
  datasets.forEach((dataset) => (byCategory[dataset.category] ||= []).push(dataset));
  document.getElementById("dataset-checkboxes").innerHTML = Object.entries(byCategory).map(([category, list]) =>
    `<h4>${escapeHtml(category.replace("_", " "))}</h4>` + list.map((dataset) => `
      <label class="${dataset.available ? "" : "disabled"}" title="${dataset.available ? "" : "needs Docker"}">
        <input type="checkbox" name="datasets" value="${escapeHtml(dataset.name)}" ${dataset.available ? "" : "disabled"}>
        ${escapeHtml(dataset.name)} <small>(${dataset.count})</small>
      </label>`).join("")).join("");
}

document.getElementById("preset-select").addEventListener("change", (event) => {
  const preset = config.test_model_presets[event.target.value];
  const commandBox = document.querySelector("#run-form [name=command]");
  if (preset) { commandBox.value = preset.command; commandBox.dispatchEvent(new Event("input")); }
});

document.querySelector("#run-form [name=command]").addEventListener("input", (event) => {
  document.getElementById("trust-warning").classList.toggle("hidden", !event.target.value.includes("--trust-remote-code"));
});

document.getElementById("run-form").addEventListener("submit", safely(async (event) => {
  event.preventDefault();
  const form = event.target;
  const sizeChoice = form.querySelector("[name=size]:checked").value;
  const runForm = {
    name: form.name.value,
    command: form.command.value,
    system_prompt: form.system_prompt.value,
    datasets: [...form.querySelectorAll("[name=datasets]:checked")].map((box) => box.value),
    size: sizeChoice === "custom" ? document.getElementById("custom-size").value : sizeChoice,
    judge: form.judge.value,
    temperature: form.temperature.value,
    max_tokens: form.max_tokens.value,
    top_p: form.top_p.value,
    top_k: form.top_k.value,
    presence_penalty: form.presence_penalty.value,
  };
  const { run_id } = await api("/api/runs", { method: "POST", body: runForm });
  showMessage(`Run started: ${run_id}`, "ok");
  openRun(run_id);
}));

// ---------- Runs list ----------

async function loadRuns() {
  const runs = await api("/api/runs");
  document.getElementById("runs-table").innerHTML = runs.map((run) => {
    const progress = isRunning(run.status) && run.progress.total
      ? ` ${run.progress.done}/${run.progress.total}` : "";
    const score = run.summary ? percent(run.summary.overall.score) : "";
    return `<tr>
      <td><input type="checkbox" class="compare-check" value="${escapeHtml(run.id)}"></td>
      <td>${escapeHtml(run.name)}</td>
      <td>${escapeHtml(run.model_path)}</td>
      <td>${escapeHtml(run.created)}</td>
      <td class="${escapeHtml(run.status)}">${escapeHtml(run.status)}${progress}</td>
      <td>${score}</td>
      <td><button data-open="${escapeHtml(run.id)}">Open</button></td>
    </tr>`;
  }).join("") || '<tr><td colspan="7">No runs yet.</td></tr>';
}

document.getElementById("runs-table").addEventListener("click", (event) => {
  if (event.target.dataset.open) openRun(event.target.dataset.open);
});

// The table is redrawn every 3 s, so remember ticked runs across redraws
const tickedRuns = new Set();
document.getElementById("runs-table").addEventListener("change", (event) => {
  if (!event.target.classList.contains("compare-check")) return;
  event.target.checked ? tickedRuns.add(event.target.value) : tickedRuns.delete(event.target.value);
});
new MutationObserver(() => {
  document.querySelectorAll(".compare-check").forEach((box) => { box.checked = tickedRuns.has(box.value); });
}).observe(document.getElementById("runs-table"), { childList: true });

document.getElementById("compare-button").addEventListener("click", safely(async () => {
  if (tickedRuns.size < 2) throw new Error("Tick at least two runs to compare.");
  await showCompare([...tickedRuns]);
}));

// ---------- Run detail ----------

function openRun(runId) {
  currentRunId = runId;
  showPage("run-detail");
  history.replaceState(null, "", `#run=${encodeURIComponent(runId)}`);  // shareable link to this run
  document.getElementById("log-box").classList.add("hidden");
  safely(loadRunDetail)();
  refreshTimer = setInterval(safely(loadRunDetail), 3000);
}

async function loadRunDetail() {
  const run = await api(`/api/runs/${currentRunId}`);
  document.getElementById("detail-title").textContent = `Run: ${run.name}`;

  const progress = run.progress || {};
  const share = progress.total ? Math.round((100 * progress.done) / progress.total) : 0;
  document.getElementById("detail-status").innerHTML = `
    <p>Status: <b class="${escapeHtml(run.status)}">${escapeHtml(run.status)}</b>
      ${isRunning(run.status) && progress.total ? ` — ${escapeHtml(progress.stage)} ${progress.done}/${progress.total}` : ""}</p>
    ${isRunning(run.status) ? `<div class="progress-bar"><div style="width:${share}%"></div></div>` : ""}
    ${run.error ? `<pre class="error">${escapeHtml(run.error)}</pre>` : ""}`;
  document.getElementById("cancel-button").classList.toggle("hidden", !isRunning(run.status));

  const summary = run.summary || {};
  const overallLast = ([a], [b]) => (a === "overall") - (b === "overall") || a.localeCompare(b);
  document.getElementById("score-table").innerHTML = Object.entries(summary).sort(overallLast).map(([dataset, counts]) => `
    <tr><td>${dataset === "overall" ? "<b>overall</b>" : escapeHtml(dataset)}</td><td>${percent(counts.score)}</td>
    <td>${counts.correct}</td><td>${counts.incorrect}</td><td>${counts.error}</td><td>${counts.total}</td></tr>`).join("")
    || '<tr><td colspan="6">Scores appear when the run is done.</td></tr>';

  const shownSettings = { ...run };
  ["summary", "progress", "error", "id"].forEach((key) => delete shownSettings[key]);
  document.getElementById("detail-settings").textContent = JSON.stringify(shownSettings, null, 2);

  // Stop polling once finished, after one last load of the results
  if (!isRunning(run.status)) clearInterval(refreshTimer);
  currentResults = await api(`/api/runs/${currentRunId}/results`);
  renderResults();
}

function renderResults() {
  const wrongOnly = document.getElementById("wrong-only").checked;
  const shown = currentResults.filter((result) => !wrongOnly || result.verdict !== "correct");
  document.getElementById("results-list").innerHTML = shown.map((result) => `
    <details>
      <summary><span class="${escapeHtml(result.verdict)}">${escapeHtml(result.verdict)}</span>
        · ${escapeHtml(result.dataset)} · ${escapeHtml(String(result.question).slice(0, 110))}</summary>
      <h5>Question</h5><pre>${escapeHtml(result.question)}</pre>
      <h5>Ground truth</h5><pre>${escapeHtml(typeof result.ground_truth === "string" ? result.ground_truth : JSON.stringify(result.ground_truth, null, 1))}</pre>
      <h5>Model answer</h5><pre>${escapeHtml(result.answer)}</pre>
      ${result.tool_calls && result.tool_calls.length ? `<h5>Tool calls</h5><pre>${escapeHtml(JSON.stringify(result.tool_calls, null, 1))}</pre>` : ""}
      ${result.reasoning ? `<h5>${result.grader === "agent" ? "Agent steps" : "Thinking"}</h5><pre>${escapeHtml(result.reasoning)}</pre>` : ""}
      <h5>Why this verdict</h5><pre>${escapeHtml(result.reason)}</pre>
      <small>${result.seconds ?? ""} s</small>
    </details>`).join("") || "<p>No questions to show.</p>";
}

document.getElementById("wrong-only").addEventListener("change", renderResults);

document.getElementById("cancel-button").addEventListener("click", safely(async () => {
  await api(`/api/runs/${currentRunId}/cancel`, { method: "POST" });
  showMessage("Cancel requested. The run stops after the questions in progress.", "ok");
}));

document.querySelectorAll("[data-log]").forEach((button) => {
  button.addEventListener("click", safely(async () => {
    const box = document.getElementById("log-box");
    box.textContent = await api(`/api/runs/${currentRunId}/log/${button.dataset.log}`);
    box.classList.remove("hidden");
  }));
});

// ---------- Compare ----------

async function showCompare(runIds) {
  const data = await api(`/api/compare?ids=${encodeURIComponent(runIds.join(","))}`);
  showPage("compare");
  history.replaceState(null, "", `#compare=${encodeURIComponent(runIds.join(","))}`);
  const runs = data.runs;
  document.getElementById("compare-head").innerHTML =
    `<tr><th>Dataset</th>${runs.map((run) => `<th>${escapeHtml(run.name)}<br><small>${escapeHtml(run.id)}</small></th>`).join("")}</tr>`;

  const datasetNames = [...new Set(runs.flatMap((run) => Object.keys(run.summary || {})))]
    .sort((a, b) => (a === "overall") - (b === "overall") || a.localeCompare(b));
  document.getElementById("compare-scores").innerHTML = datasetNames.map((dataset) => `
    <tr><td>${dataset === "overall" ? "<b>overall</b>" : escapeHtml(dataset)}</td>
    ${runs.map((run) => {
      const counts = (run.summary || {})[dataset];
      return `<td>${counts ? `${percent(counts.score)} <small>(${counts.correct}/${counts.total})</small>` : "—"}</td>`;
    }).join("")}</tr>`).join("");

  document.getElementById("difference-count").textContent = data.differences.length;
  document.getElementById("difference-head").innerHTML =
    `<tr><th>Dataset</th><th>Question</th>${runs.map((run) => `<th>${escapeHtml(run.name)}</th>`).join("")}</tr>`;
  document.getElementById("difference-rows").innerHTML = data.differences.map((difference) => `
    <tr><td>${escapeHtml(difference.dataset)}</td><td>${escapeHtml(String(difference.question).slice(0, 160))}</td>
    ${runs.map((run) => `<td class="${escapeHtml(difference.verdicts[run.id])}">${escapeHtml(difference.verdicts[run.id])}</td>`).join("")}</tr>`).join("")
    || `<tr><td colspan="${runs.length + 2}">No differences on shared questions.</td></tr>`;
}

// ---------- Datasets ----------

async function loadDatasets() {
  const datasets = await api("/api/datasets");
  document.getElementById("datasets-table").innerHTML = datasets.map((dataset) => `
    <tr><td>${escapeHtml(dataset.name)}</td><td>${escapeHtml(dataset.category)}</td><td>${dataset.count}</td>
    <td>${dataset.available ? "ready" : '<span class="error">needs Docker</span>'}</td></tr>`).join("");
}

function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("could not read the file"));
    reader.readAsText(file);
  });
}

document.getElementById("upload-form").addEventListener("submit", safely(async (event) => {
  event.preventDefault();
  const form = event.target;
  const file = form.file.files[0];
  const result = await api("/api/datasets/upload", {
    method: "POST",
    body: { name: form.name.value, filename: file.name, content: await readFileAsText(file) },
  });
  showMessage(`Uploaded ${result.name} with ${result.count} questions.`, "ok");
  form.reset();
  await loadDatasets();
  await loadNewRunForm();
}));

// ---------- Settings ----------

// Run settings shown on the Settings page: [config key, label, help text, input step]
const RUN_SETTING_GROUPS = [
  ["Questions", [
    ["quick_size", "Quick run size", "Questions per dataset for a Quick run", "1"],
    ["seed", "Seed", "Same seed + same size = same questions every run", "1"],
  ]],
  ["Sampling (test model)", [
    ["temperature", "Temperature", "0.6 recommended for thinking models; 0 can loop", "0.1"],
    ["top_p", "Top p", "0.95 recommended", "0.05"],
    ["top_k", "Top k", "20 recommended; -1 = off", "1"],
    ["presence_penalty", "Presence penalty", "1.5 reduces repetition loops; 0 = off", "0.1"],
    ["max_tokens", "Max tokens", "Thinking + answer. Reasoning evals use 16k-32k", "1"],
  ]],
  ["Speed", [
    ["parallel_requests", "Parallel requests", "Questions/judge requests sent at once", "1"],
  ]],
  ["Timeouts and retries", [
    ["request_timeout_sec", "Test request timeout (s)", "One question to the test model", "1"],
    ["judge_timeout_sec", "Judge request timeout (s)", "One judge request", "1"],
    ["request_retries", "Retries", "Extra tries after a failed request", "1"],
    ["server_start_timeout_sec", "vLLM start timeout (s)", "Wait for a model to load", "1"],
    ["code_test_timeout_sec", "Code test timeout (s)", "One HumanEval/MBPP test in Docker", "1"],
  ]],
  ["Judge", [
    ["judge_max_tokens", "Judge max tokens", "Judge replies are one short JSON", "1"],
  ]],
  ["Agent benchmarks (SWE-bench, Terminal-Bench)", [
    ["agent_max_steps", "Max steps", "Shell commands per task", "1"],
    ["agent_step_max_tokens", "Tokens per step", "Per model reply; whole chat must fit in --max-model-len", "1"],
    ["agent_command_timeout_sec", "Command timeout (s)", "One shell command", "1"],
    ["agent_output_chars", "Output shown (chars)", "End of each command's output sent to the model", "1"],
    ["agent_task_timeout_sec", "Task time limit (s)", "SWE-bench; Terminal-Bench uses each task's own limit", "1"],
    ["agent_verifier_timeout_sec", "Test run timeout (s)", "SWE-bench test script", "1"],
  ]],
];

function renderRunSettings() {
  document.getElementById("default-judge-select").innerHTML = config.judges.map((judge) =>
    `<option ${judge.name === config.default_judge ? "selected" : ""}>${escapeHtml(judge.name)}</option>`).join("");
  const groups = RUN_SETTING_GROUPS.map(([title, fields]) => `
    <fieldset class="settings-group"><legend>${title}</legend><div class="settings-grid">
      ${fields.map(([key, label, help, step]) => `
        <label>${label}
          <input name="${key}" type="number" step="${step}" value="${escapeHtml(config.run[key] ?? "")}">
          <small>${help}</small>
        </label>`).join("")}
    </div></fieldset>`).join("");
  document.getElementById("run-settings-fields").innerHTML = groups + `
    <fieldset class="settings-group"><legend>Default system prompt</legend>
      <label><textarea name="default_system_prompt" rows="3">${escapeHtml(config.run.default_system_prompt ?? "")}</textarea>
      <small>Pre-filled on the New run page</small></label>
    </fieldset>`;
}

document.getElementById("run-settings-form").addEventListener("submit", safely(async (event) => {
  event.preventDefault();
  const form = event.target;
  const run = {};
  RUN_SETTING_GROUPS.forEach(([, fields]) => fields.forEach(([key]) => { run[key] = form[key].value; }));
  run.default_system_prompt = form.default_system_prompt.value;
  await api("/api/settings", { method: "PUT", body: { run, default_judge: form.default_judge.value } });
  showMessage("Run settings saved.", "ok");
  await loadSettings();
  await loadNewRunForm();
}));

async function loadSettings() {
  config = await api("/api/config");
  renderRunSettings();
  document.getElementById("judges-table").innerHTML = config.judges.map((judge) => `
    <tr><td>${escapeHtml(judge.name)}${judge.name === config.default_judge ? " <small>(default)</small>" : ""}</td>
    <td class="command">${escapeHtml(judge.command)}</td>
    <td>${judge.name === config.default_judge ? "" : `<button class="danger" data-delete-judge="${escapeHtml(judge.name)}">Delete</button>`}</td></tr>`).join("");
  document.getElementById("presets-table").innerHTML = (config.test_model_presets || []).map((preset) => `
    <tr><td>${escapeHtml(preset.name)}</td><td class="command">${escapeHtml(preset.command)}</td>
    <td><button class="danger" data-delete-preset="${escapeHtml(preset.name)}">Delete</button></td></tr>`).join("");
}

document.getElementById("settings").addEventListener("click", safely(async (event) => {
  const judgeName = event.target.dataset.deleteJudge;
  const presetName = event.target.dataset.deletePreset;
  if (judgeName && confirm(`Delete judge "${judgeName}"?`)) {
    await api(`/api/judges/${encodeURIComponent(judgeName)}`, { method: "DELETE" });
  } else if (presetName && confirm(`Delete preset "${presetName}"?`)) {
    await api(`/api/presets/${encodeURIComponent(presetName)}`, { method: "DELETE" });
  } else {
    return;
  }
  await loadSettings();
  await loadNewRunForm();
}));

document.getElementById("judge-form").addEventListener("submit", safely(async (event) => {
  event.preventDefault();
  const form = event.target;
  await api("/api/judges", {
    method: "POST",
    body: { name: form.name.value, command: form.command.value },
  });
  showMessage("Judge added.", "ok");
  form.reset();
  await loadSettings();
  await loadNewRunForm();
}));

document.getElementById("preset-form").addEventListener("submit", safely(async (event) => {
  event.preventDefault();
  const form = event.target;
  await api("/api/presets", { method: "POST", body: { name: form.name.value, command: form.command.value } });
  showMessage("Preset added.", "ok");
  form.reset();
  await loadSettings();
  await loadNewRunForm();
}));

// ---------- start ----------

safely(loadNewRunForm)();

// Links like #run=<id> or #compare=<id>,<id> open that page directly
const [linkType, linkValue] = decodeURIComponent(location.hash.slice(1)).split("=");
if (linkType === "run" && linkValue) openRun(linkValue);
if (linkType === "compare" && linkValue) safely(showCompare)(linkValue.split(","));
if (["runs", "datasets", "settings"].includes(linkType)) showPage(linkType);  // e.g. #settings
