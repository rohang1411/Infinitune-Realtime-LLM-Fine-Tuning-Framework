/* Render saved measurements first. Raw stage JSON and file listings are lazy. */
const {token, dsSuites} = window.EVAL_DASHBOARD;
const $ = id => document.getElementById(id);
const presets = {
  integrity: ['integrity'],
  ds: ['integrity','classification','calibration','e2e','baseline','continual'],
  systems: ['integrity','serving','freshness','pipeline','precision','soak','swap_stress'],
};
const view = {selected:null, detail:null, runs:[], request:0, timer:null,
              logOffset:0, logRun:null, filesRun:null};
const suiteBoxes = [...document.querySelectorAll('[data-suite]')];
const selectedSuites = () => suiteBoxes.filter(box => box.checked).map(box => box.dataset.suite);

function el(tag, content, className) {
  const node = document.createElement(tag);
  if (content !== undefined) node.textContent = content;
  if (className) node.className = className;
  return node;
}

function format(value, digits=2) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
}

function percent(value, digits=0) {
  return typeof value === 'number' && Number.isFinite(value) ? `${(100*value).toFixed(digits)}%` : '—';
}

function stageResult(stages, name) {
  const stage = stages?.[name];
  return stage?.status === 'complete' && stage.result && typeof stage.result === 'object' ? stage.result : null;
}

function panel(title, subtitle, note) {
  const box = el('article', undefined, 'result-panel');
  box.append(el('h3', title));
  if (subtitle) box.append(el('p', subtitle));
  if (note) box.append(el('p', note, 'result-note'));
  return box;
}

function addValue(box, value, label) {
  box.append(el('div', value, 'value'));
  if (label) box.append(el('p', label));
}

function addBar(box, label, value, max, displayed, color) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return;
  const row = el('div', undefined, 'bar-row');
  const caption = el('div', undefined, 'bar-label');
  caption.append(el('span', label), el('strong', displayed));
  const track = el('div', undefined, 'bar-track');
  const fill = el('div', undefined, 'bar-fill');
  fill.style.width = `${Math.max(0, Math.min(100, 100*value/max))}%`;
  if (color) fill.style.background = color;
  track.setAttribute('role', 'meter');
  track.setAttribute('aria-label', `${label}: ${displayed}`);
  track.setAttribute('aria-valuemin', '0');
  track.setAttribute('aria-valuemax', String(max));
  track.setAttribute('aria-valuenow', String(value));
  track.append(fill); row.append(caption, track); box.append(row);
}

function addMatrix(box, matrix, tasks, seed) {
  if (!Array.isArray(matrix) || matrix.length !== 3 || !matrix.every(row => Array.isArray(row) && row.length === 3)) return;
  box.append(el('p', `Accuracy after each training phase · seed ${seed}`));
  const grid = el('div', undefined, 'heatmap');
  grid.append(el('span', 'Train ↓ / Test →', 'heat-label'));
  for (const task of tasks) grid.append(el('span', task, 'heat-label'));
  matrix.forEach((row, index) => {
    grid.append(el('span', tasks[index], 'heat-label'));
    row.forEach(score => {
      const cell = el('span', percent(score));
      cell.style.background = `rgba(113, 219, 176, ${0.10 + 0.75*Math.max(0, Math.min(1, score || 0))})`;
      grid.append(cell);
    });
  });
  box.append(grid);
}

function renderDataScience(manifest, summary) {
  const host = $('ds-summary'); host.replaceChildren();
  const stages = manifest?.stages || {};
  const spec = manifest?.spec || {};
  const base = stageResult(stages, 'classification/base');
  const adapted = stageResult(stages, 'classification/checkpoint');
  const takeaways = [];
  if (base && adapted && adapted.n)
    takeaways.push(`IMDb adapter: ${adapted.correct}/${adapted.n} correct; base: ${base.correct}/${base.n}. This small test is diagnostic.`);
  if (base || adapted) {
    const count = adapted?.n ?? base?.n;
    const box = panel('IMDb generated-label accuracy',
      `${count ?? '?'} fixed test reviews · greedy generation`,
      count < 100 ? 'Small diagnostic sample. Do not use this percentage as a resume claim.' : null);
    if (base) addBar(box, 'Base model', base.accuracy, 1, `${percent(base.accuracy)} · ${base.correct}/${base.n}`);
    if (adapted) addBar(box, 'Saved adapter', adapted.accuracy, 1, `${percent(adapted.accuracy)} · ${adapted.correct}/${adapted.n}`);
    if (base?.other_rate === 1) box.append(el('p', 'The base model gave off-format answers on every review in this sample.'));
    if (adapted) box.append(el('p', `Target-class macro F1: ${format(adapted.f1_macro_target_classes, 3)} · off-format rate: ${percent(adapted.other_rate)}`));
    host.append(box);
  }
  const calibrationBase = stageResult(stages, 'calibration/base');
  const calibrationAdapter = stageResult(stages, 'calibration/checkpoint');
  if (calibrationBase && calibrationAdapter && calibrationAdapter.ece_10_equal_width > calibrationBase.ece_10_equal_width)
    takeaways.push(`Forced-choice accuracy rose, but calibration error worsened (${format(calibrationBase.ece_10_equal_width, 3)} → ${format(calibrationAdapter.ece_10_equal_width, 3)}).`);
  if (calibrationBase || calibrationAdapter) {
    const box = panel('Forced-choice calibration',
      `${calibrationAdapter?.n ?? calibrationBase?.n ?? '?'} reviews · restricted two-label probabilities`,
      'ECE is a calibration error: lower is better, but unstable on 16 examples. This is not free-generation confidence.');
    if (calibrationBase) addBar(box, 'Base accuracy', calibrationBase.forced_choice_accuracy, 1, percent(calibrationBase.forced_choice_accuracy, 1));
    if (calibrationAdapter) addBar(box, 'Adapter accuracy', calibrationAdapter.forced_choice_accuracy, 1, percent(calibrationAdapter.forced_choice_accuracy, 1));
    box.append(el('p', `ECE base ${format(calibrationBase?.ece_10_equal_width, 3)} · adapter ${format(calibrationAdapter?.ece_10_equal_width, 3)}`));
    host.append(box);
  }
  const baseline = stageResult(stages, 'baseline/summary');
  if (baseline) {
    const scores = Array.isArray(baseline.per_seed) ? baseline.per_seed : [];
    const box = panel('Ordered vs offline shuffled',
      `${scores.length} seed${scores.length === 1 ? '' : 's'} · matched train/test data`,
      'This compares example order in an offline process; it does not measure Kafka overhead.');
    for (const row of scores) {
      addBar(box, `Seed ${row.seed} ordered`, row.ordered_accuracy, 1, percent(row.ordered_accuracy));
      addBar(box, `Seed ${row.seed} shuffled`, row.offline_accuracy, 1, percent(row.offline_accuracy));
    }
    if (scores.length === 1 && scores[0].ordered_accuracy === 0 && scores[0].offline_accuracy === 0)
      box.append(el('p', 'Neither arm showed acquired accuracy in this run.', 'result-note'));
    host.append(box);
  }
  const continual = Object.entries(stages).filter(([name, stage]) =>
    /^continual\/seed_\d+\/(no_replay|reservoir_replay)\/summary$/.test(name) && stage.status === 'complete');
  for (const method of ['no_replay','reservoir_replay']) {
    const rows = continual.filter(([name]) => name.includes(`/${method}/`)).map(([, stage]) => stage.result);
    if (!rows.length) continue;
    const title = method === 'no_replay' ? 'Continual: no replay' : 'Continual: reservoir replay';
    const box = panel(title, 'Final average accuracy after IMDb → Yelp → Amazon',
      rows.every(row => row.final_average_accuracy === 0) ? 'All final scores are zero here; forgetting and transfer are not meaningful evidence of retention.' : null);
    for (const row of rows) addBar(box, `Seed ${row.seed}`, row.final_average_accuracy, 1, percent(row.final_average_accuracy));
    if (rows.length === 1) addMatrix(box, rows[0].matrix, rows[0].tasks || ['IMDb','Yelp','Amazon'], rows[0].seed);
    host.append(box);
  }
  const e2e = stageResult(stages, 'e2e/generation');
  if (e2e) {
    const box = panel('E2E data-to-text generation',
      `${e2e.n_unique_mrs} unique meaning representations · multiple references`,
      e2e.n_unique_mrs < 100 ? 'Only a tiny diagnostic set was evaluated; this is not a leaderboard result.' : 'Protocol differs from published E2E leaderboard runs.');
    addValue(box, `BLEU ${format(e2e.bleu)}`, `Max-reference ROUGE-L F1 ${format(e2e.rouge_l_f1_max_reference, 3)}`);
    host.append(box);
  }
  if (baseline && Array.isArray(baseline.per_seed) && baseline.per_seed.length &&
      baseline.per_seed.every(row => row.ordered_accuracy === 0 && row.offline_accuracy === 0))
    takeaways.push('Neither ordered nor shuffled training acquired generated-label accuracy in this run.');
  if (takeaways.length) {
    const box = panel('What this run shows', '', null); box.classList.add('takeaway');
    const list = el('ul');
    takeaways.slice(0, 3).forEach(line => list.append(el('li', line)));
    box.append(list); host.prepend(box);
  }
  if (!host.children.length) host.append(el('div',
    'This run has no model-quality measurements. Select a saved run containing classification, calibration, baseline, continual, or E2E suites.', 'empty-state'));
  if (spec.profile === 'smoke' && host.children.length && !host.querySelector('.empty-state'))
    host.prepend(el('p', 'Smoke profile: these results check the measurement pipeline on a small sample.', 'result-note'));
  if (summary?.code_matches_current === false)
    host.prepend(el('p', 'Historical code version: these saved numbers were produced before the current benchmark edits.', 'result-note'));
}

function renderEngineering(manifest) {
  const host = $('systems-summary'); host.replaceChildren();
  const stages = manifest?.stages || {};
  const serving = stageResult(stages, 'serving/paired_summary');
  if (serving && Array.isArray(serving.pairs) && serving.pairs.length) {
    const box = panel('Serving throughput', `${serving.pairs.length} matched pair${serving.pairs.length === 1 ? '' : 's'} · achieved closed-loop QPS`,
      'One small pair does not establish a speedup or maximum capacity.');
    const top = Math.max(1, ...serving.pairs.flatMap(row => [row.router_qps || 0, row.legacy_qps || 0]));
    serving.pairs.forEach(row => {
      addBar(box, `Pair ${row.repeat} router`, row.router_qps, top, `${format(row.router_qps)} QPS`);
      addBar(box, `Pair ${row.repeat} legacy`, row.legacy_qps, top, `${format(row.legacy_qps)} QPS`);
    });
    host.append(box);
  }
  const freshness = stageResult(stages, 'freshness/adapter_propagation');
  if (freshness) {
    const box = panel('Adapter publication visibility', 'Kafka snapshot → serving version',
      'This does not measure learning from a new training example.');
    addValue(box, `${format(freshness.latency_s, 3)} s`, 'One measured publication'); host.append(box);
  }
  const pipeline = stageResult(stages, 'pipeline/record_to_version');
  if (pipeline) {
    const box = panel('Record-to-version pipeline', `${pipeline.record_count} acknowledged records · ${pipeline.expected_optimizer_steps} optimizer steps`,
      'Measures version visibility, not a demonstrated improvement in generated answers.');
    addValue(box, `${format(pipeline.latency_s, 3)} s`, 'Last record publication → visible adapter'); host.append(box);
  }
  const precision = stageResult(stages, 'precision/fp32');
  if (precision) {
    const box = panel('Optimizer step timing', `${precision.device} · ${precision.steps} steps · ${precision.active_precision}`,
      'Short synthetic timing; unsupported precision modes are not speed comparisons.');
    addValue(box, `${format(precision.median_step_s, 3)} s`, 'Median measured steady step'); host.append(box);
  }
  const soak = stageResult(stages, 'soak/inference');
  if (soak) {
    const box = panel('Inference soak', `${soak.requests} requests · ${soak.failures} failures · ${format(soak.duration_s, 1)} s`,
      soak.duration_s < 3600 ? 'Too short to infer a memory leak or long-term reliability.' : null);
    addValue(box, `${format(soak.rss_first_mb, 1)} → ${format(soak.rss_last_mb, 1)} MB`, 'Loaded server process RSS'); host.append(box);
  }
  const swap = stageResult(stages, 'swap_stress/concurrent');
  if (swap) {
    const box = panel('Adapter swap stress', `${swap.requests} requests · ${swap.swaps} swaps · ${swap.failures} failures`,
      'These are observed lock-held swap times for this small run.');
    addValue(box, `${format(swap.swap_median_ms, 0)} ms median`, `${format(swap.swap_max_ms, 0)} ms maximum`); host.append(box);
  }
  if (!host.children.length) {
    const integrity = stageResult(stages, 'integrity/pytest');
    host.append(el('div', integrity?.passed ?
      'Integrity tests passed. This run did not measure serving or streaming performance.' :
      'This run has no completed engineering measurements.', 'empty-state'));
  }
}

function renderHistory() {
  const box = $('history'); box.replaceChildren();
  if (!view.runs.length) { box.append(el('div', 'No saved runs were found.', 'empty-state')); return; }
  for (const run of view.runs) {
    const button = el('button', undefined, 'run-row' + (run.id === view.selected ? ' active' : ''));
    const title = el('div', undefined, 'run-title');
    title.append(el('strong', run.id), el('span', run.profile || 'unknown profile', 'pill'));
    button.append(title);
    button.append(el('span', `${run.status} · ${run.counts.complete} completed stage${run.counts.complete === 1 ? '' : 's'}`, 'history-brief'));
    if (run.headlines?.length) button.append(el('span', run.headlines.join(' · '), 'history-brief'));
    button.addEventListener('click', () => selectRun(run.id));
    box.append(button);
  }
}

function renderHeader(data) {
  const {summary, manifest, state} = data;
  $('selected-run-label').textContent = `${summary.id} · ${summary.profile || 'unknown profile'}`;
  const box = $('run-header'); box.replaceChildren();
  const meta = el('div', undefined, 'run-meta');
  meta.append(el('strong', summary.id), el('span', summary.status,
    summary.status === 'failed' ? 'bad' : summary.status === 'complete' ? 'good' : 'warn'),
    el('span', `${summary.counts.complete} complete · ${summary.counts.failed} failed · ${summary.counts.running} running`, 'pill'));
  box.append(meta);
  box.append(el('p', `Profile: ${summary.profile || 'unknown'} · Suites: ${summary.suites.join(', ') || 'none'}`));
  if (state) box.append(el('p', `Exit code: ${state.exit_code ?? 'pending'} · Started: ${state.started_utc || 'unknown'}`));
  if (!state && summary.has_manifest) box.append(el('p', 'This older CLI run has no recorded process exit code; use its stage statuses and raw files to audit it.', 'result-note'));
  if (summary.code_matches_current === false) box.append(el('p',
    'Historical code snapshot: the benchmark source changed after this run. Its measurements remain historical evidence; rerun for current-code results.', 'result-note'));
  if (manifest?.suite_failures && Object.keys(manifest.suite_failures).length)
    box.append(el('p', `Suite failures: ${JSON.stringify(manifest.suite_failures)}`, 'bad'));
  $('resume').hidden = !(state && state.status !== 'running');
}

function renderStageDetails() {
  const host = $('stage-details'); host.replaceChildren();
  const stages = view.detail?.manifest?.stages || {};
  const names = Object.keys(stages);
  if (!names.length) { host.append(el('p', 'No stages were recorded.')); return; }
  for (const name of names) {
    const stage = stages[name];
    const item = el('details', undefined, 'stage');
    item.append(el('summary', `${stage.status === 'complete' ? '✓' : stage.status === 'failed' ? '✕' : '•'} ${name}`));
    item.addEventListener('toggle', () => {
      if (item.open && !item.querySelector('pre')) item.append(el('pre', JSON.stringify(stage, null, 2)));
    });
    host.append(item);
  }
}

async function renderFiles() {
  const id = view.selected;
  if (!id || view.filesRun === id) return;
  const host = $('files'); host.replaceChildren(el('div', 'Loading file names…', 'loading'));
  try {
    const files = await api(`/api/runs/${encodeURIComponent(id)}/files`);
    if (view.selected !== id) return;
    view.filesRun = id; host.replaceChildren();
    for (const file of files) {
      const link = el('a', `${file.path} (${Math.ceil(file.bytes/1024)} KiB)`);
      link.href = `/api/runs/${encodeURIComponent(id)}/file/${file.path.split('/').map(encodeURIComponent).join('/')}`;
      link.download = file.path.split('/').pop(); host.append(link);
    }
    if (!files.length) host.append(el('p', 'No downloadable files were found.'));
  } catch (error) { host.replaceChildren(el('p', `Could not load file names: ${error.message}`, 'bad')); }
}

function parseProgress(text) {
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith('START ')) {
      $('progress-wrap').hidden = false; $('progress').removeAttribute('value'); $('progress-text').textContent = line;
    }
    const match = line.match(/(\d+)\/(\d+).*ETA\s+(\d+)s/);
    if (match && Number(match[2]) > 0) {
      $('progress-wrap').hidden = false;
      $('progress').value = 100*Number(match[1])/Number(match[2]);
      $('progress-text').textContent = line.trim();
    }
  }
}

async function pollLog() {
  const id = view.selected;
  if (!id) return;
  if (view.logRun !== id) {
    view.logRun = id; view.logOffset = 0; $('console').textContent = '';
    $('progress-wrap').hidden = true;
  }
  const data = await api(`/api/runs/${encodeURIComponent(id)}/log?offset=${view.logOffset}`);
  if (view.selected !== id) return;
  if (!data.available) {
    $('console').textContent = 'This earlier CLI run has no combined console log. Open All stage details or Raw evidence below.';
    return;
  }
  if (view.logOffset === 0) $('console').textContent = '';
  view.logOffset = data.size;
  if (data.text) {
    $('console').textContent += data.text;
    if ($('console').textContent.length > 150000) $('console').textContent = $('console').textContent.slice(-120000);
    $('console').scrollTop = $('console').scrollHeight;
    parseProgress(data.text);
  }
}

async function api(url, options={}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function loading() {
  $('selected-run-label').textContent = view.selected || 'selected run';
  $('run-header').replaceChildren(el('div', 'Reading saved manifest; this does not start a new evaluation.', 'loading'));
  $('ds-summary').replaceChildren(el('div', 'Loading measured results…', 'loading'));
  $('systems-summary').replaceChildren();
  $('console').textContent = 'Loading saved log…';
  $('progress-wrap').hidden = true;
  $('files').replaceChildren(); view.filesRun = null;
  $('stage-details').replaceChildren();
}

async function selectRun(id) {
  view.selected = id; view.detail = null; view.logRun = null;
  const serial = ++view.request;
  clearTimeout(view.timer); renderHistory(); loading();
  try {
    const data = await api(`/api/runs/${encodeURIComponent(id)}`);
    if (serial !== view.request) return;
    view.detail = data;
    renderHeader(data);
    renderDataScience(data.manifest, data.summary);
    renderEngineering(data.manifest);
    if ($('stage-shell').open) renderStageDetails();
    if ($('files-shell').open) renderFiles();
    await pollLog();
    scheduleActiveRefresh();
  } catch (error) {
    if (serial === view.request) $('run-header').replaceChildren(el('p', `Could not load run: ${error.message}`, 'bad'));
  }
}

function scheduleActiveRefresh() {
  clearTimeout(view.timer);
  const status = view.detail?.summary?.status;
  if (status === 'running' || status === 'launching') {
    view.timer = setTimeout(async () => {
      try {
        const id = view.selected;
        const data = await api(`/api/runs/${encodeURIComponent(id)}`);
        if (id !== view.selected) return;
        const changed = JSON.stringify(data.manifest) !== JSON.stringify(view.detail?.manifest);
        view.detail = data; renderHeader(data);
        if (changed) {
          renderDataScience(data.manifest, data.summary); renderEngineering(data.manifest);
          if ($('stage-shell').open) renderStageDetails();
          view.filesRun = null;
        }
        await pollLog();
        if (data.summary.status !== 'running' && data.summary.status !== 'launching') await loadHistory(false);
        scheduleActiveRefresh();
      } catch (error) {
        $('run-header').append(el('p', `Live refresh failed: ${error.message}`, 'bad'));
        view.timer = setTimeout(scheduleActiveRefresh, 5000);
      }
    }, 2000);
  }
}

async function loadHistory(autoSelect=true) {
  view.runs = await api('/api/runs');
  renderHistory();
  if (autoSelect && !view.selected && view.runs.length) {
    // Open the most complete run with model-quality evidence, when available.
    const candidate = [...view.runs].sort((a,b) => b.ds_stages-a.ds_stages || b.counts.complete-a.counts.complete)[0];
    await selectRun(candidate.id);
  }
}

function updateFields() {
  const s = new Set(selectedSuites());
  const enabled = {seeds:s.has('baseline')||s.has('continual'), concurrency:s.has('serving')||s.has('swap_stress'),
    port:['serving','freshness','pipeline','soak'].some(x=>s.has(x)), kafka_bootstrap:s.has('freshness')||s.has('pipeline'),
    pipeline_timeout:s.has('pipeline'), checkpoint:['classification','calibration','serving','freshness','soak','swap_stress'].some(x=>s.has(x)),
    e2e_checkpoint:s.has('e2e'), soak_seconds:s.has('soak'), soak_rate:s.has('soak'), compile:s.has('precision')};
  for (const [id, active] of Object.entries(enabled)) $(id).disabled = !active;
}

function settings() {
  const s = new Set(selectedSuites());
  const raw = {suites:[...s], profile:$('profile').value, seed:Number($('seed').value),
    seeds:$('seeds').value.split(',').map(x=>Number(x.trim())), max_seq_length:Number($('max_seq_length').value), offline:$('offline').checked};
  if (s.has('serving')||s.has('swap_stress')) raw.concurrency = Number($('concurrency').value);
  if (['serving','freshness','pipeline','soak'].some(x=>s.has(x))) raw.port = Number($('port').value);
  if (s.has('freshness')||s.has('pipeline')) raw.kafka_bootstrap = $('kafka_bootstrap').value.trim();
  if (s.has('pipeline')) raw.pipeline_timeout = Number($('pipeline_timeout').value);
  if (['classification','calibration','serving','freshness','soak','swap_stress'].some(x=>s.has(x)) && $('checkpoint').value.trim()) raw.checkpoint = $('checkpoint').value.trim();
  if (s.has('e2e') && $('e2e_checkpoint').value.trim()) raw.e2e_checkpoint = $('e2e_checkpoint').value.trim();
  if (s.has('soak')) { if ($('soak_seconds').value.trim()) raw.soak_seconds = Number($('soak_seconds').value); raw.soak_rate = Number($('soak_rate').value); }
  if (s.has('precision')) raw.compile = $('compile').checked;
  return raw;
}

suiteBoxes.forEach(box => box.addEventListener('change', updateFields));
document.querySelectorAll('[data-preset]').forEach(button => button.addEventListener('click', () => {
  suiteBoxes.forEach(box => box.checked = presets[button.dataset.preset].includes(box.dataset.suite)); updateFields();
}));
$('profile').addEventListener('change', () => {
  const p = $('profile').value;
  $('max_seq_length').value = {smoke:128,standard:256,full:512}[p];
  $('seeds').value = {smoke:'42',standard:'42,1337,2026',full:'42,1337,2026,7,11'}[p];
});
$('stage-shell').addEventListener('toggle', () => { if ($('stage-shell').open) renderStageDetails(); });
$('files-shell').addEventListener('toggle', () => { if ($('files-shell').open) renderFiles(); });
$('start').addEventListener('click', async () => {
  $('form-error').textContent = ''; $('start').disabled = true;
  try {
    const result = await api('/api/runs', {method:'POST', headers:{'Content-Type':'application/json','X-Evaluation-Token':token}, body:JSON.stringify(settings())});
    await loadHistory(false); await selectRun(result.run_id);
  } catch (error) { $('form-error').textContent = error.message; }
  finally { $('start').disabled = false; }
});
$('resume').addEventListener('click', async () => {
  if (!view.selected) return;
  try {
    await api(`/api/runs/${encodeURIComponent(view.selected)}/resume`, {method:'POST', headers:{'X-Evaluation-Token':token}});
    await selectRun(view.selected);
  } catch (error) { $('form-error').textContent = error.message; }
});
$('refresh').addEventListener('click', async () => {
  try { await loadHistory(false); if (view.selected) await selectRun(view.selected); }
  catch (error) { $('form-error').textContent = error.message; }
});
updateFields();
loadHistory(true).catch(error => $('history').replaceChildren(el('p', `Could not list runs: ${error.message}`, 'bad')));
