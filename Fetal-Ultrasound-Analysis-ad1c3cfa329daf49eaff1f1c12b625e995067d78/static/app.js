/* Fetal Ultrasound Analysis — Main JavaScript */
const API = window.location.origin;
const $ = sel => document.querySelector(sel);
const $$ = sel => document.querySelectorAll(sel);

/* ── CLASS DATA ──────────────────── */
const CLASS_ICONS = {
  "Fetal abdomen": `<svg viewBox="0 0 28 28" fill="none"><ellipse cx="14" cy="14" rx="10" ry="11" stroke="#fff" stroke-width="2"/><ellipse cx="14" cy="15" rx="5" ry="6" stroke="#fff" stroke-width="1.5" stroke-dasharray="3 2"/></svg>`,
  "Fetal brain": `<svg viewBox="0 0 28 28" fill="none"><path d="M14 4c-5 0-9 4-9 9 0 3.5 2 6.5 5 8h8c3-1.5 5-4.5 5-8 0-5-4-9-9-9z" stroke="#fff" stroke-width="2"/></svg>`,
  "Fetal femur": `<svg viewBox="0 0 28 28" fill="none"><path d="M7 7c0 2 1 3 3 3h1l5 8h1c2 0 3 1 3 3" stroke="#fff" stroke-width="2.5" stroke-linecap="round"/></svg>`,
  "Fetal thorax": `<svg viewBox="0 0 28 28" fill="none"><path d="M14 5v18" stroke="#fff" stroke-width="1.5"/><path d="M8 8c3 1 5 0 6 0M20 8c-3 1-5 0-6 0M8 12c3 1 5 0 6 0M20 12c-3 1-5 0-6 0" stroke="#fff" stroke-width="1.5" stroke-linecap="round"/></svg>`,
  "Other": `<svg viewBox="0 0 28 28" fill="none"><circle cx="14" cy="14" r="10" stroke="#fff" stroke-width="2"/><text x="14" y="19" text-anchor="middle" fill="#fff" font-size="14" font-weight="bold">?</text></svg>`
};
function getIcon(cls) {
  return CLASS_ICONS[cls] || CLASS_ICONS["Fetal brain"] || CLASS_ICONS["Other"];
}

/* ── ROUTING ─────────────────────── */
let trainingPollInterval = null;

function navigate(hash) {
  const page = (hash || '#home').replace('#','');
  $$('.page').forEach(p => p.classList.remove('active'));
  const el = $(`#page-${page}`);
  if (el) el.classList.add('active');
  else if ($('#page-home')) $('#page-home').classList.add('active');
  $$('.nav-links a').forEach(a => {
    a.classList.toggle('active', a.getAttribute('href') === '#' + page);
  });
  $('.nav-links')?.classList.remove('open');

  // Load data for relevant pages
  if (page === 'evaluation') loadEvaluation();
  if (page === 'dataset') loadDatasetStatus();
  if (page === 'training') startTrainingPolling();
  else stopTrainingPolling();
}
window.addEventListener('hashchange', () => navigate(location.hash));
document.addEventListener('DOMContentLoaded', () => navigate(location.hash));

/* ── MOBILE NAV ──────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  const toggle = $('.nav-toggle');
  if (toggle) toggle.addEventListener('click', () => {
    $('.nav-links')?.classList.toggle('open');
  });
});

/* ── FILE UPLOAD ─────────────────── */
let currentFile = null, currentDataURL = null, currentResult = null;

document.addEventListener('DOMContentLoaded', () => {
  const zone = $('#uploadZone');
  const input = $('#fileInput');
  const btnBrowse = $('#btnBrowse');
  const preview = $('#previewArea');
  const previewImg = $('#previewImg');
  const btnRemove = $('#btnRemove');
  const btnAnalyze = $('#btnAnalyze');

  if (!zone) return;

  btnBrowse?.addEventListener('click', () => input.click());
  zone.addEventListener('click', e => {
    if (!e.target.closest('button')) input.click();
  });
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  input?.addEventListener('change', () => {
    if (input.files.length) handleFile(input.files[0]);
  });
  btnRemove?.addEventListener('click', removeFile);
  btnAnalyze?.addEventListener('click', analyzeImage);
});

function handleFile(file) {
  const valid = ['image/jpeg','image/png','image/webp','image/bmp'];
  if (!valid.includes(file.type)) { alert('Please upload JPG, PNG, WEBP, or BMP.'); return; }
  if (file.size > 10*1024*1024) { alert('Max file size is 10 MB.'); return; }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    currentDataURL = e.target.result;
    $('#previewImg').src = currentDataURL;
    $('#previewArea').classList.add('visible');
    $('#uploadZone').style.display = 'none';
    $('#fileName').textContent = file.name;
    $('#fileSize').textContent = (file.size/1024).toFixed(1)+' KB';
    $('#btnAnalyze').disabled = false;
  };
  reader.readAsDataURL(file);
}

function removeFile() {
  currentFile = null; currentDataURL = null; currentResult = null;
  $('#previewImg').src = '';
  $('#previewArea').classList.remove('visible');
  $('#uploadZone').style.display = '';
  $('#btnAnalyze').disabled = true;
  $('#fileInput').value = '';
  $('#resultPlaceholder').style.display = '';
  $('#resultContent').classList.remove('visible');
  $('#annotatedSection').classList.remove('visible');
  $('#modelAlert').style.display = 'none';
}

/* ── ANALYZE ─────────────────────── */
async function analyzeImage() {
  if (!currentFile) return;
  const btn = $('#btnAnalyze');
  btn.classList.add('loading'); btn.disabled = true;
  $('#resultPlaceholder').style.display = 'none';
  $('#resultContent').classList.remove('visible');
  $('#modelAlert').style.display = 'none';

  try {
    const fd = new FormData();
    fd.append('image', currentFile);
    const resp = await fetch(API + '/predict', { method: 'POST', body: fd });
    const data = await resp.json();

    if (!resp.ok || data.error) {
      if (resp.status === 503) {
        $('#modelAlert').style.display = 'block';
        $('#modelAlert').textContent = data.detail || 'Model not found. Please train or upload a trained model.';
      } else {
        alert(data.error || data.detail || 'Prediction failed');
      }
      return;
    }
    currentResult = data;
    displayResults(data);
  } catch (err) {
    alert('Server not running. Start with: python app.py');
  } finally {
    btn.classList.remove('loading'); btn.disabled = false;
  }
}

function displayResults(r) {
  $('#resultContent').classList.add('visible');

  const cls = r.top_prediction;
  const conf = r.confidence;
  const isInvalid = r.is_ultrasound === false || cls === 'Not ultrasound';

  // Detection card
  $('#detectionIcon').innerHTML = getIcon(cls);
  $('#detectionClass').textContent = cls;
  $('#modelTypeLabel').textContent = 'Model: ' + (r.model_type || 'Unknown');

  const badge = $('#statusBadge');
  const statusText = $('#statusText');
  if (isInvalid) {
    badge.className = 'status-badge abnormal'; statusText.textContent = 'INVALID INPUT';
  } else if ((r.status||'').includes('Normal')) {
    badge.className = 'status-badge normal'; statusText.textContent = 'NORMAL';
  } else if ((r.status||'').includes('Uncertain')) {
    badge.className = 'status-badge uncertain'; statusText.textContent = 'UNCERTAIN';
  } else {
    badge.className = 'status-badge abnormal'; statusText.textContent = 'REVIEW NEEDED';
  }

  // Confidence
  const fill = $('#confBarFill');
  fill.className = 'conf-bar-fill ' + (conf > .8 ? 'high' : conf > .6 ? 'med' : 'low');
  requestAnimationFrame(() => fill.style.width = (conf*100)+'%');
  $('#confValue').textContent = (conf*100).toFixed(1) + '%';

  // Top-K
  const topK = $('#topKContainer'); topK.innerHTML = '';
  (r.top_k_predictions || []).forEach((p,i) => {
    const row = document.createElement('div'); row.className = 'topk-row';
    row.innerHTML = `<span class="topk-rank">#${i+1}</span><span class="topk-name">${p.class}</span><div class="topk-bar"><div class="topk-bar-fill" id="tk${i}"></div></div><span class="topk-conf">${(p.confidence*100).toFixed(1)}%</span>`;
    topK.appendChild(row);
    setTimeout(() => { const b = $(`#tk${i}`); if(b) b.style.width = (p.confidence*100)+'%'; }, 200+i*150);
  });
  if (!r.top_k_predictions?.length) topK.innerHTML = '<div class="topk-row"><span class="topk-name">No predictions</span></div>';

  // Clinical note
  $('#clinicalNote').textContent = r.clinical_note || '';

  // Annotated
  showAnnotated(r);
}

function showAnnotated(r) {
  const section = $('#annotatedSection');
  section.classList.add('visible');
  const canvas = $('#annotatedCanvas');
  const ctx = canvas.getContext('2d');
  const img = new Image();
  img.onload = () => {
    canvas.width = 448; canvas.height = 448;
    ctx.drawImage(img, 0, 0, 448, 448);
    const isNormal = (r.status||'').includes('Normal');
    const label = `${r.model_type||'Model'}: ${r.top_prediction} (${(r.confidence*100).toFixed(1)}%)`;
    ctx.font = 'bold 15px "DM Sans",sans-serif';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = isNormal ? '#0D6E6E' : '#E05C5C';
    ctx.fillRect(12, 12, tw+20, 28);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, 22, 32);
  };
  img.src = currentDataURL;
}

/* ── EVALUATION ──────────────────── */
async function loadEvaluation() {
  try {
    const resp = await fetch(API + '/evaluation/status');
    const data = await resp.json();

    const msg = $('#evalMessage');
    const grid = $('#evalMetricsGrid');
    const clsGrid = $('#clsMetricsGrid');

    if (data.available && (Object.keys(data.metrics||{}).length || Object.keys(data.classification_metrics||{}).length)) {
      msg.textContent = 'Real model evaluation metrics loaded from YOLO training artifacts.';
      msg.className = 'alert alert-info';
    } else {
      msg.textContent = data.message || 'No evaluation results available. Train a model first, then run YOLO validation.';
      msg.className = 'alert alert-warning';
    }

    renderMetrics(grid, data.metrics);
    renderMetrics(clsGrid, data.classification_metrics);

    // Confusion matrices
    const cmWrap = $('#confusionWrap');
    if (data.confusion_matrix_url) {
      $('#confusionImg').src = data.confusion_matrix_url + '?t=' + Date.now();
      cmWrap.classList.add('visible');
    } else cmWrap.classList.remove('visible');

    const clsCmWrap = $('#clsConfusionWrap');
    if (data.classification_confusion_matrix_url) {
      $('#clsConfusionImg').src = data.classification_confusion_matrix_url + '?t=' + Date.now();
      clsCmWrap.classList.add('visible');
    } else clsCmWrap.classList.remove('visible');

    // Files
    const files = data.files || {};
    const list = $('#evalFilesList');
    const entries = [
      ['Detection model', files.best_pt],
      ['Detection results.csv', files.results_csv],
      ['Detection confusion matrix', files.confusion_matrix],
      ['Classification model', files.classification_best_pt],
      ['Classification results.csv', files.classification_results_csv],
      ['Classification confusion matrix', files.classification_confusion_matrix],
      ['data.yaml', files.data_yaml],
    ];
    list.innerHTML = entries.map(([k,v]) =>
      `<li><strong>${k}:</strong> ${v ? '✅ Found' : '❌ Not found'}</li>`
    ).join('');

    if (data.command) $('#evalCommand').textContent = data.command;
    if (data.classification_command) $('#clsEvalCommand').textContent = data.classification_command;
  } catch {
    $('#evalMessage').textContent = 'Could not connect to server. Start with: python app.py';
    $('#evalMessage').className = 'alert alert-danger';
  }
}

function renderMetrics(grid, metrics) {
  grid.innerHTML = '';
  const m = metrics || {};
  const keys = Object.keys(m);
  if (!keys.length) {
    grid.innerHTML = '<div class="empty-state">No real metrics available. Train a model and run validation first.</div>';
    return;
  }
  const labels = {accuracy:'Accuracy',precision:'Precision',recall:'Recall',f1_score:'F1 Score',map50:'mAP@0.5',map50_95:'mAP@0.5:0.95',top5_accuracy:'Top-5 Accuracy'};
  keys.forEach(k => {
    if (m[k] == null) return;
    const card = document.createElement('div'); card.className = 'stat-card';
    card.innerHTML = `<div class="stat-value">${Number(m[k]).toFixed(1)}%</div><div class="stat-label">${labels[k]||k}</div>`;
    grid.appendChild(card);
  });
}

/* ── DATASET STATUS ──────────────── */
async function loadDatasetStatus() {
  const container = $('#datasetContainer');
  try {
    const resp = await fetch(API + '/dataset/status');
    const data = await resp.json();

    let html = '';
    if (data.message) {
      html += `<div class="alert alert-warning">${data.message}</div>`;
    }

    // Detection
    const det = data.detection_dataset || {};
    html += `<div class="dataset-card"><h4>Detection Dataset</h4>`;
    html += `<span class="badge ${det.available ? 'badge-ok' : 'badge-missing'}">${det.available ? 'Available' : 'Not Found'}</span>`;
    if (det.available && det.splits) {
      html += '<ul style="margin-top:8px;padding-left:20px">';
      Object.entries(det.splits).forEach(([k,v]) => html += `<li>${k}: ${v} images</li>`);
      html += '</ul>';
    }
    if (det.data_yaml) html += `<p style="margin-top:6px;font-size:.85rem;color:var(--muted)">data.yaml: ${det.data_yaml}</p>`;
    html += '</div>';

    // Classification
    const cls = data.classification_dataset || {};
    html += `<div class="dataset-card"><h4>Classification Dataset</h4>`;
    html += `<span class="badge ${cls.available ? 'badge-ok' : 'badge-missing'}">${cls.available ? 'Available' : 'Not Found'}</span>`;
    if (cls.available && cls.splits) {
      html += '<ul style="margin-top:8px;padding-left:20px">';
      Object.entries(cls.splits).forEach(([k,v]) => html += `<li>${k}: ${v.images} images, ${v.classes?.length||0} classes</li>`);
      html += '</ul>';
    }
    html += '</div>';

    // Instructions
    html += `<div class="card" style="margin-top:20px"><div class="card-header">How to Add a Dataset</div><div class="card-body">
      <p style="margin-bottom:12px"><strong>Detection dataset</strong> (bounding boxes):</p>
      <code class="command-box">dataset/images/train/*.jpg  +  dataset/labels/train/*.txt  +  dataset/data.yaml</code>
      <p style="margin:16px 0 12px"><strong>Classification dataset</strong> (folder-labeled):</p>
      <code class="command-box">dataset_cls/train/fetal_brain/*.jpg  dataset_cls/val/fetal_brain/*.jpg</code>
      <p style="margin-top:16px"><strong>Validate:</strong></p>
      <code class="command-box">python check_dataset.py --dataset dataset</code>
    </div></div>`;

    container.innerHTML = html;
  } catch {
    container.innerHTML = '<div class="alert alert-danger">Could not connect to server. Start with: python app.py</div>';
  }
}

/* ═══════════════════════════════════════════════════
   TRAINING DASHBOARD (SPA integration)
   ═══════════════════════════════════════════════════ */

const trainFmt = (v, s='') => v === null || v === undefined ? '--' : `${v}${s}`;

function startTrainingPolling() {
  refreshTrainingStatus();
  if (trainingPollInterval) clearInterval(trainingPollInterval);
  trainingPollInterval = setInterval(refreshTrainingStatus, 5000);
}

function stopTrainingPolling() {
  if (trainingPollInterval) {
    clearInterval(trainingPollInterval);
    trainingPollInterval = null;
  }
}

async function trainPostAction(url) {
  try {
    const r = await fetch(url, { method: 'POST' });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) alert(d.message || 'Action failed');
    await refreshTrainingStatus();
  } catch {
    alert('Could not connect to server.');
  }
}

function drawTrainingChart(canvasId, history, key, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(280, Math.floor(rect.width * dpr));
  canvas.height = Math.floor(150 * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = canvas.width / dpr, h = canvas.height / dpr;
  ctx.clearRect(0, 0, w, h);

  // Grid lines
  ctx.strokeStyle = 'rgba(13,110,110,0.12)';
  ctx.lineWidth = 1;
  for (let i = 0; i < 4; i++) {
    const y = 14 + i * ((h - 28) / 3);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  const pts = history.filter(r => r[key] != null);
  if (pts.length < 2) {
    ctx.fillStyle = '#91b8b8';
    ctx.font = '12px "DM Sans",sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for data...', w / 2, h / 2);
    return;
  }

  const vals = pts.map(r => Number(r[key]));
  const mn = Math.min(...vals), mx = Math.max(...vals), sp = mx - mn || 1;

  // Draw line
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  pts.forEach((r, i) => {
    const x = (i / (pts.length - 1)) * (w - 16) + 8;
    const y = h - 14 - ((Number(r[key]) - mn) / sp) * (h - 28);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Draw dots at last point
  const lastX = (pts.length - 1) / (pts.length - 1) * (w - 16) + 8;
  const lastY = h - 14 - ((vals[vals.length - 1] - mn) / sp) * (h - 28);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
  ctx.fill();

  // Value label at last point
  ctx.fillStyle = '#1A2B2B';
  ctx.font = 'bold 11px "DM Sans",sans-serif';
  ctx.textAlign = 'right';
  const lastVal = vals[vals.length - 1];
  ctx.fillText(key === 'loss' ? lastVal.toFixed(3) : lastVal.toFixed(1) + '%', lastX - 6, lastY - 8);
}

function setTrainingArtifact(id, url) {
  const el = document.getElementById(id);
  if (!el) return false;
  if (!url) { el.classList.remove('visible'); return false; }
  el.querySelector('img').src = `${url}?t=${Date.now()}`;
  el.classList.add('visible');
  return true;
}

async function refreshTrainingStatus() {
  try {
    const r = await fetch(API + '/training/status');
    const d = await r.json();

    // Status dot + message
    const dot = $('#trainDot');
    if (dot) {
      dot.className = 'training-dot ' + (d.running ? 'running' : d.success ? 'done' : d.completed ? 'error' : '');
    }
    const msg = $('#trainStatusMsg');
    if (msg) msg.textContent = d.message || 'Training idle.';

    // Progress
    const fill = $('#trainProgressFill');
    if (fill) fill.style.width = `${d.progress || 0}%`;

    // Epoch + GPU
    const epoch = $('#trainEpochText');
    if (epoch) epoch.textContent = `Epoch ${d.epoch || 0} / ${d.epochs || 50}`;

    const gpu = d.gpu || {};
    const gpuEl = $('#trainGpuText');
    if (gpuEl) gpuEl.textContent = gpu.cuda ? `GPU: ${gpu.name} (${gpu.vram_gb} GB)` : 'Device: CPU';

    // Completion
    const complete = $('#trainCompleteText');
    if (complete) complete.innerHTML = d.success ? '<span class="training-success">✅ Training completed successfully.</span>' : '';

    // Metric values
    const lv = $('#trainLossVal'); if (lv) lv.textContent = trainFmt(d.loss);
    const pv = $('#trainPrecisionVal'); if (pv) pv.textContent = trainFmt(d.precision, '%');
    const rv = $('#trainRecallVal'); if (rv) rv.textContent = trainFmt(d.recall, '%');
    const mv = $('#trainMapVal'); if (mv) mv.textContent = trainFmt(d.map50, '%');

    // Output paths
    const bp = $('#trainBestPath'); if (bp) bp.textContent = d.best_model || 'runs/detect/train/weights/best.pt';
    const lp = $('#trainLastPath'); if (lp) lp.textContent = d.last_model || 'runs/detect/train/weights/last.pt';

    // Logs
    const logs = $('#trainLogs');
    if (logs) logs.textContent = (d.logs || []).join('\n') || 'Waiting for logs...';

    // Buttons
    const startBtn = $('#btnTrainStart'); if (startBtn) startBtn.disabled = d.running;
    const stopBtn = $('#btnTrainStop'); if (stopBtn) stopBtn.disabled = !d.running;

    // Charts
    const h = d.history || [];
    drawTrainingChart('trainLossChart', h, 'loss', '#0D6E6E');
    drawTrainingChart('trainPrecisionChart', h, 'precision', '#00C9A7');
    drawTrainingChart('trainRecallChart', h, 'recall', '#E0A030');
    drawTrainingChart('trainF1Chart', h, 'f1', '#5B9BD5');

    // Artifacts
    const a = d.artifacts || {};
    const shown = [
      setTrainingArtifact('trainArtResults', a.results),
      setTrainingArtifact('trainArtPR', a.pr),
      setTrainingArtifact('trainArtF1', a.f1),
      setTrainingArtifact('trainArtConfusion', a.confusion),
    ].some(Boolean);
    const hint = $('#trainArtHint');
    if (hint) hint.style.display = shown ? 'none' : 'block';

  } catch {
    const msg = $('#trainStatusMsg');
    if (msg) msg.textContent = 'Could not connect to server. Start with: python app.py';
  }
}

/* ── TRAINING BUTTON HANDLERS ────── */
document.addEventListener('DOMContentLoaded', () => {
  $('#btnTrainStart')?.addEventListener('click', () => trainPostAction(API + '/training/start'));
  $('#btnTrainStop')?.addEventListener('click', () => trainPostAction(API + '/training/stop'));
  $('#btnTrainOpen')?.addEventListener('click', () => trainPostAction(API + '/training/open-results'));
});

/* ── DOWNLOAD ANNOTATED ──────────── */
document.addEventListener('DOMContentLoaded', () => {
  $('#btnDownload')?.addEventListener('click', () => {
    const canvas = $('#annotatedCanvas');
    canvas.toBlob(blob => {
      if (!blob) return;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'annotated_result.png';
      a.click();
    }, 'image/png');
  });
});
