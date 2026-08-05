// Real-voice self-test for on-device speaker verification.
// Enroll YOUR voice live, then test yourself (should PASS) vs someone else (should FAIL).
// No pre-known target numbers — the score is computed live from the mic each time.
// Uses the same custom sherpa-onnx WASM build as test.html (served from this dir on :3010).

// --- speaker-embedding extractor over the WASM C-API (names confirmed from the build) ---
class SpeakerEmbeddingExtractor {
  constructor(Module, modelPath) {
    this.Module = Module;
    const c = initSherpaOnnxSpeakerEmbeddingExtractorConfig(
        {model: modelPath, numThreads: 1, debug: 0, provider: 'cpu'}, Module);
    this.handle = Module._SherpaOnnxCreateSpeakerEmbeddingExtractor(c.ptr);
    Module._free(c.buffer); Module._free(c.ptr);
    if (!this.handle) throw new Error('extractor creation failed');
    this.dim = Module._SherpaOnnxSpeakerEmbeddingExtractorDim(this.handle);
  }
  embed(samples, sampleRate) {
    const M = this.Module;
    const stream = M._SherpaOnnxSpeakerEmbeddingExtractorCreateStream(this.handle);
    const ptr = M._malloc(samples.length * 4);
    M.HEAPF32.set(samples, ptr / 4);
    M._SherpaOnnxOnlineStreamAcceptWaveform(stream, sampleRate, ptr, samples.length);
    M._free(ptr);
    M._SherpaOnnxOnlineStreamInputFinished(stream);
    const ep = M._SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(this.handle, stream);
    const v = M.HEAPF32.subarray(ep / 4, ep / 4 + this.dim).slice();
    if (M._SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding) M._SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(ep);
    M._SherpaOnnxDestroyOnlineStream(stream);
    // L2-normalize so cosine == dot product
    let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; s = Math.sqrt(s);
    if (s > 0) for (let i = 0; i < v.length; i++) v[i] /= s;
    return v;
  }
}
function cosine(a, b) { let d = 0; for (let i = 0; i < a.length; i++) d += a[i] * b[i]; return d; }

// --- mic recording (ScriptProcessor: deprecated but universal; fine for a test page) ---
let audioCtx, micStream, micSource;
async function ensureMic() {
  if (micStream) return;
  micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  micSource = audioCtx.createMediaStreamSource(micStream);
}
async function recordSeconds(secs) {
  await ensureMic();
  if (audioCtx.state === 'suspended') await audioCtx.resume();
  const proc = audioCtx.createScriptProcessor(4096, 1, 1);
  const sink = audioCtx.createGain(); sink.gain.value = 0; // mute -> no feedback
  const chunks = [];
  proc.onaudioprocess = (e) => chunks.push(Float32Array.from(e.inputBuffer.getChannelData(0)));
  micSource.connect(proc); proc.connect(sink); sink.connect(audioCtx.destination);
  await new Promise((r) => setTimeout(r, secs * 1000));
  micSource.disconnect(proc); proc.disconnect(); sink.disconnect();
  let len = 0; for (const c of chunks) len += c.length;
  const out = new Float32Array(len); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; }
  return { samples: out, sampleRate: audioCtx.sampleRate };
}

// --- UI / state ---
const $ = (id) => document.getElementById(id);
function log(msg) { const el = $('log'); el.textContent += msg + '\n'; el.scrollTop = el.scrollHeight; }
function setStatus(t) { $('status').textContent = t; }

let extractor = null;
let enrollEmbs = [];     // list of normalized embeddings
let centroid = null;     // normalized mean

function recomputeCentroid() {
  if (!enrollEmbs.length) { centroid = null; return; }
  const c = new Float32Array(extractor.dim);
  for (const e of enrollEmbs) for (let i = 0; i < c.length; i++) c[i] += e[i];
  let s = 0; for (let i = 0; i < c.length; i++) { c[i] /= enrollEmbs.length; s += c[i] * c[i]; }
  s = Math.sqrt(s); if (s > 0) for (let i = 0; i < c.length; i++) c[i] /= s;
  centroid = c;
}

async function countdownRecord(label, secs) {
  for (let i = 3; i >= 1; i--) { setStatus(`${label} in ${i}…`); await new Promise(r => setTimeout(r, 600)); }
  setStatus(`🔴 ${label} — say "hey ozwell" now`);
  const clip = await recordSeconds(secs);
  setStatus('processing…');
  return clip;
}

async function onEnroll() {
  try {
    const clip = await countdownRecord('Enrolling', 1.6);
    const emb = extractor.embed(clip.samples, clip.sampleRate);
    enrollEmbs.push(emb); recomputeCentroid();
    log(`✓ enrollment sample ${enrollEmbs.length} captured (${clip.samples.length} samples @ ${clip.sampleRate}Hz)`);
    setStatus(`Enrolled ${enrollEmbs.length} sample(s). Add more, or test a voice.`);
    $('verifyBtn').disabled = enrollEmbs.length === 0;
  } catch (e) { log('❌ ' + e); setStatus('error — see log'); }
}

async function onVerify() {
  try {
    if (!centroid) { setStatus('enroll at least one sample first'); return; }
    const clip = await countdownRecord('Testing', 1.6);
    const emb = extractor.embed(clip.samples, clip.sampleRate);
    const score = cosine(emb, centroid);
    const thr = parseFloat($('thr').value);
    const pass = score >= thr;
    log(`${pass ? '✅ PASS' : '🔒 BLOCKED'}  score = ${score.toFixed(3)}  (threshold ${thr.toFixed(2)})`);
    setStatus(`${pass ? '✅ MATCH — would wake Ozwell' : '🔒 NOT the enrolled voice — blocked'}  (score ${score.toFixed(3)})`);
  } catch (e) { log('❌ ' + e); setStatus('error — see log'); }
}

function onClear() {
  enrollEmbs = []; centroid = null; $('verifyBtn').disabled = true;
  log('— enrollment cleared —'); setStatus('Cleared. Enroll a voice to begin.');
}

// --- bootstrap WASM (glue loaded after this script fires onRuntimeInitialized) ---
var Module = {};
Module.onRuntimeInitialized = function () {
  try {
    extractor = new SpeakerEmbeddingExtractor(Module, './nemo_en_titanet_small.onnx');
    setStatus('Ready. Click “Enroll my voice” and say "hey ozwell" a few times.');
    log('WASM ready — embedding dim ' + extractor.dim);
    $('enrollBtn').disabled = false;
    $('enrollBtn').onclick = onEnroll;
    $('verifyBtn').onclick = onVerify;
    $('clearBtn').onclick = onClear;
  } catch (e) { log('❌ init: ' + e); setStatus('init failed — see log'); }
};
