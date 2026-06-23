// Standalone speaker-embedding extractor wrapper over the custom sherpa-onnx WASM
// build (the one that exports our 9 SpeakerEmbeddingExtractor C-API functions).
//
// Reuses initSherpaOnnxSpeakerEmbeddingExtractorConfig() from the shipped
// sherpa-onnx-speaker-diarization.js (it already marshals this exact config
// struct: {model ptr, numThreads i32, debug i32, provider ptr} = 16 bytes).
//
// This file is the in-browser feasibility test: load TitaNet (preloaded into the
// WASM FS at ./nemo_en_titanet_small.onnx), embed 3 synthetic "hey ozwell" clips,
// and confirm the genuine/impostor cosine separation matches the Python spike.

class SpeakerEmbeddingExtractor {
  constructor(Module, modelPath) {
    this.Module = Module;
    const c = initSherpaOnnxSpeakerEmbeddingExtractorConfig(
        {model: modelPath, numThreads: 1, debug: 0, provider: 'cpu'}, Module);
    this.handle = Module._SherpaOnnxCreateSpeakerEmbeddingExtractor(c.ptr);
    Module._free(c.buffer);
    Module._free(c.ptr);
    if (!this.handle) throw new Error('CreateSpeakerEmbeddingExtractor returned null — model failed to load');
    this.dim = Module._SherpaOnnxSpeakerEmbeddingExtractorDim(this.handle);
  }

  // samples: Float32Array in [-1,1]; sampleRate: the TRUE rate of `samples`
  // (sherpa resamples internally to the model's 16k). Returns a Float32Array embedding.
  compute(samples, sampleRate) {
    const M = this.Module;
    const stream = M._SherpaOnnxSpeakerEmbeddingExtractorCreateStream(this.handle);
    const ptr = M._malloc(samples.length * 4);
    M.HEAPF32.set(samples, ptr / 4);
    // accept/input-finished are the GENERIC online-stream calls (stream only, no extractor arg)
    M._SherpaOnnxOnlineStreamAcceptWaveform(stream, sampleRate, ptr, samples.length);
    M._free(ptr);
    M._SherpaOnnxOnlineStreamInputFinished(stream);
    const ready = M._SherpaOnnxSpeakerEmbeddingExtractorIsReady(this.handle, stream);
    const ep = M._SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(this.handle, stream);
    const emb = M.HEAPF32.subarray(ep / 4, ep / 4 + this.dim).slice(); // copy out of heap
    // NOTE: ComputeEmbedding returns a malloc'd float* that ought to be freed via
    // SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding — we didn't export it, so
    // there's a tiny per-call leak. Fine for this test; export it before shipping.
    M._SherpaOnnxDestroyOnlineStream(stream);
    return {emb, ready};
  }
}

// --- minimal WAV reader (PCM16, any rate, mono / first channel) ---
function parseWav(arrayBuffer) {
  const dv = new DataView(arrayBuffer);
  let off = 12; // past 'RIFF'<size>'WAVE'
  let fmt = null, dataOff = -1, dataLen = 0;
  while (off + 8 <= dv.byteLength) {
    const id = String.fromCharCode(dv.getUint8(off), dv.getUint8(off + 1), dv.getUint8(off + 2), dv.getUint8(off + 3));
    const sz = dv.getUint32(off + 4, true);
    if (id === 'fmt ') {
      fmt = {channels: dv.getUint16(off + 10, true), sampleRate: dv.getUint32(off + 12, true), bits: dv.getUint16(off + 22, true)};
    } else if (id === 'data') { dataOff = off + 8; dataLen = sz; }
    off += 8 + sz + (sz & 1);
  }
  if (!fmt || dataOff < 0) throw new Error('not a PCM wav');
  const bps = fmt.bits / 8, ch = fmt.channels;
  const n = Math.floor(dataLen / (bps * ch));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = dv.getInt16(dataOff + i * bps * ch, true) / 32768;
  return {samples: out, sampleRate: fmt.sampleRate};
}

function cosine(a, b) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / (Math.sqrt(na) * Math.sqrt(nb));
}

function log(msg) {
  console.log(msg);
  const el = document.getElementById('out');
  if (el) el.textContent += msg + '\n';
}

// --- bootstrap: emscripten glue (loaded after this file) calls onRuntimeInitialized ---
var Module = {};
Module.onRuntimeInitialized = async function () {
  try {
    log('WASM runtime ready. Creating extractor (TitaNet from WASM FS)…');
    const ex = new SpeakerEmbeddingExtractor(Module, './nemo_en_titanet_small.onnx');
    log('embedding dim = ' + ex.dim + '\n');

    const names = ['liam_a', 'liam_b', 'matilda'];
    const emb = {};
    for (const n of names) {
      const buf = await (await fetch('./test-clips/' + n + '.wav')).arrayBuffer();
      const { samples, sampleRate } = parseWav(buf);
      emb[n] = ex.compute(samples, sampleRate).emb;
      log(`${n}: ${samples.length} samples @ ${sampleRate}Hz  ->  embedding[${emb[n].length}]`);
    }

    const gen = cosine(emb.liam_a, emb.liam_b);
    const imp = cosine(emb.liam_a, emb.matilda);
    log('');
    log(`GENUINE   liam_a vs liam_b  : ${gen.toFixed(3)}   (Python ground truth: 0.659)`);
    log(`IMPOSTOR  liam_a vs matilda : ${imp.toFixed(3)}   (Python ground truth: 0.302)`);
    log('');
    const ok = gen > 0.55 && imp < 0.45 && (gen - imp) > 0.2;
    log(ok ? '✅ SEPARATION CONFIRMED IN BROWSER — speaker verification works on-device.'
           : '⚠️ Unexpected scores — numbers should be ~0.66 genuine / ~0.30 impostor.');
  } catch (e) {
    log('❌ ERROR: ' + (e && e.stack ? e.stack : e));
  }
};
