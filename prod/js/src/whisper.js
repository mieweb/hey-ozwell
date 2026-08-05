// On-device speech-to-text via Transformers.js + Whisper. The model runs entirely in the
// browser (WebGPU, or single-threaded WASM fallback) — audio never leaves the page (PHI).
// Exposes window.Whisper.{ ready, isLoaded, transcribe, modelName }.
//
// Loaded as a module from index.html. First load downloads the model (cached afterward).
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

// Single-threaded WASM avoids SharedArrayBuffer (the demo page has no COOP/COEP headers).
env.backends.onnx.wasm.numThreads = 1;
env.allowLocalModels = false;

// Accuracy-first DEFAULT on WebGPU: large-v3-turbo (near-large-v3 accuracy, much faster;
// ~800MB, needs a GPU). It's multilingual, so transcribe() pins it to English.
const TURBO_MODEL = "onnx-community/whisper-large-v3-turbo"; // strong — default
const SMALL_MODEL = "Xenova/whisper-small.en";              // light/fast — for quick dev reloads

// Dev toggle: append ?model=small to the URL to use the light model (fast warmup); default
// is the strong model. small.en is also the automatic fallback if the big model won't load.
const wantSmall = /^(small|fast)$/i.test(new URLSearchParams(location.search).get("model") || "");

let transcriber = null, isMultilingual = false, readyResolve, readyReject;
const readyPromise = new Promise((res, rej) => { readyResolve = res; readyReject = rej; });

async function loadSmall(reason) {
  try { transcriber = await pipeline("automatic-speech-recognition", SMALL_MODEL, { device: "webgpu" }); }
  catch (e) { transcriber = await pipeline("automatic-speech-recognition", SMALL_MODEL); } // WASM
  isMultilingual = false;
  console.log(`[Whisper] loaded ${SMALL_MODEL}${reason ? " (" + reason + ")" : ""}`);
}

async function init() {
  if (wantSmall) {
    await loadSmall("?model=small");
  } else {
    try {
      // fp16 encoder keeps accuracy; q4 decoder keeps it fast on WebGPU.
      transcriber = await pipeline("automatic-speech-recognition", TURBO_MODEL, {
        device: "webgpu",
        dtype: { encoder_model: "fp16", decoder_model_merged: "q4" },
      });
      isMultilingual = true;
      console.log("[Whisper] loaded on WebGPU:", TURBO_MODEL, "— append ?model=small for fast dev reloads");
    } catch (e) {
      console.warn("[Whisper] large model failed to load — falling back to small.en.", e);
      await loadSmall("fallback");
    }
  }
  readyResolve();
}

function resampleTo16k(x, sr) {
  if (sr === 16000) return x;
  const n = Math.round(x.length * 16000 / sr);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i * sr / 16000, i0 = Math.floor(t), i1 = Math.min(i0 + 1, x.length - 1);
    out[i] = x[i0] + (x[i1] - x[i0]) * (t - i0);
  }
  return out;
}

window.Whisper = {
  ready: () => readyPromise,
  isLoaded: () => transcriber !== null,
  modelName: () => (isMultilingual ? TURBO_MODEL : SMALL_MODEL),
  // samples: Float32Array; returns the transcribed text (on-device).
  async transcribe(samples, sampleRate) {
    await readyPromise;
    const audio = resampleTo16k(samples, sampleRate); // whisper wants 16kHz mono
    // chunk_length_s lets it handle multi-minute sessions (>30s). Pin language for the
    // multilingual model so it doesn't auto-detect or translate.
    const opts = { chunk_length_s: 30, stride_length_s: 5 };
    if (isMultilingual) { opts.language = "english"; opts.task = "transcribe"; }
    const out = await transcriber(audio, opts);
    return (out && out.text ? out.text : "").trim();
  },
};

init();
