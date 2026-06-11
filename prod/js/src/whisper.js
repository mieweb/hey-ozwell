// On-device speech-to-text via Transformers.js + Whisper. The model runs entirely in the
// browser (WebGPU, or single-threaded WASM fallback) — audio never leaves the page (PHI).
// Exposes window.Whisper.{ ready, isLoaded, transcribe, modelName }.
//
// Loaded as a module from index.html. First load downloads the model (~240MB for small.en),
// cached by the browser afterward.
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

// Single-threaded WASM avoids SharedArrayBuffer (the demo page has no COOP/COEP headers).
env.backends.onnx.wasm.numThreads = 1;
env.allowLocalModels = false;

// Accuracy-leaning English model. Swap to "onnx-community/whisper-large-v3-turbo" (bigger,
// WebGPU strongly recommended) for max accuracy, or "Xenova/whisper-base.en" for speed.
const MODEL = "Xenova/whisper-small.en";

let transcriber = null, readyResolve, readyReject;
const readyPromise = new Promise((res, rej) => { readyResolve = res; readyReject = rej; });

async function init() {
  try {
    try {
      transcriber = await pipeline("automatic-speech-recognition", MODEL, { device: "webgpu" });
      console.log("[Whisper] loaded on WebGPU:", MODEL);
    } catch (e) {
      console.warn("[Whisper] WebGPU unavailable — falling back to WASM (slower).", e);
      transcriber = await pipeline("automatic-speech-recognition", MODEL);
      console.log("[Whisper] loaded on WASM:", MODEL);
    }
    readyResolve();
  } catch (e) {
    console.error("[Whisper] failed to load:", e);
    readyReject(e);
  }
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
  modelName: () => MODEL,
  // samples: Float32Array; returns the transcribed text (on-device).
  async transcribe(samples, sampleRate) {
    await readyPromise;
    const audio = resampleTo16k(samples, sampleRate); // whisper wants 16kHz mono
    const out = await transcriber(audio);
    return (out && out.text ? out.text : "").trim();
  },
};

init();
