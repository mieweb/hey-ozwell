/** @module models/asr-whisper */

/**
 * Whisper ASR engine for the stage-2 verifier (fully client-side via sherpa-onnx WASM).
 *
 * Wraps the sherpa-onnx offline (whisper-tiny.en) recognizer and exposes the single method the
 * Verifier needs:  async transcribe(Float32Array @16kHz) -> string.
 *
 * Runs ONLY when stage-1 fires (rare), so the ~100MB model's speed is a non-issue; it's loaded once.
 *
 * LOADING: the sherpa WASM build produces three artifacts (place them under prod/js/asr/):
 *   - sherpa-onnx-wasm-main-asr.js   (emscripten glue; defines the Module factory)
 *   - sherpa-onnx-wasm-main-asr.wasm
 *   - sherpa-onnx-wasm-main-asr.data (preloaded whisper-tiny.en model)
 *   - sherpa-onnx-asr.js             (provides the OfflineRecognizer class on the Module)
 * This module loads the glue, waits for runtime init, then builds an OfflineRecognizer with whisper
 * config pointing at the in-virtual-FS asset names we bundled.
 */
export class WhisperASR {
    /**
     * @param {Object} [options]
     * @param {string} [options.basePath="./asr"] - Where the sherpa WASM artifacts are served from.
     */
    constructor(options = {}) {
        this.basePath = options.basePath || "./asr";
        this.recognizer = null;
        this.Module = null;
        this.ready = false;
    }

    /**
     * Load the WASM module + build the whisper recognizer. Call once before use.
     * @returns {Promise<void>}
     */
    async init() {
        if (this.ready) return;
        if (this._initPromise) return this._initPromise; // a fire during load must not start a 2nd load
        this._initPromise = this._doInit();
        return this._initPromise;
    }

    /** @private */
    async _doInit() {
        // Load the sherpa helper (defines OfflineRecognizer etc.) and the emscripten glue.
        // These are classic (non-ESM) scripts that attach to a Module; we load them dynamically.
        const Module = await this._loadModule();
        this.Module = Module;
        // Build the offline whisper recognizer against the bundled (preloaded) model files.
        const config = {
            featConfig: { sampleRate: 16000, featureDim: 80 },
            modelConfig: {
                whisper: {
                    encoder: "tiny.en-encoder.int8.onnx",
                    decoder: "tiny.en-decoder.int8.onnx",
                    language: "en",
                    task: "transcribe",
                    tailPaddings: -1,
                },
                tokens: "tiny.en-tokens.txt",
                numThreads: 1,
                provider: "cpu",
                debug: 0,
            },
            decodingMethod: "greedy_search",
        };
        // OfflineRecognizer is provided by sherpa-onnx-asr.js (loaded alongside the glue).
        this.recognizer = new Module.OfflineRecognizer
            ? new Module.OfflineRecognizer(config, Module)
            : new globalThis.OfflineRecognizer(config, Module);
        this.ready = true;
    }

    /**
     * Transcribe ~1-2s of 16kHz mono audio.
     * @param {Float32Array} samples
     * @returns {Promise<string>}
     */
    async transcribe(samples) {
        if (!this.ready) await this.init();
        const stream = this.recognizer.createStream();
        stream.acceptWaveform(16000, samples);
        this.recognizer.decode(stream);
        const result = this.recognizer.getResult(stream);
        stream.free();
        return (result && result.text) ? result.text : "";
    }

    /**
     * Load the emscripten Module + sherpa helper. Resolves once the WASM runtime is initialized.
     * @private
     */
    _loadModule() {
        return new Promise((resolve, reject) => {
            const g = globalThis;
            // sherpa's glue calls Module.onRuntimeInitialized when ready.
            g.Module = g.Module || {};
            g.Module.locateFile = (path) => `${this.basePath}/${path}`;
            g.Module.onRuntimeInitialized = () => resolve(g.Module);
            // Load helper first (defines OfflineRecognizer), then the glue (boots the runtime).
            this._loadScript(`${this.basePath}/sherpa-onnx-asr.js`)
                .then(() => this._loadScript(`${this.basePath}/sherpa-onnx-wasm-main-asr.js`))
                .catch(reject);
        });
    }

    /** @private */
    _loadScript(src) {
        return new Promise((resolve, reject) => {
            const s = document.createElement("script");
            s.src = src;
            s.onload = resolve;
            s.onerror = () => reject(new Error(`failed to load ${src}`));
            document.head.appendChild(s);
        });
    }
}
