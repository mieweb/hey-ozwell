/** @module models/acoustic-verifier */
import { ONNX } from "../onnx.js";

/**
 * Stage-2 ACOUSTIC verifier.
 *
 * Stage-1 (the tiny wake-word model) fires loosely (high recall). This re-scores the SAME
 * speech-embedding window pass-1 just scored with a small, independently-trained classifier and only
 * confirms if it agrees. The classifier was trained with hard-negatives = real false-fires MINED at
 * scale from People's Speech + VoxPopuli + AMI (noise-augmented), positives = the phrase. It judges
 * the SOUND, not a transcript — unlike the ASR verifier ("ozwell" is a made-up word → Whisper
 * hallucinates on short clips). Offline it rejects ~96% of false-fires on an independent corpus (AMI)
 * while keeping ~85% of real wakes. See model/eval/verifier_final.py.
 *
 * WHY this is cheap: it runs ONLY on a stage-1 fire, and it reuses the [16,96] embedding pass-1
 * already computed — NO extra audio model, NO re-running mel/embedding. The only cost is one tiny
 * MLP forward pass (input 1536 -> 256 -> 64 -> 2) per fire.
 *
 * Interface matches the ASR Verifier so it drops into the same `options.verifier` slot:
 *   async verify(audioFloat32, wakeWordName, embeddingBuffer) -> bool
 * It uses `embeddingBuffer` (the [frames, dim] tensor pass-1 scored), NOT the raw audio.
 */
export class AcousticVerifier {
    /**
     * @param {Object} models - Map of wakeWordName -> { modelPath, threshold }.
     *   `threshold` = reject the fire if P(wake) < threshold. Lower = keep more real wakes (and let
     *   more false-fires through). Tuned offline per phrase (see eval/verifier_final.py sweep).
     *   Example: { "ozwell-i'm-done": { modelPath: "../models/ozwell-i'm-done-verifier.onnx", threshold: 0.3 } }
     * @param {Object} [options]
     * @param {boolean} [options.debug=false]
     */
    constructor(models, options = {}) {
        this.debug = options.debug ?? false;
        this.models = {}; // name -> { session, threshold }
        for (const name in models) {
            const { modelPath, threshold } = models[name];
            const entry = { session: null, threshold: threshold ?? 0.3 };
            this.models[name] = entry;
            // Load lazily; verify() awaits readiness.
            ONNX.createInferenceSession(modelPath, { executionProviders: ["wasm"] })
                .then((s) => { entry.session = s; if (this.debug) console.log(`[acoustic-verifier] loaded ${name}`); })
                .catch((e) => console.error(`[acoustic-verifier] failed to load ${name}:`, e));
        }
    }

    /**
     * @param {Float32Array} _audio - Unused (kept for interface compatibility with the ASR Verifier).
     * @param {string} wakeWordName - Which wake word fired (model file stem, e.g. "ozwell-i'm-done").
     * @param {Object} embeddingBuffer - The ONNX tensor pass-1 scored ([frames, dim], e.g. [16,96]).
     * @returns {Promise<boolean>} - true if the acoustic verifier confirms the fire.
     */
    async verify(_audio, wakeWordName, embeddingBuffer) {
        const entry = this.models[wakeWordName];
        if (!entry) return true; // no verifier for this word -> don't block (fail open)
        if (!embeddingBuffer || !embeddingBuffer.data) {
            if (this.debug) console.warn(`[acoustic-verifier] no embedding for ${wakeWordName}, passing`);
            return true;
        }
        // Wait for the session if it's still loading; fail open if it never loaded.
        if (entry.session === null) {
            for (let i = 0; i < 200 && entry.session === null; i++) await new Promise((r) => setTimeout(r, 10));
            if (entry.session === null) return true;
        }

        // Flatten the [frames, dim] embedding window to [1, frames*dim] for the MLP.
        const flat = embeddingBuffer.data instanceof Float32Array
            ? embeddingBuffer.data
            : Float32Array.from(embeddingBuffer.data);
        if (this.debug) {
            let mn = Infinity, mx = -Infinity, sum = 0;
            for (let i = 0; i < flat.length; i++) { const v = flat[i]; if (v < mn) mn = v; if (v > mx) mx = v; sum += v; }
            console.log(`[acoustic-verifier] INPUT dims=${JSON.stringify(embeddingBuffer.dims)} len=${flat.length} min=${mn.toFixed(3)} max=${mx.toFixed(3)} mean=${(sum/flat.length).toFixed(3)} first5=[${Array.from(flat.slice(0,5)).map(v=>v.toFixed(3)).join(",")}]`);
        }
        const input = await ONNX.createTensor("float32", flat, [1, flat.length]);
        const out = await entry.session.run({ input });
        let p = AcousticVerifier.probabilityOf(out);

        // DIAGNOSTIC: also score the TRANSPOSED input (frame/dim swap) to find the correct flatten order.
        // embeddingBuffer.dims = [frames, dim]; transpose to [dim, frames] flatten: T[d*F+f] = flat[f*D+d].
        if (this.debug) {
            const F = embeddingBuffer.dims[0], D = embeddingBuffer.dims[1];
            const T = new Float32Array(F * D);
            for (let f = 0; f < F; f++) for (let d = 0; d < D; d++) T[d * F + f] = flat[f * D + d];
            const tIn = await ONNX.createTensor("float32", T, [1, T.length]);
            const pT = AcousticVerifier.probabilityOf(await entry.session.run({ input: tIn }));
            console.log(`[acoustic-verifier] ORDER PROBE  as-is P=${p.toFixed(3)}  transposed P=${pT.toFixed(3)}  (whichever is ~1.0 on a REAL wake is the right order)`);
            // One-time full-vector dump: copy this whole line so we can diagnose permutation vs representation.
            if (!AcousticVerifier._dumped) {
                AcousticVerifier._dumped = true;
                console.log("FULLVEC " + JSON.stringify(Array.from(flat).map(v => Math.round(v * 1000) / 1000)));
            }
        }
        const ok = p >= entry.threshold;
        if (this.debug) {
            console.log(`[acoustic-verifier] ${wakeWordName}: P(wake)=${p.toFixed(3)} thr=${entry.threshold} -> ${ok ? "CONFIRM" : "reject"}`);
        }
        return ok;
    }

    /**
     * Pull P(wake) out of the ONNX output. skl2onnx (zipmap=False) emits "output_probability" as a
     * [1,2] float tensor (column 1 = positive class). Fall back to scanning outputs for robustness.
     * @param {Object} out - The ONNX run() result map.
     * @returns {number} P(wake) in [0,1].
     */
    static probabilityOf(out) {
        const probs = out.output_probability ?? out.probabilities ?? out.output ?? Object.values(out).find((t) => t && t.data && t.data.length >= 2);
        const d = probs?.data;
        if (!d) return 1.0; // can't read -> fail open
        return d.length >= 2 ? d[1] * 1 : d[0] * 1; // [neg, pos] -> pos; single value -> as-is
    }
}