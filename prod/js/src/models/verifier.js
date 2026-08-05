/** @module models/verifier */

/**
 * Stage-2 ASR verifier.
 *
 * Stage-1 (the tiny wake-word model) fires loosely. This re-checks the buffered audio with an ASR
 * engine and only confirms if the transcript matches the wake phrase — rejecting "confident on random
 * speech" false fires (validated offline: 11/11 random fires rejected, all real detections kept).
 *
 * The ASR engine is injected (so the heavy model/WASM is pluggable): it must expose
 *   async transcribe(Float32Array @16kHz) -> string
 *
 * NOTE on targets: whisper hears "ozwell" as "as well"/"oz well", so each phrase lists those variants.
 * Matching is fuzzy (Levenshtein ratio) so minor ASR errors still match. Close confusables ("as well
 * I'm done") may pass — that's acceptable; the goal is to kill clearly-different random speech.
 */
export class Verifier {
    /**
     * @param {Object} asr - ASR engine with `async transcribe(Float32Array) -> string`.
     * @param {Object} [options]
     * @param {Object} [options.targets] - Map of wakeWordName -> [phrase variants]. Sensible defaults below.
     * @param {number} [options.threshold=0.6] - Min fuzzy similarity (0-1) to accept.
     * @param {boolean} [options.debug=false]
     */
    constructor(asr, options = {}) {
        this.asr = asr;
        this.threshold = options.threshold ?? 0.6;
        this.debug = options.debug ?? false;
        this.targets = options.targets ?? {
            "hey-ozwell": ["hey ozwell", "hey oz well", "hey as well", "hey oswald"],
            "ozwell-i'm-done": ["ozwell i'm done", "oz well i'm done", "as well i'm done", "ozwell im done", "oswald i'm done"],
        };
    }

    /** Normalize text for matching: lowercase, keep letters/apostrophes/spaces. */
    static normalize(text) {
        return (text || "").toLowerCase().replace(/[^a-z' ]/g, " ").replace(/\s+/g, " ").trim();
    }

    /** Levenshtein-based similarity ratio in [0,1] (1 = identical). */
    static similarity(a, b) {
        if (!a.length && !b.length) return 1;
        const m = a.length, n = b.length;
        const dp = new Array(n + 1);
        for (let j = 0; j <= n; j++) dp[j] = j;
        for (let i = 1; i <= m; i++) {
            let prev = dp[0];
            dp[0] = i;
            for (let j = 1; j <= n; j++) {
                const tmp = dp[j];
                dp[j] = Math.min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1));
                prev = tmp;
            }
        }
        return 1 - dp[n] / Math.max(m, n);
    }

    /**
     * @param {Float32Array} audio - Buffered audio @16kHz around the stage-1 fire.
     * @param {string} wakeWordName - Which wake word fired (model file stem, e.g. "ozwell-i'm-done").
     * @returns {Promise<boolean>} - true if the transcript matches the phrase.
     */
    async verify(audio, wakeWordName) {
        const targets = this.targets[wakeWordName];
        if (!targets) return true; // no targets configured for this word -> don't block
        const transcript = Verifier.normalize(await this.asr.transcribe(audio));
        let best = 0;
        for (const t of targets) best = Math.max(best, Verifier.similarity(transcript, t));
        const ok = best >= this.threshold;
        if (this.debug) {
            console.log(`[verifier] ${wakeWordName}: "${transcript}" sim=${best.toFixed(2)} -> ${ok ? "CONFIRM" : "reject"}`);
        }
        return ok;
    }
}
