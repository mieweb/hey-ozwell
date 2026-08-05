/** @module models/enrollment */

/**
 * Per-user voice ENROLLMENT for the stage-2 verifier (on-device, no training, no backend).
 *
 * Decided 2026-06-17 after a head-to-head: for few-shot per-user data, SIMILARITY beats training
 * (94-99% conversation rejection at ~90% real-wake retention, and it wins more when reps are few).
 * So enrollment = store the user's own phrase embeddings as templates, and at detection accept a fire
 * if it's close (cosine) to one of them. Fully client-side: templates live in localStorage.
 *
 * We store the PEAK window per spoken rep (the single highest stage-1 score in that utterance = the
 * cleanest full-phrase moment), not every firing frame — avoids the noisy/partial edge windows.
 *
 * Interface matches the verifier so it drops into the same slot: verify(audio, name, embeddingBuffer).
 */
const LS_KEY = "ozwell.enrollment.v1";

function l2norm(v) {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    s = Math.sqrt(s) + 1e-9;
    const out = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) out[i] = v[i] / s;
    return out;
}
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

export class Enrollment {
    /**
     * @param {Object} [options]
     * @param {number} [options.threshold=0.55] - accept if max cosine to a template >= threshold. Tunable.
     * @param {number} [options.repsPerPhrase=4] - how many reps to collect per phrase.
     * @param {boolean} [options.debug=false]
     */
    constructor(options = {}) {
        this.threshold = options.threshold ?? 0.55;
        this.repsPerPhrase = options.repsPerPhrase ?? 4;
        this.debug = options.debug ?? false;
        this.templates = this._load();   // { name: [Float32Array(normalized), ...] }
        this.lastScore = null;
    }

    _load() {
        try {
            const raw = JSON.parse(localStorage.getItem(LS_KEY) || "{}");
            const out = {};
            for (const name in raw) out[name] = raw[name].map((a) => Float32Array.from(a));
            return out;
        } catch { return {}; }
    }
    _save() {
        const raw = {};
        for (const name in this.templates) raw[name] = this.templates[name].map((t) => Array.from(t));
        localStorage.setItem(LS_KEY, JSON.stringify(raw));
    }

    isEnrolled(name) { return (this.templates[name]?.length || 0) > 0; }
    count(name) { return this.templates[name]?.length || 0; }
    clear(name) { if (name) delete this.templates[name]; else this.templates = {}; this._save(); }

    /** Add one enrolled template (a peak-window embedding, length 1536). Stored L2-normalized. */
    addTemplate(name, embedding) {
        const v = embedding instanceof Float32Array ? embedding : Float32Array.from(embedding);
        (this.templates[name] ||= []).push(l2norm(v));
        this._save();
        if (this.debug) console.log(`[enroll] ${name}: ${this.templates[name].length} template(s)`);
    }

    /** Max cosine similarity of an embedding to this phrase's enrolled templates (or null if none). */
    score(name, embedding) {
        const ts = this.templates[name];
        if (!ts || !ts.length) return null;
        const q = l2norm(embedding instanceof Float32Array ? embedding : Float32Array.from(embedding));
        let best = -1;
        for (const t of ts) { const c = dot(q, t); if (c > best) best = c; }
        return best;
    }

    /**
     * Verifier-compatible: confirm a fire if it's close to the user's enrolled templates.
     * @returns {boolean} - true to confirm. If not enrolled for this phrase, returns true (defer to floor).
     */
    verify(_audio, name, embeddingBuffer) {
        if (!this.isEnrolled(name)) { this.lastScore = null; return true; } // not enrolled -> don't block here
        const d = embeddingBuffer?.data;
        if (!d) { this.lastScore = null; return true; }
        const s = this.score(name, d);
        this.lastScore = s;
        const ok = s >= this.threshold;
        if (this.debug) console.log(`[enroll] ${name}: cos=${s.toFixed(3)} thr=${this.threshold} -> ${ok ? "CONFIRM" : "reject"}`);
        return ok;
    }
}

if (typeof window !== "undefined") window.Enrollment = Enrollment;
