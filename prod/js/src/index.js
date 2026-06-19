/**
 * Turns floating-point audio samples to a Wave blob.
 * @param {Float32Array} audioSamples - The audio samples to play.
 * @param {number} sampleRate - The sample rate of the audio samples.
 * @param {number} numChannels - The number of channels in the audio. Defaults to 1 (mono).
 * @return {Blob} A blob of type `audio/wav`
 */
function samplesToBlob(audioSamples, sampleRate = 16000, numChannels = 1) {
    // Helper to write a string to the DataView
    const writeString = (view, offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    // Helper to convert Float32Array to Int16Array (16-bit PCM)
    const floatTo16BitPCM = (output, offset, input) => {
        for (let i = 0; i < input.length; i++, offset += 2) {
            let s = Math.max(-1, Math.min(1, input[i])); // Clamping to [-1, 1]
            output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true); // Convert to 16-bit PCM
        }
    };

    const byteRate = sampleRate * numChannels * 2; // 16-bit PCM = 2 bytes per sample

    // Calculate sizes
    const blockAlign = numChannels * 2; // 2 bytes per sample for 16-bit audio
    const wavHeaderSize = 44;
    const dataLength = audioSamples.length * numChannels * 2; // 16-bit PCM data length
    const buffer = new ArrayBuffer(wavHeaderSize + dataLength);
    const view = new DataView(buffer);

    // Write WAV file headers
    writeString(view, 0, 'RIFF'); // ChunkID
    view.setUint32(4, 36 + dataLength, true); // ChunkSize
    writeString(view, 8, 'WAVE'); // Format
    writeString(view, 12, 'fmt '); // Subchunk1ID
    view.setUint32(16, 16, true); // Subchunk1Size (PCM = 16)
    view.setUint16(20, 1, true); // AudioFormat (PCM = 1)
    view.setUint16(22, numChannels, true); // NumChannels
    view.setUint32(24, sampleRate, true); // SampleRate
    view.setUint32(28, byteRate, true); // ByteRate
    view.setUint16(32, blockAlign, true); // BlockAlign
    view.setUint16(34, 16, true); // BitsPerSample (16-bit PCM)
    writeString(view, 36, 'data'); // Subchunk2ID
    view.setUint32(40, dataLength, true); // Subchunk2Size

    // Convert the Float32Array audio samples to 16-bit PCM and write them to the DataView
    floatTo16BitPCM(view, wavHeaderSize, audioSamples);

    // Create and return the Blob
    return new Blob([view], { type: 'audio/wav' });
}

/**
 * Renders a blob to an audio element with controls.
 * Use `appendChild(result)` to add to the document or a node.
 * @param {Blob} audioBlob - A blob with a valid audio type.
 * @see samplesToBlob
 */
function blobToAudio(audioBlob) {
    const url = URL.createObjectURL(audioBlob);
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = url;
    return audio;
}

/** Configuration */
const colors = {
    "hey ozwell": [255, 209, 0],
    "ozwell i'm done": [255, 0, 255],
    "speech": [22,200,206],
    "frame budget": [25,255,25]
};
const rootUrl = "https://huggingface.co/benjamin-paine/hey-buddy/resolve/main";
const wakeWords = ["hey ozwell", "ozwell i'm done"];
const canvasSize = { width: 640, height: 100 };
const graphLineWidth = 1;
const options = {
    debug: true,
    modelPath: wakeWords.map((word) => `../models/${word.replace(/ /g, '-')}.onnx`),
    // Per-word BASE (front-end) thresholds. The old values (hey 0.8 / done 0.5) were tuned for "<1 FP/hr"
    // with NO downstream gate. We now have WHO+WHAT precision gates, so the base model should run LOOSE for
    // recall (two-pass design: high-recall front-end, strict back-end) and let the gates reject false fires.
    // hey-ozwell lowered 0.8 -> 0.6 because at 0.8 it failed to register at all in noise / when not crisply
    // enunciated ("doesn't even register a guess"). window.__heyThr / live tuning can push it further to 0.5.
    wakeWordThresholds: {
        "hey-ozwell": 0.6,
        "ozwell-i'm-done": 0.5,
    },
    // Voiceprint match threshold (tuned from the live readout): the enrolled phrase peaks
    // ~0.85, the other phrase ~0.57, silence -1 — so ~0.72 fires on YOUR phrase only.
    voiceprintThreshold: 0.72,
    // DROP RECALL (2026-06-17 decision): the voiceprint is used for PRECISION only — reject false
    // fires in runWakeGate — not to amplify weak wakes. Recall wasn't the problem (false positives
    // were), and amplifying fights precision. Flip true to restore the recall fallback.
    voiceprintRecall: false,
    // DEBOUNCE (2026-06-19): require the phrase to clear threshold for N CONSECUTIVE frames before firing.
    // Browser-faithful eval: cuts false fires ~30x at ~1pt recall cost. PER-PHRASE: the long "ozwell i'm done"
    // sustains many frames so it can afford N=3; the SHORT "hey ozwell" fires in a brief 1-2 frame burst, so
    // N=3 ate real wakes (the "skinny bar, no score" non-detections) — it runs at N=1. window.__debounceFrames
    // (a number) overrides all for quick testing.
    debounceFrames: { "hey-ozwell": 1, "ozwell-i'm-done": 3 },
    vadModelPath: `${rootUrl}/pretrained/silero-vad.onnx`,
    spectrogramModelPath: `${rootUrl}/pretrained/mel-spectrogram.onnx`,
    embeddingModelPath: `${rootUrl}/pretrained/speech-embedding.onnx`,
};

/** Main */
document.addEventListener("DOMContentLoaded", async () => {
    /** DOM elements */
    const graphsContainer = document.getElementById("graphs");
    const audioContainer = document.getElementById("audio");

    /** Memory for drawing */
    const graphs = {};
    const history = {};
    const current = {};
    const active = {};

    /** Get user media to request permission and start the microphone */
    try {
        await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (error) {
        alert("Microphone access has been denied, this demo will not function. Please reset audio permissions and refresh the page to try again.");
        return;
    }

    /** Instantiate */
    const heyBuddy = new HeyBuddy(options);

    /** Add callbacks */

    // When processed, update state for next draw
    heyBuddy.onProcessed((result) => {
        current["frame budget"] = heyBuddy.frameTimeEma;
        current["speech"] = result.speech.probability || 0.0;
        active["speech"] = result.speech.active;
        for (let wakeWord in result.wakeWords) {
            current[wakeWord.replace(/-/g, ' ')] = result.wakeWords[wakeWord].probability || 0.0;
            active[wakeWord.replace(/-/g, ' ')] = result.wakeWords[wakeWord].active;
        }
        if (result.recording) {
            audioContainer.innerHTML = "Recording&hellip;";
        }
    });

    // When recording is complete, replace the audio element
    heyBuddy.onRecording((audioSamples) => {
        const audioBlob = samplesToBlob(audioSamples);
        const audioElement = blobToAudio(audioBlob);
        audioContainer.innerHTML = "";
        audioContainer.appendChild(audioElement);
        // Enrollment capture mode: route this utterance to the enroller (same audio path as
        // verification). Only accept it if the wake that fired matches the phrase we asked for.
        if (svCapture) {
            const target = svCapture.targetName, r = svCapture.resolve;
            const firedName = pendingWake ? pendingWake.name : null;
            pendingWake = null; svCapture = null;
            // matched phrase -> capture BOTH the audio (speaker voiceprint) and the fire-time
            // embedding (phrase voiceprint); a DIFFERENT wake -> reject immediately (no waiting).
            r(firedName === target ? { audio: audioSamples, embedding: heyBuddy.lastWakeEmbedding } : { wrong: firedName });
            return;
        }
        // Otherwise: run the speaker-verification gate on the wake utterance.
        runWakeGate(audioSamples);
    });

    // --- Integration (Option B): wake word drives the REAL Ozwell widget ---------
    // Same HeyBuddy instance the bars use: onProcessed feeds the graphs (above),
    // and here onDetected opens + drives the embedded Ozwell widget. So this page
    // shows your new model's confidence bars AND the chatbot responding, together.
    // Guard: the Ozwell widget loads from a separate server (:3000). If that's down, don't let
    // it crash the whole page — the wake/verify/transcribe layers work without the chat widget.
    const ozwellReady = typeof OzwellChat !== "undefined";
    if (ozwellReady) OzwellChat.mount();
    else console.warn("[demo] Ozwell widget unavailable (:3000 not running) — wake/verify/transcribe still work, it just won't drive the chat.");

    const integ = document.createElement("div");
    integ.style = "margin:1em 0;padding:0.75em;border:1px solid #4af;border-radius:6px;" +
                  "font-family:monospace;color:#cde;background:#0b1622;line-height:1.5";
    integ.innerHTML = "<b>Ozwell integration</b> — say “hey ozwell” to open Ozwell.";
    document.body.insertBefore(integ, document.body.firstChild);

    // PERSISTENT gate-score log (its own box, append-only, survives dictation) — for tuning thresholds:
    // every detection drops a row here with WHO + WHAT scores + verdict, so the numbers don't vanish when
    // the integration box flips to "dictating…". Newest on top, last 10 kept.
    const gateBox = document.createElement("div");
    gateBox.style = "margin:1em 0;padding:0.6em 0.75em;border:1px solid #2b3a4a;border-radius:6px;" +
                    "font-family:monospace;font-size:12px;color:#9fb6cc;background:#0b1622";
    gateBox.innerHTML = "<b style='color:#cde'>Gate scores</b> — WHO(voice)=speaker (thr 0.4) · WHAT(phrase)=voiceprint cosine (per-phrase thr) · newest first";
    const gateRows = document.createElement("div");
    gateRows.style = "margin-top:6px";
    gateBox.appendChild(gateRows);
    document.body.insertBefore(gateBox, integ.nextSibling);
    function logGate(name, v, sim, rej, outcome, color) {
        const who = v ? `WHO ${v.score.toFixed(2)} ${v.pass ? "✓" : "✗"}` : "WHO —off";
        const what = (sim != null) ? `WHAT ${sim.toFixed(2)} ${sim >= rej ? "✓" : "✗"}` : "WHAT —off";
        const row = document.createElement("div");
        row.style = `color:${color || "#cde"};padding:1px 0`;
        row.textContent = `${name.padEnd(15)} ${who.padEnd(13)} · ${what.padEnd(14)} → ${outcome}`;
        gateRows.prepend(row);
        while (gateRows.children.length > 10) gateRows.removeChild(gateRows.lastChild);
    }

    let widgetReady = false, pendingContent = null;
    window.addEventListener("message", (e) => {
        if (e.data && e.data.type === "ready") {
            widgetReady = true;
            if (pendingContent) { sendToWidget(pendingContent); pendingContent = null; }
        }
    });
    function sendToWidget(content) {
        const iframe = document.querySelector('iframe[title="Ozwell Chat"]');
        if (iframe && iframe.contentWindow) {
            iframe.contentWindow.postMessage(
                { source: "ozwell-chat-parent", type: "ozwell:send-message", payload: { content } },
                "*"
            );
        }
    }
    function driveOzwell(content, label) {
        if (window.__ozEnrollActive) return; // ignore detections during enrollment
        integ.innerHTML = label;
        if (ozwellReady) OzwellChat.open();
        if (widgetReady) sendToWidget(content);
        else { pendingContent = content; setTimeout(() => { if (pendingContent) { sendToWidget(pendingContent); pendingContent = null; } }, 2500); }
    }

    // Wake detection now only STASHES the intent; the actual decision to drive Ozwell
    // happens in runWakeGate() once we have the recorded utterance (onRecording), so we
    // can verify the speaker first. Anyone can trigger the wake word; only the enrolled
    // doctor's voice acts on it.
    let pendingWake = null;
    let svCapture = null; // { targetName, resolve } while enrollment is waiting for an utterance
    // WHAT-precision reject threshold on RAW cosine — PER-PHRASE, because the two phrases have very
    // different cosine distributions: the long "ozwell i'm done" real wakes sit ~0.85-0.94, but the
    // SHORT "hey ozwell" real wakes sit ~0.64-0.93 (measured live), so a single 0.82 rejected most real
    // "hey ozwell". Set hey-ozwell lower until peak-capture tightens it; re-measure & narrow the gap.
    // window.__wakeRejectSim (a number) overrides all for quick testing.
    const WAKE_REJECT_SIM = { "hey-ozwell": 0.62, "ozwell-i'm-done": 0.82 };
    // Raw cosine (no background-mean subtraction — matches the demo's validated metric, NOT the
    // product's voiceprintSimilarity) of a wake embedding to the phrase's enrolled voiceprints (max).
    function phraseCosine(name, vec) {
        const set = heyBuddy.voiceprints && heyBuddy.voiceprints[name];
        if (!set || !set.length || !vec) return null;
        let qn = 0; for (let i = 0; i < vec.length; i++) qn += vec[i] * vec[i]; qn = Math.sqrt(qn) + 1e-9;
        let best = -1;
        for (const t of set) {
            let d = 0, tn = 0;
            for (let i = 0; i < vec.length; i++) { d += vec[i] * t[i]; tn += t[i] * t[i]; }
            const c = d / (qn * (Math.sqrt(tn) + 1e-9));
            if (c > best) best = c;
        }
        return best;
    }
    heyBuddy.onDetected("hey-ozwell", () => {
        pendingWake = { name: "hey-ozwell", content: "Hey Ozwell — a clinician just started a session.",
                        label: "🔔 <b>“hey ozwell” detected</b>" };
        // Instant feedback that the wake word registered (the verify + "dictating" cue follows
        // a beat later, once you pause). So you're never left wondering if it heard you.
        if (!sessionActive && !window.__ozEnrollActive && !svCapture) {
            integ.innerHTML = `🔔 <b>heard “hey ozwell”</b> — pause a beat, then talk…`;
        }
    });
    heyBuddy.onDetected("ozwell-i'm-done", () => {
        pendingWake = { name: "ozwell-i'm-done", content: "Ozwell, I'm done — please wrap up and summarize the session.",
                        label: "🛑 <b>“ozwell i'm done” detected</b>" };
    });

    // Stage-2 verifier: does the Whisper transcript of the wake buffer actually match the phrase?
    // Whisper hears "ozwell" as "as well"/"oz well"/"oswell", so we list variants and fuzzy-match
    // (Levenshtein ratio). Validated offline on real audio: rejects 11/11 random false-fires, keeps
    // every real detection. Kills false wakes on the doctor's OWN conversation (which pass speaker-
    // verify because it's their voice) — e.g. the stray "*pain*"/"Azul" snippets that leaked through.
    const WAKE_TARGETS = {
        "hey-ozwell": ["hey ozwell", "hey oz well", "hey as well", "hey oswell", "hey oswald"],
        "ozwell-i'm-done": ["ozwell i'm done", "oz well i'm done", "as well i'm done", "oswell i'm done", "ozwell im done", "oswald i'm done"],
    };
    function wakePhraseMatches(text, name) {
        const targets = WAKE_TARGETS[name];
        if (!targets) return true;
        const norm = (text || "").toLowerCase().replace(/[^a-z' ]/g, " ").replace(/\s+/g, " ").trim();
        const sim = (a, b) => {
            const m = a.length, n = b.length;
            if (!m && !n) return 1;
            const d = Array.from({ length: n + 1 }, (_, j) => j);
            for (let i = 1; i <= m; i++) {
                let p = d[0]; d[0] = i;
                for (let j = 1; j <= n; j++) { const t = d[j]; d[j] = Math.min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] === b[j - 1] ? 0 : 1)); p = t; }
            }
            return 1 - d[n] / Math.max(m, n);
        };
        return targets.some((t) => sim(norm, t) >= 0.6);
    }

    // Called from onRecording with the captured wake utterance (16 kHz). Verifies the
    // speaker against THIS phrase's enrolled centroid before acting.
    async function runWakeGate(audioSamples) {
        if (!pendingWake || window.__ozEnrollActive) { pendingWake = null; return; }
        const { content, label, name } = pendingWake; pendingWake = null;
        const sv = window.SpeakerVerify;
        let v = null;
        if (sv && sv.isLoaded() && sv.hasEnrollment(name)) {
            const t0 = performance.now();
            v = sv.verify(name, audioSamples, 16000);
            // [latency] wake-detection per-frame compute (the <250ms target) + one-shot speaker verify
            console.log(`[latency] ${name}: wake-frame ~${heyBuddy.frameTimeEma.toFixed(0)}ms | speaker-verify ${(performance.now() - t0).toFixed(0)}ms | score ${v.score.toFixed(2)} pass ${v.pass}`);
        }
        // --- Compute BOTH gate scores up front so we can SHOW them in one readout ---
        // WHO (speaker/TitaNet): is it the enrolled doctor's voice. WHAT (phrase voiceprint cosine):
        // did they actually say the phrase. Different references — a wake can pass one and fail the
        // other; that's expected, NOT a contradiction. Each gate shows "— off" until enrolled.
        let sim = null;
        const rej = (typeof window.__wakeRejectSim === "number") ? window.__wakeRejectSim
            : (WAKE_REJECT_SIM[name] ?? 0.82);
        if (heyBuddy.hasVoiceprint(name) && heyBuddy.lastWakeEmbedding) {
            // Single fire-frame cosine. (Tried per-utterance aggregate 2026-06-18: it BACKFIRED —
            // "median of top-k windows" hunts each utterance's best-aligned window, which RAISED
            // near-misses into the real-wake range and erased the clean separation single-frame had
            // [real 0.85-0.92 vs near-miss 0.68-0.78]. Reverted to single frame.)
            sim = phraseCosine(name, heyBuddy.lastWakeEmbedding);
        }
        const whoStr  = v        ? `WHO(voice) ${v.score.toFixed(2)} ${v.pass ? "✓" : "✗"}`        : `WHO(voice) —off`;
        const whatStr = sim != null ? `WHAT(phrase) ${sim.toFixed(2)} ${sim >= rej ? "✓" : "✗"}` : `WHAT(phrase) —off`;
        const gates = `<div style="font-size:12px;color:#9fb6cc;margin-top:4px">${whoStr} &nbsp;·&nbsp; ${whatStr}</div>`;
        console.log(`[gates] ${name} → WHO ${v ? v.score.toFixed(2) + (v.pass ? " pass" : " FAIL") : "off"} | WHAT ${sim != null ? sim.toFixed(2) + (sim >= rej ? " pass" : " FAIL") : "off"}`);

        // WHAT precision: reject the doctor's own non-phrase false fires (it's their voice, so WHO
        // passes, but it isn't the phrase). Replaces the dead Whisper transcript gate.
        if (sim != null && sim < rej) {
            integ.innerHTML = `${label} → 🛑 <b>not the phrase</b> — ignored ${gates}`;
            logGate(name, v, sim, rej, "🛑 not the phrase (WHAT)", "#f88");
            return;
        }

        // During an active dictation session, only a verified "ozwell i'm done" ends it.
        if (sessionActive) {
            if (name === "ozwell-i'm-done" && v && v.pass) {
                logGate(name, v, sim, rej, "✅ stop dictation", "#8f8");
                const trimSamples = Math.max(0, audioSamples.length - Math.round(0.6 * 16000));
                stopAndTranscribe(true, trimSamples);
            } else { logGate(name, v, sim, rej, "… session ongoing", "#9fb6cc"); showSessionUI(); }
            return;
        }

        // WHO gate: if enrolled, require the doctor's voice; otherwise off.
        if (v && !v.pass) {
            integ.innerHTML = `${label} → 🔒 <b>not the enrolled doctor</b> — ignored ${gates}`;
            logGate(name, v, sim, rej, "🔒 not the doctor (WHO)", "#f88");
            return;
        }
        logGate(name, v, sim, rej, "✅ verified → " + (name === "hey-ozwell" ? "dictation" : "no session"), "#8f8");
        if (name === "hey-ozwell") {
            integ.innerHTML = `${label} → ✅ <b>verified</b> → starting dictation… ${gates}`;
            if (ozwellReady) OzwellChat.open();
            beep(880); // "go" chime — session is live, start dictating now
            startSession();
        } else {
            integ.innerHTML = `${label} → ✅ verified — no active dictation ${gates}`;
        }
    }
    // ---------------------------------------------------------------------------

    // ===================== Doctor speaker-ID enrollment ========================
    // Enroll the doctor's VOICE (TitaNet speaker embedding) so only they wake Ozwell.
    // Distinct from the content-voiceprint POC below: this GATES (act/no-act on the
    // speaker), that one BOOSTS recall for the doctor's accented phrase. They compose.
    const SV_BTN = "font:13px monospace;padding:6px 12px;border-radius:6px;cursor:pointer;border:1px solid #2ecc71;background:#12233a;color:#cde";
    const svSleep = (ms) => new Promise((r) => setTimeout(r, ms));

    const svPanel = document.createElement("div");
    svPanel.style = "margin:1em 0;padding:0.75em;border:1px solid #2ecc71;border-radius:6px;" +
                    "font-family:monospace;color:#cde;background:#0b1622;line-height:1.6";
    function renderSvPanel() {
        const sv = window.SpeakerVerify;
        const whoList = sv ? sv.enrolledPhrases() : [];                                   // WHO = speaker voiceprint
        const phraseNames = wakeWords.map((w) => w.replace(/ /g, "-"));
        const whatList = phraseNames.filter((n) => heyBuddy.hasVoiceprint && heyBuddy.hasVoiceprint(n)); // WHAT = phrase voiceprint
        const bothMissing = !whoList.length && !whatList.length;
        const mismatch = whoList.length && !whatList.length; // speaker enrolled but phrase gate not (e.g. old enrollment)
        svPanel.innerHTML =
            "<b>🩺 Doctor enrollment</b> — one enroll sets BOTH gates: WHO (your voice) + WHAT (the phrase).<br>" +
            "<span style='color:#9fb6cc'>" +
            (bothMissing ? "not enrolled — gates off (anyone can wake)"
                : "WHO(voice): " + (whoList.length ? whoList.join(", ") : "—none") +
                  " &nbsp;·&nbsp; WHAT(phrase): " + (whatList.length ? whatList.join(", ") : "—none") +
                  (mismatch ? " &nbsp;⚠️ phrase gate not set — click Enroll again" : "")) +
            "</span><div id='sv-status' style='margin:.3em 0;min-height:1.3em;color:#ffd60a'></div>";
        const row = document.createElement("div");
        const enrollBtn = document.createElement("button");
        enrollBtn.textContent = "Enroll doctor voice"; enrollBtn.style = SV_BTN;
        enrollBtn.onclick = () => enrollDoctor(enrollBtn);
        const clearBtn = document.createElement("button");
        clearBtn.textContent = "Clear"; clearBtn.style = SV_BTN + ";margin-left:.5em;border-color:#27aae1";
        clearBtn.onclick = () => {
            if (window.SpeakerVerify) window.SpeakerVerify.clearEnrollment();         // clear WHO
            wakeWords.map((w) => w.replace(/ /g, "-")).forEach((n) => heyBuddy.clearVoiceprint && heyBuddy.clearVoiceprint(n)); // clear WHAT
            voiceprints = {}; saveVoiceprints(voiceprints);                            // clear WHAT storage
            renderSvPanel();
        };
        row.appendChild(enrollBtn); row.appendChild(clearBtn);
        svPanel.appendChild(row);
    }

    // Samples per phrase. More templates (phraseCosine = max over templates) = more robust, and lets
    // you ENROLL ACROSS CONDITIONS — e.g. 6 = 3 masked + 3 clear so it matches whether you're masked or
    // not (each condition gets ≥2 templates, matched by the closest). 3 = quick single-condition.
    const SV_N_ENROLL = 3;

    // Enrollment captures the doctor's wake utterance through hey-buddy's OWN recording path
    // (onRecording) — the SAME path used at verification — so the embeddings actually match.
    // (Recording enrollment on a separate mic at a different rate made even the same voice
    // score ~0.46.) The matching wake must fire to capture, which doubles as the phrase check.
    function captureWakeUtterance(targetName, timeoutMs = 3000) {
        return new Promise((resolve) => {
            svCapture = { targetName, resolve };
            setTimeout(() => { if (svCapture && svCapture.resolve === resolve) { svCapture = null; resolve(null); } }, timeoutMs);
        });
    }

    async function enrollDoctor(btn) {
        const sv = window.SpeakerVerify;
        btn.disabled = true; window.__ozEnrollActive = true; // suppress wake drives while enrolling
        const st = () => document.getElementById("sv-status");
        try {
            // Best-effort load the speaker model (WHO). If its WASM is missing, DON'T hang — still
            // enroll the phrase voiceprint (WHAT). So the precision gate is testable without TitaNet.
            let svOk = false;
            if (sv) {
                st().textContent = "loading verifier…";
                try { await Promise.race([sv.ready(), new Promise((_, rej) => setTimeout(() => rej(new Error("sv timeout")), 8000))]); svOk = sv.isLoaded(); }
                catch (e) { console.warn("[enroll] speaker model unavailable — enrolling phrase voiceprint only:", e); }
            }
            for (const ph of ENROLL_PHRASES) {
                const clips = [];
                const vecs = []; // fire-time embeddings -> phrase voiceprint (WHAT precision gate)
                while (clips.length < SV_N_ENROLL) {
                    st().innerHTML = `🗣️ <b>say “${ph.label}”</b> &nbsp;<span style='color:#9fb6cc'>(sample ${clips.length + 1}/${SV_N_ENROLL})</span>`;
                    beep(660); await svSleep(180); // "speak now" cue (let the chime finish before they talk)
                    const res = await captureWakeUtterance(ph.name);
                    if (!res) { st().innerHTML = `⏳ didn’t catch “${ph.label}” — say it again, clearly`; await svSleep(700); continue; }
                    if (res.wrong) { st().innerHTML = `❌ that was the other phrase — please say “${ph.label}”`; await svSleep(900); continue; }
                    clips.push({ samples: res.audio, sampleRate: 16000 });
                    if (res.embedding) vecs.push(res.embedding);
                    beep(990); // "got it" confirmation chime
                    st().textContent = "✓ got it"; await svSleep(1200); // let the ~2s wake cooldown pass before the next
                }
                if (svOk) sv.enroll(ph.name, clips);     // WHO: speaker voiceprint (only the doctor acts) — if model loaded
                if (vecs.length) {                       // WHAT: phrase voiceprint (reject the doctor's own false fires) — always
                    heyBuddy.setVoiceprint(ph.name, vecs);
                    voiceprints[ph.name] = vecs;
                    saveVoiceprints(voiceprints);
                }
            }
            st().textContent = svOk
                ? "✅ enrolled both phrases — speaker + phrase voiceprints set (one enrollment, both gates)."
                : "✅ enrolled both phrases — phrase voiceprint set (precision gate). Speaker model not loaded — WHO gate off.";
        } catch (e) {
            st().textContent = "error: " + e;
        } finally {
            window.__ozEnrollActive = false; btn.disabled = false; renderSvPanel();
        }
    }

    renderSvPanel();
    document.body.insertBefore(svPanel, document.body.firstChild);
    // ===========================================================================

    // ============== On-device dictation SESSION (Whisper) ======================
    // "hey ozwell" (verified) starts a session: record continuously until "ozwell i'm done"
    // (verified) or the Stop button, then transcribe the whole thing in-browser (audio never
    // leaves the page) and send the text to Ozwell. Transformers.js chunks long audio, so a
    // multi-minute session is fine.
    let cmdAudioCtx, cmdStream, cmdSource;
    let sessionActive = false, sessionProc = null, sessionSink = null, sessionChunks = [];

    async function startSession() {
        if (sessionActive) return;
        if (!cmdStream) {
            cmdStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            cmdAudioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            cmdSource = cmdAudioCtx.createMediaStreamSource(cmdStream);
        }
        if (cmdAudioCtx.state === "suspended") await cmdAudioCtx.resume();
        sessionChunks = [];
        sessionProc = cmdAudioCtx.createScriptProcessor(4096, 1, 1);
        sessionSink = cmdAudioCtx.createGain(); sessionSink.gain.value = 0; // mute -> no feedback
        // Track silence so a session can auto-finish if the stop phrase never fires — the
        // invisible safety net that replaces the old Stop button. ~8s of quiet after some
        // speech ends it. (Saying "ozwell i'm done" is loud, so it resets this and stops first.)
        let silentSamples = 0, sawSpeech = false;
        const SILENCE_THRESH = 0.015, SILENCE_LIMIT = 8 * 16000;
        sessionProc.onaudioprocess = (e) => {
            const d = e.inputBuffer.getChannelData(0);
            sessionChunks.push(Float32Array.from(d));
            let p = 0; for (let i = 0; i < d.length; i++) { const a = Math.abs(d[i]); if (a > p) p = a; }
            if (p > SILENCE_THRESH) { sawSpeech = true; silentSamples = 0; }
            else if (sawSpeech && (silentSamples += d.length) > SILENCE_LIMIT) {
                silentSamples = 0; setTimeout(() => stopAndTranscribe(false), 0); // auto-finish on silence
            }
        };
        cmdSource.connect(sessionProc); sessionProc.connect(sessionSink); sessionSink.connect(cmdAudioCtx.destination);
        sessionActive = true;
        showSessionUI();
    }

    function showSessionUI() {
        integ.innerHTML = `🎤 <b>dictating…</b> say <b>“ozwell i'm done”</b> when you're finished`;
    }

    // The stop phrase "ozwell i'm done" is always the final utterance, but Whisper renders the
    // made-up word "ozwell" inconsistently (Ozwell / All's well / Alls well / As well / Oswald)
    // and sometimes drops "I'm done". Strip those trailing variants from the transcript.
    function stripStopPhrase(text) {
        // Peel the stop phrase off the end. Two problems Whisper causes: (1) it renders the
        // made-up "ozwell" inconsistently (Ozwell / All's well / Alls well / As well / Oswald)
        // and writes "i'm done" as "I am done"; (2) when the stop phrase is QUIET it doesn't
        // transcribe it at all and instead hallucinates filler ("that's all", "that was all",
        // "thank you", "bye"). So we repeatedly strip any of those from the END, one at a time,
        // until the tail is clean. Bare "as well" is intentionally NOT peeled alone (too common
        // as real speech) — only when it's joined to "i'm done".
        const tail = /(?:\b(?:oz\s*well|all['’]?s?\s*well|as\s*well|also|oswald)\b\s*,?\s*i(?:['’]?m|\s+am)\s+done\b|\b(?:oz\s*well|all['’]?s\s*well|also|oswald)\b|\bi(?:['’]?m|\s+am)\s+done\b|\b(?:that\s+was\s+|was\s+)?well\s+done\b|\bthat['’]?s?\s+(?:was\s+)?all\b|\bthank(?:s|\s+you)(?:\s+for\s+watching)?\b|\bbye\b)[\s.,!?-]*$/i;
        let prev = null;
        while (text !== prev && text) { prev = text; text = text.replace(tail, "").replace(/[\s.,!?-]+$/, ""); }
        return text.trim();
    }

    async function stopAndTranscribe(stripStop, trimSamples) {
        if (!sessionActive) return;
        sessionActive = false;
        try { cmdSource.disconnect(sessionProc); sessionProc.disconnect(); sessionSink.disconnect(); } catch (e) {}
        let len = 0; for (const c of sessionChunks) len += c.length;
        let samples = new Float32Array(len); let o = 0; for (const c of sessionChunks) { samples.set(c, o); o += c.length; }
        // Cut the measured stop utterance (minus margin) off the tail so Whisper doesn't see it.
        if (trimSamples > 0 && samples.length > trimSamples) {
            samples = samples.subarray(0, samples.length - trimSamples);
            len = samples.length;
        }
        let peak = 0; for (let i = 0; i < samples.length; i++) { const a = Math.abs(samples[i]); if (a > peak) peak = a; }
        const secs = (len / 16000).toFixed(1);
        console.log(`[dictate] session ${secs}s, ${len} samples, peak ${peak.toFixed(3)}`);
        const w = window.Whisper;
        if (!w || peak < 0.01) {
            integ.innerHTML = `🎤 session ended (${secs}s) — ${peak < 0.01 ? "no audio captured" : "transcription unavailable"}`;
            return;
        }
        integ.innerHTML = `🎤 <b>transcribing ${secs}s on-device…</b>` + (w.isLoaded() ? "" : " <span style='color:#9fb6cc'>(loading model, first time)</span>");
        try {
            const t0 = performance.now();
            let text = (await w.transcribe(samples, 16000)).replace(/\[BLANK_AUDIO\]/gi, "").trim();
            // [latency] one-shot transcription at session end (not subject to the 250ms target)
            console.log(`[latency] transcribe ${secs}s of audio: ${((performance.now() - t0) / 1000).toFixed(1)}s (${w.modelName()})`);
            // Ended by voice: strip the trailing stop phrase from the TEXT (precise — keeps real
            // words a blind audio trim would eat). Silence auto-stop passes false.
            if (stripStop) text = stripStopPhrase(text);
            if (text) { integ.innerHTML = `🗣️ <b>“${text}”</b> → sent to Ozwell`; sendToWidget(text); }
            else { integ.innerHTML = "🗣️ (no speech recognized)"; }
        } catch (e) { integ.innerHTML = `transcription error: ${e}`; }
    }
    // ===========================================================================

    // ===================== Voice enrollment (POC) ==============================
    // Per-user layer on top of the general model: record each phrase a few times, store
    // the embedding "fingerprints" (vectors, not audio) in localStorage, and let the
    // runtime fire on a voiceprint match when the general model is unsure. Phase 0 (in
    // hey-buddy.js) does the matching; this is the capture loop + storage + UI.
    const ENROLL_PHRASES = [
        { name: "hey-ozwell", label: "hey ozwell" },
        { name: "ozwell-i'm-done", label: "ozwell I'm done" },
    ];
    const ENROLL_REPS = 3;
    const ENROLL_GATE = 0.6; // the model must CLEARLY recognize the phrase to accept a rep —
                             // blocks enrolling a different phrase ("hey doug" peaked ~0.55).
                             // Enrollment is deliberate/one-time, so we can be strict here;
                             // the runtime voiceprintGate (0.3) stays lenient for the boost.
    const VP_KEY = "ozwellVoiceprints";
    const BTN_CSS = "font:13px monospace;padding:6px 12px;border-radius:6px;cursor:pointer;border:1px solid #27aae1;background:#12233a;color:#cde";

    const vnorm = (v) => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; return Math.sqrt(s); };
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    function beep(freq) {
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const o = ctx.createOscillator(), g = ctx.createGain();
            o.frequency.value = freq || 880; o.connect(g); g.connect(ctx.destination);
            g.gain.setValueAtTime(0.12, ctx.currentTime);
            o.start(); o.stop(ctx.currentTime + 0.12);
            o.onended = () => ctx.close();
        } catch (e) {}
    }

    function loadVoiceprints() {
        try {
            const obj = JSON.parse(localStorage.getItem(VP_KEY) || "{}");
            const out = {};
            for (const name in obj) out[name] = obj[name].map((a) => Float32Array.from(a));
            return out;
        } catch (e) { return {}; }
    }
    function saveVoiceprints(vp) {
        const obj = {};
        for (const name in vp) obj[name] = vp[name].map((a) => Array.from(a));
        try { localStorage.setItem(VP_KEY, JSON.stringify(obj)); } catch (e) { console.warn("voiceprint save failed", e); }
    }
    function applyVoiceprints(vp) { for (const name in vp) heyBuddy.setVoiceprint(name, vp[name]); }

    let voiceprints = loadVoiceprints();
    applyVoiceprints(voiceprints); // re-apply any saved voiceprint to the runtime on load
    // The SV panel rendered earlier (before this load), so it showed WHAT(phrase) as not-enrolled even
    // when it IS persisted. Re-render now that heyBuddy has the loaded voiceprints so the status is honest
    // and you don't re-enroll the phrase gate on every refresh.
    renderSvPanel();

    // Capture hook: a 2nd onProcessed that snapshots the live embedding while capturing,
    // and tracks live voiceprint similarity (for threshold tuning).
    const enroll = { capturing: false, snaps: [] };
    const liveSim = {};
    const livePeak = {};
    heyBuddy.onProcessed((result) => {
        if (enroll.capturing && result.embedding) {
            // capture every window during the cue (don't depend on VAD); filter by loudness
            // below. record per-phrase model prob too, so we can REJECT a wrong phrase.
            const probs = {};
            if (result.wakeWords) for (const n in result.wakeWords) probs[n] = result.wakeWords[n].probability || 0;
            enroll.snaps.push({
                v: Float32Array.from(result.embedding),
                e: vnorm(result.embedding),
                speaking: !!(result.speech && result.speech.active),
                probs: probs
            });
        }
        if (result.wakeWords) {
            for (const name in result.wakeWords) {
                const s = result.wakeWords[name].voiceprintSim;
                if (typeof s === "number") { liveSim[name] = s; if (s > (livePeak[name] ?? -1)) livePeak[name] = s; }
            }
        }
    });

    const ePanel = document.createElement("div");
    ePanel.style = "margin:1em 0;padding:0.75em;border:1px solid #27aae1;border-radius:6px;" +
                   "font-family:monospace;color:#cde;background:#0b1622;line-height:1.6";
    function renderPanel() {
        const enrolled = ENROLL_PHRASES.filter((p) => (voiceprints[p.name] || []).length > 0).length;
        ePanel.innerHTML =
            "<b>🎙️ Voice enrollment (POC)</b> — boosts detection for <i>your</i> voice.<br>" +
            "<span style='color:#9fb6cc'>Enrolled: " + enrolled + " / " + ENROLL_PHRASES.length + " phrases.</span>" +
            "<div id='enroll-phrase' style='margin:.5em 0;font-size:15px;min-height:1.2em'></div>" +
            "<div id='enroll-status' style='margin:.3em 0;min-height:1.3em;color:#ffd60a'></div>" +
            "<div id='enroll-live' style='color:#7fb;font-size:12px'></div>";
        const row = document.createElement("div");
        const enrollBtn = document.createElement("button");
        enrollBtn.textContent = "Enroll my voice"; enrollBtn.style = BTN_CSS;
        enrollBtn.onclick = () => runEnrollment(enrollBtn);
        const clearBtn = document.createElement("button");
        clearBtn.textContent = "Clear"; clearBtn.style = BTN_CSS + ";margin-left:.5em";
        clearBtn.onclick = () => {
            voiceprints = {};
            for (const p of ENROLL_PHRASES) heyBuddy.clearVoiceprint(p.name);
            try { localStorage.removeItem(VP_KEY); } catch (e) {}
            renderPanel();
        };
        row.appendChild(enrollBtn); row.appendChild(clearBtn);
        ePanel.appendChild(row);
    }
    // NOTE: the old content-voiceprint POC panel is HIDDEN on this (speaker-verification)
    // branch to keep the demo to one clear panel. The boost logic still exists; only its UI
    // is suppressed. The full voiceprint POC lives on branch jlocala/voice-enrollment.
    const SHOW_CONTENT_VOICEPRINT_PANEL = false;
    if (SHOW_CONTENT_VOICEPRINT_PANEL) {
        renderPanel();
        document.body.insertBefore(ePanel, document.body.firstChild);
    }

    // live similarity readout (helps tune the match threshold)
    setInterval(() => {
        for (const p of ENROLL_PHRASES) if (livePeak[p.name] !== undefined) livePeak[p.name] = Math.max(-1, livePeak[p.name] - 0.04);
        const el = document.getElementById("enroll-live");
        if (el) el.textContent = "live match (recent peak) — " + ENROLL_PHRASES.map((p) => p.label + ": " + (livePeak[p.name] ?? 0).toFixed(2)).join("   ");
    }, 200);

    async function captureRep(name) {
        const st = document.getElementById("enroll-status");
        st.textContent = "Get ready…"; await sleep(700);
        beep(); st.innerHTML = "🔴 <b>Speak now</b>"; // the PING cue for when to talk
        enroll.snaps = []; enroll.capturing = true;
        await sleep(2000);
        enroll.capturing = false;
        const snaps = enroll.snaps;
        const speaking = snaps.filter((s) => s.speaking);
        const peakProb = snaps.reduce((m, s) => Math.max(m, (s.probs && s.probs[name]) || 0), 0);
        console.log("[enroll]", name, "| frames:", snaps.length, "| VAD-speaking:", speaking.length, "| peak model prob:", peakProb.toFixed(2));
        const pool = speaking.length >= 3 ? speaking : snaps;
        if (pool.length < 3) { st.textContent = "Didn't catch that — try again, a bit louder?"; await sleep(1000); return null; }
        // GATE: the general model must at least weakly recognize the phrase, so you can't enroll
        // a DIFFERENT phrase (e.g. "hey doug") into this slot — the voiceprint only boosts the real one.
        if (peakProb < ENROLL_GATE) { st.innerHTML = "That didn't sound like the wake word — please say it as written."; await sleep(1500); return null; }
        // keep the highest-energy windows (the phrase-centered ones)
        return pool.slice().sort((a, b) => b.e - a.e).slice(0, 3).map((s) => s.v);
    }

    async function runEnrollment(btn) {
        btn.disabled = true;
        window.__ozEnrollActive = true; // suppress widget pop-ups while enrolling
        const collected = {};
        for (const ph of ENROLL_PHRASES) {
            collected[ph.name] = [];
            const setPhrase = (extra) => {
                const el = document.getElementById("enroll-phrase");
                if (el) el.innerHTML = "🗣️ Say: <b>" + ph.label + "</b>" + (extra || "");
            };
            setPhrase("");
            document.getElementById("enroll-status").textContent = "Get ready to enroll this phrase…";
            await sleep(1500); // give a new user time to read the phrase first
            let got = 0;
            while (got < ENROLL_REPS) {
                setPhrase(" &nbsp;<span style='color:#9fb6cc'>(" + (got + 1) + " of " + ENROLL_REPS + ")</span>");
                const rep = await captureRep(ph.name); // captureRep drives the status cue; the phrase stays put
                if (rep) {
                    collected[ph.name].push.apply(collected[ph.name], rep);
                    got++;
                    document.getElementById("enroll-status").textContent = "✓ got it";
                    await sleep(600);
                }
            }
        }
        document.getElementById("enroll-phrase").textContent = "";
        voiceprints = collected;
        saveVoiceprints(voiceprints);
        applyVoiceprints(voiceprints);
        window.__ozEnrollActive = false;
        btn.disabled = false;
        renderPanel();
        document.getElementById("enroll-status").textContent = "✅ Enrollment complete — your voice is now boosted.";
    }
    // ===========================================================================

    /** Add graphs */
    for (let graphName of ["wake words", "speech", "frame budget"]) {
        // Create containers for the graph and its label
        const graphContainer = document.createElement("div");
        const graphLabel = document.createElement("label");
        graphLabel.textContent = graphName;

        // Create a canvas for the graph
        const graphCanvas = document.createElement("canvas");
        graphCanvas.className = "graph";
        graphCanvas.width = canvasSize.width;
        graphCanvas.height = canvasSize.height;
        graphs[graphName] = graphCanvas;

        // Add the canvas to the container and the container to the document
        graphContainer.appendChild(graphCanvas);
        graphContainer.appendChild(graphLabel);
        graphsContainer.appendChild(graphContainer);

        // If this is the wake-word graph, also add legend
        if (graphName === "wake words") {
            const graphLegend = document.createElement("div");
            graphLegend.className = "legend";
            for (let wakeWord of wakeWords) {
                const legendItem = document.createElement("div");
                const [r,g,b] = colors[wakeWord];
                legendItem.style.color = `rgb(${r},${g},${b})`;
                legendItem.textContent = wakeWord;
                graphLegend.appendChild(legendItem);
            }
            graphLabel.appendChild(graphLegend);
        }
    }

    /** Define draw loop */
    const draw = () => {
        // Draw speech and model graphs
        for (let graphName in graphs) {
            const isWakeWords = graphName === "wake words";
            const isFrameBudget = graphName === "frame budget";
            const subGraphs = isWakeWords ? wakeWords : [graphName];

            let isFirst = true;
            for (let name of subGraphs) {
                // Update history
                history[name] = history[name] || [];
                if (isFrameBudget) {
                    history[name].push((current[name] || 0.0) / 120.0); // 120ms budget
                } else {
                    history[name].push(current[name] || 0.0);
                }

                // Trim history
                if (history[name].length > canvasSize.width) {
                    history[name] = history[name].slice(history[name].length - canvasSize.width);
                }

                // Draw graph
                const canvas = graphs[graphName];
                const ctx = canvas.getContext("2d");
                const [r,g,b] = colors[name];
                const opacity = isFrameBudget || active[name] ? 1.0 : 0.5;
                
                if (isFirst) {
                    // Clear canvas on first draw
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    isFirst = false;
                }

                ctx.strokeStyle = `rgba(${r},${g},${b},${opacity})`;
                ctx.fillStyle = `rgba(${r},${g},${b},${opacity/2})`;
                ctx.lineWidth = graphLineWidth;

                // Draw from left to right (the frame shifts right to left)
                ctx.beginPath();
                let lastX;
                for (let i = 0; i < history[name].length; i++) {
                    const x = i;
                    const y = canvas.height - history[name][i] * canvas.height;
                    if (i === 0) {
                        ctx.moveTo(1, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                    lastX = x;
                }
                // extend downwards to make a polygon
                ctx.lineTo(lastX, canvas.height);
                ctx.lineTo(0, canvas.height);
                ctx.closePath();
                ctx.fill();
                ctx.stroke();
            }
        }

        // Request next frame
        requestAnimationFrame(draw);
    };

    /** Start the loop */
    requestAnimationFrame(draw);
});
