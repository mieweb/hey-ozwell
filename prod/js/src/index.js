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
    // Per-word operating thresholds from offline eval (each meets <1 FP/hr at its point):
    //   hey-ozwell @0.8 (0 FP/hr), ozwell-i'm-done @0.5 (0.6 FP/hr).
    wakeWordThresholds: {
        "hey-ozwell": 0.8,
        "ozwell-i'm-done": 0.5,
    },
    // Voiceprint match threshold (tuned from the live readout): the enrolled phrase peaks
    // ~0.85, the other phrase ~0.57, silence -1 — so ~0.72 fires on YOUR phrase only.
    voiceprintThreshold: 0.72,
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
        // We finally have the wake utterance — run the speaker-verification gate on it.
        runWakeGate(audioSamples);
    });

    // --- Integration (Option B): wake word drives the REAL Ozwell widget ---------
    // Same HeyBuddy instance the bars use: onProcessed feeds the graphs (above),
    // and here onDetected opens + drives the embedded Ozwell widget. So this page
    // shows your new model's confidence bars AND the chatbot responding, together.
    OzwellChat.mount(); // floating Ozwell widget

    const integ = document.createElement("div");
    integ.style = "margin:1em 0;padding:0.75em;border:1px solid #4af;border-radius:6px;" +
                  "font-family:monospace;color:#cde;background:#0b1622;line-height:1.5";
    integ.innerHTML = "<b>Ozwell integration</b> — say “hey ozwell” to open Ozwell.";
    document.body.insertBefore(integ, document.body.firstChild);

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
        OzwellChat.open();
        if (widgetReady) sendToWidget(content);
        else { pendingContent = content; setTimeout(() => { if (pendingContent) { sendToWidget(pendingContent); pendingContent = null; } }, 2500); }
    }

    // Wake detection now only STASHES the intent; the actual decision to drive Ozwell
    // happens in runWakeGate() once we have the recorded utterance (onRecording), so we
    // can verify the speaker first. Anyone can trigger the wake word; only the enrolled
    // doctor's voice acts on it.
    let pendingWake = null;
    heyBuddy.onDetected("hey-ozwell", () => {
        pendingWake = { content: "Hey Ozwell — a clinician just started a session.",
                        label: "🔔 <b>“hey ozwell” detected</b>" };
    });
    heyBuddy.onDetected("ozwell-i'm-done", () => {
        pendingWake = { content: "Ozwell, I'm done — please wrap up and summarize the session.",
                        label: "🛑 <b>“ozwell i'm done” detected</b>" };
    });

    // Called from onRecording with the captured wake utterance (16 kHz).
    function runWakeGate(audioSamples) {
        if (!pendingWake || window.__ozEnrollActive) { pendingWake = null; return; }
        const { content, label } = pendingWake; pendingWake = null;
        const sv = window.SpeakerVerify;
        if (sv && sv.isLoaded() && sv.hasEnrollment()) {
            // Gate ACTIVE: cosine-match the utterance against the enrolled doctor centroid.
            const { score, pass } = sv.verify(audioSamples, 16000);
            if (pass) {
                driveOzwell(content, `${label} → ✅ <b>doctor verified</b> (${score.toFixed(2)}) → opening Ozwell…`);
            } else {
                integ.innerHTML = `${label} → 🔒 <b>not the enrolled doctor</b> (match ${score.toFixed(2)}) — ignored`;
            }
        } else {
            // Gate OFF (no doctor enrolled yet, or verifier still loading): behave as before.
            const note = (sv && sv.hasEnrollment())
                ? " <span style='color:#9fb6cc'>(verifier still loading…)</span>"
                : " <span style='color:#9fb6cc'>(no doctor enrolled — gate off)</span>";
            driveOzwell(content, `${label} → opening Ozwell…${note}`);
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
        const enrolled = sv && sv.hasEnrollment();
        svPanel.innerHTML =
            "<b>🩺 Doctor speaker-ID</b> — gate the wake word to <i>your</i> voice only.<br>" +
            "<span style='color:#9fb6cc'>Status: " +
            (enrolled ? "enrolled ✅ — gate ON (only your voice wakes Ozwell)" : "not enrolled — gate off (anyone can wake)") +
            "</span><div id='sv-status' style='margin:.3em 0;min-height:1.3em;color:#ffd60a'></div>";
        const row = document.createElement("div");
        const enrollBtn = document.createElement("button");
        enrollBtn.textContent = "Enroll doctor voice (3×)"; enrollBtn.style = SV_BTN;
        enrollBtn.onclick = () => enrollDoctor(enrollBtn);
        const clearBtn = document.createElement("button");
        clearBtn.textContent = "Clear"; clearBtn.style = SV_BTN + ";margin-left:.5em;border-color:#27aae1";
        clearBtn.onclick = () => { if (window.SpeakerVerify) window.SpeakerVerify.clearEnrollment(); renderSvPanel(); };
        row.appendChild(enrollBtn); row.appendChild(clearBtn);
        svPanel.appendChild(row);
    }

    // Dedicated short recorder for enrollment (own mic stream, independent of hey-buddy's).
    let svAudioCtx, svStream, svSource;
    async function svRecord(secs) {
        if (!svStream) {
            svStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            svAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
            svSource = svAudioCtx.createMediaStreamSource(svStream);
        }
        if (svAudioCtx.state === "suspended") await svAudioCtx.resume();
        const proc = svAudioCtx.createScriptProcessor(4096, 1, 1);
        const sink = svAudioCtx.createGain(); sink.gain.value = 0; // mute -> no feedback
        const chunks = [];
        proc.onaudioprocess = (e) => chunks.push(Float32Array.from(e.inputBuffer.getChannelData(0)));
        svSource.connect(proc); proc.connect(sink); sink.connect(svAudioCtx.destination);
        await svSleep(secs * 1000);
        svSource.disconnect(proc); proc.disconnect(); sink.disconnect();
        let len = 0; for (const c of chunks) len += c.length;
        const out = new Float32Array(len); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; }
        return { samples: out, sampleRate: svAudioCtx.sampleRate };
    }

    async function enrollDoctor(btn) {
        const sv = window.SpeakerVerify;
        if (!sv) return;
        btn.disabled = true; window.__ozEnrollActive = true; // suppress wake drives while enrolling
        const st = () => document.getElementById("sv-status");
        try {
            st().textContent = "loading verifier…"; await sv.ready();
            const clips = [];
            for (let i = 1; i <= 3; i++) {
                for (let c = 3; c >= 1; c--) { st().textContent = `sample ${i}/3 — say “hey ozwell” in ${c}…`; await svSleep(600); }
                st().innerHTML = `🔴 sample ${i}/3 — <b>say “hey ozwell” now</b>`;
                clips.push(await svRecord(1.6));
                st().textContent = "✓ got it"; await svSleep(400);
            }
            sv.enroll(clips);
            st().textContent = "✅ enrolled — only your voice will wake Ozwell now.";
        } catch (e) {
            st().textContent = "error: " + e;
        } finally {
            window.__ozEnrollActive = false; btn.disabled = false; renderSvPanel();
        }
    }

    renderSvPanel();
    document.body.insertBefore(svPanel, document.body.firstChild);
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
    renderPanel();
    document.body.insertBefore(ePanel, document.body.firstChild);

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
