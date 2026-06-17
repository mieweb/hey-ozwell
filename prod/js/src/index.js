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
    // DEBOUNCE TEST: require N consecutive over-threshold frames to fire. 1 = original (fires on
    // single-frame spikes). Set to 2 to kill conversational/TV false-fires (they're single-frame;
    // real spoken phrases sustain 3-6 frames). Try 2 first; 3 is stricter but costs recall. Toggle live.
    wakeWordDebounces: {
        "hey-ozwell": 2,
        "ozwell-i'm-done": 2,
    },
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

    /** Stage-2 ACOUSTIC verifier (embedding MLP — the one we ship). Re-scores the embedding pass-1 used
     *  and suppresses junk fires; judges SOUND, not text (so it works on the made-up word "ozwell" where
     *  ASR hallucinates). Offline: kills ~99% of false-fires, keeps ~99% of diverse voices; real-voice
     *  recall is the open risk (one recording dropped ~40% — fixed by 5-rep enrollment). TOGGLE:
     *    "off"    = stage-1 only
     *    "shadow" = runs + logs what it WOULD kill, but FIRES ANYWAY (zero recall risk — use to A/B live)
     *    "active" = actually suppresses junk fires
     *  Start in "shadow" to see its calls on your live speech, then flip to "active". */
    const VERIFIER_MODE = "active"; // "off" | "shadow" | "active"  (shadow = capture/A-B without suppressing)
    // Per-user ENROLLMENT (similarity, on-device). If the user has enrolled a phrase, use their personal
    // similarity check for it; otherwise fall back to the general trained verifier (the floor).
    const enrollment = new Enrollment({ threshold: 0.88, repsPerPhrase: 3, debug: true });
    if (typeof window !== "undefined") window.__enrollment = enrollment; // live-tune: window.__enrollment.threshold = 0.6
    if (VERIFIER_MODE !== "off" && typeof AcousticVerifier !== "undefined") {
        const general = new AcousticVerifier(
            {
                "ozwell-i'm-done": { modelPath: "../models/ozwell-i'm-done-verifier.onnx", threshold: 0.80 },
                "hey-ozwell":      { modelPath: "../models/hey-ozwell-verifier.onnx",      threshold: 0.80 },
            },
            { debug: true }
        );
        // Combined verifier: enrolled phrase -> personal similarity; else -> general floor.
        options.verifier = {
            lastScore: null,
            async verify(audio, name, emb) {
                if (enrollment.isEnrolled(name)) {
                    const ok = enrollment.verify(audio, name, emb);
                    this.lastScore = enrollment.lastScore;
                    if (this.debug) console.log(`🧑 enrolled-verify ${name}: ${ok ? "CONFIRM" : "reject"} (cos ${this.lastScore?.toFixed(3)})`);
                    return ok;
                }
                const ok = await general.verify(audio, name, emb);
                this.lastScore = general.lastScore;
                return ok;
            },
            debug: true,
        };
        options.verifierShadow = (VERIFIER_MODE === "shadow");
        console.log(`🔬 verifier mode: ${VERIFIER_MODE} | enrolled: ${["hey-ozwell","ozwell-i'm-done"].filter(n=>enrollment.isEnrolled(n)).join(", ")||"none"}`);
    }

    /** Instantiate */
    const heyBuddy = new HeyBuddy(options);

    /** Voice enrollment UI (demo): record a few reps per phrase -> personal on-device similarity check.
     *  Each rep stores the PEAK window of the utterance (cleanest full-phrase moment). */
    const ePanel = document.createElement("div");
    ePanel.style.cssText = "margin:4px 0 16px;padding:12px 14px;border:1px solid rgba(39,170,225,.4);border-radius:12px;background:rgba(39,170,225,.06)";
    const eHead = document.createElement("div");
    eHead.style.cssText = "font:600 12px system-ui,sans-serif;letter-spacing:.08em;color:#27aae1;margin-bottom:8px;text-transform:none;position:static;max-width:none;text-align:left";
    eHead.textContent = "Voice enrollment (personalize on-device)";
    const eStatus = document.createElement("div");
    eStatus.style.cssText = "font:500 14px system-ui,sans-serif;color:#e6edf3;margin-bottom:10px;min-height:20px";
    const eBtns = document.createElement("div");
    eBtns.style.cssText = "display:flex;gap:8px;flex-wrap:wrap";
    ePanel.append(eHead, eStatus, eBtns);
    graphsContainer.parentNode.insertBefore(ePanel, graphsContainer);

    const PHRASES = ["hey-ozwell", "ozwell-i'm-done"];
    const REPS = enrollment.repsPerPhrase;
    const pn = (n) => n.replace(/-/g, " ");
    const refreshStatus = () => {
        eStatus.textContent = PHRASES.map(n => `${pn(n)}: ${enrollment.count(n) >= REPS ? "✓ enrolled" : enrollment.count(n) + "/" + REPS}`).join("    ·    ");
    };
    // Chime (Web Audio) to cue WHEN to speak — like Hey Siri's ping.
    let _actx = null;
    const chime = (freq = 880, dur = 0.12) => {
        try {
            _actx = _actx || new (window.AudioContext || window.webkitAudioContext)();
            const o = _actx.createOscillator(), g = _actx.createGain();
            o.frequency.value = freq; o.type = "sine"; o.connect(g); g.connect(_actx.destination);
            const t = _actx.currentTime;
            g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.2, t + 0.01);
            g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
            o.start(t); o.stop(t + dur);
        } catch (e) {}
    };
    const ENROLL_GATE = 0.85;          // base model must be quite sure it's the phrase (rejects FP garbage)
    const CONSISTENCY = 0.5;           // later reps must resemble the first (cosine) — same phrase each time
    let enrolling = false;
    const enrollPhrase = (name) => {
        if (enrolling) return;
        enrolling = true;
        enrollment.clear(name);
        refreshStatus();               // FIX: reflect cleared state on the marker immediately (no stale "enrolled")
        let got = 0;
        let lastCap = 0;
        let armed = true;              // only accept a capture when armed (one per prompt)
        const prompt = () => { armed = true; eStatus.textContent = `🔵 Say “${pn(name)}” now — ${got}/${REPS}`; chime(); };
        prompt();
        heyBuddy.startEnroll(name, (emb, score) => {
            const now = Date.now();
            // One rep per distinct, paced saying: must be armed AND >=1.2s since the last accepted rep.
            // Stops a single utterance (or VAD flap / lingering audio) from counting as multiple reps.
            if (!armed || now - lastCap < 1200) return;
            // Consistency: each rep after the first must resemble the first (same phrase), else don't count.
            if (got > 0) {
                const sim = enrollment.score(name, emb);
                if (sim !== null && sim < CONSISTENCY) {
                    eStatus.textContent = `⚠️ that sounded different — say “${pn(name)}” again (${got}/${REPS})`;
                    chime(440, 0.18);
                    return;
                }
            }
            armed = false;             // disarm until the next prompt
            lastCap = now;
            enrollment.addTemplate(name, emb);
            got++;
            refreshStatus();           // update marker every step
            if (got >= REPS) {
                heyBuddy.stopEnroll(); enrolling = false;
                eStatus.textContent = `✓ enrolled “${pn(name)}” (${got} reps) — now verified by your voice`;
                chime(1320, 0.18);     // success tone
            } else {
                eStatus.textContent = `✓ got rep ${got}/${REPS} (${Math.round(score * 100)}%) — pause…`;
                setTimeout(prompt, 800);   // pause, then chime + prompt for the next distinct rep
            }
        }, ENROLL_GATE);
    };
    for (const name of PHRASES) {
        const b = document.createElement("button");
        b.textContent = `Enroll “${pn(name)}”`;
        b.style.cssText = "padding:8px 12px;border-radius:8px;border:1px solid #27aae1;background:transparent;color:#27aae1;font:600 13px system-ui,sans-serif;cursor:pointer";
        b.onclick = () => enrollPhrase(name);
        eBtns.appendChild(b);
    }
    const clr = document.createElement("button");
    clr.textContent = "Clear enrollment";
    clr.style.cssText = "padding:8px 12px;border-radius:8px;border:1px solid #6b7280;background:transparent;color:#9aa6b2;font:600 13px system-ui,sans-serif;cursor:pointer";
    clr.onclick = () => { heyBuddy.stopEnroll(); enrolling = false; enrollment.clear(); refreshStatus(); };
    eBtns.appendChild(clr);
    refreshStatus();

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
    });

    /** Clean STAGE-2 VERIFIER feed (so you don't have to watch the console fire 10x/phrase).
     *  One row per phrase: shows "checking…" while you talk, then SETTLES to a single green CONFIRMED /
     *  red REJECTED verdict (no flicker). A row settles ~700ms after the last frame of the utterance. */
    const vpanel = document.createElement("div");
    vpanel.id = "verifier-feed";
    vpanel.style.cssText = "margin:4px 0 20px;padding:12px 14px;border:1px solid rgba(255,255,255,.12);border-radius:12px;background:rgba(255,255,255,.03)";
    const vhead = document.createElement("div");
    vhead.textContent = "Stage-2 verifier — real wake vs false fire";
    vhead.style.cssText = "font:600 12px system-ui,sans-serif;letter-spacing:.08em;color:#8a93a6;margin-bottom:8px;position:static;text-transform:none;text-align:left;max-width:none";
    const vlist = document.createElement("div");
    vpanel.append(vhead, vlist);
    graphsContainer.parentNode.insertBefore(vpanel, graphsContainer);
    const prettyName = (n) => n.replace(/-/g, " ");
    const vstate = {};
    heyBuddy.onVerifierDecision((name, d) => {
        const now = Date.now();
        let g = vstate[name];
        if (!g || now - g.ts > 1500) { // new utterance -> new row
            g = vstate[name] = { ts: now, fired: false, peak: 0, row: document.createElement("div"), timer: null };
            g.row.style.cssText = "display:flex;justify-content:space-between;align-items:center;padding:10px 14px;margin:6px 0;border-radius:10px;font:600 15px system-ui,sans-serif;background:rgba(148,163,184,.12);color:#94a3b8";
            g.row.innerHTML = `<span>“${prettyName(name)}”</span><span>checking…</span>`;
            vlist.prepend(g.row);
            while (vlist.children.length > 8) vlist.removeChild(vlist.lastChild);
        }
        g.ts = now;
        if (d.accepted) g.fired = true;                              // fired if any frame crossed the bar
        if (typeof d.score === "number") g.peak = Math.max(g.peak, d.score);
        clearTimeout(g.timer);
        g.timer = setTimeout(() => {                                 // settle to one final verdict
            const ok = g.fired;
            g.row.style.background = ok ? "rgba(34,197,94,.16)" : "rgba(239,68,68,.16)";
            g.row.style.color = ok ? "#22c55e" : "#f87171";
            g.row.innerHTML = `<span>${ok ? "✅" : "🛑"} “${prettyName(name)}”</span>` +
                `<span style="font-variant-numeric:tabular-nums">${ok ? "CONFIRMED" : "REJECTED"} · ${Math.round(g.peak * 100)}%</span>`;
        }, 700);
    });

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
