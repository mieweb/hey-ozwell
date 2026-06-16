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
    if (VERIFIER_MODE !== "off" && typeof AcousticVerifier !== "undefined") {
        // Only the stop phrase has a verifier so far (it's the one that over-fires / cuts off dictation).
        // hey-ozwell passes through untouched until we train/export its verifier.
        options.verifier = new AcousticVerifier(
            {
                "ozwell-i'm-done": { modelPath: "../models/ozwell-i'm-done-verifier.onnx", threshold: 0.65 },
                "hey-ozwell":      { modelPath: "../models/hey-ozwell-verifier.onnx",      threshold: 0.65 },
            },
            { debug: true }
        );
        options.verifierShadow = (VERIFIER_MODE === "shadow");
        console.log(`🔬 acoustic verifier mode: ${VERIFIER_MODE}`);
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
    });

    /** Clean STAGE-2 VERIFIER feed (so you don't have to watch the console fire 10x/phrase).
     *  One row per phrase you say: green CONFIRMED / red REJECTED + the verifier's confidence.
     *  Collapses the per-frame decisions of a single utterance (within 1.5s) into one row. */
    const vpanel = document.createElement("div");
    vpanel.style.cssText = "margin:8px 0 16px";
    vpanel.innerHTML = '<label style="font:600 12px system-ui;letter-spacing:.12em;color:#8a93a6">STAGE-2 VERIFIER (real wake vs false fire)</label>';
    const vlist = document.createElement("div");
    vlist.style.cssText = "margin-top:6px";
    vpanel.appendChild(vlist);
    (graphsContainer || document.body).prepend(vpanel);
    const prettyName = (n) => n.replace(/-/g, " ");
    const vstate = {};
    heyBuddy.onVerifierDecision((name, d) => {
        const now = Date.now();
        let g = vstate[name];
        if (!g || now - g.ts > 1500) { // new utterance -> new row
            g = vstate[name] = { ts: now, anyAccepted: false, maxScore: 0, row: document.createElement("div") };
            g.row.style.cssText = "display:flex;justify-content:space-between;align-items:center;padding:9px 14px;margin:5px 0;border-radius:10px;font:600 15px system-ui;transition:background .15s";
            vlist.prepend(g.row);
            while (vlist.children.length > 9) vlist.removeChild(vlist.lastChild);
        }
        g.ts = now;
        if (d.accepted) g.anyAccepted = true;
        if (typeof d.score === "number") g.maxScore = Math.max(g.maxScore, d.score);
        const ok = g.anyAccepted;
        g.row.style.background = ok ? "rgba(34,197,94,.16)" : "rgba(239,68,68,.16)";
        g.row.style.color = ok ? "#22c55e" : "#f87171";
        g.row.innerHTML = `<span>${ok ? "✅" : "🛑"} “${prettyName(name)}”</span><span style="font-variant-numeric:tabular-nums">${ok ? "CONFIRMED" : "REJECTED"} · ${Math.round(g.maxScore * 100)}%</span>`;
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
