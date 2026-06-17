/** @module hey-buddy */
import { ONNX } from "./onnx.js";
import { AudioBatcher } from "./audio.js";
import {
    SileroVAD,
    SpeechEmbedding,
    MelSpectrogram,
    WakeWord
} from "./models.js";
import { AcousticVerifier } from "./models/acoustic-verifier.js";
import { Enrollment } from "./models/enrollment.js";

/**
 * Combines an array of embedding buffers into a single embedding tensor.
 *
 * @async
 * @function
 * @param {Float32Array[]} embeddingBufferArray - An array of embedding buffers, where each buffer is a Float32Array.
 * @param {number} numFramesPerEmbedding - The number of frames per embedding.
 * @param {number} embeddingDim - The dimensionality of each embedding.
 * @returns {Promise<Object>} A promise that resolves to an ONNX tensor containing the combined embeddings.
 */
async function embeddingBufferArrayToEmbedding(embeddingBufferArray, numFramesPerEmbedding, embeddingDim){
    // Create empty buffer of the right size
    const combinedEmptyData = new Float32Array(numFramesPerEmbedding * embeddingBufferArray.length * embeddingDim);
    
    // Create tensor with the empty buffer
    const embeddingBuffer = await ONNX.createTensor(
        "float32",
        combinedEmptyData,
        [numFramesPerEmbedding * embeddingBufferArray.length, embeddingDim]
    );

    // Fill the buffer with data
    for (let i = 0; i < embeddingBufferArray.length; i++) {
        const embedding = embeddingBufferArray[i];
        embeddingBuffer.data.set(embedding.data, i * numFramesPerEmbedding * embeddingDim);
    }
    return embeddingBuffer;
}

/**
 * HeyBuddy class for running wake word detection.
 */
export class HeyBuddy {
    /**
     * Create a HeyBuddy instance.
     * @param {Object} [options] - Options object.
     * @param {number} [options.positiveVadThreshold=0.5] - VAD threshold for speech.
     * @param {number} [options.negativeVadThreshold=0.25] - VAD threshold for silence.
     * @param {number} [options.negativeVadCount=8] - Number of negative VADs to trigger silence.
     * @param {number} [options.wakeWordThreads=4] - Number of threads for wake word detection.
     * @param {number} [options.wakeWordThreshold=0.5] - Wake word detection threshold.
     * @param {string|string[]} [options.modelPath="/models/hey-buddy.onnx"] - Path to wake word model.
     * @param {string} [options.vadModelPath="/pretrained/silero-vad.onnx"] - Path to VAD model.
     * @param {string} [options.embeddingModelPath="/pretrained/speech-embedding.onnx"] - Path to speech embedding model.
     * @param {string} [options.spectrogramModelPath="/pretrained/mel-spectrogram.onnx"] - Path to mel spectrogram model.
     * @param {number} [options.batchSeconds=1.08] - Number of seconds per batch.
     * @param {number} [options.batchIntervalSeconds=0.12] - Number of seconds between batches.
     * @param {number} [options.targetSampleRate=16000] - Target sample rate for audio.
     * @param {number} [options.spectrogramMelBins=32] - Number of mel bins for spectrogram.
     * @param {number} [options.embeddingDim=96] - Dimension of speech embedding.
     * @param {number} [options.embeddingWindowSize=76] - Window size for speech embedding.
     * @param {number} [options.embeddingWindowStride=8] - Window stride for speech embedding.
     */
    constructor (options) {
        options = options || {};
        // Get options or use defaults for runtime
        this.debug = options.debug || false;
        options.positiveVadThreshold = options.positiveVadThreshold || 0.65;
        options.negativeVadThreshold = options.negativeVadThreshold || 0.4;
        options.negativeVadCount = options.negativeVadCount || 8;
        this.wakeWordThreads = options.wakeWordThreads || 4;
        this.wakeWordThreshold = options.wakeWordThreshold || 0.5;
        // Per-word thresholds (keyed by model file name, e.g. "hey-ozwell"); falls back to wakeWordThreshold.
        this.wakeWordThresholds = options.wakeWordThresholds || {};
        // Debounce: consecutive over-threshold frames required to fire. 1 = original behavior.
        // Set 2 (or 3) to kill single-frame false-fires (conversational/TV). Per-word overrides via
        // wakeWordDebounces, e.g. { "hey-ozwell": 2, "ozwell-i'm-done": 2 }.
        this.wakeWordDebounce = options.wakeWordDebounce || 1;
        this.wakeWordDebounces = options.wakeWordDebounces || {};
        // Stage-2 verifier (optional): an object with `async verify(audioFloat32, wakeWordName) -> bool`.
        // When set, a debounced stage-1 fire is only accepted if the verifier confirms it (e.g. an ASR
        // re-check of the buffered audio). Rejects "confident on random speech" fires. null = stage-1 only.
        this.verifier = options.verifier || null;
        // Shadow mode: when true, the verifier still runs and LOGS what it would suppress, but does NOT
        // block the fire. Use for a zero-risk live A/B (measure real suppression before gating).
        this.verifierShadow = options.verifierShadow || false;
        this.verifierAudioSamples = Math.round((options.verifierAudioSeconds || 2.0) * 16000);
        this.recentAudio = new Float32Array(0); // rolling raw-audio buffer for the verifier
        // Enrollment: when set via startEnroll(name, onCapture), track the PEAK (highest stage-1)
        // embedding per utterance and hand it to onCapture at speech end (one clean template per rep).
        this.enroll = null;
        this._enrollPeak = { score: 0, emb: null };
        this.wakeWordInterval = options.wakeWordInterval || 2.0; // How often a wake word can be uttered

        // Get options or use defaults for models
        const modelPath = options.modelPath || "/models/hey-buddy.onnx";
        const modelArray = Array.isArray(modelPath) ? modelPath : [modelPath];
        const vadModelPath = options.vadModelPath || "/pretrained/silero-vad.onnx";
        const embeddingModelPath = options.embeddingModelPath || "/pretrained/speech-embedding.onnx";
        const spectrogramModelPath = options.spectrogramModelPath || "/pretrained/mel-spectrogram.onnx";
        const batchSeconds = options.batchSeconds || 1.08; // 1080ms * 16khz = 17280 samples
        const batchIntervalSeconds = options.batchIntervalSeconds || 0.12; // 120ms * 16khz = 1920 samples
        const targetSampleRate = options.targetSampleRate || 16000;
        const spectrogramMelBins = options.spectrogramMelBins || 32;
        const embeddingDim = options.embeddingDim || 96;
        const embeddingWindowSize = options.embeddingWindowSize || 76;
        const embeddingWindowStride = options.embeddingWindowStride || 8;
        const wakeWordEmbeddingFrames = options.wakeWordEmbeddingFrames || 16;

        // Initialize shared models
        this.vad = new SileroVAD(vadModelPath, this.targetSampleRate, options.positiveVadThreshold, options.negativeVadThreshold, options.negativeVadCount);
        this.vad.test(this.debug);

        this.spectrogram = new MelSpectrogram(spectrogramModelPath);
        this.spectrogram.test(this.debug);
        this.spectrogramMelBins = spectrogramMelBins;

        this.embedding = new SpeechEmbedding(
            embeddingModelPath,
            embeddingDim,
            embeddingWindowSize,
            embeddingWindowStride,
        );
        this.embedding.test(this.debug);
        this.embeddingDim = embeddingDim;
        this.embeddingWindowSize = embeddingWindowSize;
        this.embeddingWindowStride = embeddingWindowStride;
        this.embeddingBuffer = null;
        this.embeddingBufferArray = []

        // Initialize wake word models
        this.wakeWords = {};
        this.wakeWordTimes = {};
        this.wakeWordEmbeddingFrames = wakeWordEmbeddingFrames;
        for (let model of modelArray) {
            let modelName = model.split("/").pop().split(".")[0];
            let modelThreshold = this.wakeWordThresholds[modelName] ?? this.wakeWordThreshold;
            this.wakeWords[modelName] = new WakeWord(model, modelThreshold);
            this.wakeWords[modelName].debounceFrames = this.wakeWordDebounces[modelName] ?? this.wakeWordDebounce;
            this.wakeWords[modelName].test(this.debug);
        }

        // Initialize state
        this.recording = false;
        this.audioBuffer = null;
        this.frameIntervalEma = 0;
        this.frameIntervalEmaWeight = 0.1;
        this.frameTimeEma = 0;
        this.frameTimeEmaWeight = 0.1;

        this.speechStartCallbacks = [];
        this.speechEndCallbacks = [];
        this.recordingCallbacks = [];
        this.processedCallbacks = [];
        this.detectedCallbacks = [];
        this.verifierDecisionCallbacks = []; // (name, {accepted, score, stage1}) per stage-2 decision

        // Initialize batcher and add callback
        this.batcher = new AudioBatcher(
            batchSeconds,
            batchIntervalSeconds,
            targetSampleRate
        );
        this.batcher.onBatch((batch) => this.process(batch));
    }

    /**
     * Gets the names of wake words, chunked for threaded wake word detection.
     * @returns {string[][]} - Names of wake words.
     */
    get chunkedWakeWords() {
        return Object.keys(this.wakeWords).reduce((carry, name, i) => {
            const chunkIndex = Math.floor(i / this.wakeWordThreads);
            if (!carry[chunkIndex]) {
                carry[chunkIndex] = [];
            }
            carry[chunkIndex].push(name);
            return carry;
        }, []);
    }

    /**
     * Add a callback for when a wake word is detected.
     * @param {string|string[]} names - Name of wake word.
     * @param {Function} callback - Callback function.
     */
    onDetected(names, callback) {
        this.detectedCallbacks.push({names, callback});
    }

    /**
     * Add a callback for each stage-2 verifier decision (confirm/reject), for clean UI display.
     * @param {Function} callback - called with (name, {accepted, score, stage1}).
     */
    onVerifierDecision(callback) {
        this.verifierDecisionCallbacks.push(callback);
    }

    /**
     * Begin enrollment for a phrase: the peak embedding of each spoken rep is handed to onCapture.
     * @param {string} name - wake word stem (e.g. "hey-ozwell").
     * @param {Function} onCapture - called (embeddingFloat32, peakScore) once per spoken rep at speech end.
     */
    startEnroll(name, onCapture) { this.enroll = { name, onCapture }; this._enrollPeak = { score: 0, emb: null }; }
    /** Stop enrollment capture. */
    stopEnroll() { this.enroll = null; this._enrollPeak = { score: 0, emb: null }; }

    /**
     * Add a callback for processed data.
     * @param {Function} callback - Callback function.
     */
    onProcessed(callback) {
        this.processedCallbacks.push(callback);
    }

    /**
     * Add a callback for speech start.
     * @param {Function} callback - Callback function.
     */
    onSpeechStart(callback) {
        this.speechStartCallbacks.push(callback);
    }

    /**
     * Add a callback for speech end.
     * @param {Function} callback - Callback function.
     */
    onSpeechEnd(callback) {
        this.speechEndCallbacks.push(callback);
    }

    /**
     * Add a callback for recording.
     * @param {Function} callback - Callback function.
     */
    onRecording(callback) {
        this.recordingCallbacks.push(callback);
    }

    /**
     * Trigger speech start event.
     */
    speechStart() {
        if (this.debug) {
            console.log("Speech start");
        }
        for (let callback of this.speechStartCallbacks) {
            callback();
        }
    }

    /**
     * Trigger speech end event.
     */
    speechEnd() {
        if (this.debug) {
            console.log("Speech end");
        }
        // Enrollment: at the end of an utterance, hand the peak embedding to the capture callback (one rep).
        if (this.enroll && this._enrollPeak.emb && this._enrollPeak.score >= 0.3) {
            this.enroll.onCapture(this._enrollPeak.emb, this._enrollPeak.score);
        }
        this._enrollPeak = { score: 0, emb: null };
        for (let callback of this.speechEndCallbacks) {
            callback();
        }
        if (this.recording) {
            this.dispatchRecording();
            this.recording = false;
        }
    }

    /**
     * Dispatch recording to all recording callbacks.
     */ 
    dispatchRecording() {
        if (this.audioBuffer === null) {
            console.error("No recording to dispatch");
            return;
        }
        if (this.debug) {
            const recordingLength = this.audioBuffer.length;
            const recordedDuration = recordingLength / this.batcher.targetSampleRate;
            console.log(`Dispatching recording with ${recordingLength} frames (${recordedDuration} s)`);
        }
        for (let callback of this.recordingCallbacks) {
            callback(this.audioBuffer);
        }
        this.audioBuffer = null;
    }

    /**
     * Trigger wake word detection event.
     * @param {string} name - Name of wake word.
     */
    wakeWordDetected(name) {
        const now = Date.now();
        if (this.wakeWordTimes[name] && (now - this.wakeWordTimes[name]) < this.wakeWordInterval * 1000) {
            return;
        }
        if (this.debug) {
            console.log("Wake word detected:", name);
        }
        this.recording = true;
        this.wakeWordTimes[name] = now;

        for (let {names, callback} of this.detectedCallbacks) {
            if (Array.isArray(names) && names.includes(name) || names === name) {
                callback();
            }
        }
    }

    /**
     * Trigger processed event.
     * @param {Object} data - Processed data.
     */
    processed(data) {
        for (let callback of this.processedCallbacks) {
            callback(data);
        }
    }

    /**
     * Runs wake word detection on a subset of wake words.
     * @param {string[]} wakeWordNames - Names of wake words to check.
     * @returns {Promise} - Promise that resolves when wake word detection is complete.
     */
    async checkWakeWordSubset(wakeWordNames) {
        return await Promise.all(
            wakeWordNames.map(name => this.wakeWords[name].checkWakeWordCalled(this.embeddingBuffer))
        );
    }

    /**
     * Run wake word detection on audio.
     * @returns {Promise} - Promise that resolves when wake word detection is complete.
     */
    async checkWakeWords() {
        const returnMap = {};
        for (let nameChunk of this.chunkedWakeWords) {
            const wakeWordsCalled = await this.checkWakeWordSubset(nameChunk);
            for (let i = 0; i < nameChunk.length; i++) {
                const name = nameChunk[i];
                const wordCalled = wakeWordsCalled[i];
                returnMap[name] = wordCalled;
            }
        }
        // Winner-take-all (MIE 2026-06-09): the two phrases share the word "ozwell" and can
        // co-fire on a single utterance. Since start/stop are mutually exclusive, when more than
        // one wake word crosses its threshold in the same window, fire ONLY the most CONFIDENT one
        // (highest raw probability). NOTE: do NOT compare by margin (prob - threshold) — that biases
        // toward the lower-threshold word (ozwell-done @0.5 out-margins hey-ozwell @0.8 even when you
        // clearly said "hey ozwell"). Raw probability is the correct comparison here.
        // Live debounce visibility: show every threshold-crossing and whether debounce suppressed it.
        // On conversational/TV audio you should see lots of "suppressed (run 1)" and few/no "FIRED".
        for (let name in returnMap) {
            const r = returnMap[name];
            if (r.probability >= this.wakeWords[name].threshold) {
                const db = this.wakeWords[name].debounceFrames || 1;
                console.log(`${r.detected ? "🔔 FIRED" : "🔇 suppressed"} ${name} prob=${r.probability.toFixed(2)} run=${r.run}/${db}`);
            }
        }
        // Enrollment: track the peak (highest stage-1 probability) embedding for the phrase being enrolled.
        if (this.enroll && returnMap[this.enroll.name]) {
            const p = returnMap[this.enroll.name].probability;
            if (p > this._enrollPeak.score) {
                this._enrollPeak = { score: p, emb: Float32Array.from(this.embeddingBuffer.data) };
            }
        }
        let best = null;
        for (let name in returnMap) {
            if (returnMap[name].detected) {
                const prob = returnMap[name].probability;
                if (best === null || prob > best.prob) {
                    best = { name, prob };
                }
            }
        }
        if (best !== null) {
            // Stage-2: re-check the buffered audio with the verifier (e.g. ASR). Only fire if confirmed.
            if (this.verifier) {
                let confirmed = false;
                try {
                    // Pass the embedding window pass-1 just scored (3rd arg) for the ACOUSTIC verifier;
                    // the ASR verifier ignores it and uses this.recentAudio instead.
                    confirmed = await this.verifier.verify(this.recentAudio, best.name, this.embeddingBuffer);
                } catch (e) {
                    console.error("verifier error (failing open):", e);
                    confirmed = true; // don't let a verifier crash block detection
                }
                // Emit a clean decision event for the UI (one per fire frame; UI collapses per utterance).
                const score = (this.verifier && typeof this.verifier.lastScore === "number") ? this.verifier.lastScore : null;
                for (let cb of this.verifierDecisionCallbacks) cb(best.name, {accepted: confirmed, score, stage1: best.prob});
                if (!confirmed) {
                    if (this.verifierShadow) {
                        // shadow: log what we WOULD have killed, but fire anyway (zero recall risk)
                        console.log(`👻 stage-2 WOULD reject ${best.name} (SHADOW — firing anyway) stage-1 prob ${best.prob.toFixed(2)}`);
                    } else {
                        console.log(`🛑 stage-2 REJECTED ${best.name} (stage-1 prob ${best.prob.toFixed(2)})`);
                        return returnMap;
                    }
                } else {
                    console.log(`✅ stage-2 confirmed ${best.name}`);
                }
            }
            this.wakeWordDetected(best.name);
        }
        return returnMap;
    }

    /**
     * Process audio batch.
     * @param {Float32Array} audio - Audio samples.
     */
    async process(audio) {
        // Start timer
        this.frameStart = (new Date()).getTime();

        if (this.frameEnd !== undefined && this.frameEnd !== null) {
            this.frameInterval = this.frameStart - this.frameEnd;
        } else {
            this.frameInterval = 0;
        }
        if (this.frameIntervalEma === 0) {
            this.frameIntervalEma = this.frameInterval;
        } else {
            this.frameIntervalEma = this.frameIntervalEma * (1 - this.frameIntervalEmaWeight) + this.frameInterval * this.frameIntervalEmaWeight;
        }

        // Get the last batch of samples
        const lastBatch = audio.subarray(audio.length - this.batcher.batchIntervalSamples);

        // Maintain a rolling raw-audio buffer (~verifierAudioSeconds) for the stage-2 verifier,
        // by appending each new increment. Only when a verifier is attached (else skip the work).
        if (this.verifier) {
            const merged = new Float32Array(this.recentAudio.length + lastBatch.length);
            merged.set(this.recentAudio, 0);
            merged.set(lastBatch, this.recentAudio.length);
            this.recentAudio = merged.length > this.verifierAudioSamples
                ? merged.slice(merged.length - this.verifierAudioSamples)
                : merged;
        }

        // Calculate the spectrogram for this buffer, assert it is exactly one window
        const spectrograms = await this.spectrogram.run(audio);
        const embedding = await this.embedding.getEmbeddingFromMelSpectrogramOutput(spectrograms);
        const numFramesPerEmbedding = embedding.dims[0];
        const maxEmbeddings = this.wakeWordEmbeddingFrames/numFramesPerEmbedding;


        // We want to run it via a "window" of audio samples at a time
        // so we add a new element, remove the first element, then analyze the new section of audio
        // (or rather audio embeddings) to see if the voice keyword is detected there
        this.embeddingBufferArray.push(embedding);
        if (this.embeddingBufferArray.length > maxEmbeddings) this.embeddingBufferArray.shift();

        this.embeddingBuffer = await embeddingBufferArrayToEmbedding(this.embeddingBufferArray, numFramesPerEmbedding, this.embeddingDim);
        const {isSpeaking, speechProbability, justStoppedSpeaking, justStartedSpeaking} = await this.vad.hasSpeechAudio(lastBatch);

        if(justStartedSpeaking) this.speechStart();
        if(justStoppedSpeaking) this.speechEnd();

        if (isSpeaking && this.embeddingBuffer.dims[0] === this.wakeWordEmbeddingFrames) {
            // If we're listening, run wake word detection
            const wakeWordsCalled = await this.checkWakeWords();
            // Trigger callbacks with processed data
            this.processed({
                listening: true,
                recording: this.recording,
                speech: {probability: speechProbability, active: isSpeaking},
                wakeWords: wakeWordsCalled
            });
        } else {
            // Trigger callbacks right away if we're not listening
            this.processed({
                listening: false,
                recording: this.recording,
                speech: {probability: speechProbability, active: isSpeaking},
                wakeWords: Object.entries(this.wakeWords).reduce(
                    (carry, [name, model]) => {
                        carry[name] = {
                            probability: 0.0,
                            active: false
                        };
                        return carry;
                    },
                    {}
                )
            });
        }

        // If we're recording, append audio to buffer
        if (this.recording) {
            if (this.audioBuffer === null) {
                this.audioBuffer = new Float32Array(audio.length);
                this.audioBuffer.set(audio);
            } else {
                const concatenated = new Float32Array(this.audioBuffer.length + lastBatch.length);
                concatenated.set(this.audioBuffer);
                concatenated.set(lastBatch, this.audioBuffer.length);
                this.audioBuffer = concatenated;
            }
        }

        // Stop timer
        this.frameEnd = (new Date()).getTime();
        this.frameTime = this.frameEnd - this.frameStart;
        if (this.frameTimeEma === 0) {
            this.frameTimeEma = this.frameTime;
        } else {
            this.frameTimeEma = this.frameTimeEma * (1 - this.frameTimeEmaWeight) + this.frameTime * this.frameTimeEmaWeight;
        }
    }
};

if (typeof window !== "undefined") {
    window.HeyBuddy = HeyBuddy;
    window.AcousticVerifier = AcousticVerifier;  // stage-2 ACOUSTIC verifier (embedding MLP — the one we ship)
    window.Enrollment = Enrollment;              // per-user on-device enrollment (similarity)
}