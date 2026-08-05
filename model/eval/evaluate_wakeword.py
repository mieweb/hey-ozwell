#!/usr/bin/env python3
"""
Offline wake-word evaluation harness.

Scores a trained wake-word ONNX model against labeled positive/negative wav clips,
replicating the inference preprocessing in prod/js/src (mel-spectrogram ->
speech-embedding -> wake-word classifier).

Inference only — does NOT need piper-phonemize, so it runs on macOS where the
training CLI cannot. Established June 2026 as the first evaluation of the shipped
heybuddy MLP models, which previously had no recorded metrics.

Pipeline (matches prod/js/src/models/*.js):
  1. mel-spectrogram:  raw audio [1, N] -> [frames, 32], then post-scale x/10 + 2
  2. speech-embedding: 76-frame windows (stride 8) -> one 96-d embedding per window
  3. wake-word:        a [1, 16, 96] window of embeddings -> probability in [0,1]
A clip is scored by sliding a 16-embedding window across it and taking the MAX
probability ("did the phrase fire anywhere in the clip").

IMPORTANT CAVEATS (read before quoting any number):
  * The data.zip clips are ElevenLabs TTS — clean, synthetic speech. Results are an
    OPTIMISTIC ceiling, not real-world performance (real clinicians, noise, accents).
  * "per-clip FPR" (fraction of negative clips that fire) is NOT "false positives per
    hour". The <1 FP/hour production target needs a separate streaming test on real
    continuous audio.
  * Short clips are center-padded with silence to TARGET samples so they yield >=16
    embedding frames (the model needs ~2s of context).

Usage:
  python evaluate_wakeword.py \
      --model ../../prod/js/models/hey-ozwell.onnx \
      --positives /path/to/test/positive \
      --negatives /path/to/test/negative \
      --pretrained-dir ./pretrained \
      --label "hey ozwell"

See README.md for how to fetch the shared ONNX models and extract test clips.
"""
import argparse, glob, os, sys
import numpy as np
import soundfile as sf
import onnxruntime as ort

WIN, STRIDE, EMB_FRAMES, EMB_DIM = 76, 8, 16, 96
TARGET = 48000  # 3s @ 16kHz — enough mel frames for >=16 embedding frames


def load_16k_mono(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype("float32")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-5:                               # peak-normalize loudness (matches training; see
        audio = audio / peak                      # model/docs/audio-scale-mismatch.md)
    if len(audio) < TARGET:                       # center the clip in silence
        pad = TARGET - len(audio)
        audio = np.concatenate(
            [np.zeros(pad // 2, "float32"), audio, np.zeros(pad - pad // 2, "float32")]
        )
    return audio.astype("float32")


def _deterministic_session(path):
    """Single-threaded, sequential execution so scores are reproducible run-to-run.
    Multi-threaded CPU inference is otherwise slightly non-deterministic, which swings
    the recall number for clips sitting near the decision boundary."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(path, sess_options=so)


class WakeWordEvaluator:
    def __init__(self, model_path, pretrained_dir):
        self.mel = _deterministic_session(os.path.join(pretrained_dir, "mel-spectrogram.onnx"))
        self.emb = _deterministic_session(os.path.join(pretrained_dir, "speech-embedding.onnx"))
        self.wake = _deterministic_session(model_path)

    def score_clip(self, path):
        """Return the max wake-word probability over the clip, or None if too short."""
        audio = load_16k_mono(path)
        # 1) mel -> [frames, 32] (32 is the contiguous last dim, so reshape(-1,32) is safe)
        mel_out = self.mel.run(None, {"input": audio[None, :]})[0]
        mel_frames = (mel_out.reshape(-1, 32) / 10.0 + 2.0).astype("float32")
        if mel_frames.shape[0] < WIN:
            return None
        # 2) windowed speech-embedding -> [num_windows, 96]
        n = mel_frames.shape[0]
        n_trunc = n - (n - WIN) % STRIDE
        starts = range(0, n_trunc - WIN + 1, STRIDE)
        windows = np.stack([mel_frames[s:s + WIN] for s in starts])[..., None].astype("float32")
        embeddings = self.emb.run(None, {"input_1": windows})[0].reshape(-1, EMB_DIM).astype("float32")
        if embeddings.shape[0] < EMB_FRAMES:
            return None
        # 3) slide a 16-frame window through the wake-word model, take the max prob
        return max(
            float(self.wake.run(None, {"input": embeddings[s:s + EMB_FRAMES][None].astype("float32")})[0].reshape(-1)[0])
            for s in range(0, embeddings.shape[0] - EMB_FRAMES + 1)
        )

    def score_folder(self, folder):
        scores = [self.score_clip(p) for p in sorted(glob.glob(os.path.join(folder, "*.wav")))]
        return np.array([s for s in scores if s is not None])


def main():
    ap = argparse.ArgumentParser(description="Offline wake-word evaluation harness.")
    ap.add_argument("--model", required=True, help="Path to the wake-word .onnx model")
    ap.add_argument("--positives", required=True, help="Folder of positive (phrase) wavs")
    ap.add_argument("--negatives", required=True, help="Folder of negative (non-phrase) wavs")
    ap.add_argument("--pretrained-dir", default="./pretrained",
                    help="Folder containing mel-spectrogram.onnx and speech-embedding.onnx")
    ap.add_argument("--label", default="wake word", help="Display label for this phrase")
    ap.add_argument("--thresholds", default="0.3,0.5,0.7,0.9",
                    help="Comma-separated decision thresholds to report")
    args = ap.parse_args()

    ev = WakeWordEvaluator(args.model, args.pretrained_dir)
    pos = ev.score_folder(args.positives)
    neg = ev.score_folder(args.negatives)
    if len(pos) == 0 or len(neg) == 0:
        sys.exit(f"No scorable clips found (pos={len(pos)}, neg={len(neg)}). Check the folders.")

    print(f"\n=== {args.label}: max wake-word probability per clip (SYNTHETIC test audio) ===")
    for name, arr in [("POSITIVE", pos), ("NEGATIVE", neg)]:
        print(f"{name:9s} n={len(arr):4d}  mean={arr.mean():.3f}  median={np.median(arr):.3f}  "
              f"p10={np.percentile(arr, 10):.3f}  p90={np.percentile(arr, 90):.3f}")

    print("=== recall (% positives detected) / per-clip FPR (% negatives firing) ===")
    for t in [float(x) for x in args.thresholds.split(",")]:
        recall = (pos >= t).mean() * 100
        fpr = (neg >= t).mean() * 100
        print(f"  thr={t:.2f}   recall={recall:5.1f}%   per-clip FPR={fpr:5.1f}%")
    print("\nReminder: synthetic audio (optimistic); per-clip FPR != FP/hour.")


if __name__ == "__main__":
    main()
