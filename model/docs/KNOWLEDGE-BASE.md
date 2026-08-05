# Ozwell Wake-Word — Master Knowledge Base

**Single source of truth.** Synthesizes the work sessions from Jun 3–8.
If you only read one doc, read this one; it links the deeper docs where detail matters.

- Detail docs: [audio-scale-mismatch.md](audio-scale-mismatch.md) · [retrain-ozwell-im-done.md](retrain-ozwell-im-done.md) · [SESSION-RESUME-2026-06-03.md](SESSION-RESUME-2026-06-03.md)
- Reproducible scripts: [`logs/run_pipeline.sh`](../logs/run_pipeline.sh) (52.5k-neg run), [`logs/run_rebalanced.sh`](../logs/run_rebalanced.sh) (10k-neg run, has a `probe` smoke-test mode)
- Result files: [`logs/FINAL_RESULTS.txt`](../logs/FINAL_RESULTS.txt), [`logs/FINAL_RESULTS_REBAL.txt`](../logs/FINAL_RESULTS_REBAL.txt)

---

## 1. What we're building
On-device wake-word detection for Ozwell. Two phrases:
- **"hey ozwell"** — start listening. ~97% on American speech, but shares the accent blind spot
  (~11% on non-American voices). Accent retrain in progress (same recipe as below).
- **"ozwell I'm done"** — stop. Was ~24% recall (broken). After the loudness fix + accent-diverse
  retrain it now reaches 83–100% across accents with false positives under 1/hour (see section 4).

Targets: >95% recall, <1 false positive/hour, <250 ms latency. The detector is a tiny MLP that runs
in the browser over speech embeddings produced by an upstream mel→embedding model.

## 2. The two problems (keep these separate — we kept conflating them)

There were **two independent failures**, not one. This distinction matters:

### Problem A — the shipped "ozwell I'm done" was under-trained (a DATA/recipe problem)
- The shipped model scores **near-chance at every audio scale** (20% at production scale, 16% scaled up).
  Broken regardless of loudness → loudness is NOT its main issue.
- **Root cause: its training recipe was never recorded**, and the evidence points to narrow/insufficient
  training data → overfitting. Nobody knows exactly what data or settings produced it.
- This is why "hey ozwell" and "ozwell I'm done" behave so differently despite "the same" tool:
  **heybuddy is a framework, not a fixed recipe.** Same tool ≠ same recipe. They were trained at
  different times with different (unrecorded) data/configs; only one was any good.

### Problem B — train/inference audio-loudness mismatch (a PLUMBING bug)
- The mel→embedding pipeline's output scale depends on **per-clip loudness**. If the loudness the model
  trains at differs from what it sees at inference, it collapses.
- heybuddy's **trainer** historically scaled audio up (`audio *= 32767`); the **production JS runtime**
  and the **eval harness** feed `[-1,1]` floats. So a model trained with current heybuddy expects loud
  audio that production never delivers → near-zero recall.
- It gets worse *within* training: positives (Piper, loud) and negatives (freesound, very quiet) sat at
  different loudness, so the model learned to **separate clips by volume, not by content.**
- **Why "hey ozwell" dodged it:** it was trained *before/without* the `×32767` scaling — i.e. at `[-1,1]`,
  the same scale production uses. Proof: it scores **97% at `[-1,1]`, drops to 55% scaled up.** It got
  lucky on the matching side; its working masked the bug for everyone else.
- **This bug is inherited from upstream** (`painebenjamin/hey-buddy`): the Python trainer and JS runtime
  were written independently and never agreed on an audio-loudness contract. Anyone combining upstream's
  trainer + runtime inherits it. Our `prod/js` is forked from that same upstream (via
  `amandamarg/hey-ozwell-demo`).

**Fix for B (DONE, committed `ef40a06`):** peak-normalize every clip to a common loudness right before
the mel model, in **both** training (`heybuddy/embeddings.py`) and eval (`eval/evaluate_wakeword.py`).
Verified: loud Piper and quiet freesound now produce matching embedding scales. The variety must live in
*content*, never in *format/loudness*.

## 3. Why we use multiple data sources (this kept causing confusion — it's deliberate)

| Role | Source | Why this one |
|---|---|---|
| Training **positives** (the phrase) | **Piper TTS** (synthetic) | No corpus of a made-up phrase exists; synthesize ~100k diverse variants. |
| Training **negatives** (everything else) | **LibriSpeech** (real speech) + **freesound** (sounds) | Model must reject *all* other speech/sound — the whole world. No TTS covers that; need real, diverse speech + real sounds. |
| **Augmentation** (robustness) | **freesound / MIT** noise + reverb impulses | Clean synthetic positives must survive real rooms/mics. |
| **Evaluation** (honest test) | **ElevenLabs** (a *different* TTS) held-out + small in-domain Piper | Must be a source the model never trained on, or you're grading memorization. Mismatched train/test = the generalization check, on purpose. |

Key principle: positives want max variation of *the same words*; negatives want max variation of
*different content*; test must be *different from train*. One source can't fill all roles.
The multi-source design is correct — but it's exactly why Problem B bit us (sources arrive at different
loudness), which is why loudness-normalization is now mandatory discipline.

**Still missing from EVERY role: real humans saying "ozwell I'm done" into a real mic.** Even our
"honest" test (ElevenLabs) is a TTS standing in for real users. Collecting real recordings is the
highest-value next data investment.

## 4. Current model status (as of 2026-06-08)

Both original problems are resolved for "ozwell I'm done". The remaining gap is recall headroom on
specific accents, and the binding limitation is now test-set size, not the model.

- **Accents (was the product-blocker).** The synthetic positive generator was American-only, so held-out
  Indian/British/Australian recall was ~11%. Adding accent-diverse positives (Piper + Google + Azure TTS,
  audio-augmented) lifted held-out accents to 83–100% and the American base from 64% to ~92%. Diversity
  was the lever — it reduced overfitting to a single vendor/accent, which also lifted the American case.
- **False positives.** A negative-ratio sweep (more, and conversational, People's Speech negatives) cut
  false fires from ~18/hr to **0.6/hr** at the high-recall threshold. Config C (160k negs) meets <1 FP/hr
  at threshold 0.5; config B (105k) needs ~0.85. Recipe: [`logs/run_negsweep.sh`](../logs/run_negsweep.sh),
  results: [`logs/FINAL_RESULTS_NEGSWEEP.txt`](../logs/FINAL_RESULTS_NEGSWEEP.txt).
- **Remaining gap.** Neither config yet clears ≥95% recall on *every* accent group simultaneously
  (American sits ~89–92%). That is a positives-quality task (more/better positives), not an FP problem.
- **Validation limit.** Per-accent test sets are tiny (12–16 synthetic clips, capped by the number of
  distinct held-out TTS voices), so small per-accent differences are noise. Real human recordings are now
  the gate to a confident decision — see section 6. The same accent pipeline is being applied to
  "hey ozwell" ([`logs/run_heyozwell.sh`](../logs/run_heyozwell.sh)).

## 5. How to reproduce / retrain (so we never lose the recipe again)
Environment + exact commands live in [retrain-ozwell-im-done.md](retrain-ozwell-im-done.md). Essentials:
Python 3.11 + uv venv; torch cu124 (match the driver); run long jobs in **tmux**; checkpoints land in
`model/checkpoints/` (not `exports/`). The `run_*.sh` scripts log the exact training command —
**always record the command** (Problem A was caused by not doing this).

### GPU acceleration for data generation (found 2026-06-08)
The 4× V100s sat idle because the **CPU-only `onnxruntime`** was installed (no CUDA execution provider),
so the mel/embedding models silently ran on CPU even when a device was requested — the pipeline is
already GPU-wired. Switching to `onnxruntime-gpu` makes the embedding step **~47× faster** (798 →
~38,000 clips/sec on a V100). This only speeds **data generation** (extract / gen_* / augmentation), not
cached training (a tiny MLP over precomputed embeddings has no embedding work to accelerate).

Volta (V100, compute capability 7.0) needs the **CUDA-11 / cuDNN-8** stack — the newer
`onnxruntime-gpu 1.26 + cuDNN 9` fails on it ("no kernel image available"). Working recipe:
`onnxruntime-gpu==1.18.1` + `nvidia-cudnn-cu11==8.9.6.50` + the cu11 nvidia libs, with `LD_LIBRARY_PATH`
pointed at the pip lib dirs. It requires numpy<2 (the training venv runs numpy 2.x), so use a dedicated
GPU venv for data-gen — build script: [`setup-gpu-venv.sh`](../setup-gpu-venv.sh). Not yet integrated
into the live pipeline.

## 6. Open items / next steps
1. **Get real human recordings** of both phrases (clinicians, real mics). This is now the binding step:
   synthetic per-accent test sets are too small (12–16 clips) to choose between configs or to confirm
   that synthetic accent gains hold on real voices.
2. **Pick the operating config** (negative-ratio B vs C) and threshold, once real recordings can settle
   the small-sample accent differences.
3. **Apply the accent pipeline to "hey ozwell"** — same proven recipe, in progress
   ([`logs/run_heyozwell.sh`](../logs/run_heyozwell.sh)); the start word still has the American-only blind spot.
4. **Apply the loudness fix to `prod/js`** — the browser runtime still feeds un-normalized audio to the
   mel model. Required before production; nothing has been promoted to prod yet.
5. **GPU acceleration for data generation** (see section 5) — apply when regenerating large datasets.

## 7. Talking points (for Doug / stakeholders)
- We diagnosed *why* "ozwell I'm done" was broken: a documented data/recipe gap **plus** an audio-loudness
  plumbing bug inherited from the upstream library. Root cause found, not guessed.
- The loudness fix already recovered recall **24% → 74–93%**. The remaining work is a recall-vs-false-positive
  tradeoff we've now characterized and can dial — normal model iteration, not a dead end.
- Lead with this narrative, **not performance charts** — numbers are still moving; the only "finished"
  result worth showing is the recall recovery arc.
- Biggest unblocked ask: **real human recordings of the phrase.**

## 8. Work history
- **Phase 1 (Jun 3):** discovery — found the loudness/scale bug, confirmed prod is forked from upstream,
  wrote the detail docs, implemented the peak-norm fix.
- **Phase 2 (Jun 3–4):** built loudness-matched real-speech negatives, ran the 52.5k and 10k experiments,
  mapped the recall↔FP tradeoff, wrote this knowledge base.
- **Phase 3 (Jun 5–6):** found the accent blind spot (American-only positives); built the three-engine
  (Piper/Google/Azure) accent-diverse positive pipeline with held-out per-accent tests; accent recall
  11% → 90–100%, American base 64% → ~92%.
- **Phase 4 (Jun 8):** negative-ratio sweep cut false positives to under 1/hr; identified the idle-GPU
  software mismatch (47× data-gen speedup, recipe in section 5); started the accent retrain for "hey ozwell".
</content>
</invoke>
