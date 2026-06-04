# Ozwell Wake-Word — Master Knowledge Base

**Single source of truth.** Synthesizes both working sessions (Jun 3 discovery + Jun 3–4 retraining).
If you only read one doc, read this one; it links the deeper docs where detail matters.

- Detail docs: [audio-scale-mismatch.md](audio-scale-mismatch.md) · [retrain-ozwell-im-done.md](retrain-ozwell-im-done.md) · [SESSION-RESUME-2026-06-03.md](SESSION-RESUME-2026-06-03.md)
- Reproducible scripts: [`logs/run_pipeline.sh`](../logs/run_pipeline.sh) (52.5k-neg run), [`logs/run_rebalanced.sh`](../logs/run_rebalanced.sh) (10k-neg run, has a `probe` smoke-test mode)
- Result files: [`logs/FINAL_RESULTS.txt`](../logs/FINAL_RESULTS.txt), [`logs/FINAL_RESULTS_REBAL.txt`](../logs/FINAL_RESULTS_REBAL.txt)

---

## 1. What we're building
On-device wake-word detection for Ozwell. Two phrases:
- **"hey ozwell"** — start listening. **Works (~98% recall).**
- **"ozwell I'm done"** — stop. **The problem child. Was ~24% recall (broken). Active retraining target.**

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

## 4. Current model status (as of 2026-06-04)

Recall is recoverable; **false positives are now the binding constraint.** We mapped the recall↔FP
tradeoff via fast-signal (5k) runs on the held-out ElevenLabs test set:

| Run | Ext. negatives | Recall @0.5 | FPR @0.5 | Note |
|---|---|---|---|---|
| Pre-loudness-fix shipped | — | ~24% | — | broken (Problem A) |
| Post-fix, default negs | freesound only | ~76–93% | 58–71% | recall fixed, fires on all speech |
| `run_pipeline` | 52.5k (libri+free) | 44% | 32% | over-suppressed: too many negs crushed recall |
| `run_rebalanced` | 10k (libri+free) | **74%** | **41%** | recall recovered; FP went up — direct tradeoff |

**Confirmed mechanism:** negative:positive **balance is the recall lever.** ~10:1 negs crushes recall;
~2:1 recovers it but raises FP. The usable point is a middle ratio — and still needs FP far lower.
**Stubborn issue:** negatives have a hard-firing tail (`p90≈0.998`) at *every* ratio — clips that
genuinely sound like the phrase. No ratio fixes those; they need targeted hard-negatives and/or more
positive diversity (the main argument for the full 100k run).

Caveat: the 15-clip in-domain Piper eval scored *lower* than the held-out set (40% vs 74%) — backwards
and probably just a tiny-sample artifact; don't trust that number.

## 5. How to reproduce / retrain (so we never lose the recipe again)
Environment + exact commands live in [retrain-ozwell-im-done.md](retrain-ozwell-im-done.md). Essentials:
Python 3.11 + uv venv; torch cu124 (match the driver); run long jobs in **tmux**; checkpoints land in
`model/checkpoints/` (not `exports/`). Both `run_pipeline.sh` and `run_rebalanced.sh` log the exact
training command — **always record the command** (Problem A was caused by not doing this).

## 6. Open items / next steps
1. **PUSH THE WORK (urgent).** Commit `ef40a06` (the loudness fix) is **local-only**. The remote uses
   HTTPS w/ password auth (GitHub-disabled) and SSH isn't set up → pushes fail. Set up a PAT or SSH key
   and `git push`. Until then the most important fix exists only on this machine.
2. **Find the middle-ratio operating point** — one more fast run (~20–30k negs or neg-weighting) targeting
   recall ≥70% with FP trending down.
3. **Full 100k production run** overnight (in tmux) once the ratio is chosen — positive diversity is the
   main weapon against the hard-negative tail.
4. **Apply the loudness fix to `prod/js`** — the browser runtime still feeds un-normalized audio to the
   mel model. REQUIRED before production; without it the shipped model won't match training.
5. **Get real human recordings** of "ozwell I'm done" (clinicians, real mics) for the test set (and ideally
   some in training). This is the gap between good eval numbers and real-world performance.

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
</content>
</invoke>
