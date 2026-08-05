#!/usr/bin/env python3
"""Charter visuals: (1) operating-point chart (recall vs FP/hour, both models, target zone);
(2) score-separation histograms. FP/hour values are the measured 1.59h People's Speech numbers."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_wakeword import WakeWordEvaluator

OUT = "../charter_figures"; os.makedirs(OUT, exist_ok=True)
THRS = [0.5, 0.7, 0.8, 0.9]
# measured FP/hour over 1.59h held-out People's Speech (prod-style single-frame)
FP_HR = {
    "hey ozwell":      {0.5: 1.3, 0.7: 1.3, 0.8: 0.6, 0.9: 0.0},
    "ozwell i'm done": {0.5: 18.3, 0.7: 5.7, 0.8: 2.5, 0.9: 1.3},
}
MODELS = {
    "hey ozwell":      ("../../prod/js/models/hey-ozwell.onnx", "/tmp/eval/hey/pos", "/tmp/eval/hey/neg"),
    "ozwell i'm done": ("../checkpoints/scratch-onnx/ozwell_done_100k.onnx", "/tmp/eval/done/pos", "/tmp/eval/done/neg"),
}
COLORS = {"hey ozwell": "#2a7", "ozwell i'm done": "#c33"}

scored = {}
for name, (m, posd, negd) in MODELS.items():
    ev = WakeWordEvaluator(m, "pretrained")
    scored[name] = (ev.score_folder(posd), ev.score_folder(negd))

# ---- (1) operating-point chart ----
fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.add_patch(plt.Rectangle((0, 95), 1, 7, color="#7c7", alpha=0.4, zorder=0))
ax.annotate("TARGET\n≥95% recall, ≤1 FP/hr", xy=(0.5, 97.5), xytext=(5.0, 98.5),
            fontsize=8.5, color="#262", ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="#262", lw=1.2))
for name, (pos, neg) in scored.items():
    xs = [FP_HR[name][t] for t in THRS]
    ys = [(pos >= t).mean() * 100 for t in THRS]
    ax.plot(xs, ys, "-o", color=COLORS[name], label=name, lw=2, ms=6)
    for t, x, y in zip(THRS, xs, ys):
        ax.annotate(f"thr {t}", (x, y), textcoords="offset points", xytext=(6, 5), fontsize=7, color=COLORS[name])
ax.set_xlabel("False alarms per hour (held-out real speech)")
ax.set_ylabel("Recall % (synthetic test set)")
ax.set_title("Wake-word operating points: recall vs false-alarms/hour")
ax.set_xlim(-0.5, 20); ax.set_ylim(35, 102); ax.grid(alpha=0.3); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/operating_point.png", dpi=140); plt.close(fig)

# ---- (2) score-separation histograms ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
bins = np.linspace(0, 1, 26)
for ax, (name, (pos, neg)) in zip(axes, scored.items()):
    ax.hist(neg, bins=bins, alpha=0.6, color="#c33", label=f"negatives (n={len(neg)})", density=True)
    ax.hist(pos, bins=bins, alpha=0.6, color="#2a7", label=f"positives (n={len(pos)})", density=True)
    ax.axvline(0.8, color="k", ls="--", lw=1, alpha=0.6)
    ax.set_title(f"{name}\n(recall {(pos>=0.8).mean()*100:.0f}% @thr 0.8)")
    ax.set_xlabel("wake-word score"); ax.legend(fontsize=8)
axes[0].set_ylabel("density")
fig.suptitle("Score separation — positives vs negatives (well-separated = good)", y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/score_separation.png", dpi=140, bbox_inches="tight"); plt.close(fig)

print("wrote", os.path.abspath(f"{OUT}/operating_point.png"))
print("wrote", os.path.abspath(f"{OUT}/score_separation.png"))
