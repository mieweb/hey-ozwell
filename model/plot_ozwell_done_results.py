#!/usr/bin/env python3
"""Visual summary of the 'ozwell I'm done' retrain — accent fix + negative-rebalance results.
Data from logs/FINAL_RESULTS_NEGSWEEP.txt + baseline in docs (recall = held-out @ threshold)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# palette — muted, distinct, colorblind-friendly-ish
GRAY   = "#aeb6c2"   # before / baseline
GREEN  = "#2a9d8f"   # after / good / config C
AMBER  = "#e9a23b"   # config B (intermediate)
BLUE   = "#1f4e9c"   # American line
RED    = "#c0392b"   # FP + target lines
BLUE_B = "#6b8cce"   # config B bars

plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#cccccc"})
fig, ax = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('"ozwell I\'m done" — accent fix + false-positive rebalance  (held-out synthetic eval)',
             fontsize=16, fontweight="bold", y=0.99)

def label_bars(a, xs, vals, color, dy, fw="normal"):
    for x, v in zip(xs, vals):
        a.text(x, v + dy, f"{v}%", ha="center", va="bottom", fontsize=9.5, color=color, fontweight=fw)

# --- (0,0) BEFORE vs AFTER accent recall ---
a = ax[0, 0]
groups = ["American", "Indian", "British", "Australian"]
before = [64, 11, 12, 11]; after = [92, 83, 94, 100]
x = np.arange(len(groups)); w = 0.38
a.bar(x - w/2, before, w, label="Before (American-only training)", color=GRAY)
a.bar(x + w/2, after,  w, label="After (accent-diverse, config C @0.5)", color=GREEN)
label_bars(a, x - w/2, before, "#6b7280", 1.5)
label_bars(a, x + w/2, after, GREEN, 1.5, "bold")
a.axhline(95, ls="--", lw=1.2, color=RED, zorder=0)
a.text(3.55, 95, "95% target", color=RED, fontsize=9, ha="right", va="bottom")
a.set_title("1. Accent recall: before vs after", fontweight="bold", fontsize=13, pad=10)
a.set_ylabel("Recall @0.5 (%)"); a.set_xticks(x); a.set_xticklabels(groups)
a.set_ylim(0, 130); a.grid(axis="y", alpha=0.25, zorder=0)
a.legend(loc="upper center", ncol=2, fontsize=9.5, framealpha=0.95, bbox_to_anchor=(0.5, 1.0))

# --- (0,1) FALSE POSITIVES across the negative-rebalance journey ---
a = ax[0, 1]
labels = ["Baseline\n(52.5k neg)", "Config B\n(105k neg)", "Config C\n(160k neg)"]
fp = [18.3, 5.7, 0.6]; bars = a.bar(labels, fp, color=[GRAY, AMBER, GREEN], width=0.6)
for b, v in zip(bars, fp):
    a.text(b.get_x()+b.get_width()/2, v+0.35, f"{v} /hr", ha="center", va="bottom",
           fontweight="bold", fontsize=10.5)
a.axhline(1.0, ls="--", lw=1.4, color=RED, zorder=0)
a.text(2.45, 1.4, "< 1/hr target", color=RED, fontsize=9.5, ha="right", va="bottom")
a.set_title("2. False positives / hour (@0.5) — more negatives fix it", fontweight="bold", fontsize=13, pad=10)
a.set_ylabel("False fires per hour (real speech)"); a.set_ylim(0, 21)
a.grid(axis="y", alpha=0.25, zorder=0)

# --- (1,0) CONFIG C OPERATING CURVE (recall + FP vs threshold) ---
a = ax[1, 0]
thr=[0.3,0.4,0.5,0.7,0.85,0.9]; amer=[96,95,92,86,79,75]; indn=[92,83,83,83,83,83]; fp_hr=[4.4,1.9,0.6,0.6,0.6,0.0]
a.axvspan(0.47, 0.53, color=GREEN, alpha=0.12, zorder=0)
l1,=a.plot(thr, amer, "-o", color=BLUE, lw=2, label="American recall")
l2,=a.plot(thr, indn, "-o", color=GREEN, lw=2, label="Indian recall")
a.axhline(95, ls="--", lw=1.2, color="#888", zorder=0)
a.text(0.305, 95.6, "95% recall", color="#666", fontsize=8.5, va="bottom")
a.set_xlabel("Detection threshold"); a.set_ylabel("Recall (%)")
a.set_ylim(60, 104); a.set_xlim(0.27, 0.93)
a.set_title("3. Config C operating curve — 0.5 is the sweet spot", fontweight="bold", fontsize=13, pad=10)
a.text(0.5, 61.5, "operating\npoint (0.5)", ha="center", va="bottom", fontsize=8.5, color="#1d7a6f")
a2 = a.twinx()
l3,=a2.plot(thr, fp_hr, "--s", color=RED, lw=1.8, alpha=0.85, label="FP / hour")
a2.axhline(1.0, ls=":", lw=1.4, color=RED, zorder=0)
a2.text(0.915, 1.15, "1/hr", color=RED, fontsize=8.5, ha="right", va="bottom")
a2.set_ylabel("False positives / hour", color=RED); a2.set_ylim(0, 10)
a2.tick_params(axis="y", colors=RED)
a.legend(handles=[l1, l2, l3], loc="upper right", fontsize=9.5, framealpha=0.95)

# --- (1,1) B vs C at their SHIPPABLE operating points (both ~0.6 FP/hr) ---
a = ax[1, 1]
acc=["American","Indian","British","Aussie","en-US"]
b_r=[89,100,100,100,87]   # config B @0.85  (FP 0.6/hr)
c_r=[92,83,94,100,96]     # config C @0.5   (FP 0.6/hr) — chosen
x=np.arange(len(acc)); w=0.38
a.bar(x - w/2, b_r, w, label="Config B @0.85  (0.6 FP/hr)", color=BLUE_B)
a.bar(x + w/2, c_r, w, label="Config C @0.5  (0.6 FP/hr) ← chosen", color=GREEN)
label_bars(a, x - w/2, b_r, "#4a6298", 1.5)
label_bars(a, x + w/2, c_r, GREEN, 1.5, "bold")
a.set_title("4. Shippable points (both ~0.6 FP/hr): B & C similar — C chosen for FP robustness",
            fontweight="bold", fontsize=11.5, pad=10)
a.set_ylabel("Recall (%)"); a.set_xticks(x); a.set_xticklabels(acc); a.set_ylim(0, 128)
a.grid(axis="y", alpha=0.25, zorder=0)
a.legend(loc="upper center", ncol=2, fontsize=9, framealpha=0.95, bbox_to_anchor=(0.5, 1.0))

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("ozwell_done_results.png", dpi=130, bbox_inches="tight")
print("saved ozwell_done_results.png")
