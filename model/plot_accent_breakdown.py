#!/usr/bin/env python3
"""Per-wake-word accent breakdown with ACTUAL false-alarm rate. Two frames, 1080x1920 each."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREEN="#2a9d8f"; DARK="#1d3557"; GRAY="#b9c0cc"; RED="#c0392b"; AMBER="#e9a23b"

def breakdown(fname, phrase, role, rows, fp_text, fp_color, note=None):
    # rows = list of (accent, recall%)
    fig=plt.figure(figsize=(6.75,12)); fig.patch.set_facecolor("white")
    ax=fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,18)
    # header
    ax.add_patch(plt.Rectangle((0,16.0),10,2.0,color=DARK))
    ax.text(5,17.25,phrase,ha="center",va="center",fontsize=24,fontweight="bold",color="white")
    ax.text(5,16.4,role+"  ·  recall by accent",ha="center",va="center",fontsize=14,color="#9fc0d6",fontweight="bold")

    # horizontal bars
    bx=fig.add_axes([0.30,0.34,0.60,0.46])
    labels=[r[0] for r in rows]; vals=[r[1] for r in rows]
    y=np.arange(len(rows))[::-1]
    bx.barh(y, vals, color=GREEN, height=0.62, zorder=3)
    bx.axvline(11, ls="--", lw=1.6, color=RED, zorder=2)
    for yi,v in zip(y,vals):
        bx.text(v-2, yi, f"{v:.0f}%" if v==int(v) else f"{v:.1f}%", va="center", ha="right",
                fontsize=15, fontweight="bold", color="white", zorder=4)
    bx.set_yticks(y); bx.set_yticklabels(labels, fontsize=16)
    bx.set_xlim(0,100); bx.set_xticks([])
    for s in ["top","right","bottom"]: bx.spines[s].set_visible(False)
    bx.spines["left"].set_color("#ccc")
    bx.text(11, len(rows)-0.35, "  was ~11%", color=RED, fontsize=11, ha="left", va="bottom")

    # FALSE ALARM number — big and explicit
    fig.text(0.5,0.225,"FALSE ALARMS",ha="center",fontsize=14,color="#555",fontweight="bold")
    fig.text(0.5,0.145,fp_text,ha="center",fontsize=40,fontweight="bold",color=fp_color)
    if note:
        fig.text(0.5,0.075,note,ha="center",fontsize=12.5,color="#888",style="italic")
    fig.savefig(fname,dpi=160,facecolor="white"); plt.close(fig); print("saved",fname)

# hey ozwell — operating threshold 0.85, FP 0.0/hr
breakdown("accent_breakdown_hey_ozwell.png",'"hey ozwell"',"START word",
          [("Indian",100),("British",100),("Australian",100),("US",95.7)],
          "0.0 / hour", GREEN, note="target: under 1/hr  ✓   (at detection threshold 0.85)")

# ozwell i'm done — operating threshold 0.5, FP 0.6/hr
breakdown("accent_breakdown_ozwell_done.png",'"ozwell I\'m done"',"STOP word",
          [("American",92),("Indian",83),("British",94),("Australian",100),("US",96)],
          "0.6 / hour", GREEN, note="target: under 1/hr  ✓   (at detection threshold 0.5)")
