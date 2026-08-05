#!/usr/bin/env python3
"""Combined scorecard: both Ozwell wake words now handle accents. 1080x1920."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

GREEN="#2a9d8f"; DARK="#1d3557"; GRAY="#9aa3b2"; AMBER="#e9a23b"
fig=plt.figure(figsize=(6.75,12)); fig.patch.set_facecolor("white")
ax=fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,18)

# header
ax.add_patch(plt.Rectangle((0,15.9),10,2.1,color=DARK))
ax.text(5,17.25,"OZWELL WAKE WORDS",ha="center",va="center",fontsize=24,fontweight="bold",color="white")
ax.text(5,16.35,"both now understand accents",ha="center",va="center",fontsize=15,color="#9fc0d6",style="italic")

def card(y, phrase, role, before, after):
    ax.add_patch(FancyBboxPatch((0.6,y),8.8,4.6,boxstyle="round,pad=0.15,rounding_size=0.3",
                                fc="#f4f7fa",ec="#dce3ec",lw=1.5))
    ax.text(1.1,y+3.95,phrase,ha="left",fontsize=21,fontweight="bold",color=DARK)
    ax.text(8.9,y+3.95,role,ha="right",fontsize=13,color=AMBER,fontweight="bold")
    # accent recall before -> after (own line, full width)
    ax.text(1.1,y+2.95,"accent recall",ha="left",fontsize=12.5,color="#777")
    ax.text(1.2,y+1.75,before,ha="left",fontsize=30,fontweight="bold",color=GRAY)
    ax.text(3.55,y+1.9,"→",ha="center",fontsize=24,color="#bbb")
    ax.text(4.35,y+1.75,after,ha="left",fontsize=33,fontweight="bold",color=GREEN)
    # false-alarm line (below, full width)
    ax.text(1.1,y+0.55,"✓  false alarms under target",ha="left",fontsize=15,color=GREEN,fontweight="bold")

card(9.9, '"hey ozwell"', "START word", "~11%", "100%")
card(4.6, '"ozwell I\'m done"', "STOP word", "~11%", "83–100%")

# footer
ax.add_patch(plt.Rectangle((0,0),10,3.0,color="#eef2f6"))
ax.text(5,2.25,"Indian · British · Australian accents",ha="center",fontsize=14.5,fontweight="bold",color=DARK)
ax.text(5,1.55,"from barely working  →  reliably detected",ha="center",fontsize=13,color=GREEN,fontweight="bold")
ax.text(5,0.85,"next: validate with real voices",ha="center",fontsize=12.5,color="#888",style="italic")

fig.savefig("both_wakewords_scorecard.png",dpi=160,facecolor="white"); print("saved both_wakewords_scorecard.png")
