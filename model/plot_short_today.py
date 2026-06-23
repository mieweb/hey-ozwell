#!/usr/bin/env python3
"""Vertical (9:16) hero graphic for a daily YouTube short — today's wake-word progress."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

GREEN="#2a9d8f"; GRAY="#aeb6c2"; DARK="#1d3557"; AMBER="#e9a23b"; RED="#c0392b"
fig=plt.figure(figsize=(6.75,12)); fig.patch.set_facecolor("white")
ax=fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,18)

# header
ax.add_patch(plt.Rectangle((0,16.2),10,1.8,color=DARK))
ax.text(5,17.35,"WAKE-WORD PROGRESS",ha="center",va="center",fontsize=23,fontweight="bold",color="white")
ax.text(5,16.65,"making Ozwell understand every accent",ha="center",va="center",fontsize=12.5,color="#a8c0d8",style="italic")

# phrase 1 — DONE
ax.text(5,15.4,'"ozwell I\'m done"  ·  STOP word',ha="center",fontsize=15.5,fontweight="bold",color=DARK)
ax.text(5,14.85,"✓  now at production spec",ha="center",fontsize=13,color=GREEN,fontweight="bold")

# big stat 1 — accents
ax.text(5,13.7,"ACCENT RECALL  (Indian · British · Australian)",ha="center",fontsize=11.5,color="#555",fontweight="bold")
ax.text(2.4,12.5,"11%",ha="center",fontsize=40,fontweight="bold",color=GRAY)
ax.text(4.6,12.55,"→",ha="center",fontsize=28,color="#999")
ax.text(7.2,12.5,"83–100%",ha="center",fontsize=34,fontweight="bold",color=GREEN)

# big stat 2 — false alarms
ax.text(5,10.9,"FALSE ALARMS PER HOUR",ha="center",fontsize=11.5,color="#555",fontweight="bold")
ax.text(3.1,9.7,"18",ha="center",fontsize=42,fontweight="bold",color=RED)
ax.text(5,9.75,"→",ha="center",fontsize=34,color="#999")
ax.text(7.0,9.7,"0.6",ha="center",fontsize=42,fontweight="bold",color=GREEN)
ax.text(7.0,8.9,"(target: under 1)",ha="center",fontsize=10.5,color=GREEN,style="italic")

# mini before/after bar
bx=fig.add_axes([0.17,0.30,0.66,0.16]);
g=["American","Indian","British","Aussie"]; before=[64,11,12,11]; after=[92,83,94,100]
x=np.arange(4); w=0.38
bx.bar(x-w/2,before,w,color=GRAY,label="before")
bx.bar(x+w/2,after,w,color=GREEN,label="after")
for i,(b,a2) in enumerate(zip(before,after)):
    bx.text(i+w/2,a2+3,f"{a2}",ha="center",fontsize=8.5,fontweight="bold",color=GREEN)
bx.set_ylim(0,118); bx.set_xticks(x); bx.set_xticklabels(g,fontsize=9); bx.set_yticks([])
for s in ["top","right","left"]: bx.spines[s].set_visible(False)
bx.legend(fontsize=8,loc="upper left",frameon=False,ncol=2)
bx.set_title("recall by accent (%)",fontsize=9.5,color="#555")

# footer — hey ozwell in progress
ax.add_patch(plt.Rectangle((0,0),10,3.0,color="#f0f3f7"))
ax.text(5,2.4,'"hey ozwell"  ·  START word',ha="center",fontsize=14.5,fontweight="bold",color=DARK)
ax.text(5,1.75,"same proven fix — training now",ha="center",fontsize=12.5,color=AMBER,fontweight="bold")
ax.text(5,1.15,"results tomorrow",ha="center",fontsize=12,color="#888",style="italic")

plt.savefig("progress_short_today.png",dpi=160,facecolor="white")
print("saved progress_short_today.png")
