#!/usr/bin/env python3
"""Two talk-over frames for the daily short: accent before/after + false-alarm drop. 1080x1920."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREEN="#2a9d8f"; GRAY="#b9c0cc"; DARK="#1d3557"; RED="#c0392b"
plt.rcParams.update({"font.family":"DejaVu Sans"})

def base_fig():
    fig=plt.figure(figsize=(6.75,12)); fig.patch.set_facecolor("white")
    return fig

def header(fig, title, sub):
    ax=fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,18)
    ax.add_patch(plt.Rectangle((0,16.0),10,2.0,color=DARK))
    ax.text(5,17.3,title,ha="center",va="center",fontsize=22,fontweight="bold",color="white")
    ax.text(5,16.5,sub,ha="center",va="center",fontsize=15,color="#9fc0d6",fontweight="bold")
    return ax

# ---------- accent chart (with `show_after` toggle for the reveal build) ----------
def accent_frame(show_after, fname):
    fig=base_fig(); header(fig,'Accent recall  —  "ozwell I\'m done"',"Last Friday  →  Today")
    bx=fig.add_axes([0.13,0.30,0.78,0.50])
    g=["American","Indian","British","Aussie"]; before=[64,11,12,11]; after=[92,83,94,100]
    x=np.arange(4); w=0.40
    bx.bar(x-w/2,before,w,color=GRAY,label="Last Friday")
    for i,b in enumerate(before): bx.text(i-w/2,b+2,f"{b}%",ha="center",fontsize=12,color="#6b7280",fontweight="bold")
    if show_after:
        bx.bar(x+w/2,after,w,color=GREEN,label="Today")
        for i,a in enumerate(after): bx.text(i+w/2,a+2,f"{a}%",ha="center",fontsize=13,color=GREEN,fontweight="bold")
    bx.set_ylim(0,116); bx.set_xticks(x); bx.set_xticklabels(g,fontsize=14)
    bx.set_yticks([]);
    for s in ["top","right","left"]: bx.spines[s].set_visible(False)
    bx.legend(fontsize=13,loc="upper left",frameon=False,ncol=2)
    # big takeaway
    fig.text(0.5,0.18,"11%  →  83–100%" if show_after else "non-American: ~11%",
             ha="center",fontsize=30 if show_after else 24,fontweight="bold",
             color=GREEN if show_after else RED)
    fig.text(0.5,0.12,"accents that used to fail" if not show_after else "Indian · British · Australian",
             ha="center",fontsize=14,color="#777",style="italic")
    fig.savefig(fname,dpi=160,facecolor="white"); plt.close(fig); print("saved",fname)

# ---------- false alarms ----------
def fp_frame(fname):
    fig=base_fig(); ax=header(fig,"False alarms per hour","Last Friday  →  Today")
    ax.text(2.9,10.6,"18",ha="center",fontsize=80,fontweight="bold",color=RED)
    ax.text(2.9,8.8,"last Friday",ha="center",fontsize=15,color="#888")
    ax.text(5.0,10.2,"→",ha="center",fontsize=48,color="#aaa")
    ax.text(7.2,10.6,"0.6",ha="center",fontsize=80,fontweight="bold",color=GREEN)
    ax.text(7.2,8.8,"today",ha="center",fontsize=15,color="#888")
    ax.text(5,5.6,"✓  target: under 1 per hour",ha="center",fontsize=20,color=GREEN,fontweight="bold")
    ax.text(5,4.4,"fixed by rebalancing with more\nreal conversational negative examples",
            ha="center",fontsize=14,color="#777",style="italic")
    fig.savefig(fname,dpi=160,facecolor="white"); plt.close(fig); print("saved",fname)

accent_frame(False,"short_A1_accent_before.png")
accent_frame(True, "short_A2_accent_after.png")
fp_frame("short_B_false_alarms.png")
