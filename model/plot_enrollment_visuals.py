#!/usr/bin/env python3
"""Two talk-over visuals for the daily short: (1) two-layer enrollment flow, (2) UI mockup. 1080x1920."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

DARK="#1d3557"; GREEN="#2a9d8f"; BLUE="#2c6fbb"; AMBER="#e9a23b"; GRAY="#6b7280"; LGRAY="#eef2f6"

def header(ax, title, sub):
    ax.add_patch(plt.Rectangle((0,16.1),10,1.9,color=DARK))
    ax.text(5,17.3,title,ha="center",va="center",fontsize=23,fontweight="bold",color="white")
    ax.text(5,16.5,sub,ha="center",va="center",fontsize=14,color="#9fc0d6",style="italic")

def box(ax,x,y,w,h,fc,ec,title,body,tcolor="white",bcolor="white",tsize=15,bsize=11.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.1,rounding_size=0.25",fc=fc,ec=ec,lw=2))
    ax.text(x+w/2,y+h-0.55,title,ha="center",va="center",fontsize=tsize,fontweight="bold",color=tcolor)
    if body: ax.text(x+w/2,y+h/2-0.35,body,ha="center",va="center",fontsize=bsize,color=bcolor)

def arrow(ax,x1,y1,x2,y2,color=GRAY):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=22,lw=2.4,color=color))

# ============ VISUAL 1 — two-layer flow ============
fig=plt.figure(figsize=(6.75,12)); fig.patch.set_facecolor("white")
ax=fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,18)
header(ax,"HOW IT WORKS","personal voiceprint + general model")

box(ax,2.6,14.0,4.8,1.5,DARK,DARK,'You say "Hey Ozwell"',"",tsize=16)
arrow(ax,5,14.0,3.3,12.7); arrow(ax,5,14.0,6.7,12.7)
# two parallel paths
box(ax,0.5,10.7,4.0,2.0,BLUE,BLUE,"GENERAL MODEL","trained on many accents\n· works for everyone\n· always on",tsize=13,bsize=10.5)
box(ax,5.5,10.7,4.0,2.0,GREEN,GREEN,"YOUR VOICEPRINT","from your setup\n· matches your voice\n· your accent",tsize=13,bsize=10.5)
arrow(ax,2.5,10.7,4.6,9.2); arrow(ax,7.5,10.7,5.4,9.2)
# converge
box(ax,2.2,7.4,5.6,1.6,AMBER,AMBER,"Fires if EITHER matches","",tsize=15)
arrow(ax,5,7.4,5,6.1)
box(ax,3.0,4.4,4.0,1.6,DARK,DARK,"Ozwell wakes up","",tsize=16)
# fallback note
ax.add_patch(plt.Rectangle((0,0.4),10,2.4,color=LGRAY))
ax.text(5,2.0,"Didn't set up your voice?",ha="center",fontsize=13.5,fontweight="bold",color=DARK)
ax.text(5,1.25,"Still works on the general model — setup just makes it better for you",
        ha="center",fontsize=11.5,color=GRAY,style="italic")
fig.savefig("enroll_flow.png",dpi=160,facecolor="white"); plt.close(fig); print("saved enroll_flow.png")

# ============ VISUAL 2 — UI mockup (3 phone screens) ============
from matplotlib.patches import Arc
fig=plt.figure(figsize=(6.75,12)); fig.patch.set_facecolor("white")
ax=fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,18)
header(ax,"WHAT SETUP LOOKS LIKE","~30 seconds, like 'Hey Siri'")

def draw_mic(cx,yc):  # drawn mic (emoji glyph isn't in the font)
    ax.add_patch(FancyBboxPatch((cx-0.16,yc-0.02),0.32,0.5,boxstyle="round,pad=0.02,rounding_size=0.16",fc=GREEN,ec="none"))
    ax.add_patch(Arc((cx,yc+0.05),0.64,0.72,theta1=200,theta2=340,color=GREEN,lw=3))
    ax.plot([cx,cx],[yc-0.30,yc-0.52],color=GREEN,lw=3)
    ax.plot([cx-0.18,cx+0.18],[yc-0.52,yc-0.52],color=GREEN,lw=3)

def phone(cx, items, caption):
    w,h=2.8,7.8; x,y=cx-w/2, 4.9
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.1,rounding_size=0.45",fc="white",ec=DARK,lw=2.5))
    ax.add_patch(FancyBboxPatch((x+0.25,y+0.3),w-0.5,h-0.6,boxstyle="round,pad=0.05,rounding_size=0.25",fc="#f7f9fb",ec="none"))
    yy=y+h-1.0
    for it in items:
        if it[0]=="mic":
            draw_mic(cx,yy); yy-=1.05
        elif it[0]=="btn":  # green pill, white label
            ax.add_patch(FancyBboxPatch((cx-0.82,yy-0.26),1.64,0.52,boxstyle="round,pad=0.02,rounding_size=0.26",fc=GREEN,ec="none"))
            ax.text(cx,yy,it[1],ha="center",va="center",fontsize=11.5,color="white",fontweight="bold"); yy-=0.95
        else:
            ax.text(cx,yy,it[1],ha="center",va="center",fontsize=it[2],color=it[3],fontweight=it[4]); yy-=0.9
    ax.text(cx,4.1,caption,ha="center",fontsize=12,fontweight="bold",color=DARK)

phone(2.0,[("mic",),("txt","Set up\nyour voice",13,DARK,"bold"),
           ("txt","Say 2 short\nphrases a few times",10,GRAY,"normal"),
           ("btn","Start"),("txt","Skip for now",10,GRAY,"normal")],"1. Intro")
phone(5.0,[("txt","Say:",11,GRAY,"normal"),("txt",'"Hey Ozwell"',13,DARK,"bold"),
           ("txt","◉",30,GREEN,"bold"),("txt","● ● ○",16,GREEN,"bold"),
           ("txt","2 of 3",10,GRAY,"normal"),("txt","✓ got it",11,GREEN,"bold")],"2. Record")
phone(8.0,[("txt","✓",30,GREEN,"bold"),("txt","All set!",14,DARK,"bold"),
           ("txt","Ozwell now\nknows your voice",10,GRAY,"normal"),("btn","Done")],"3. Done")

ax.text(5,3.0,"skippable · redo anytime in Settings · gets better with use",
        ha="center",fontsize=11,color=GRAY,style="italic")
fig.savefig("enroll_ui.png",dpi=160,facecolor="white"); plt.close(fig); print("saved enroll_ui.png")
