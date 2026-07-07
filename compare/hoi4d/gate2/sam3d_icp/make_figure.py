"""Diagnostic figure for the SAM-3D registration test (kettle_N15).
Four stacked panels over frame index: visible-depth error, surface-fit
residual, per-frame free scale (sim), visible object mask area (occlusion).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
C = {"rc": "#2a78d6", "rigid": "#1baf7a", "sim": "#eda100"}
LBL = {"rc": "RC pipeline", "rigid": "ICP rigid", "sim": "ICP free-scale"}

fair = json.load(open(os.path.join(OUT, "fair_metrics.json")))
res = json.load(open(os.path.join(OUT, "results.json")))
T = len(fair["rc"]["vis_cm"])
x = np.arange(T)

fig, axes = plt.subplots(4, 1, figsize=(9, 9), dpi=150)
fig.subplots_adjust(hspace=0.45, left=0.09, right=0.98, top=0.95, bottom=0.06)

ax = axes[0]
for k in ("rc", "rigid", "sim"):
    ax.plot(x, fair[k]["vis_cm"], color=C[k], lw=1.8, label=LBL[k])
ax.axhline(0, color="#999", lw=0.8, ls=":")
ax.set_title("Visible-surface depth error (mesh front − GT), cm", loc="left", fontsize=10)
ax.legend(loc="lower right", fontsize=8, ncol=3, frameon=False)

ax = axes[1]
for k in ("rc", "rigid", "sim"):
    ax.plot(x, fair[k]["fit_mm"], color=C[k], lw=1.8)
ax.set_title("GT cloud → posed mesh surface residual (median), mm", loc="left", fontsize=10)

ax = axes[2]
sc = 100 * (np.array(res["sim_scales"]) / np.mean(res["sim_scales"]) - 1)
ax.plot(x, sc, color=C["sim"], lw=1.8)
ax.axhline(0, color="#999", lw=0.8, ls=":")
ax.set_title("Per-frame free scale, % deviation from mean (sim variant)", loc="left", fontsize=10)

ax = axes[3]
# recompute area cheaply from stored per-frame nan pattern? use saved corr data:
# areas were not saved; reload masks
import cv2
RUN = "/workspace/code/hoi_recon/render_and_compare/runs/kettle_gt"
ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
areas = [int((cv2.erode(np.load(f"{RUN}/stage1_detect_track/masks/{t:05d}.npy")
                        .astype(np.uint8), ker) > 0).sum()) for t in range(T)]
ax.fill_between(x, 0, np.array(areas) / 1000, color="#c3c2b7", alpha=0.6)
ax.set_title("Visible object mask area (kpx) — occlusion proxy", loc="left", fontsize=10)
ax.set_xlabel("frame", fontsize=9)

for ax in axes:
    ax.grid(color="#eee", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, T - 1)

fig.savefig(os.path.join(OUT, "curves.png"))
print("saved", os.path.join(OUT, "curves.png"))
