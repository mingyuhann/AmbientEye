"""
IoU vs. Local Brightness & Solar Altitude — Three-panel figure
==============================================================
(A) IoU vs. whole-frame mean brightness   : Sun-facing vs. Sun-occluded
(B) IoU vs. solar altitude (binned)       : Sun-facing vs. Sun-occluded
(C) IoU vs. GT pupil brightness           : Lateral (eye1) vs. Medial (eye0)
    (pred_brightness column = mean pixel inside EllSeg-predicted pupil at 320×240)

Data source: brightness_iou.csv (output of evaluate_brightness_iou.py)
  - All contour-annotated frames, sub-sampled
  - columns: p_num, cond, eye, frame_idx, iou, brightness, pred_brightness

Output: <OUTPUT_DIR>/iou_by_brightness.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

BRIGHTNESS_CSV = Path("")   # brightness_iou.csv from evaluate_brightness_iou.py
SOLAR_REF_PATH = Path("")   # solar_position_reference.md from dataset
OUTPUT_DIR     = Path("")   # output directory for figures

ALTITUDE_BINS  = np.arange(15, 91, 15)

mpl.use("Agg")
_BASE = 28
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": _BASE, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

C_FACE = "#C0392B"; C_AWAY = "#2471A3"
C_LAT  = "#8E44AD"; C_MED  = "#E67E22"
FS = _BASE; FS_LEG = _BASE-3; FS_TICK = _BASE-2
FS_ANN = _BASE-4; FS_LABEL = int(_BASE*1.5)


def load_solar_altitude():
    txt = SOLAR_REF_PATH.read_text()
    mapping = {}
    in_table = False
    for line in txt.split("\n"):
        if "| Participant" in line: in_table = True; continue
        if not in_table: continue
        if not line.startswith("|"): 
            if line.strip(): break
            continue
        if "---" in line: continue
        cols = [c.strip() for c in line.split("|")[1:-1]]
        if len(cols) < 6: continue
        try:
            p_num = int(cols[0].lstrip("P"))
            cond  = cols[1]  # "sunfacing" or "sunoccluded"
            mapping[(p_num, cond)] = float(cols[5])
        except (ValueError, IndexError):
            continue
    return mapping


def bin_iou(df, col, bins):
    df = df.dropna(subset=[col, "iou"])
    centres, means, stds = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = df[(df[col] >= lo) & (df[col] < hi)]["iou"]
        if len(sub) < 5: continue
        centres.append((lo + hi) / 2)
        means.append(sub.mean()); stds.append(sub.std())
    return np.array(centres), np.array(means), np.array(stds)


def draw_line_panel(ax, datasets, col, bins, xlabel, title):
    for label, df, color, marker, ls in datasets:
        ctrs, means, stds = bin_iou(df, col, bins)
        if len(ctrs) == 0: continue
        ax.plot(ctrs, means, color=color, ls=ls, lw=2,
                marker=marker, markersize=8, label=label, zorder=4)
        ax.fill_between(ctrs, means-stds, means+stds, alpha=0.10, color=color)
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel("Mean IoU", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, lw=0.4, ls="--", color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=FS_LEG, loc="lower right")
    ax.set_title(title, fontsize=FS_LABEL, loc="left", fontweight="bold")


def main():
    print("[1] Loading brightness_iou.csv ...")
    df = pd.read_csv(BRIGHTNESS_CSV)
    print(f"    {len(df):,} frames  |  "
          f"sunfacing={len(df[df.cond=='sunfacing']):,}  sunoccluded={len(df[df.cond=='sunoccluded']):,}")

    print("[2] Loading solar altitude ...")
    solar = load_solar_altitude()
    df["altitude"] = df.apply(lambda r: solar.get((int(r["p_num"]), r["cond"]), np.nan), axis=1)

    face_df = df[df["cond"] == "sunfacing"]
    away_df = df[df["cond"] == "sunoccluded"]
    lat_df  = df[df["eye"] == "eye1"]
    med_df  = df[df["eye"] == "eye0"]

    # Panel A bins: whole-frame brightness
    br_lo = int(df["brightness"].min() // 10) * 10
    br_hi = int(df["brightness"].max() // 10 + 1) * 10
    br_bins = np.arange(br_lo, br_hi + 10, 10)
    print(f"    Brightness range: {br_lo}–{br_hi}  ({len(br_bins)-1} bins)")

    # Panel C bins: GT pupil brightness (pred_brightness column)
    pb = df["pred_brightness"].dropna()
    pb_lo = int(pb.min() // 10) * 10
    pb_hi = int(pb.max() // 10 + 1) * 10
    pb_bins = np.arange(pb_lo, pb_hi + 10, 10)
    print(f"    Pred-brightness range: {pb_lo}–{pb_hi}  ({len(pb_bins)-1} bins)")

    print("[3] Plotting ...")
    fig, axes = plt.subplots(1, 3, figsize=(36, 9))
    fig.subplots_adjust(wspace=0.45, left=0.07, right=0.98, top=0.88, bottom=0.22)

    draw_line_panel(axes[0], [
        ("Sun-facing",   face_df, C_FACE, "o", "-"),
        ("Sun-occluded", away_df, C_AWAY, "s", "--"),
    ], col="brightness", bins=br_bins,
    xlabel="Mean Frame Brightness (pixel value)",
    title="(A)  IoU vs. Frame Brightness")

    draw_line_panel(axes[1], [
        ("Sun-facing",   face_df, C_FACE, "o", "-"),
        ("Sun-occluded", away_df, C_AWAY, "s", "--"),
    ], col="altitude", bins=ALTITUDE_BINS,
    xlabel="Solar Altitude (°)",
    title="(B)  IoU vs. Solar Altitude")

    draw_line_panel(axes[2], [
        ("Lateral", lat_df, C_LAT, "o", "-"),
        ("Medial",  med_df, C_MED, "s", "--"),
    ], col="pred_brightness", bins=pb_bins,
    xlabel="GT Pupil Brightness (pixel value)",
    title="(C)  Lateral vs. Medial")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in [(".pdf", 300), (".png", 200)]:
        out = OUTPUT_DIR / f"iou_by_brightness{suffix}"
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
        print(f"[saved] {out}")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
