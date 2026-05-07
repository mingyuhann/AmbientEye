"""
IR Irradiance Analysis Figure
==============================
(A) Number of evaluation frames per local brightness bin
    (sun-facing vs. sun-occluded), from brightness_iou.csv

(B) Pupil IoU across datasets and conditions
    Bar chart: OpenEDS | TEyeD | Ours sunfacing/sunoccluded × medial/lateral

Output: <FIGURE_OUT>/ir_analysis.{png,pdf}
Usage:  python ir_analysis.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ELLSEG_OUT  = Path("")   # directory containing brightness_iou.csv, openeds.csv, teyed.csv, ours_*.csv
FIGURE_OUT  = Path("")   # output directory for figures
OURS_GROUPS = ["ours_sunfacing_eye0", "ours_sunfacing_eye1",
               "ours_sunoccluded_eye0", "ours_sunoccluded_eye1"]

BRIGHTNESS_CSV = ELLSEG_OUT / "brightness_iou.csv"

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

C_FACE    = "#C0392B"
C_AWAY    = "#2471A3"
C_OPENEDS = "#2980B9"
C_TEYED   = "#27AE60"
C_LAT     = "#8E44AD"
C_MED     = "#E67E22"
FS_LABEL  = int(_BASE * 1.5)
TARGET_N  = 2376


def load_ours_mean_iou():
    """Return mean IoU per group using balanced 2,376 sample."""
    results = {}
    for g in OURS_GROUPS:
        path = ELLSEG_OUT / f"{g}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        # balanced sample: TARGET_N // 4 per group
        n = TARGET_N // len(OURS_GROUPS)
        sample = df.sample(n=min(n, len(df)), random_state=42)
        results[g] = (sample["iou"].mean(), sample["iou"].std())
    return results


def main():
    fig, axes = plt.subplots(1, 2, figsize=(24, 9))
    fig.subplots_adjust(wspace=0.45, left=0.08, right=0.98, top=0.88, bottom=0.20)

    # ── Panel A: Frame count per brightness bin ───────────────────────────────
    ax = axes[0]
    if BRIGHTNESS_CSV.exists():
        df = pd.read_csv(BRIGHTNESS_CSV)
        bins = np.arange(0, 261, 10)
        face_df = df[df["cond"] == "sunfacing"]
        away_df = df[df["cond"] == "sunoccluded"]
        ax.hist(face_df["brightness"].clip(0, 250), bins=bins,
                color=C_FACE, alpha=0.65, label="Sun-facing", zorder=3)
        ax.hist(away_df["brightness"].clip(0, 250), bins=bins,
                color=C_AWAY, alpha=0.65, label="Sun-occluded", zorder=3)
        ax.set_xlabel("Local Mean Brightness (pixel value)", fontsize=FS_LABEL)
        ax.set_ylabel("Number of Frames", fontsize=FS_LABEL)
        ax.tick_params(labelsize=_BASE - 2)
        ax.yaxis.grid(True, lw=0.4, ls="--", color="#cccccc", zorder=0)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=_BASE - 3)
        ax.set_title("(A)  Frames per IR Intensity Bin", fontsize=FS_LABEL,
                     loc="left", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "brightness_iou.csv not ready",
                ha="center", va="center", transform=ax.transAxes)

    # ── Panel B: Mean IoU per dataset / condition ─────────────────────────────
    ax = axes[1]

    # Reference datasets
    bars = []
    openeds_path = ELLSEG_OUT / "openeds.csv"
    teyed_path   = ELLSEG_OUT / "teyed.csv"

    if openeds_path.exists():
        df = pd.read_csv(openeds_path)
        bars.append(("OpenEDS",        df["iou"].mean(), df["iou"].std(), C_OPENEDS))
    if teyed_path.exists():
        df = pd.read_csv(teyed_path)
        bars.append(("TEyeD",          df["iou"].mean(), df["iou"].std(), C_TEYED))

    # Ours: balanced sample per group
    ours = load_ours_mean_iou()
    label_map = {
        "ours_sunfacing_eye0":  ("Sunfacing\nMedial",   C_FACE),
        "ours_sunfacing_eye1":  ("Sunfacing\nLateral",  C_FACE),
        "ours_sunoccluded_eye0":("Sunoccluded\nMedial", C_AWAY),
        "ours_sunoccluded_eye1":("Sunoccluded\nLateral",C_AWAY),
    }
    for g, (lbl, color) in label_map.items():
        if g in ours:
            mean, std = ours[g]
            bars.append((lbl, mean, std, color))

    if bars:
        labels = [b[0] for b in bars]
        means  = [b[1] for b in bars]
        stds   = [b[2] for b in bars]
        colors = [b[3] for b in bars]
        x = np.arange(len(bars))

        rects = ax.bar(x, means, color=colors, alpha=0.85,
                       edgecolor="white", linewidth=0.6, zorder=3)
        ax.errorbar(x, means, yerr=stds,
                    fmt="none", color="#333333", capsize=4, lw=1.5, zorder=4)
        for rect, m in zip(rects, means):
            ax.text(rect.get_x() + rect.get_width() / 2, m + 0.02,
                    f"{m:.3f}", ha="center", va="bottom",
                    fontsize=_BASE - 6, color="#333333")

        # Separator between reference and ours
        n_ref = sum(1 for b in bars if b[0] in ("OpenEDS", "TEyeD"))
        if n_ref > 0 and len(bars) > n_ref:
            ax.axvline(n_ref - 0.5, color="#999999", lw=1.0, ls="--", zorder=2)
            ax.text(n_ref - 0.5, 1.02, "Ours →",
                    ha="left", va="bottom", fontsize=_BASE - 6,
                    color="#555555", transform=ax.get_xaxis_transform())

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=_BASE - 4)
        ax.set_ylabel("Mean IoU", fontsize=FS_LABEL)
        ax.set_ylim(0, 1.12)
        ax.yaxis.grid(True, lw=0.4, ls="--", color="#cccccc", zorder=0)
        ax.set_axisbelow(True)
        ax.set_title("(B)  Pupil IoU Across Datasets and Conditions",
                     fontsize=FS_LABEL, loc="left", fontweight="bold")

    FIGURE_OUT.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in [(".png", 200), (".pdf", 300)]:
        out = FIGURE_OUT / f"ir_analysis{suffix}"
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
        print(f"[saved] {out}")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
