"""
IoU vs. Pupil Area — Three-panel figure
========================================
(A) All datasets  : OpenEDS, TEyeD, Combined, Ours (balanced 2,376)
(B) Off-axis      : Ours Lateral (eye1) vs. Medial (eye0), each 2,376
(C) Sun condition : Ours Facesun vs. Awaysun, each 2,376

Output: <OUTPUT_DIR>/iou_by_area.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import OPENEDS_NORM_AREA, TEYED_NORM_AREA, OURS_NORM_AREA

# ============================================================================
#  CONFIG
# ============================================================================

OPENEDS_ROOT    = Path("")   # OpenEDS dataset root (with train/validation/test sub-dirs)
TEYED_ROOT      = Path("")   # TEyeD dataset root (with ANNOTATIONS/ and VIDEOS/ sub-dirs)
OURS_VIDEO_DIR  = Path("")   # Our dataset Video/ directory (P{N}_{cond}/ sessions)

ELLSEG_OPENEDS  = Path("")   # openeds.csv from evaluate_ellseg.py
ELLSEG_TEYED    = Path("")   # teyed.csv from evaluate_ellseg.py
ELLSEG_OURS_DIR = Path("")   # directory containing ours_sunfacing_eye0.csv etc.
OURS_GROUPS     = ["ours_sunfacing_eye0", "ours_sunfacing_eye1",
                   "ours_sunoccluded_eye0", "ours_sunoccluded_eye1"]

OUTPUT_DIR      = Path("")   # output directory for figures
TARGET_N        = 2376

# ============================================================================
#  Style
# ============================================================================

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

C_OPENEDS = "#2980B9"
C_TEYED   = "#27AE60"
C_BOTH    = "#555555"
C_OURS    = "#C0392B"
C_LAT     = "#8E44AD"
C_MED     = "#E67E22"
C_FACE    = "#C0392B"
C_AWAY    = "#2471A3"


# ============================================================================
#  GT stats  (identical to iou_by_aspect_ratio.py)
# ============================================================================

def gt_stats_openeds(iou_df):
    lbl_roots = {s: OPENEDS_ROOT / s / "labels" for s in ("train", "validation", "test")}
    ratios, areas = [], []
    for row in iou_df.itertuples():
        fid = str(row.file_id).zfill(6)
        try:
            lbl = np.load(str(lbl_roots[row.split] / f"{fid}.npy"))
        except Exception:
            ratios.append(np.nan); areas.append(np.nan); continue
        mask = (lbl == 3).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts: ratios.append(np.nan); areas.append(np.nan); continue
        cnt = max(cnts, key=cv2.contourArea)
        if len(cnt) < 5: ratios.append(np.nan); areas.append(np.nan); continue
        try: (_, _), (d1, d2), _ = cv2.fitEllipse(cnt.astype(np.float32))
        except cv2.error: ratios.append(np.nan); areas.append(np.nan); continue
        ma, mb = max(d1, d2), min(d1, d2)
        ratios.append(mb / ma)
        areas.append(np.pi * (ma / 2) * (mb / 2) / OPENEDS_NORM_AREA * 100)
    df = iou_df.copy(); df["aspect_ratio"] = ratios; df["area_pct"] = areas
    return df


def gt_stats_teyed(iou_df):
    annot_dir = TEYED_ROOT / "ANNOTATIONS"

    def load_eli(video_name):
        table = {}
        eli_path = annot_dir / f"{video_name}pupil_eli.txt"
        if not eli_path.exists(): return table
        for line in eli_path.read_text().splitlines()[1:]:
            p = line.strip().split(";")
            if len(p) < 6: continue
            try:
                fi = int(p[0]); ew, eh = float(p[4]), float(p[5])
                table[fi] = (max(ew, eh), min(ew, eh))
            except ValueError: pass
        return table

    def from_seg(seg_cap, fidx):
        seg_cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, fr = seg_cap.read()
        if not ok or fr is None: return None
        if fr.ndim == 3: fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        mask = (fr >= 128).astype(np.uint8)
        if mask.sum() < 80: return None
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts: return None
        cnt = max(cnts, key=cv2.contourArea)
        if len(cnt) < 5: return None
        try: (_, _), (d1, d2), _ = cv2.fitEllipse(cnt.astype(np.float32))
        except cv2.error: return None
        ma, mb = max(d1, d2), min(d1, d2)
        return (ma, mb) if ma >= 1.0 else None

    ratios, areas = [], []
    prev_video, eli_table, seg_cap = None, {}, None
    for row in iou_df.itertuples():
        if row.video != prev_video:
            if seg_cap: seg_cap.release()
            eli_table = load_eli(row.video)
            seg_path  = annot_dir / f"{row.video}pupil_seg_2D.mp4"
            seg_cap   = cv2.VideoCapture(str(seg_path)) if seg_path.exists() else None
            prev_video = row.video
        entry = eli_table.get(int(row.frame_idx))
        if entry is None and seg_cap: entry = from_seg(seg_cap, int(row.frame_idx))
        if entry is None: ratios.append(np.nan); areas.append(np.nan); continue
        ma, mb = entry
        ratios.append(mb / ma)
        areas.append(np.pi * (ma / 2) * (mb / 2) / TEYED_NORM_AREA * 100)
    if seg_cap: seg_cap.release()
    df = iou_df.copy(); df["aspect_ratio"] = ratios; df["area_pct"] = areas
    return df


def gt_stats_ours(iou_df):
    ratios, areas = [], []
    cj_cache = {}
    for row in iou_df.itertuples():
        p_num, cond, eye, fi = int(row.p_num), row.cond, row.eye, int(row.frame_idx)
        key = f"P{p_num}_{cond}_{eye}"
        if key not in cj_cache:
            cj_path = OURS_VIDEO_DIR / f"P{p_num}_{cond}" / f"{eye}_contours_reviewed.json"
            try:
                cj = json.loads(cj_path.read_text())
                cj_cache[key] = {int(e["frame"]): e for e in cj.get("frames", [])}
            except Exception:
                cj_cache[key] = {}
        entry = cj_cache[key].get(fi)
        if not entry: ratios.append(np.nan); areas.append(np.nan); continue
        cs = entry.get("contours") or []
        if not cs: ratios.append(np.nan); areas.append(np.nan); continue
        best = max(cs, key=lambda c: len(c.get("points") or []))
        pts_raw = best.get("points")
        if not pts_raw or len(pts_raw) < 5: ratios.append(np.nan); areas.append(np.nan); continue
        pts = np.array(pts_raw, dtype=np.float32).reshape(-1, 1, 2)
        try: (_, _), (d1, d2), _ = cv2.fitEllipse(pts)
        except cv2.error: ratios.append(np.nan); areas.append(np.nan); continue
        ma, mb = max(d1, d2), min(d1, d2)
        if ma < 2.0: ratios.append(np.nan); areas.append(np.nan); continue
        ratios.append(mb / ma)
        areas.append(np.pi * (ma / 2) * (mb / 2) / OURS_NORM_AREA * 100)
    df = iou_df.copy(); df["aspect_ratio"] = ratios; df["area_pct"] = areas
    return df


# ============================================================================
#  Bin helper
# ============================================================================

def bin_iou(df, col, bins):
    df = df.dropna(subset=[col, "iou"])
    centres, means, stds, counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = df[(df[col] >= lo) & (df[col] < hi)]["iou"]
        if len(sub) < 5: continue
        centres.append((lo + hi) / 2)
        means.append(sub.mean()); stds.append(sub.std()); counts.append(len(sub))
    return np.array(centres), np.array(means), np.array(stds), np.array(counts, dtype=int)


# ============================================================================
#  Figure
# ============================================================================

def plot_figure(oe, te, ou_bal, ou_lat, ou_med, ou_face, ou_away, output_dir):
    combined = pd.concat([oe, te], ignore_index=True)
    BINS     = np.arange(0, 2.25, 0.25)
    FS       = _BASE
    FS_TICK  = _BASE - 2
    FS_ANN   = _BASE - 5
    FS_LABEL = int(_BASE * 1.5)

    fig, axes = plt.subplots(1, 3, figsize=(36, 9))
    fig.subplots_adjust(wspace=0.45, left=0.07, right=0.98, top=0.88, bottom=0.22)

    def draw(ax, datasets, title):
        for label, df, color, marker, ls in datasets:
            ctrs, means, stds, _ = bin_iou(df, "area_pct", BINS)
            if len(ctrs) == 0: continue
            ax.plot(ctrs, means, color=color, ls=ls, lw=2, marker=marker,
                    markersize=8, label=label, zorder=4)
            ax.fill_between(ctrs, means - stds, means + stds, alpha=0.10, color=color)
        ax.set_xlabel("Pupil Area  (% of image area)", fontsize=FS_LABEL)
        ax.set_ylabel("Mean IoU", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlim(0, 2.2); ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, lw=0.4, ls="--", color="#cccccc", zorder=0)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=FS_ANN, loc="lower right")
        ax.set_title(title, fontsize=FS_LABEL, loc="left", fontweight="bold")

    draw(axes[0], [
        ("OpenEDS", oe,       C_OPENEDS, "o", "--"),
        ("TEyeD",   te,       C_TEYED,   "s", "-."),
        ("Combined", combined, C_BOTH,   "^", ":"),
        ("Ours",    ou_bal,   C_OURS,    "D", "-"),
    ], "(A)  IoU vs. Pupil Area")

    draw(axes[1], [
        ("Lateral", ou_lat, C_LAT, "o", "-"),
        ("Medial",  ou_med, C_MED, "s", "--"),
    ], "(B)  Lateral vs. Medial")

    draw(axes[2], [
        ("Sun-facing",  ou_face, C_FACE, "o", "-"),
        ("Sun-occluded", ou_away, C_AWAY, "s", "--"),
    ], "(C)  Sun-facing vs. Sun-occluded")

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in [(".pdf", 300), (".png", 200)]:
        out = output_dir / f"iou_by_area{suffix}"
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
        print(f"[saved] {out}")
    plt.close()


# ============================================================================
#  Main
# ============================================================================

def load_and_enrich():
    print("[1] Loading IoU CSVs ...")
    oe_iou = pd.read_csv(ELLSEG_OPENEDS)
    te_iou = pd.read_csv(ELLSEG_TEYED)
    group_dfs = {g: pd.read_csv(ELLSEG_OURS_DIR / f"{g}.csv") for g in OURS_GROUPS}

    print("[2] OpenEDS GT stats ...")
    oe = gt_stats_openeds(oe_iou)
    print(f"    valid: {oe['aspect_ratio'].notna().sum():,}")

    print("[3] TEyeD GT stats ...")
    te = gt_stats_teyed(te_iou)
    print(f"    valid: {te['aspect_ratio'].notna().sum():,}")

    print("[4] Ours GT stats ...")
    enriched = {}
    for g, df in group_dfs.items():
        enriched[g] = gt_stats_ours(df)
        print(f"    {g}: {enriched[g]['aspect_ratio'].notna().sum():,}")

    n_each = TARGET_N // len(OURS_GROUPS)
    ou_bal = pd.concat([
        enriched[g].dropna(subset=["aspect_ratio","area_pct"]).sample(
            n=min(n_each, len(enriched[g].dropna(subset=["aspect_ratio","area_pct"]))),
            random_state=42)
        for g in OURS_GROUPS], ignore_index=True)

    def merge_eye(eye_tag):
        parts = []
        for cond in ("sunfacing", "sunoccluded"):
            g = f"ours_{cond}_{eye_tag}"
            part = enriched[g].dropna(subset=["aspect_ratio","area_pct"])
            parts.append(part)
        return pd.concat(parts, ignore_index=True)

    def merge_cond(cond_tag):
        parts = []
        for eye in ("eye0", "eye1"):
            g = f"ours_{cond_tag}_{eye}"
            part = enriched[g].dropna(subset=["aspect_ratio","area_pct"])
            parts.append(part)
        return pd.concat(parts, ignore_index=True)

    ou_lat  = merge_eye("eye1")
    ou_med  = merge_eye("eye0")
    ou_face = merge_cond("sunfacing")
    ou_away = merge_cond("sunoccluded")

    print(f"\n    Balanced (A): {len(ou_bal):,}  "
          f"Lateral (B): {len(ou_lat):,}  Medial (B): {len(ou_med):,}  "
          f"Facesun (C): {len(ou_face):,}  Awaysun (C): {len(ou_away):,}")
    return oe, te, ou_bal, ou_lat, ou_med, ou_face, ou_away


def main():
    data = load_and_enrich()
    print("\n[5] Plotting ...")
    plot_figure(*data, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
