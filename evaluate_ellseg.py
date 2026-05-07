"""
Pupil segmentation evaluation with EllSeg (ritnet_v2).

Evaluates cross-dataset pupil IoU on three datasets:
  - Our dataset  (400×400 eye videos + contour JSON annotations)
  - OpenEDS      (640×400 PNG images + class label .npy)
  - TEyeD        (384×288 video + pupil_seg_2D.mp4 annotation)

All images are preprocessed to 320×240 via dataset-specific transforms
defined in preprocess.py before being fed to EllSeg.
IoU is computed at 320×240.

Usage
-----
python evaluate_ellseg.py \
    --dataset ours \
    --video_dir  /path/to/our/dataset/Video \
    --weights    /path/to/EllSeg/weights/all.git_ok \
    --ellseg_dir /path/to/EllSeg \
    --output     results_ours.csv

python evaluate_ellseg.py \
    --dataset openeds \
    --data_dir  /path/to/openEDS \
    --weights   /path/to/EllSeg/weights/all.git_ok \
    --ellseg_dir /path/to/EllSeg \
    --output    results_openeds.csv

python evaluate_ellseg.py \
    --dataset teyed \
    --data_dir  /path/to/TEyeD/Dikablis \
    --weights   /path/to/EllSeg/weights/all.git_ok \
    --ellseg_dir /path/to/EllSeg \
    --output    results_teyed.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Import EllSeg model
# ---------------------------------------------------------------------------
def import_ellseg(ellseg_dir: str):
    sys.path.insert(0, ellseg_dir)
    from modelSummary import model_dict  # noqa: E402  (EllSeg repo)
    return model_dict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_W, TARGET_H = 320, 240
ELLSEG_PUPIL_CLASS = 2   # class index for pupil in EllSeg output
OPENEDS_PUPIL_CLASS = 3  # class index for pupil in OpenEDS labels


# ---------------------------------------------------------------------------
# Preprocessing  (see preprocess.py for full documentation)
# ---------------------------------------------------------------------------
def preprocess_ours(img: np.ndarray) -> np.ndarray:
    """400×400 → zero-pad to 533×400 → resize to 320×240."""
    h, w = img.shape[:2]
    padded_w = int(round(h * TARGET_W / TARGET_H))
    pad_left = (padded_w - w) // 2
    pad_right = padded_w - w - pad_left
    padded = np.pad(img, ((0, 0), (pad_left, pad_right)))
    return cv2.resize(padded, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)


def preprocess_openeds(img: np.ndarray, scleral_center=None) -> tuple[np.ndarray, tuple]:
    """640×400 → crop 400×300 around scleral center → resize to 320×240."""
    h, w = img.shape[:2]
    cx, cy = scleral_center if scleral_center is not None else (w // 2, h // 2)
    crop_w, crop_h = 400, 300
    x1 = max(0, min(cx - crop_w // 2, w - crop_w))
    y1 = max(0, min(cy - crop_h // 2, h - crop_h))
    cropped = img[y1:y1 + crop_h, x1:x1 + crop_w]
    resized = cv2.resize(cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    return resized, (x1, y1)


def preprocess_teyed(img: np.ndarray) -> np.ndarray:
    """384×288 → direct downsample to 320×240."""
    return cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)


def transform_mask_ours(mask: np.ndarray) -> np.ndarray:
    """Apply same spatial transform as preprocess_ours to a binary mask."""
    h, w = mask.shape[:2]
    padded_w = int(round(h * TARGET_W / TARGET_H))
    pad_left = (padded_w - w) // 2
    pad_right = padded_w - w - pad_left
    padded = np.pad(mask, ((0, 0), (pad_left, pad_right)))
    return cv2.resize(padded, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)


def transform_mask_openeds(label: np.ndarray, x1: int, y1: int) -> np.ndarray:
    """Crop and resize OpenEDS label map to binary pupil mask at 320×240."""
    crop = label[y1:y1 + 300, x1:x1 + 400]
    pupil = (crop == OPENEDS_PUPIL_CLASS).astype(np.uint8)
    return cv2.resize(pupil, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)


def transform_mask_teyed(mask: np.ndarray) -> np.ndarray:
    """Direct resize TEyeD binary mask to 320×240."""
    return cv2.resize(mask, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(weights_path: str, ellseg_dir: str, device: torch.device) -> torch.nn.Module:
    model_dict = import_ellseg(ellseg_dir)
    model = model_dict["ritnet_v2"]
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def predict(model: torch.nn.Module, img_320: np.ndarray, device) -> np.ndarray:
    """Run EllSeg on a (240, 320) grayscale image. Returns class map."""
    img_f = img_320.astype(np.float32)
    img_n = (img_f - img_f.mean()) / (img_f.std() + 1e-8)
    t = torch.from_numpy(img_n).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
    x4, x3, x2, x1, x = model.enc(t)
    seg = model.dec(x4, x3, x2, x1, x)
    return seg.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------
def pupil_iou(pred: np.ndarray, gt: np.ndarray) -> float | None:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union) if union > 0 else None


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

# --- Our dataset ---
def _load_contour_json(path: Path) -> dict | None:
    try:
        return json.load(open(path))
    except Exception:
        return None

def _index_contours(cj: dict) -> dict[int, np.ndarray]:
    by_frame = {}
    for entry in cj.get("frames", []):
        fid = entry.get("frame")
        cs = entry.get("contours") or []
        if fid is None or not cs:
            continue
        best = max(cs, key=lambda c: len(c.get("points") or []))
        pts = best.get("points") or []
        if len(pts) >= 3:
            by_frame[int(fid)] = np.asarray(pts, dtype=np.int32)
    return by_frame

_OURS_SESSION_RE = re.compile(r"^P(\d+)_(sunfacing|sunoccluded)$")

def iter_ours(video_dir: Path):
    """Yield (img_320, gt_320, meta) for every annotated frame in video_dir.

    Meta columns are emitted as p_num, cond, eye, frame_idx so the resulting
    CSV is directly compatible with the downstream analysis scripts
    (iou_by_area.py, iou_by_aspect_ratio.py, iou_by_brightness.py, ir_analysis.py).
    """
    for sess_dir in sorted(video_dir.iterdir()):
        if not sess_dir.is_dir():
            continue
        m = _OURS_SESSION_RE.match(sess_dir.name)
        if m is None:
            continue
        p_num, cond = int(m.group(1)), m.group(2)
        for eye in ("eye0", "eye1"):
            mp4 = sess_dir / f"{eye}.mp4"
            cj_path = sess_dir / f"{eye}_contours_reviewed.json"
            if not mp4.exists() or not cj_path.exists():
                continue
            cj = _load_contour_json(cj_path)
            if cj is None:
                continue
            contour_idx = _index_contours(cj)
            cap = cv2.VideoCapture(str(mp4))
            if not cap.isOpened():
                continue
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for fi, pts in sorted(contour_idx.items()):
                if fi >= n_frames:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, fr = cap.read()
                if not ok or fr is None:
                    continue
                if fr.ndim == 3:
                    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                h, w = fr.shape[:2]
                gt_raw = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(gt_raw, [pts], 1)
                img_320 = preprocess_ours(fr)
                gt_320 = transform_mask_ours(gt_raw)
                yield img_320, gt_320, {
                    "p_num": p_num,
                    "cond": cond,
                    "eye": eye,
                    "frame_idx": fi,
                }
            cap.release()


# --- OpenEDS ---
def iter_openeds(data_dir: Path):
    """Yield (img_320, gt_320, meta) for every image in data_dir/{split}/images."""
    for split in ("validation", "test"):
        img_dir = data_dir / split / "images"
        lbl_dir = data_dir / split / "labels"
        if not img_dir.exists():
            continue
        for png in sorted(img_dir.iterdir()):
            if png.suffix != ".png":
                continue
            npy = lbl_dir / (png.stem + ".npy")
            if not npy.exists():
                continue
            img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            lbl = np.load(str(npy))
            if img is None or img.shape != lbl.shape:
                continue
            img_320, (x1, y1) = preprocess_openeds(img)
            gt_320 = transform_mask_openeds(lbl, x1, y1)
            yield img_320, gt_320, {"split": split, "file_id": png.stem}


# --- TEyeD ---
def iter_teyed(data_dir: Path):
    """Yield (img_320, gt_320, meta) for every frame with non-empty pupil mask."""
    vid_dir = data_dir / "VIDEOS"
    ann_dir = data_dir / "ANNOTATIONS"
    if not vid_dir.exists():
        return
    for v_path in sorted(vid_dir.iterdir()):
        seg_path = ann_dir / f"{v_path.name}pupil_seg_2D.mp4"
        if not seg_path.exists():
            continue
        cap = cv2.VideoCapture(str(v_path))
        seg_cap = cv2.VideoCapture(str(seg_path))
        if not cap.isOpened() or not seg_cap.isOpened():
            cap.release(); seg_cap.release()
            continue
        n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(seg_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        for fi in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            seg_cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok1, fr = cap.read()
            ok2, sg = seg_cap.read()
            if not ok1 or not ok2:
                continue
            if fr.ndim == 3:
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if sg.ndim == 3:
                sg = cv2.cvtColor(sg, cv2.COLOR_BGR2GRAY)
            gt_raw = (sg >= 128).astype(np.uint8)
            if gt_raw.sum() == 0:
                continue
            img_320 = preprocess_teyed(fr)
            gt_320 = transform_mask_teyed(gt_raw)
            yield img_320, gt_320, {"video": v_path.name, "frame_idx": fi}
        cap.release(); seg_cap.release()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate EllSeg pupil segmentation IoU on a single dataset."
    )
    ap.add_argument("--dataset", required=True, choices=["ours", "openeds", "teyed"])
    ap.add_argument("--data_dir",  type=str, default=None,
                    help="Root directory for openeds or teyed dataset.")
    ap.add_argument("--video_dir", type=str, default=None,
                    help="Root Video/ directory for our dataset.")
    ap.add_argument("--weights",   type=str, required=True,
                    help="Path to EllSeg weight file (e.g. weights/all.git_ok).")
    ap.add_argument("--ellseg_dir", type=str, required=True,
                    help="Path to the EllSeg repository root.")
    ap.add_argument("--output",    type=str, default="results.csv",
                    help="Output CSV path.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device: {device}")

    model = load_model(args.weights, args.ellseg_dir, device)
    print("[*] EllSeg loaded")

    if args.dataset == "ours":
        assert args.video_dir, "--video_dir required for ours"
        data_iter = iter_ours(Path(args.video_dir))
    elif args.dataset == "openeds":
        assert args.data_dir, "--data_dir required for openeds"
        data_iter = iter_openeds(Path(args.data_dir))
    else:
        assert args.data_dir, "--data_dir required for teyed"
        data_iter = iter_teyed(Path(args.data_dir))

    rows = []
    for i, (img_320, gt_320, meta) in enumerate(data_iter):
        pred = predict(model, img_320, device)
        pred_pupil = (pred == ELLSEG_PUPIL_CLASS).astype(np.uint8)
        iou = pupil_iou(pred_pupil, gt_320)
        if iou is None:
            continue
        meta["iou"] = iou
        rows.append(meta)
        if (i + 1) % 500 == 0:
            print(f"  {i+1} frames processed ...")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"[*] {len(df)} frames -> mean IoU = {df['iou'].mean():.4f}")
    print(f"[*] saved: {args.output}")


if __name__ == "__main__":
    main()
