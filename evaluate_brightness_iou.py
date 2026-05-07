"""
Evaluate EllSeg IoU + brightness on ALL contour-annotated frames.

Iterates every session in Video/, every annotated frame in contours_reviewed.json,
runs EllSeg, computes IoU, whole-frame mean brightness, and mean brightness
inside the EllSeg-predicted pupil region.

Output CSV columns: p_num, cond, eye, frame_idx, iou, brightness, pred_brightness
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import cv2
import numpy as np
import torch

OURS_VIDEO_DIR = Path("")   # Our dataset Video/ directory (contains P{N}_{cond}/ sessions)
ELLSEG_DIR     = Path("")   # EllSeg repository root
WEIGHTS_PATH   = ELLSEG_DIR / "weights" / "all.git_ok"
OUTPUT_CSV     = Path("")   # output CSV path
TARGET_W, TARGET_H = 320, 240
ELLSEG_PUPIL_CLASS = 2

def load_model(device):
    sys.path.insert(0, str(ELLSEG_DIR))
    from modelSummary import model_dict
    model = model_dict["ritnet_v2"]
    ckpt = torch.load(str(WEIGHTS_PATH), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()

@torch.no_grad()
def predict(model, img_320, device):
    img_f = img_320.astype(np.float32)
    img_n = (img_f - img_f.mean()) / (img_f.std() + 1e-8)
    t = torch.from_numpy(img_n).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
    x4,x3,x2,x1,x = model.enc(t)
    seg = model.dec(x4,x3,x2,x1,x)
    return seg.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

def preprocess(img):
    h, w = img.shape[:2]
    pw = int(round(h * TARGET_W / TARGET_H))
    pl = (pw - w) // 2; pr = pw - w - pl
    return cv2.resize(np.pad(img, ((0,0),(pl,pr))), (TARGET_W,TARGET_H), interpolation=cv2.INTER_AREA)

def transform_mask(mask):
    h, w = mask.shape[:2]
    pw = int(round(h * TARGET_W / TARGET_H))
    pl = (pw - w) // 2; pr = pw - w - pl
    return cv2.resize(np.pad(mask, ((0,0),(pl,pr))), (TARGET_W,TARGET_H), interpolation=cv2.INTER_NEAREST)

def pupil_iou(pred, gt):
    inter = np.logical_and(pred.astype(bool), gt.astype(bool)).sum()
    union = np.logical_or(pred.astype(bool), gt.astype(bool)).sum()
    return float(inter)/float(union) if union > 0 else None

def contour_to_mask(pts, h=400, w=400):
    mask = np.zeros((h,w), dtype=np.uint8)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts.reshape(-1,1,2).astype(np.int32)], 1)
    return mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print("[1] Loading model ...")
    model = load_model(device)

    sessions = sorted([d for d in OURS_VIDEO_DIR.iterdir()
                       if d.is_dir() and re.match(r'P\d+_(sunfacing|sunoccluded)$', d.name)])
    print(f"[2] Processing {len(sessions)} sessions ...")

    # Resume: skip already-completed (p_num, cond, eye) triples
    import csv as _csv
    done_triples = set()
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV) as _f:
            for _r in _csv.DictReader(_f):
                done_triples.add((_r["p_num"], _r["cond"], _r["eye"]))
        print(f"    Resuming: {len(done_triples)} (p_num,cond,eye) already done")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    need_header = (not OUTPUT_CSV.exists()) or OUTPUT_CSV.stat().st_size == 0
    out_f = open(OUTPUT_CSV, "a")
    if need_header:
        out_f.write("p_num,cond,eye,frame_idx,iou,brightness,pred_brightness\n")

    total_written = 0
    for si, sess_dir in enumerate(sessions):
        m = re.match(r'P(\d+)_(sunfacing|sunoccluded)$', sess_dir.name)
        p_num, cond = int(m.group(1)), m.group(2)
        print(f"[{si+1}/{len(sessions)}] {sess_dir.name}", flush=True)

        for eye in ("eye0", "eye1"):
            if (str(p_num), cond, eye) in done_triples:
                print(f"  skipping {eye} (already done)", flush=True)
                continue
            cj_path = sess_dir / f"{eye}_contours_reviewed.json"
            vid_path = sess_dir / f"{eye}.mp4"
            if not cj_path.exists() or not vid_path.exists():
                continue

            cj = json.loads(cj_path.read_text())
            frame_map = {}
            for entry in cj.get("frames", []):
                fid = entry.get("frame")
                cs = entry.get("contours") or []
                if fid is None or not cs: continue
                best = max(cs, key=lambda c: len(c.get("points") or []))
                pts = best.get("points") or []
                if len(pts) >= 3:
                    frame_map[int(fid)] = np.array(pts, dtype=np.int32)

            if not frame_map: continue
            cap = cv2.VideoCapture(str(vid_path))
            if not cap.isOpened(): continue

            SAMPLE_EVERY = 30  # 1 frame per 30 annotated frames
            n_frames = len(frame_map)
            for fi, (frame_idx, pts) in enumerate(sorted(frame_map.items())):
                if fi % SAMPLE_EVERY != 0:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None: continue
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                img_320 = preprocess(frame)
                pred = predict(model, img_320, device)
                pred_bin = (pred == ELLSEG_PUPIL_CLASS).astype(np.uint8)

                gt_mask = contour_to_mask(pts)
                gt_320 = transform_mask(gt_mask)

                iou = pupil_iou(pred_bin, gt_320)
                if iou is None: continue

                br = float(frame.mean())
                pb_pixels = img_320.astype(np.float32)[pred_bin == 1]
                pb_str = f"{pb_pixels.mean():.2f}" if pb_pixels.size > 0 else ""

                out_f.write(f"{p_num},{cond},{eye},{frame_idx},{iou:.6f},{br:.2f},{pb_str}\n")
                total_written += 1

                if total_written % 10000 == 0:
                    out_f.flush()
                    print(f"  written {total_written:,} frames so far", flush=True)

            cap.release()

    out_f.close()
    print(f"\nDone. Total written: {total_written:,}")
    print(f"[saved] {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
