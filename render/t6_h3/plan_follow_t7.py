#!/usr/bin/env python
"""How well an H3 render follows OUR plan (the Blender future video), not the real outcome.

Motion masks are made the Physics-IQ way (gray, 5x5 Gaussian blur, running-average background with weight 0.3,
|frame - background| > 10, 5x5 open + close) on both videos at a common 216x120, on the renders' 24 fps timeline (render
frame i <-> Blender frame round((i + 1) * 30 / 24) - 1). Per pair: spatial IoU of the two "ever moved" maps and the mean
per-frame IoU over frames where at least one video moves (frames where both are still are left out, so stillness earns
nothing). Reported per run as the mean over pairs.

    python t6_h3/plan_follow_t7.py ours_h3full_t7 h3only_t7          # runs under runs/

Writes review/plan_follow_t7.json (summary + per pair).
"""
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REASONING_RUN  # noqa: E402

OURS = REASONING_RUN / "T7"
SIZE = (216, 120)
KERNEL = np.ones((5, 5), np.uint8)


def frames(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.resize(f, SIZE, interpolation=cv2.INTER_AREA))
    return out


def masks(video):
    out, avg = [np.zeros(SIZE[::-1], bool)], None
    for f in video:
        g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if avg is None:
            avg = g.astype("float")
            continue
        cv2.accumulateWeighted(g, avg, 0.3)
        _, b = cv2.threshold(cv2.absdiff(g, cv2.convertScaleAbs(avg)), 10, 255, cv2.THRESH_BINARY)
        b = cv2.morphologyEx(cv2.morphologyEx(b, cv2.MORPH_OPEN, KERNEL), cv2.MORPH_CLOSE, KERNEL)
        out.append(b > 0)
    return np.stack(out)


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else None


def one(job):
    case, runs = job
    blender = frames(OURS / case / "video.mp4")
    if not blender:
        return case, None
    result = {}
    for run in runs:
        render = frames(ROOT / "runs" / run / "T7" / case / "video.mp4")
        if not render:
            continue
        aligned = [blender[min(max(round((i + 1) * 30 / 24) - 1, 0), len(blender) - 1)] for i in range(len(render))]
        mb, mr = masks(aligned), masks(render)
        per_frame = [v for v in (iou(x, y) for x, y in zip(mb, mr)) if v is not None]
        result[run] = {"spatial": iou(mb.any(0), mr.any(0)), "per_frame_moving": float(np.mean(per_frame)) if per_frame else None,
                       "moving_frames": len(per_frame)}
    return case, result


def main():
    runs = sys.argv[1:]
    cases = sorted(p.parent.name for p in OURS.glob("*/video.mp4"))
    with get_context("fork").Pool(8) as pool:
        results = dict(pool.map(one, [(c, runs) for c in cases]))
    summary = {}
    for run in runs:
        vals = [r[run] for r in results.values() if r and run in r]
        mean = lambda k: float(np.mean([v[k] for v in vals if v[k] is not None]))
        summary[run] = {"pairs": len(vals), "spatial_iou_vs_plan": round(mean("spatial"), 3),
                        "moving_frame_iou_vs_plan": round(mean("per_frame_moving"), 3)}
        print(f"{run:24s} pairs {len(vals):3d}  spatial IoU vs plan {summary[run]['spatial_iou_vs_plan']:.3f}  "
              f"per-frame IoU vs plan (moving frames) {summary[run]['moving_frame_iou_vs_plan']:.3f}")
    out = ROOT / "review" / "plan_follow_t7.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_pair": results}, indent=1, default=float))


if __name__ == "__main__":
    main()
