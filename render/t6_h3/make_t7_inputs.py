#!/usr/bin/env python3
"""T7 inputs for MiniMax-H3. For every pair whose reasoning run left a scene and a prediction
(<run>/T7/<id>/work/final.blend and video.mp4), writes inputs/T7/<id>/:

  last_observed.png   the last frame of the 3 s conditioning clip: H3's keyframe at frame 0
  case.json           id, category, the Physics-IQ description, and the observed / predicted frame range of the scene

    python t6_h3/make_t7_inputs.py        # the run is $VIDEOCOS_REASONING_RUN (default ../reasoning/runs/full_v1)
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from common import REASONING_RUN, bench_path, samples

ROOT = Path(__file__).resolve().parents[1]
OURS = REASONING_RUN / "T7"
FUTURE = 150                           # 5 s at the scene's 30 fps


def last_frame(path):
    cap = cv2.VideoCapture(str(path))
    last, n = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last, n = frame, n + 1
    return last, n


def one(sample):
    case = sample["id"]
    if not (OURS / case / "work" / "final.blend").is_file() or not (OURS / case / "video.mp4").is_file():
        return f"{case}: no scene/video, skipped"
    d = ROOT / "inputs" / "T7" / case
    if (d / "last_observed.png").is_file() and (d / "case.json").is_file():
        return f"{case}: exists"
    d.mkdir(parents=True, exist_ok=True)
    frame, observed = last_frame(bench_path(sample["input"]["video"]))   # 90 frames = 3 s at 30 fps
    cv2.imwrite(str(d / "last_observed.png"), frame)
    description = sample["input"]["description"]
    (d / "case.json").write_text(json.dumps({
        "id": case, "category": sample["category"], "subgroup": sample["subgroup"], "perspective": sample["perspective"],
        "description": description, "prompt": "Fixed Camera. " + description, "observed_frames": observed,
        "scene_frames": [observed, observed + FUTURE], "source_video": str(OURS / case / "video.mp4")}, indent=1))
    return f"{case}: last_observed.png ({observed} observed frames)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    with ThreadPoolExecutor(a.workers) as pool:
        for line in pool.map(one, samples("T7")):
            print(line, flush=True)


if __name__ == "__main__":
    main()
