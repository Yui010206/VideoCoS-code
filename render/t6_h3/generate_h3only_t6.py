#!/usr/bin/env python
"""The H3-only baseline on T6: the same MiniMax-H3 Ref2VA request as generate.py, without our plan.

Ref2VA refuses a request with a keyframe and no reference, so the real start photo is both the keyframe at frame 0
(<Picture 1>) and the only reference (<Picture 2>). The prompt is prompt_template_h3only_t6.txt with the goal; the
sampling is generate.py's (50 steps, flow shift 12 / 3, seed 0, short edge 480, 15 s at 24 fps). All 110 cases: a case
prepared by prepare.py uses inputs/T6/<id>/start.png, any other the upright bench photo (inputs_h3only/T6/<id>/).

    python t6_h3/generate_h3only_t6.py --server http://127.0.0.1:30021 [--ids pilot|all|id,id] [--dry-run]

Writes runs/h3only_t6/T6/<id>/video.mp4 with request.json and job.json; claim files let several generators share
the list, like generate.py.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

from PIL import Image, ImageOps

from common import INPUTS, PILOT, RUNS, bench_path, probe, samples, write_json
from generate import DONE, FAILED, RATIOS, claim, http, wait_ready

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).resolve().parent / "prompt_template_h3only_t6.txt"
PROMPTS = ROOT / "prompts_h3only" / "T6"


def keyframe(sample):
    path = INPUTS / sample["id"] / "start.png"
    if path.is_file():
        return path
    path = ROOT / "inputs_h3only" / "T6" / sample["id"] / "start.png"
    if not path.is_file():
        image = ImageOps.exif_transpose(Image.open(bench_path(sample["input"]["start_image"]))).convert("RGB")
        s = 720 / min(image.size)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.resize((round(image.width * s), round(image.height * s)), Image.LANCZOS).save(path)
    return path


def aspect_ratio(path):
    with Image.open(path) as image:
        ratio = image.width / image.height
    return min(RATIOS, key=lambda k: abs(math.log(ratio / RATIOS[k])))


def request_body(sample, a):
    key = a.keyframes[sample["id"]]
    return {"task": "ref2va", "prompt": (Path(a.prompts) / f"{sample['id']}.txt").read_text().strip(),
            "conditions": [{"type": "image", "uri": f"file://{key}", "role": "keyframe", "frame_index": 0},
                           {"type": "image", "uri": f"file://{key}", "role": "reference"}],
            "target": {"short_edge": a.short_edge, "aspect_ratio": aspect_ratio(key), "duration_seconds": a.seconds},
            "num_inference_steps": a.steps, "flow_shift": a.flow_shift, "audio_flow_shift": a.audio_flow_shift,
            "seed": a.seed}


def render(sample, a):
    case_id = sample["id"]
    out = RUNS / a.name / "T6" / case_id
    if (out / "video.mp4").is_file():
        return "exists"
    lock = RUNS / a.name / "T6" / f"{case_id}.claim"
    if not claim(lock):
        return "claimed by another generator"
    try:
        if (out / "video.mp4").is_file():
            return "exists"
        body = request_body(sample, a)
        write_json(out / "request.json", body)
        started = time.time()
        job = http("POST", a.server + "/v1/videos", body)
        while True:
            time.sleep(a.poll)
            job = http("GET", f"{a.server}/v1/videos/{job['id']}")
            status = str(job.get("status", "")).lower()
            if status in DONE or status in FAILED or time.time() - started > a.timeout_hours * 3600:
                break
        record = {"job": job, "seconds": round(time.time() - started, 1), "server": a.server}
        if status not in DONE:
            write_json(out / "job.json", {**record, "outcome": status or "timeout"})
            return f"{status or 'timeout'} after {record['seconds']:.0f} s"
        tmp = out / "video.tmp.mp4"
        tmp.write_bytes(http("GET", f"{a.server}/v1/videos/{job['id']}/content", timeout=600))
        os.replace(tmp, out / "video.mp4")
        record["video"] = probe(out / "video.mp4")
        write_json(out / "job.json", {**record, "outcome": "completed"})
        v = record["video"]
        return f"{v['width']}x{v['height']} {v['frames']} frames {v['seconds']} s in {record['seconds'] / 60:.1f} min"
    finally:
        lock.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="all")
    ap.add_argument("--server", default="http://127.0.0.1:30021")
    ap.add_argument("--name", default="h3only_t6")
    ap.add_argument("--short-edge", type=int, default=480)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--flow-shift", type=float, default=12.0)
    ap.add_argument("--audio-flow-shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--timeout-hours", type=float, default=4.0)
    ap.add_argument("--wait-minutes", type=float, default=60.0)
    ap.add_argument("--template", default=str(TEMPLATE))
    ap.add_argument("--prompts", default=str(PROMPTS))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    S = samples()
    if a.ids == "pilot":
        S = [s for s in S if s["id"] in PILOT]
    elif a.ids != "all":
        wanted = set(a.ids.split(","))
        S = [s for s in S if s["id"] in wanted]
    S.sort(key=lambda s: (s["id"] not in PILOT, s["id"]))                  # the paper figure's cases first
    Path(a.prompts).mkdir(parents=True, exist_ok=True)
    template = Path(a.template).read_text()
    for s in S:
        (Path(a.prompts) / f"{s['id']}.txt").write_text(template.replace("{goal}", s["input"]["goal"].strip()))
    a.keyframes = {s["id"]: keyframe(s) for s in S}
    if a.dry_run:
        body = request_body(S[0], a)
        body["prompt"] = body["prompt"][:300] + " ..."
        print(len(S), "cases;", json.dumps(body, indent=1))
        return
    wait_ready(a.server, a.wait_minutes)
    for s in S:
        try:
            print(f"{s['id']}: {render(s, a)}", flush=True)
        except Exception as error:                       # keep going; the case is retried on the next run
            print(f"{s['id']}: ERROR {error}", flush=True)


if __name__ == "__main__":
    main()
