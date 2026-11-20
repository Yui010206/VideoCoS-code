#!/usr/bin/env python
"""T7 with MiniMax-H3 Ref2VA: our Blender prediction rendered photoreal from the last observed frame.

One ref2va request per pair, served by SGLang (serve_h3.sh): the last observed real frame is the keyframe at frame 0
(<Picture 1>, inputs/T7/<id>/last_observed.png from make_t7_inputs.py), our Blender prediction of the next 5 s is the
reference video (<Video 1>, <reasoning run>/T7/<id>/video.mp4, see common.REASONING_RUN), and the prompt is
prompt_template_t7_ref.txt with the scene's description. Sampling: 50 steps, flow shift 12, audio flow shift 3, seed 0,
short edge 480, 16:9, 5 s at 24 fps. Frame 0 of the output is the keyframe; video.mp4 keeps frames 1..120, so frame k
shows k/24 s after the last observed frame. raw.mp4 is what the server returned.

--no-reference is the H3-only baseline (all 66 pairs, no plan): Ref2VA refuses a keyframe without a reference, so the
last observed frame is given again as the only (image) reference, with prompt_template_t7_h3only.txt.

    python t6_h3/generate_t7_ref2va.py --server http://127.0.0.1:30021 [--ids pilot|all|id,id] [--no-reference] [--dry-run]

Several generators (one per server) share the case list through claim files, like generate.py.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BENCH, REASONING_RUN, bench_path, probe, write_json  # noqa: E402
from generate import claim, http, wait_ready, DONE, FAILED  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TASKS = BENCH / "meta" / "tasks" / "T7.json"
PLANS = REASONING_RUN / "T7"
INPUTS = ROOT / "inputs" / "T7"
TEMPLATE = Path(__file__).resolve().parent / "prompt_template_t7_ref.txt"
H3ONLY_TEMPLATE = Path(__file__).resolve().parent / "prompt_template_t7_h3only.txt"
PILOT = ["T7_0009_right_ball-hits-duck", "T7_0146_center_roll-behind-box", "T7_0034_left_block-domino",
         "T7_0191_center_weight-on-paper"]
FRAMES, FPS = 120, 24


def keyframe(s):
    """The last observed frame: inputs/T7/<id>/last_observed.png, or (pairs without a plan) the last frame of the
    bench's 3 s input clip, extracted once to inputs_h3only/T7/<id>/."""
    path = INPUTS / s["id"] / "last_observed.png"
    if path.is_file():
        return path
    path = ROOT / "inputs_h3only" / "T7" / s["id"] / "last_observed.png"
    if not path.is_file():
        import cv2
        cap, last = cv2.VideoCapture(str(bench_path(s["input"]["video"]))), None
        while True:
            ok, f = cap.read()
            if not ok:
                break
            last = f
        if last is None:
            raise SystemExit(f"{s['id']}: no conditioning clip at {bench_path(s['input']['video'])} (see bench/README.md)")
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), last)
    return path


def cases(first=PILOT, need_plan=True):
    """Pairs with a Blender prediction (unless need_plan is off) and a last observed frame, the `first` ones first."""
    S = json.loads(TASKS.read_text())["samples"]
    ready = [s for s in S if not need_plan or ((PLANS / s["id"] / "video.mp4").is_file() and (INPUTS / s["id"] / "last_observed.png").is_file())]
    ready.sort(key=lambda s: (s["id"] not in first, first.index(s["id"]) if s["id"] in first else 0, s["id"]))
    return ready


def write_prompts(S, template_path, prompts):
    prompts.mkdir(parents=True, exist_ok=True)
    template = Path(template_path).read_text()
    for s in S:
        (prompts / f"{s['id']}.txt").write_text(template.replace("{description}", s["input"]["description"].strip()))


def request_body(case_id, a):
    conditions = [{"type": "image", "uri": f"file://{a.keyframes[case_id]}", "role": "keyframe", "frame_index": 0}]
    if not a.no_reference:
        conditions.append({"type": "video", "uri": f"file://{PLANS / case_id / 'video.mp4'}", "role": "reference"})
    else:               # Ref2VA refuses keyframe-only requests: the same real frame as the (only) reference
        conditions.append({"type": "image", "uri": f"file://{a.keyframes[case_id]}", "role": "reference"})
    return {"task": "ref2va", "prompt": (Path(a.prompts) / f"{case_id}.txt").read_text().strip(),
            "conditions": conditions,
            "target": {"short_edge": a.short_edge, "aspect_ratio": "16:9", "duration_seconds": a.seconds},
            "num_inference_steps": a.steps, "flow_shift": a.flow_shift, "audio_flow_shift": a.audio_flow_shift,
            "seed": a.seed}


def trim(raw, out):
    """Frames 1..120 of the server's video (frame 0 is the keyframe) -> the 5 s prediction at 24 fps."""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-an", "-vf",
                    f"select='between(n,1,{FRAMES})',setpts=N/{FPS}/TB", "-r", str(FPS), "-c:v", "libx264", "-crf", "12",
                    "-pix_fmt", "yuv420p", str(out)], check=True)


def render(case_id, a):
    out = ROOT / "runs" / a.name / "T7" / case_id
    if (out / "video.mp4").is_file():
        return "exists"
    lock = ROOT / "runs" / a.name / "T7" / f"{case_id}.claim"
    if not claim(lock):
        return "claimed by another generator"
    try:
        if (out / "video.mp4").is_file():
            return "exists"
        body = request_body(case_id, a)
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
        (out / "raw.tmp.mp4").write_bytes(http("GET", f"{a.server}/v1/videos/{job['id']}/content", timeout=600))
        (out / "raw.tmp.mp4").replace(out / "raw.mp4")
        record["raw"] = probe(out / "raw.mp4")
        if record["raw"]["frames"] < FRAMES + 1:
            write_json(out / "job.json", {**record, "outcome": "too few frames"})
            return f"too few frames: {record['raw']['frames']}"
        trim(out / "raw.mp4", out / "video.tmp.mp4")
        (out / "video.tmp.mp4").replace(out / "video.mp4")
        record["video"] = probe(out / "video.mp4")
        write_json(out / "job.json", {**record, "outcome": "completed", "kept": f"frames 1..{FRAMES} of raw.mp4"})
        v = record["video"]
        return f"raw {record['raw']['frames']} frames -> {v['width']}x{v['height']} {v['frames']} frames in {record['seconds'] / 60:.1f} min"
    finally:
        lock.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="all")
    ap.add_argument("--server", default="http://127.0.0.1:30021")
    ap.add_argument("--name", help="output run under runs/ (default: ours_h3full_t7, with --no-reference h3only_t7)")
    ap.add_argument("--short-edge", type=int, default=480)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--flow-shift", type=float, default=12.0)
    ap.add_argument("--audio-flow-shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poll", type=float, default=15.0)
    ap.add_argument("--timeout-hours", type=float, default=3.0)
    ap.add_argument("--wait-minutes", type=float, default=120.0)
    ap.add_argument("--template", help="prompt template (default: prompt_template_t7_ref.txt, with --no-reference "
                                       "prompt_template_t7_h3only.txt)")
    ap.add_argument("--prompts", help="per-pair prompt folder written from --template (default: prompts_t7_ref/T7, with "
                                      "--no-reference prompts_t7_h3only/T7)")
    ap.add_argument("--no-reference", action="store_true",
                    help="the H3-only baseline: no plan video; all 66 pairs, the real frame as the only reference")
    ap.add_argument("--first", default="", help="comma-separated ids to render first (default: the pilot pairs)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.name = a.name or ("h3only_t7" if a.no_reference else "ours_h3full_t7")
    a.template = a.template or str(H3ONLY_TEMPLATE if a.no_reference else TEMPLATE)
    a.prompts = a.prompts or str(ROOT / ("prompts_t7_h3only" if a.no_reference else "prompts_t7_ref") / "T7")
    first = [i for i in a.first.split(",") if i] or PILOT
    S = cases(first, need_plan=not a.no_reference)
    ids = [s["id"] for s in S]
    if a.ids == "pilot":
        ids = [i for i in first if i in ids]
    elif a.ids != "all":
        ids = [i for i in a.ids.split(",") if i in ids]
    if not ids:
        raise SystemExit("no pair to render (a pair needs our prediction and inputs/T7/<id>/, see README)")
    S = [s for s in S if s["id"] in ids]
    a.keyframes = {s["id"]: keyframe(s) for s in S}
    write_prompts(S, a.template, Path(a.prompts))
    if a.dry_run:
        body = request_body(ids[0], a)
        body["prompt"] = body["prompt"][:400] + " ..."
        print(len(ids), "cases;", json.dumps(body, indent=1))
        return
    wait_ready(a.server, a.wait_minutes)
    for case_id in ids:
        try:
            print(f"{case_id}: {render(case_id, a)}", flush=True)
        except Exception as error:                       # keep going; the case is retried on the next run
            print(f"{case_id}: ERROR {error}", flush=True)


if __name__ == "__main__":
    main()
