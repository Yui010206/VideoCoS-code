#!/usr/bin/env python
"""Render each case with MiniMax-H3 Ref2VA through the SGLang server started by serve_h3.sh.

One hybrid ref2va request per case: the real start frame is a pixel-aligned keyframe at frame 0 (<Picture 1>), our
plan video is the reference (<Video 1>), and the prompt is prompts_v4/T6/<id>.txt (make_prompts.py). Defaults follow
the SGLang cookbook's Ref2VA requests (50 steps, flow shift 12, audio flow shift 3) and the format of Ego2Act's own
MiniMax-H3 videos (24 fps, 15 s, short edge 480), so the result sits next to that baseline row.

    python t6_h3/generate.py --ids pilot [--server http://127.0.0.1:30011] [--short-edge 480] [--steps 50]
    python t6_h3/generate.py --ids pilot --dry-run          # print the first request, contact nothing

Writes runs/<name>/T6/<id>/video.mp4 with request.json and job.json beside it; finished cases are skipped. Several
generators (one per server) can share one case list: each claims a case with runs/<name>/T6/<id>.claim (its PID)
before submitting it, skips cases claimed by a live process, and takes over claims of dead ones. With --ids all the
pilot cases come first.
"""
import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from common import INPUTS, PILOT, PROMPTS, RUNS, chosen, probe, write_json

DONE, FAILED = {"completed", "succeeded", "success"}, {"failed", "error", "cancelled", "canceled", "expired"}


def http(method, url, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data, {"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return payload if url.endswith("/content") else json.loads(payload or b"{}")


RATIOS = {"21:9": 21 / 9, "16:9": 16 / 9, "4:3": 4 / 3, "1:1": 1.0, "3:4": 3 / 4, "9:16": 9 / 16}   # H3's only choices


def aspect_ratio(case_dir):
    """The H3 ratio closest to the real frame's. "auto" would not do: for ref2va it always means 16:9, and the start
    keyframe is then center-cropped to the canvas."""
    plan = json.loads((case_dir / "case.json").read_text())["plan_video"]
    ratio = plan["width"] / plan["height"]
    return min(RATIOS, key=lambda k: abs(math.log(ratio / RATIOS[k])))


def request_body(case_id, a):
    case_dir = INPUTS / case_id
    return {"task": "ref2va", "prompt": (Path(a.prompts) / f"{case_id}.txt").read_text().strip(),
            "conditions": [{"type": "image", "uri": f"file://{case_dir / 'start.png'}", "role": "keyframe",
                            "frame_index": 0},
                           {"type": "video", "uri": f"file://{case_dir / 'plan.mp4'}", "role": "reference"}],
            "target": {"short_edge": a.short_edge, "aspect_ratio": aspect_ratio(case_dir), "duration_seconds": a.seconds},
            "num_inference_steps": a.steps, "flow_shift": a.flow_shift, "audio_flow_shift": a.audio_flow_shift,
            "seed": a.seed}


def wait_ready(server, minutes):
    """Loading 144 GB from the NAS takes a while; poll until the server answers."""
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        for path in ("/health", "/v1/models"):
            try:
                http("GET", server + path, timeout=10)
                return
            except (urllib.error.URLError, OSError, ValueError):
                pass
        time.sleep(20)
    raise SystemExit(f"server {server} not ready after {minutes} min")


def claim(lock):
    """Atomically take a case for this process; a claim whose process is gone is taken over."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            os.kill(int(lock.read_text().strip()), 0)
            return False
        except (ValueError, ProcessLookupError, FileNotFoundError):
            lock.unlink(missing_ok=True)
            return claim(lock)
        except PermissionError:
            return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def render(case_id, a):
    out = RUNS / a.name / "T6" / case_id
    if (out / "video.mp4").is_file():
        return "exists"
    lock = RUNS / a.name / "T6" / f"{case_id}.claim"
    if not claim(lock):
        return "claimed by another generator"
    try:
        if (out / "video.mp4").is_file():               # finished by another generator just before the claim
            return "exists"
        return submit_and_fetch(case_id, out, a)
    finally:
        lock.unlink(missing_ok=True)


def submit_and_fetch(case_id, out, a):
    body = request_body(case_id, a)
    write_json(out / "request.json", body)
    started = time.time()
    job = http("POST", a.server + "/v1/videos", body)
    job_id = job["id"]
    while True:
        time.sleep(a.poll)
        job = http("GET", f"{a.server}/v1/videos/{job_id}")
        status = str(job.get("status", "")).lower()
        if status in DONE or status in FAILED or time.time() - started > a.timeout_hours * 3600:
            break
    record = {"job": job, "seconds": round(time.time() - started, 1), "server": a.server}
    if status not in DONE:
        write_json(out / "job.json", {**record, "outcome": status or "timeout"})
        return f"{status or 'timeout'} after {record['seconds']:.0f} s"
    tmp = out / "video.tmp.mp4"
    tmp.write_bytes(http("GET", f"{a.server}/v1/videos/{job_id}/content", timeout=600))
    os.replace(tmp, out / "video.mp4")
    record["video"] = probe(out / "video.mp4")
    write_json(out / "job.json", {**record, "outcome": "completed"})
    v = record["video"]
    return f"{v['width']}x{v['height']} {v['frames']} frames {v['seconds']} s in {record['seconds'] / 60:.1f} min"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="pilot")
    ap.add_argument("--server", default="http://127.0.0.1:30011")
    ap.add_argument("--name", default="ours_h3_v4", help="output run name under runs/")
    ap.add_argument("--prompts", default=str(PROMPTS), help="prompt folder written by make_prompts.py (default: prompts_v4/T6)")
    ap.add_argument("--short-edge", type=int, default=480)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--flow-shift", type=float, default=12.0)
    ap.add_argument("--audio-flow-shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--timeout-hours", type=float, default=4.0)
    ap.add_argument("--wait-minutes", type=float, default=60.0, help="how long to wait for the server to come up")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ids = [i for i in chosen(a.ids) if (Path(a.prompts) / f"{i}.txt").is_file()]
    ids = [i for i in PILOT if i in ids] + [i for i in ids if i not in PILOT]      # pilot cases first
    if a.dry_run:
        body = request_body(ids[0], a)
        body["prompt"] = body["prompt"][:300] + " ..."
        print(json.dumps(body, indent=1))
        return
    wait_ready(a.server, a.wait_minutes)
    for case_id in ids:
        try:
            print(f"{case_id}: {render(case_id, a)}", flush=True)
        except Exception as error:                       # keep going; the case is retried on the next run
            print(f"{case_id}: ERROR {error}", flush=True)


if __name__ == "__main__":
    main()
