"""Shared paths and helpers for the render stage: MiniMax-H3 animates the real start frame of a T6 case along our
simulated plan video. The real frame is the starting point; the plan says which objects move, where and when;
H3 renders it photoreal and adds what the interaction needs (the hand)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Read only: the bench (task manifests and media) and the reasoning run whose plan videos are rendered. Defaults:
# this repository's bench/ and reasoning/runs/full_v1; override with $VIDEOCOS_BENCH and $VIDEOCOS_REASONING_RUN.
BENCH = Path(os.environ.get("VIDEOCOS_BENCH", ROOT.parent / "bench"))
REASONING_RUN = Path(os.environ.get("VIDEOCOS_REASONING_RUN", ROOT.parent / "reasoning" / "runs" / "full_v1"))
PLANS = REASONING_RUN / "T6"                                                 # our T6 run: <id>/video.mp4 + result.json
INPUTS, PROMPTS, RUNS, LOGS = ROOT / "inputs" / "T6", ROOT / "prompts_v4" / "T6", ROOT / "runs", ROOT / "logs"
PLAN_FPS = 25                                                                # our renders: 375 frames, 15 s
PILOT = ["T6_headphone_mouse_book", "T6_light_bulb_glasses", "T6_shoes_sandals_boxes", "T6_stack_blocks",
         "T6_acoustic_electric_guitar"]                                      # the paper figure's four, plus a long case


def samples(task="T6"):
    return json.loads((BENCH / "meta" / "tasks" / f"{task}.json").read_text())["samples"]


def bench_path(path):
    path = Path(path)
    return path if path.is_absolute() else BENCH / path


def chosen(ids):
    """--ids value: 'pilot', 'all' or comma-separated case ids; only cases with prepared inputs."""
    ready = sorted(p.name for p in INPUTS.glob("T6_*") if (p / "case.json").is_file())
    if ids == "all":
        return ready
    wanted = PILOT if ids == "pilot" else [i for i in ids.split(",") if i]
    missing = [i for i in wanted if i not in ready]
    if missing:
        raise SystemExit(f"not prepared: {missing}")
    return wanted


def timestamp(seconds):
    minutes, seconds = divmod(max(float(seconds), 0.0), 60.0)
    return f"{int(minutes):02d}:{seconds:06.3f}"


def frames_to_times(text, fps=PLAN_FPS):
    """The plan summaries cite Blender frames (frame 1 = 0 s); rewrite them as target-video timestamps."""
    at = lambda frame: timestamp((int(frame) - 1) / fps)
    text = re.sub(r"\bframes?\s+(\d+)\s*(?:to|through|until|-|–)\s*(?:frame\s+)?(\d+)",
                  lambda m: f"{at(m[1])} to {at(m[2])}", text, flags=re.I)
    return re.sub(r"\bframe\s+(\d+)\b", lambda m: at(m[1]), text, flags=re.I)


def probe(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_read_packets:format=duration", "-of", "json",
                          str(video)], capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    num, den = map(int, stream["r_frame_rate"].split("/"))
    return {"width": int(stream["width"]), "height": int(stream["height"]), "fps": num / den,
            "frames": int(stream["nb_read_packets"]), "seconds": round(float(data["format"]["duration"]), 3)}


def write_json(path, value):
    """Atomic: a reader never sees half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, indent=1, ensure_ascii=False)
    os.replace(handle.name, path)
