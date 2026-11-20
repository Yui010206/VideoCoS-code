"""Diagnostic-bench samples (meta/tasks/T1.json ... T7.json) as codification inputs.

A sample says what its task gives a system: a video or a start image, text, and for T5 the source camera. This
module turns one sample into the media path, the task family's config and the per-sample settings (the text the
model reads, the given cameras); nothing reads a sample's eval_only part. Codification stops at the scene of the
input in every family; answering the question, rendering the target view, planning the goal or predicting the
future is the next harness's job.
"""
import json
from pathlib import Path

from . import contract

# Task family -> config (configs/tasks/<family>.json): T1-T4 ask about one walkthrough video each; the scene of a
# video is the same for all its questions, so the question is not an input of codification.
FAMILIES = {"T1": "T1-T4", "T2": "T1-T4", "T3": "T1-T4", "T4": "T1-T4", "T5": "T5", "T6": "T6", "T7": "T7"}


def find(bench_root, sample_id):
    """(task, sample) of one sample id, e.g. T7_0001_left_ball-and-block-fall."""
    task = sample_id.split("_")[0]
    if task not in FAMILIES:
        raise SystemExit(f"{sample_id}: not a bench sample id (T1_... to T7_...)")
    data = contract.read_json(Path(bench_root) / "meta" / "tasks" / f"{task}.json")
    for sample in data["samples"]:
        if sample["id"] == sample_id:
            return task, sample
    raise SystemExit(f"{sample_id} is not in {task}.json")


UPRIGHT = "input_upright.jpg"     # a T6 photo turned upright, written next to the run


def inputs(config, sample_id, run_dir):
    """media path and the settings this sample adds (merged over its family config)."""
    bench_root = config["bench_root"]
    task, sample = find(bench_root, sample_id)
    given = sample["input"]
    resolve = lambda p: str(p if Path(p).is_absolute() else Path(bench_root) / p)
    media = resolve(given.get("video") or given["start_image"])
    settings = {"task": {"sample_id": sample_id, "bench_task": task}}
    if task == "T5":
        # T5's source is the scene's high view v0 (meta/tasks/T5.json since 2026-10-26), so the three samples of a
        # scene share one codification; its camera comes from that view's JSON camera block (Kubric world frame,
        # floor at z = 0). The target cameras are for the next harness.
        from . import given_cameras
        scene_dir = Path(bench_root) / "data" / "t5_kubric4d" / sample["scene"]
        path = Path(run_dir) / "given_cameras.json"
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        given_cameras.from_kubric_view(scene_dir / f"{sample['scene']}_p0_v{sample['source_view']}.json", path)
        settings["fuse"] = {"given_cameras": str(path)}
    elif task == "T6":
        settings["task"]["text"] = (f"The scene will be used to carry out this goal from the start image: "
                                    f"{given['goal']}")
        # A photo whose EXIF orientation asks for a turn is given upright (decoding takes no display rotation).
        from PIL import Image, ImageOps
        with Image.open(media) as image:
            if image.getexif().get(274, 1) != 1:
                Path(run_dir).mkdir(parents=True, exist_ok=True)
                upright = Path(run_dir) / UPRIGHT
                ImageOps.exif_transpose(image).convert("RGB").save(upright, quality=95)
                media = str(upright)
    elif task == "T7":
        settings["task"]["text"] = f"Description of the scene given with the video: {given['description']}"
    return media, settings
