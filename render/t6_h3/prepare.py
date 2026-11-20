#!/usr/bin/env python
"""Inputs for every T6 case that has a plan (our run completed with a video):

    inputs/T6/<id>/start.png   the real start frame, at the plan video's canvas (our render was made from it)
    inputs/T6/<id>/plan.mp4    a copy of our simulated plan video (25 fps, 15 s, no hand)
    inputs/T6/<id>/case.json   goal, constraints, objects, and the plan summary with frames rewritten as times

    python t6_h3/prepare.py [--ids all|pilot|a,b]
"""
import argparse
import json
import shutil

from PIL import Image, ImageOps

from common import INPUTS, PLANS, bench_path, frames_to_times, probe, samples, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="all")
    a = ap.parse_args()
    wanted = None if a.ids == "all" else set(a.ids.split(","))
    done, skipped = 0, []
    for sample in samples():
        case = sample["id"]
        if wanted and case not in wanted:
            continue
        run = PLANS / case
        result = json.loads((run / "result.json").read_text()) if (run / "result.json").is_file() else {}
        if result.get("status") != "completed" or not (run / "video.mp4").is_file():
            skipped.append(case)
            continue
        out = INPUTS / case
        out.mkdir(parents=True, exist_ok=True)
        plan = probe(run / "video.mp4")
        raw = Image.open(bench_path(sample["input"]["start_image"]))
        image = ImageOps.exif_transpose(raw).convert("RGB")     # upright, as our plan was made from (10 cases rotate)
        if abs(image.width / image.height - plan["width"] / plan["height"]) > 0.02:
            print(f"WARNING {case}: upright start image {image.size} vs plan {plan['width']}x{plan['height']}")
        image.resize((plan["width"], plan["height"]), Image.LANCZOS).save(out / "start.png")
        shutil.copy2(run / "video.mp4", out / "plan.mp4")
        write_json(out / "case.json", {
            "id": case, "goal": sample["input"]["goal"], "constraints": sample["input"]["constraints"],
            "objects": sample.get("objects", []), "action_types": sample.get("action_types", []),
            "plan_summary": frames_to_times(result.get("summary") or ""), "plan_video": plan,
            "start_image": str(bench_path(sample["input"]["start_image"])), "start_image_size": list(raw.size)})
        done += 1
    print(f"prepared {done} cases in {INPUTS}; no plan for {len(skipped)}: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
