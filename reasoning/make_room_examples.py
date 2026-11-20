#!/usr/bin/env python3
"""T1–T4 example folders from codification outputs, for `run.py --examples`.

For every room with a codified scene (<scenes>/<scene>/codify/final/scene.blend), writes

    <out>/T1-T4/<scene>/bench_records.json   the room's questions from the T1–T4 manifests (input / eval_only)
                        scene.blend          a hard link to the codified scene (a copy across file systems)
                        scene.py

    python3 make_room_examples.py --scenes ../codification/outputs --out runs/room_examples [--rooms a,b]

This is how the paper's T1–T4 inputs were made (one folder per room, the questions of that room).
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from harness.tasks import BENCH_TASKS, EVAL_KEY, INPUT_KEY, QUERY_FAMILIES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True, help="codification outputs: <scenes>/<scene>/codify/final/scene.blend")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rooms", default="", help="comma-separated scene names (default: every codified room)")
    a = ap.parse_args()
    questions = {}
    for family in QUERY_FAMILIES:
        for sample in json.loads((BENCH_TASKS / f"{family}.json").read_text())["samples"]:
            questions.setdefault(sample["input"]["scene_name"], []).append(
                {"id": sample["id"], "task": family, INPUT_KEY: sample["input"], EVAL_KEY: sample["eval_only"]})
    wanted = [r for r in a.rooms.split(",") if r] or sorted(questions)
    made, missing = 0, []
    for room in wanted:
        final = Path(a.scenes) / room / "codify" / "final"
        if room not in questions:
            raise SystemExit(f"{room}: no T1–T4 question in {BENCH_TASKS}")
        if not (final / "scene.blend").is_file():
            missing.append(room)
            continue
        out = Path(a.out) / "T1-T4" / room
        out.mkdir(parents=True, exist_ok=True)
        (out / "bench_records.json").write_text(json.dumps(questions[room], indent=1, ensure_ascii=False) + "\n")
        for name in ("scene.blend", "scene.py"):
            if (final / name).is_file() and not (out / name).exists():
                try:
                    os.link(final / name, out / name)
                except OSError:
                    shutil.copy2(final / name, out / name)
        made += 1
    print(f"{made} rooms, {sum(len(questions[r]) for r in wanted if r not in missing)} questions -> {a.out}/T1-T4"
          + (f"; no codified scene for {len(missing)}: {', '.join(missing[:5])}" if missing else ""))


if __name__ == "__main__":
    main()
