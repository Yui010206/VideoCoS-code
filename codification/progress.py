#!/usr/bin/env python3
"""Progress bars for a queue: how many inputs are perceived and codified, and which step each running input is on.
It only reads <out>/ (lock files, step logs, reports); it starts nothing.

usage (from this folder, on any machine that sees the output folder):
  progress.py [--queue queues/vsi.json] [--out outputs]          one look
  progress.py --watch 60                                          redraw every 60 seconds (Ctrl-C to stop)"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from perception import DONE, LOCK, has_scene, queue_of  # noqa: E402

PERCEPTION = ["decode", "geometry", "camera_depth", "vocabulary", "tracks", "proposals", "proposal_names",
              "proposal_tracks", "fuse"]
STEPS = {"T1-T4": PERCEPTION + ["objects", "meshes"], "T5": PERCEPTION, "T6": PERCEPTION, "T7": PERCEPTION}


def bar(done, total, width=30):
    filled = int(round(width * done / total)) if total else 0
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def step_of(S, task):
    """(index, name) of the step a running input is on."""
    if (S / "perception" / "run.json").is_file() and json.loads((S / "perception" / "run.json").read_text()).get("evidence"):
        if task != "T1-T4":
            return len(STEPS[task]) - 1, "fuse"
        name = "meshes" if (S / "objects" / "evidence" / "manifest.json").is_file() else "objects"
        return STEPS[task].index(name), name
    log = S / "perception" / "run.log"
    current = "decode"
    if log.is_file():
        for line in log.read_text(errors="replace").splitlines():
            m = re.search(r"\[(\w+?)(?:_[0-9a-f]{10})?\] (running|resume)", line)
            if m and m.group(1) in PERCEPTION:
                current = m.group(1)
    return PERCEPTION.index(current), current


def gpu_of(S, step):
    out = S / f"{'perception' if step in PERCEPTION else step}.out"
    if not out.is_file():
        return "?"
    heads = re.findall(r"^=== \S+ \S+ (\S+) gpu (\d+)", out.read_text(errors="replace"), re.M)
    return f"{heads[-1][0].split('.')[0]} gpu {heads[-1][1]}" if heads else "?"


def show(args):
    items = queue_of(args)
    perceived, running, codified = 0, [], {}
    for item in items:
        S = args.out / item["id"]
        if (S / DONE).exists():
            perceived += 1
        elif (S / LOCK).exists():
            running.append(item)
        if has_scene(S):
            codified["scene.blend"] = codified.get("scene.blend", 0) + 1
    total, done_codify = len(items), sum(codified.values())
    print(f"{time.strftime('%F %T')}  queue {args.queue if not args.video else args.video}  ->  {args.out}")
    print(f"perception {bar(perceived, total)} {perceived}/{total}  (running {len(running)}, waiting "
          f"{total - perceived - len(running)})")
    print(f"codify     {bar(done_codify, total)} {done_codify}/{total}  "
          + (", ".join(f"{k} {v}" for k, v in sorted(codified.items())) or ""))
    if running:
        print("running:")
        for item in running:
            S = args.out / item["id"]
            index, name = step_of(S, item["task"])
            steps = len(STEPS[item["task"]])
            hours = (time.time() - (S / LOCK).stat().st_mtime) / 3600
            print(f"  {item['id'][:36]:36s} {item['task']:6s} step {index + 1:2d}/{steps} {name:16s} "
                  f"{bar(index, steps, 20)} {hours:4.1f} h  {gpu_of(S, name)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queue", default=str(HERE / "queues" / "vsi.json"))
    p.add_argument("--video", help=argparse.SUPPRESS)
    p.add_argument("--task", help=argparse.SUPPRESS)
    p.add_argument("--id", help=argparse.SUPPRESS)
    p.add_argument("--order", default="shortest", help=argparse.SUPPRESS)
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    p.add_argument("--watch", type=int, default=0, help="redraw every this many seconds")
    p.add_argument("--num-groups", type=int, default=0, help="as given to perception.py")
    p.add_argument("--group", type=int, default=0, help="with --num-groups: the group to show, 0 to N-1")
    args = p.parse_args()
    args.out = args.out.resolve()
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        show(args)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
