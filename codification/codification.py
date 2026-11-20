#!/usr/bin/env python3
"""VideoCoS codification, step 2 of 2: codification (Codex agent + sandboxed Blender). See README.md.

For every input of a queue that perception.py finished (<out>/<id>/.perceived) it writes <out>/<id>/codify/:
  T1-T4  the fast room mode: the agent's first build is base_scene.py (every evidence object, as its SAM 3D mesh where
         that agrees with its box, the room's floor, walls, ceiling and lights, then arrange_room); it compares the
         renders with the frames, fixes what is most visibly wrong and submits; 20 minutes, no GPU
  T5-T7  the raw perception layers (frames, depth, masks, cameras), SAM 3D meshes made by the agent when it needs
         them (make_objects, track_objects); 35 minutes, one GPU for SAM 3D
  codify/final/scene.blend, scene.py, inventory.json, preview.mp4 (input | reconstruction); codify/run.json

usage (on a machine with Docker, the sandbox image videocos-blender-runtime:1, a logged-in Codex CLI and Blender; with the
harness Python):
  codification.py run    [--queue queues/vsi.json] [--out outputs] [--parallel 4] [--gpus 6,7] [--watch]
  codification.py run    --video clip.mp4 --task T7 [--id my_clip] --gpus 6        one input of your own
  codification.py status [--queue queues/vsi.json] [--out outputs]
--watch keeps looking for newly perceived inputs every 5 minutes until every input of the queue is codified, so it can
run beside perception lanes. Log: <out>/codification.log."""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from perception import DONE, TASKS, has_scene, machine, queue_of, tqdm  # noqa: E402

PY = sys.executable
MINUTES = {"T1-T4": 20, "T5": 35, "T6": 35, "T7": 35}
_log_lock = threading.Lock()


def log(out, message):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + message
    (tqdm.write if tqdm else print)(line)
    with _log_lock, open(out / "codification.log", "a") as stream:
        stream.write(line + "\n")


def finished(S):
    """Done when codification left a scene.blend (a run that ended without one is tried again)."""
    return has_scene(S)


def attempts(S):
    """Earlier runs that ended without a scene, kept aside as codify_unfinished_*."""
    return len(list(S.glob("codify_unfinished_*"))) + ((S / "codify").exists() and not has_scene(S))


def command(args, item, S, gpu):
    task = item["task"]
    common = ["--output", str(S / "codify"), "--agent", "codex", "--codex-model", args.model,
              "--codex-effort", args.effort, "--codex-minutes", str(args.minutes or MINUTES[task])]
    if task == "T1-T4":                 # the given evidence (objects + meshes); only the decoded frames are reused
        return [PY, "run_codify.py", "codify", "--video", item["video"], "--config", str(S / "config_codify.json"),
                "--reuse", str(S / "decode_only"), *common]
    source = ["--sample", item["sample"]] if item.get("sample") else \
        ["--video", item["video"], "--config", str(HERE / "configs" / "tasks" / f"{task}.json")]
    return [PY, "run_codify.py", "codify", *source, "--reuse", str(S / "perception"), "--gpu", str(gpu), *common,
            "--raw", "--sam3d"]


def run_with_bar(cmd, stream, bar, S, started, budget, stage):
    """Run cmd; every 10 seconds the bar shows the minutes used against the agent's budget and the builds so far."""
    process = subprocess.Popen(cmd, cwd=HERE, stdout=stream, stderr=subprocess.STDOUT)
    while process.poll() is None:
        if bar:
            bar.n = min(budget, int((time.time() - started) / 60))
            revisions = len(list((S / "codify" / "revisions").glob("r[0-9]*"))) if (S / "codify" / "revisions").is_dir() else 0
            bar.set_postfix_str(f"{stage}, {revisions} builds")
        time.sleep(10)


def codify(args, item, gpu, position=1):
    S = args.out / item["id"]
    if (S / "codify").exists():         # an earlier run that did not finish: kept aside, started again
        shutil.move(S / "codify", S / f"codify_unfinished_{time.strftime('%Y%m%d_%H%M%S')}")
    started = time.time()
    budget = int(args.minutes or MINUTES[item["task"]])
    log(args.out, f"{item['id']}: codify started ({item['task']}{f', gpu {gpu}' if gpu is not None else ''})")
    bar = tqdm(total=budget, desc=f"{item['id']}", position=position, leave=False, dynamic_ncols=True,
               bar_format="{desc} |{bar}| {n}/{total} min{postfix}") if tqdm else None
    with open(S / "codification.out", "a") as stream:
        stream.write(f"\n=== {time.strftime('%F %T')}\n")
        stream.flush()
        run_with_bar(command(args, item, S, gpu), stream, bar, S, started, budget, "agent")
        minutes = (time.time() - started) / 60
        if (S / "codify" / "final" / "scene.blend").is_file() and not args.no_preview:
            run_with_bar([PY, "run_codify.py", "preview", "--run", str(S / "codify")], stream, bar, S, started, budget,
                         "preview video")
    if bar:
        bar.close()
    path = S / "codify" / "final_report.json"
    report = json.loads(path.read_text()) if path.is_file() else {}
    log(args.out, f"{item['id']}: {report.get('status', 'failed')} {report.get('selected_revision')} in "
                  f"{minutes:.0f} min (codify; preview "
                  f"{'ok' if (S / 'codify' / 'final' / 'preview.mp4').is_file() else 'none'})")


def ready(args, started):
    """Perceived inputs without a scene that have not used up their tries."""
    return [i for i in queue_of(args) if i["id"] not in started and (args.out / i["id"] / DONE).exists()
            and not finished(args.out / i["id"]) and attempts(args.out / i["id"]) < args.tries]


def settled(args, items):
    """Every input has a scene or has used up its tries: --watch can stop."""
    return all(finished(args.out / i["id"]) or attempts(args.out / i["id"]) >= args.tries for i in items)


def run(args):
    problems = []
    if subprocess.run(["docker", "image", "inspect", "videocos-blender-runtime:1"], capture_output=True).returncode:
        problems.append("docker or the sandbox image videocos-blender-runtime:1 is missing")
    if not shutil.which("codex"):
        problems.append("the Codex CLI (codex) is not on PATH")
    if not Path(machine()["agent"]["blender_dir"], "blender").is_file():
        problems.append(f"no Blender in {machine()['agent']['blender_dir']}")
    if problems:
        sys.exit("codification cannot run on this machine: " + "; ".join(problems))
    gpu_slots = queue.Queue()            # T5-T7 runs hold a GPU for SAM 3D, per_gpu at a time
    for gpu in args.gpus:
        for _ in range(args.per_gpu):
            gpu_slots.put(gpu)
    jobs, started, threads = queue.Queue(), set(), []
    launch = {"lock": threading.Lock(), "last": 0.0}
    items = queue_of(args)
    done_count = lambda: sum(finished(args.out / i["id"]) for i in items)
    overall = tqdm(total=len(items), initial=done_count(), desc="codified", position=0, dynamic_ncols=True,
                   bar_format="{desc} |{bar}| {n}/{total} [{elapsed}]") if tqdm else None

    def worker(position):
        while True:
            item = jobs.get()
            if item is None:
                return
            gpu = None
            if item["task"] != "T1-T4":
                if not args.gpus:
                    log(args.out, f"{item['id']}: skipped, {item['task']} needs --gpus (SAM 3D)")
                    continue
                gpu = gpu_slots.get()
            try:
                while True:
                    while (args.out / "PAUSE").exists():     # a watchdog holds new starts (the Codex quota, failures)
                        time.sleep(30)
                    with launch["lock"]:     # --stagger: Codex sessions started together can come up without their tools
                        time.sleep(max(0.0, launch["last"] + args.stagger - time.time()))
                        launch["last"] = time.time()
                    codify(args, item, gpu, position)
                    S = args.out / item["id"]
                    if finished(S) or attempts(S) >= args.tries:
                        break
                    log(args.out, f"{item['id']}: no scene; trying again ({attempts(S)} of {args.tries} tries used)")
            finally:
                if gpu is not None:
                    gpu_slots.put(gpu)
                if overall:
                    overall.n = done_count()
                    overall.refresh()
    for index in range(args.parallel):  # each worker's bar has its own line under the overall one
        threads.append(threading.Thread(target=worker, args=(index + 1,), daemon=True))
        threads[-1].start()
    log(args.out, f"start: queue {args.video or args.queue}, {args.parallel} at a time, gpus {args.gpus or 'none'}"
                  f"{', watching' if args.watch else ''}")
    while True:
        for item in ready(args, started):
            started.add(item["id"])
            jobs.put(item)
        if not args.watch or (settled(args, items) and jobs.empty()):
            break
        time.sleep(300)                  # perception lanes add inputs meanwhile
    for _ in threads:
        jobs.put(None)
    for t in threads:
        t.join()
    if overall:
        overall.n = done_count()
        overall.close()
    log(args.out, "done")


def status(args):
    items = queue_of(args)
    scenes, perceived, gave_up = 0, 0, []
    for item in items:
        S = args.out / item["id"]
        perceived += (S / DONE).exists()
        scenes += finished(S)
        if not finished(S) and attempts(S) >= args.tries:
            gave_up.append(item["id"])
    print(f"{len(items)} inputs | perceived {perceived} | with a scene.blend {scenes} | "
          f"no scene after {args.tries} tries {len(gave_up)}" + (f": {', '.join(gave_up[:20])}" if gave_up else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["run", "status"])
    p.add_argument("--queue", default=str(HERE / "queues" / "vsi.json"), help="the queue perception.py ran on")
    p.add_argument("--video", help="one input of your own (as given to perception.py)")
    p.add_argument("--task", help="with --video: " + ", ".join(TASKS))
    p.add_argument("--id", help="with --video: the output folder's name (default: the file name)")
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="output root (default: outputs/)")
    p.add_argument("--order", choices=["shortest", "longest"], default="shortest")
    p.add_argument("--parallel", type=int, default=4, help="codification runs at a time")
    p.add_argument("--gpus", default="", help="GPUs for SAM 3D in T5-T7 runs (T1-T4 runs need none)")
    p.add_argument("--per-gpu", type=int, default=2, help="T5-T7 runs per GPU at a time (SAM 3D: about 20 GB each)")
    p.add_argument("--watch", action="store_true", help="keep picking up newly perceived inputs until all are codified")
    p.add_argument("--model", default="gpt-6.1-terra", help="Codex model")
    p.add_argument("--effort", default="xhigh", help="Codex reasoning effort")
    p.add_argument("--minutes", type=float, help="the agent's time budget (default: T1-T4 20, T5-T7 35)")
    p.add_argument("--tries", type=int, default=2, help="codification runs per input before it is left without a scene")
    p.add_argument("--stagger", type=float, default=0.0, help="seconds between two codification starts")
    p.add_argument("--no-preview", action="store_true", help="leave the preview video for later (run_codify.py preview "
                                                                "--run <out>/<id>/codify); the slot goes to the next input")
    p.add_argument("--num-groups", type=int, default=0, help="as given to perception.py: the queue's groups")
    p.add_argument("--group", type=int, default=0, help="with --num-groups: the group to codify, 0 to N-1")
    args = p.parse_args()
    args.out = args.out.resolve()
    args.gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    os.environ.update(machine().get("env", {}))
    if args.command == "status":
        return status(args)
    args.out.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
