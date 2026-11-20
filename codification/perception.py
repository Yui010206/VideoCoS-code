#!/usr/bin/env python3
"""VideoCoS codification, step 1 of 2: perception (GPU). See README.md.

For every input of a queue it writes <out>/<id>/, what codification.py needs:
  T1-T4 (VSI-Bench room videos, minutes long)
    perception/              decoded frames, DA3-Streaming cameras and depth, object names (Qwen3.8-27B), SAM 3 tracks,
                             open detection (run_codify.py perceive)
    objects/evidence/        the object layer: tracks checked by the 27B model, lifted to 3D, one entry per physical
                             object (codify.instances)
    objects/evidence_meshes/ the same evidence plus a SAM 3D mesh per object (codify.prepare_room)
    config_codify.json, decode_only/
  T5, T6, T7 (Kubric 60 frames, Ego2Act single images, Physics-IQ 3-second clips)
    perception/              run_codify.py perceive
  .perceived                 written when all of the above is there

usage (from this folder, with the harness Python):
  perception.py check  --gpus 0,1,2,3                       paths, model environments, weights, free GPU memory
  perception.py run    --gpus 0,1,2,3 [--queue queues/vsi.json] [--out outputs] [--order longest] [--limit N]
  perception.py run    --gpus 0 --video clip.mp4 --task T7 [--id my_clip]       one input of your own
  perception.py status [--queue queues/vsi.json] [--out outputs]
  perception.py groups --num-groups 16                    the queue split into 16 groups of about equal GPU time
  perception.py run    --gpus 0 --num-groups 16 --group 3  only group 3 (0 to 15); inputs with a scene.blend are skipped
  perception.py unlock <id>                                 free an input whose lane died
One lane per GPU. A lane takes the next input nobody has (a lock file per input), so lanes on several machines can
share one queue. A failed step is retried (cached stages resume); an attempt that ran out of GPU memory (another job
took the room) is not counted: the step waits for 60 GB and starts again. Each lane goes over the queue twice, the
second time taking over inputs a lane gave up on for lack of GPU memory. Log: <out>/perception.log."""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:                     # plain lines instead of progress bars
    tqdm = None

HERE = Path(__file__).resolve().parent
PY = sys.executable
MACHINE = HERE / "configs" / "machine.json"
TASKS = ("T1-T4", "T5", "T6", "T7")
LOCK, DONE = ".lock", ".perceived"
STAGES = ["decode", "geometry", "camera_depth", "vocabulary", "tracks", "proposals", "proposal_names",
          "proposal_tracks", "fuse"]        # run_codify.py perceive, in order
STEPS = {"T1-T4": STAGES + ["objects", "meshes"], "T5": STAGES, "T6": STAGES, "T7": STAGES}


class Bar:
    """One input's progress over its steps on this lane's line: a tqdm bar, or a plain line per step without tqdm."""

    def __init__(self, item_id, task, gpu, position):
        self.steps, self.label, self.name = STEPS[task], f"gpu {gpu} {item_id}", None
        self.bar = tqdm(total=len(self.steps), desc=self.label, position=position, leave=True, dynamic_ncols=True,
                        bar_format="{desc} |{bar}| {n}/{total}{postfix} [{elapsed}]") if tqdm else None

    def set(self, name):
        if name == self.name:
            return
        self.name, index = name, self.steps.index(name)
        if self.bar:
            self.bar.n = index
            self.bar.set_postfix_str(name)
        else:
            print(f"{time.strftime('%H:%M:%S')} {self.label}: step {index + 1}/{len(self.steps)} {name}", flush=True)

    def close(self, ok):
        if self.bar:
            self.bar.n = len(self.steps) if ok else self.bar.n
            self.bar.set_postfix_str("done" if ok else f"FAILED at {self.name}")
            self.bar.close()
        else:
            print(f"{time.strftime('%H:%M:%S')} {self.label}: {'done' if ok else 'FAILED at ' + str(self.name)}", flush=True)


def current_stage(run):
    """The perception stage run_codify.py perceive is on, from its run.log."""
    current = "decode"
    log_path = run / "run.log"
    if log_path.is_file():
        for line in log_path.read_text(errors="replace").splitlines():
            m = re.search(r"\[(\w+?)(?:_[0-9a-f]{10})?\] (running|resume)", line)
            if m and m.group(1) in STAGES:
                current = m.group(1)
    return current


def machine():
    return json.loads(MACHINE.read_text())


def log(out, message):
    with open(out / "perception.log", "a") as stream:
        stream.write(time.strftime("%Y-%m-%d %H:%M:%S ") + f"{message}\n")


def free_gb(gpu):
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu)],
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0) / 1024


def cost(item):
    """Estimated GPU minutes of an input's perception, measured with the 27B steps in transformers (the vLLM server
    makes them shorter; only the balance matters here): a VSI video about 66 + 1.07 per second of video, a T5 clip
    12, a T6 image 4, a T7 clip 10."""
    if item["task"] == "T1-T4":
        return 66 + 1.07 * (item.get("seconds") or 0)
    return {"T5": 12, "T6": 4, "T7": 10}.get(item["task"], 10)


def split(items, count):
    """count groups with an even share of every task and about equal estimated GPU time (so about equal video
    length too): task by task, the longest inputs first, each input goes to the group with the least time so far
    among the groups holding the fewest inputs of its task. It depends only on the queue and count, so a group
    holds the same inputs however far the others have got."""
    loads, groups = [0.0] * count, [set() for _ in range(count)]
    for task in sorted({i["task"] for i in items}):
        held = [0] * count
        for item in sorted((i for i in items if i["task"] == task), key=lambda i: (-cost(i), i["id"])):
            fewest = min(held)
            k = min((g for g in range(count) if held[g] == fewest), key=lambda g: (loads[g], g))
            loads[k] += cost(item)
            held[k] += 1
            groups[k].add(item["id"])
    return groups, loads


def queue_of(args):
    if getattr(args, "video", None):
        if args.task not in TASKS:
            raise SystemExit(f"--task is one of {TASKS}")
        return [{"id": args.id or Path(args.video).stem, "task": args.task, "video": str(Path(args.video).resolve())}]
    items = json.loads(Path(args.queue).read_text())
    if getattr(args, "num_groups", 0):           # only this group's inputs, in the queue's order
        if not 0 <= args.group < args.num_groups:
            raise SystemExit(f"--group is 0 to {args.num_groups - 1}")
        members = split(items, args.num_groups)[0][args.group]
        items = [i for i in items if i["id"] in members]
    return items[::-1] if args.order == "longest" else items


def has_scene(S):
    """The input is finished: codification left a scene.blend."""
    return (S / "codify" / "final" / "scene.blend").is_file()


def out_of_memory(S, name):
    """The last attempt of step name ran out of GPU memory: its log, the model worker's log it names, or a model
    worker's log under the input written during that attempt (the 27B checks name none) says so."""
    path = S / f"{name}.out"
    if not path.is_file():
        return False
    last = path.read_text(errors="replace").rsplit("\n=== ", 1)[-1]
    logs = [Path(p) for p in re.findall(r"inspect (\S+worker\.log)", last)]
    began = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", last)
    if began:
        since = time.mktime(time.strptime(began.group(1), "%Y-%m-%d %H:%M:%S"))
        logs += [p for p in S.rglob("worker.log") if p.stat().st_mtime >= since]
    texts = [last] + [p.read_text(errors="replace")[-20000:] for p in logs if p.is_file()]
    return any("out of memory" in t.lower() or "outofmemoryerror" in t.lower() for t in texts)


def gave_up_on_memory(out, S, item_id, locked):
    """The lane holding the lock taken at time locked logged a step of this input FAILED, and that step's last
    attempt ran out of GPU memory: another job had the room, the input itself is fine."""
    if not (out / "perception.log").is_file():
        return False
    for line in (out / "perception.log").read_text(errors="replace").splitlines():
        m = re.match(rf"(\S+ \S+) {re.escape(item_id)}: (\S+) FAILED", line)
        if m and time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")) >= locked and out_of_memory(S, m.group(2)):
            return True
    return False


def take(out, S, item_id):
    """Lock an input for this lane; False when another lane (here or on another machine) has it. A lock whose lane
    gave up when its GPU ran out of memory is taken over, by one lane only (a marker per stale lock)."""
    S.mkdir(parents=True, exist_ok=True)
    stamp = f"{socket.gethostname()} pid {os.getpid()} {time.strftime('%F %T')}"
    try:
        locked = int((S / LOCK).stat().st_mtime)
    except FileNotFoundError:
        locked = None
    if locked is not None and gave_up_on_memory(out, S, item_id, locked):
        try:
            os.close(os.open(S / f"{LOCK}.takeover_{locked}", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            return False
        (S / f"{LOCK}.{os.getpid()}").write_text(stamp + " (took over after an out-of-memory failure)\n")
        os.replace(S / f"{LOCK}.{os.getpid()}", S / LOCK)
        log(out, f"{item_id}: taken over after an out-of-memory failure ({socket.gethostname()})")
        return True
    try:
        fd = os.open(S / LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, (stamp + "\n").encode())
    os.close(fd)
    return True


def step(out, S, item_id, name, command, done, gpu, need_gb, tries=3, memory_tries=10, watch=None):
    """Run command until done() holds, at most tries attempts; an attempt that ran out of GPU memory is not counted
    (at most memory_tries of them): the step waits until the GPU has 60 GB free and starts again. watch() is called
    every few seconds while the command runs (the progress bar)."""
    if done():
        return True
    attempt = shortages = 0
    while attempt < tries and shortages <= memory_tries:
        while free_gb(gpu) < (need_gb if not shortages else max(need_gb, 60)):
            time.sleep(60)
        with open(S / f"{name}.out", "a") as stream:
            stream.write(f"\n=== {time.strftime('%F %T')} {socket.gethostname()} gpu {gpu} attempt {attempt + 1}"
                         + (f" (after {shortages} out of GPU memory)" if shortages else "") + "\n")
            stream.flush()
            process = subprocess.Popen(command, cwd=HERE, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if watch:
                    watch()
                time.sleep(5)
        if done():
            break
        if out_of_memory(S, name):
            shortages += 1
            time.sleep(600)
        else:
            attempt += 1
            time.sleep(120)
    ok = done()
    log(out, f"{item_id}: {name} {'done' if ok else 'FAILED'} (gpu {gpu} on {socket.gethostname()})")
    return ok


def perceive(args, item, gpu):
    """All perception of one input; True when it is ready for codification."""
    out, task, item_id = args.out, item["task"], item["id"]
    S = out / item_id
    run = S / "perception"
    source = ["--sample", item["sample"]] if item.get("sample") else \
        ["--video", item["video"], "--config", str(HERE / "configs" / "tasks" / f"{task}.json")]
    perceived = lambda: (run / "run.json").is_file() and bool(json.loads((run / "run.json").read_text()).get("evidence"))
    bar = Bar(item_id, task, gpu, getattr(args, "position", 1))
    bar.set("decode")
    if not step(out, S, item_id, "perception", [PY, "run_codify.py", "perceive", *source, "--output", str(run),
                                                "--gpu", str(gpu)], perceived, gpu, 25,
                watch=lambda: bar.set(current_stage(run))):
        bar.close(False)
        return False
    if task == "T1-T4":
        vlm = machine()["vocabulary"]
        evidence = S / "objects" / "evidence"
        bar.set("objects")
        if not step(out, S, item_id, "objects", [PY, "-m", "codify.instances", str(run), "--out", str(S / "objects"),
                                                 "--cache", str(S / "objects_cache"), "--check-tracks", "--check",
                                                 "--gpu", str(gpu), "--vlm-python", vlm["python"], "--vlm-weights",
                                                 vlm["weights"], "--merge-revisits", "--min-views", "3", "--min-frames",
                                                 "2", "--name-share", "0.25", "--evidence", str(evidence)],
                    lambda: (evidence / "manifest.json").is_file(), gpu, 2 if args.qwen_url else 60, tries=2):
            bar.close(False)
            return False
        meshes = S / "objects" / "evidence_meshes"
        bar.set("meshes")
        if not step(out, S, item_id, "meshes", [PY, "-m", "codify.prepare_room", str(run), str(evidence), str(meshes),
                                                "--gpu", str(gpu), "--workers", "2", "--machine", str(MACHINE)],
                    meshes.is_dir, gpu, 45, tries=2):
            bar.close(False)
            return False
        (S / "config_codify.json").write_text(json.dumps({
            "include": str(HERE / "configs" / "tasks" / "T1-T4.json"), "evidence": {"given": str(meshes)},
            "task": {"sample_id": None, "bench_task": "T1"}}, indent=1) + "\n")
        if not (S / "decode_only").exists():        # codification reuses only the decoded frames (hard links)
            (S / "decode_only" / "work").mkdir(parents=True)
            subprocess.run(["cp", "-al", str(run / "work" / "decode"), str(S / "decode_only" / "work") + "/"], check=True)
            subprocess.run(["cp", str(run / "work" / "decode.stage.json"), str(S / "decode_only" / "work") + "/"], check=True)
    (S / DONE).write_text(f"{time.strftime('%F %T')} {socket.gethostname()} gpu {gpu}\n")
    log(out, f"{item_id}: perceived (gpu {gpu} on {socket.gethostname()})")
    bar.close(True)
    return True


class QwenServer:
    """This lane's Qwen server (codify/qwen.py): started before the lane's first input, kept asleep (the weights in CPU
    memory) except while a step asks the model, restarted if it died, stopped when the lane ends. Its steps find it
    through CODIFY_QWEN_URL."""

    def __init__(self, out, gpu):
        self.out, self.gpu, self.process = out, gpu, None

    def ensure(self):
        if self.process is not None and self.process.poll() is None:
            return
        from codify import qwen
        vocab = machine()["vocabulary"]
        for attempt in range(6):
            while free_gb(self.gpu) < qwen.AWAKE_GB:      # the first load puts the whole model on the card
                time.sleep(60)
            try:
                self.process, url = qwen.start(vocab["python"], vocab["weights"], self.gpu,
                                               self.out / f"qwen_server_{socket.gethostname()}_gpu{self.gpu}.log")
                qwen.sleep(url)
                os.environ[qwen.URL_ENV], os.environ[qwen.GPU_ENV] = url, str(self.gpu)
                log(self.out, f"qwen server up and asleep on gpu {self.gpu} ({socket.gethostname()})")
                return
            except RuntimeError as error:
                log(self.out, f"qwen server on gpu {self.gpu} failed to start ({error}); retrying")
                time.sleep(300)
        raise SystemExit(f"the Qwen server would not start on gpu {self.gpu}")

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.process.kill()


def lane(args, gpu):
    if args.qwen_url:                   # the shared server (perception.py qwen-server): every step asks it
        from codify import qwen
        os.environ[qwen.URL_ENV], os.environ[qwen.SHARED_ENV] = args.qwen_url, "1"
    server = QwenServer(args.out, gpu) if args.qwen_server and not args.qwen_url else None
    count = 0
    try:
        for _ in range(2):              # the second pass takes over what lanes gave up for lack of GPU memory
            for item in queue_of(args):
                S = args.out / item["id"]
                if (S / DONE).exists() or has_scene(S) or not take(args.out, S, item["id"]):
                    continue
                if server:
                    server.ensure()
                perceive(args, item, gpu)
                count += 1
                if args.limit and count >= args.limit:
                    break
            if args.limit and count >= args.limit:
                break
    finally:
        if server:
            server.stop()
    log(args.out, f"lane gpu {gpu} on {socket.gethostname()}: finished ({count} inputs)")


def check(args):
    m = machine()
    ok = True
    paths = {"harness python": PY, "DA3 python": m["geometry"]["python"], "DA3 weights": m["geometry"]["weights"],
             "DA3-Streaming code": m["geometry"]["streaming_paths"]["code_dir"],
             "SALAD weights": m["geometry"]["streaming_paths"]["salad_weights"],
             "DINOv2 hub code": m["geometry"]["streaming_paths"]["dinov2_hub_dir"],
             "VLM python": m["vocabulary"]["python"], "VLM weights (Qwen3.8-27B)": m["vocabulary"]["weights"],
             "SAM 3 python": m["tracks"]["python"], "SAM 3 weights": m["tracks"]["weights"],
             "SAM 3D python": m["objects3d"]["python"], "SAM 3D repo": m["objects3d"]["repo"],
             "SAM 3D checkpoints": m["objects3d"]["config"], "bench data": MACHINE.parent / m["bench_root"],
             "Hugging Face cache (MoGe)": Path(m["env"]["HF_HOME"]) / "hub" / "models--Ruicheng--moge-vitl",
             "torch hub (DINOv2 for SAM 3D)": Path(m["env"]["TORCH_HOME"]) / "hub" / "checkpoints"}
    for name, path in paths.items():
        exists = Path(path).exists()
        ok &= exists
        print(f"{'ok ' if exists else 'MISSING'} {name}: {path}")
    for name, python in (("harness", PY), ("DA3", m["geometry"]["python"]), ("VLM", m["vocabulary"]["python"]),
                         ("SAM 3", m["tracks"]["python"]), ("SAM 3D", m["objects3d"]["python"])):
        r = subprocess.run([python, "-c", "import sys; print(sys.version.split()[0])"], capture_output=True, text=True)
        ok &= r.returncode == 0
        print(f"{'ok ' if r.returncode == 0 else 'BROKEN'} {name} python runs: {r.stdout.strip() or r.stderr.strip()[-200:]}")
    ram = int(subprocess.run(["awk", "/MemAvailable/ {print $2}", "/proc/meminfo"], capture_output=True,
                             text=True).stdout.strip() or 0) / 2 ** 20
    need = 0 if getattr(args, "qwen_url", None) else 86 * max(1, len(args.gpus))
    ok &= ram >= need
    print(f"{'ok ' if ram >= need else 'LOW'} RAM available {ram:.0f} GB (a lane's own sleeping Qwen server keeps "
          f"about 86 GB in memory; with --qwen-url the lanes share one server: {need} GB needed)")
    for tool in ("ffmpeg", "ffprobe", "nvidia-smi"):
        found = subprocess.run(["which", tool], capture_output=True).returncode == 0
        ok &= found
        print(f"{'ok ' if found else 'MISSING'} {tool}")
    for gpu in args.gpus:
        gb = free_gb(gpu)
        print(f"{'ok ' if gb >= 72 else 'LOW'} gpu {gpu}: {gb:.0f} GB free (the lane's Qwen server starts once 72 GB "
              "are free and needs about 60 while awake)")
    try:
        args.out.mkdir(parents=True, exist_ok=True)
        probe = args.out / f".write_test_{socket.gethostname()}"
        probe.write_text("ok")
        probe.unlink()
        print(f"ok  {args.out} is writable")
    except OSError as error:
        ok = False
        print(f"MISSING write access to {args.out}: {error}")
    print("ready" if ok else "not ready: fix the lines above (paths live in configs/machine.json)")
    return ok


def groups(args):
    """The balanced split of the queue into --num-groups groups."""
    items = json.loads(Path(args.queue).read_text())
    members, loads = split(items, args.num_groups or 16)
    tasks = sorted({i["task"] for i in items})
    print(f"{len(items)} inputs in {len(members)} groups (estimated GPU hours of perception per group, measured with "
          "the transformers 27B steps; the vLLM server shortens them):")
    for k, (ids, load) in enumerate(zip(members, loads)):
        mine = [i for i in items if i["id"] in ids]
        seconds = sum(i.get("seconds") or 0 for i in mine)
        per_task = ", ".join(f"{t} {sum(i['task'] == t for i in mine)}" for t in tasks)
        done = sum(((args.out / i["id"]).joinpath(DONE).exists() or has_scene(args.out / i["id"])) for i in mine)
        print(f"  group {k:2d}: {len(ids):3d} inputs ({per_task}), {seconds / 60:5.1f} min of VSI video, "
              f"~{load / 60:5.1f} GPU h, {done} done")


def status(args):
    items = queue_of(args)
    done, running, waiting = [], [], []
    for item in items:
        S = args.out / item["id"]
        if (S / DONE).exists() or has_scene(S):
            done.append(item["id"])
        elif (S / LOCK).exists():
            age = (time.time() - (S / LOCK).stat().st_mtime) / 3600
            running.append(f"{item['id']} ({(S / LOCK).read_text().strip()}, {age:.1f} h)")
        else:
            waiting.append(item["id"])
    print(f"{len(items)} inputs | perceived {len(done)} | running {len(running)} | waiting {len(waiting)}")
    for line in running:
        print("  running:", line)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["check", "run", "status", "groups", "unlock", "qwen-server"])
    p.add_argument("item", nargs="?", help="for unlock: the input's id")
    p.add_argument("--gpus", default="", help="comma-separated GPU indices, one lane each")
    p.add_argument("--queue", default=str(HERE / "queues" / "vsi.json"),
                   help="JSON list of inputs: {id, task, video} or {id, task, sample} (queues/vsi.json, queues/t5_t7.json)")
    p.add_argument("--video", help="one input of your own (a video or an image) instead of a queue")
    p.add_argument("--task", help="with --video: T1-T4 (a room walkthrough), T5, T6 (an image) or T7 (a short clip)")
    p.add_argument("--id", help="with --video: the output folder's name (default: the file name)")
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="output root (default: outputs/)")
    p.add_argument("--order", choices=["shortest", "longest"], default="shortest",
                   help="queue order (queues/vsi.json is shortest first)")
    p.add_argument("--limit", type=int, default=0, help="inputs per lane (0: all)")
    p.add_argument("--num-groups", type=int, default=0,
                   help="split the queue into this many groups of about equal GPU time (see the groups command)")
    p.add_argument("--group", type=int, default=0, help="with --num-groups: the group to run, 0 to N-1")
    p.add_argument("--no-qwen-server", dest="qwen_server", action="store_false",
                   help="load the Qwen model in each step instead of keeping one sleeping server per lane")
    p.add_argument("--qwen-url", help="use this shared Qwen server (perception.py qwen-server) instead of one per lane, "
                                      "e.g. http://127.0.0.1:8701")
    p.add_argument("--port", type=int, default=8701, help="for qwen-server: its port")
    args = p.parse_args()
    args.out = args.out.resolve()
    args.gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    os.environ.update(machine().get("env", {}))        # the model caches (configs/machine.json env) for every model process
    if args.command == "groups":
        return groups(args)
    if args.command == "check":
        sys.exit(0 if check(args) else 1)
    if args.command == "status":
        return status(args)
    if args.command == "unlock":
        (args.out / args.item / LOCK).unlink(missing_ok=True)
        return print("unlocked", args.item)
    if args.command == "qwen-server":   # the shared server, always awake, on the first GPU given
        from codify import qwen
        if not args.gpus:
            sys.exit("give --gpus (one GPU for the shared Qwen server)")
        vocab = machine()["vocabulary"]
        args.out.mkdir(parents=True, exist_ok=True)
        return qwen.serve(vocab["python"], vocab["weights"], args.gpus[0],
                          args.out / f"qwen_shared_{socket.gethostname()}_gpu{args.gpus[0]}.log", port=args.port)
    if not args.gpus:
        sys.exit("give --gpus")
    args.out.mkdir(parents=True, exist_ok=True)
    log(args.out, f"start on {socket.gethostname()}: gpus {args.gpus}, queue "
                  f"{args.video or args.queue}, order {args.order}")
    items = queue_of(args)
    perceived = lambda: sum((args.out / i["id"] / DONE).exists() or has_scene(args.out / i["id"]) for i in items)
    overall = tqdm(total=len(items), initial=perceived(), desc="perceived", position=0, dynamic_ncols=True,
                   bar_format="{desc} |{bar}| {n}/{total} [{elapsed}]") if tqdm else None
    lanes = []
    for index, gpu in enumerate(args.gpus):
        child = os.fork()
        if child == 0:
            args.position = index + 1   # this lane's progress bar sits under the overall one
            lane(args, gpu)
            os._exit(0)
        lanes.append(child)
        time.sleep(30)                  # stagger the lanes' first stages
    while lanes:                        # the overall bar follows the queue until every lane has finished
        lanes = [c for c in lanes if os.waitpid(c, os.WNOHANG) == (0, 0)]
        if overall:
            overall.n = perceived()
            overall.refresh()
        time.sleep(30)
    if overall:
        overall.close()


if __name__ == "__main__":
    main()
