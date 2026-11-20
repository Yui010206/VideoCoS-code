"""Host side of one task's Blender session: a sandboxed, long-lived Blender holding the scene.

The worker (harness/blender/worker.py) runs in the videocos-blender-runtime Docker image with no
network, a read-only root, no capabilities, the caller's uid, CPU/memory/PID limits and no
environment beyond HOME. It sees the Blender install (read-only), this package (read-only) and the
task's work directory (read-write), where the scene is copied first so the example never changes.
``docker=False`` runs the same worker as a plain subprocess, for query-only use and tests.

Transport, timeouts and the reply marker are ported from VideoCoS engine/rpc_process.py; the id
aliasing in ``query`` from engine/scene.py, so the query tools behave as in the reported runs.
"""
from __future__ import annotations

import copy
import json
import os
import queue
import shutil
import subprocess
import threading
import uuid
from collections import deque
from pathlib import Path

from .protocol import MAX_RESPONSE_BYTES, PREFIX, sample_frames, validate_arguments

CODE_ROOT = Path(__file__).resolve().parents[1]
BLENDER_DIR = Path(os.environ.get("VIDEOCOS_BLENDER_DIR", Path.home() / "apps" / "blender-4.5.14-linux-x64"))
IMAGE = "videocos-blender-runtime:1"
QUERY_METHODS = {"scene_inventory", "inspect_entities", "measure_pair", "viewpoint_relation", "camera_history",
                 "track_motion"}
TIMEOUTS = {"run_python": 900, "reset": 300, "render": 14400, "save": 300, "check": 300}
SEMANTIC_FIELDS = ("category", "prototype_name", "motion_type", "geometry_status")


class SessionError(RuntimeError):
    """The Blender process failed, timed out or answered with something unusable."""


class QueryError(ValueError):
    """A query was rejected by the worker; the message is safe to show the model."""


class BlenderSession:
    def __init__(self, scene_blend, workdir, docker=True, cpus=4, memory="8g", timeout=90):
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.original = self.workdir / "original.blend"
        if not self.original.exists():
            shutil.copy2(scene_blend, self.original)
        self.docker, self.cpus, self.memory, self.timeout = docker, cpus, memory, timeout
        self.name = "vreason-" + uuid.uuid4().hex[:12]
        self.process, self._metadata = None, None
        self.restarts = 0
        self.start()

    # ------------------------------------------------------------------ process
    def _command(self):
        worker_args = ["--python", "harness/blender/worker.py"]
        if self.docker:
            return ["docker", "run", "-i", "--rm", "--name", self.name, "--network", "none", "--read-only",
                    "--tmpfs", "/tmp:rw,size=2g", "--user", f"{os.getuid()}:{os.getgid()}", "--cap-drop", "ALL",
                    "--security-opt", "no-new-privileges", "--pids-limit", "1024", "--cpus", str(self.cpus),
                    "--memory", self.memory, "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
                    "-w", "/code", "-v", f"{BLENDER_DIR}:/opt/blender:ro", "-v", f"{CODE_ROOT}:/code:ro",
                    # The scene the session starts from (and reset reloads) is read-only inside the
                    # sandbox, so no edit can overwrite it; outputs go to /work.
                    "-v", f"{self.original}:/scene/original.blend:ro", "-v", f"{self.workdir}:/work:rw", IMAGE,
                    "/opt/blender/blender", "--background", "--factory-startup", "--threads", str(self.cpus),
                    "/scene/original.blend", "--python", "/code/harness/blender/worker.py", "--",
                    "--original", "/scene/original.blend"]
        return [str(BLENDER_DIR / "blender"), "--background", "--factory-startup", "--disable-autoexec",
                "--threads", str(self.cpus), str(self.original), *worker_args[:1],
                str(CODE_ROOT / worker_args[1]), "--", "--original", str(self.original)]

    def start(self):
        env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL") if k in os.environ}
        self._responses, self._logs, self._id = queue.Queue(), deque(maxlen=40), 0
        self.process = subprocess.Popen(self._command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, env=env, cwd=str(CODE_ROOT))
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()
        try:
            ready = self._wait(0, 180)
        except SessionError:
            self.close()        # 2026-11-08: a sandbox that never answered is stopped, not left running as an orphan
            raise
        if ready.get("ok") is not True:
            self.close()
            raise SessionError("Blender rejected the scene: " + str(ready.get("error")))
        self.summary = ready["summary"]
        self._metadata = None

    def restart(self):
        """A fresh worker on the original scene; the caller replays its edits."""
        self.close()
        self.restarts += 1
        self.name = "vreason-" + uuid.uuid4().hex[:12]
        self.start()

    def _drain(self):
        try:
            while chunk := self.process.stderr.readline(4096):
                self._logs.append(chunk.decode("utf-8", errors="replace")[:400])
        except (OSError, ValueError):
            pass

    def _read(self):
        prefix, line_start = PREFIX.encode(), True
        try:
            while True:
                line = self.process.stdout.readline(MAX_RESPONSE_BYTES + 1)
                if not line:
                    raise SessionError("Blender exited before replying: " + " | ".join(list(self._logs)[-4:]))
                if not (line_start and line.startswith(prefix)):
                    line_start = line.endswith(b"\n")
                    if line.strip():
                        self._logs.append(line.decode("utf-8", errors="replace")[:400])
                    continue
                payload = line[len(prefix):]
                while not payload.endswith(b"\n") and len(payload) <= MAX_RESPONSE_BYTES:
                    more = self.process.stdout.readline(MAX_RESPONSE_BYTES - len(payload) + 1)
                    if not more:
                        break
                    payload += more
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise SessionError("Blender response exceeded the size limit")
                value = json.loads(payload.decode("utf-8"))
                self._responses.put(value)
        except SessionError as exc:
            self._responses.put(exc)
        except (OSError, ValueError) as exc:
            self._responses.put(SessionError(f"Blender response stream failed: {type(exc).__name__}"))

    def _wait(self, request_id, timeout):
        try:
            reply = self._responses.get(timeout=timeout)
        except queue.Empty:
            raise SessionError(f"Blender did not answer within {timeout:.0f} s") from None
        if isinstance(reply, Exception):
            raise reply
        if reply.get("id") != request_id:
            raise SessionError("Blender reply id does not match the request")
        if isinstance(reply.get("error"), dict):
            error = reply["error"]
            return {"ok": False, "error": f"{error.get('type', 'Error')}: {error.get('message', 'failed')}"}
        result = reply.get("result")
        if not isinstance(result, dict):
            raise SessionError("Blender reply has no result")
        return result

    def request(self, method, arguments, timeout=None):
        if self.process is None or self.process.poll() is not None:
            raise SessionError("Blender is not running")
        self._id += 1
        data = json.dumps({"id": self._id, "method": method, "arguments": arguments},
                          ensure_ascii=False, allow_nan=False).encode() + b"\n"
        if len(data) > 128 * 1024:
            raise QueryError("request exceeds 128 KiB")
        if timeout is None:
            heavy = method in ("camera_history", "track_motion") or arguments.get("metric") == "surface"
            timeout = TIMEOUTS.get(method, self.timeout * (3 if heavy else 1))
        try:
            self.process.stdin.write(data)
            self.process.stdin.flush()
            return self._wait(self._id, timeout)
        except (OSError, ValueError):
            self.close()
            raise SessionError("Blender request stream closed") from None
        except SessionError:
            self.close()
            raise

    def close(self):
        if self.process is None:
            return
        if self.docker:
            subprocess.run(["docker", "kill", self.name], capture_output=True)
        try:
            if self.process.poll() is None:
                self.process.stdin.close()
                self.process.wait(timeout=10)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            self.process.kill()
        self.process = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------ queries
    @property
    def frame_range(self):
        return list(self.summary["frame_range"])

    def invalidate(self):
        """Edits can add, remove or rename objects: forget the id table and re-read the frame range."""
        self._metadata = None
        summary = self.request("summary", {})
        if summary.get("ok", True):
            self.summary = {**self.summary, **summary}

    def _objects(self):
        if self._metadata is None:
            result, offset = {}, 0
            while offset is not None:
                page = self.request("_scene_metadata", {"offset": offset, "limit": 32})
                if page.get("ok") is False:
                    raise QueryError(page["error"])
                result.update(page["objects"])
                offset = page["next_offset"]
            self._metadata = result
        return self._metadata

    def query(self, name, arguments):
        """One query tool, with authored object ids ('m3') accepted wherever an entity id is."""
        if name not in QUERY_METHODS:
            raise QueryError("Unknown query")
        args = validate_arguments(name, copy.deepcopy(arguments))
        first, last = self.frame_range
        if name == "camera_history":
            frames = sample_frames(args, first, last)
            args = {k: v for k, v in args.items() if k not in {"frames", "start_frame", "end_frame", "step"}} | {"frames": frames}
        elif name not in ("scene_inventory", "track_motion"):
            frame = args.get("frame", first)
            if type(frame) is not int or not first <= frame <= last:
                raise QueryError(f"frame must be an integer within permitted range {[first, last]}")
            args["frame"] = frame
        keys = [k for k in ("a", "b", "observer", "facing", "subject", "entity_id") if k in args]
        requested = [args[k] for k in keys] + args.get("ids", []) + args.get("entity_ids", [])
        generic = (self.summary.get("semantic_object_count") == 0
                   and all(uid.startswith(("mesh:", "collection:")) for uid in requested))
        metadata = {} if generic or not requested else self._objects()
        aliases, no_geometry = {}, {}

        def alias(uid):
            if uid not in metadata:
                return uid
            entity = metadata[uid].get("entity_id")
            if entity is None:
                if name == "inspect_entities" and args.get("detail", "basic") in {"basic", "transform"}:
                    no_geometry[uid] = metadata[uid]
                    return None
                raise QueryError("This object has no mesh geometry")
            aliases[entity] = metadata[uid]
            return entity
        for key in keys:
            args[key] = alias(args[key])
        if "ids" in args:
            args["ids"] = [e for uid in args["ids"] if (e := alias(uid)) is not None]
        if "entity_ids" in args:
            args["entity_ids"] = list(dict.fromkeys(alias(uid) for uid in args["entity_ids"]))
        if name != "inspect_entities":
            result = self.request(name, args)
        else:
            result = (self.request(name, args) if args["ids"] else
                      {"ok": True, "frame": args["frame"], "detail": args.get("detail", "basic"), "entities": [],
                       "units": "meters"})
        if result.get("ok") is False:
            raise QueryError(result["error"])
        if name != "inspect_entities":
            return result
        for entity in result["entities"]:
            if entity.get("id") in aliases:
                semantic = aliases[entity["id"]]
                entity["object_id"] = semantic["object_id"]
                entity["semantic"] = {k: semantic[k] for k in SEMANTIC_FIELDS if k in semantic and semantic[k] != "unknown"}
        transform = args.get("detail") == "transform"
        pose_ids = list(no_geometry) + ([e.get("object_id", e["id"]) for e in result["entities"]] if transform else [])
        if pose_ids:
            states = self.request("_scene_states", {"ids": pose_ids[:20], "frame": args["frame"]})
            if states.get("ok") is False:
                raise QueryError(states["error"])
            poses = states["states"]
            for uid, semantic in no_geometry.items():
                result["entities"].append({
                    "id": uid, "object_id": uid, "name": semantic["name"], "kind": "object",
                    "center": poses[uid]["position"], "bbox_min": None, "bbox_max": None, "dimensions": None,
                    "member_count": 0, "member_preview": [], "geometry_available": False,
                    "geometry_note": "Authored non-mesh object: its live pose is available, but it has no mesh bounds.",
                    "semantic": {k: semantic[k] for k in SEMANTIC_FIELDS if k in semantic and semantic[k] != "unknown"}})
            if transform:
                for entity in result["entities"]:
                    key = entity.get("object_id", entity["id"])
                    if key in poses:
                        entity.update(poses[key])
        return result

    def distance(self, a, b, frame):
        """measure_pair with metric=center, shaped as the reported runs returned it."""
        result = self.query("measure_pair", {"a": str(a), "b": str(b), "frame": frame, "metric": "center"})
        return {**{k: v for k, v in result.items() if k != "ok"}, "object_a": str(a), "object_b": str(b)}
