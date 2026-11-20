"""Blender entry point: one long-lived session per task, answering one JSON line at a time.

    blender --background --factory-startup SCENE.blend --python worker.py -- --original SCENE.blend

It serves the read-only queries (the same handlers the VSI runs used) and, for update tasks, live
edits: run_python executes the model's code in a namespace that persists for the whole session.
Replies are single lines prefixed with RPC_PREFIX; anything else on stdout is Blender's own log.
Started inside the Docker sandbox by harness.session, so the model's code has no network, no
credentials and a read-only file system apart from the task's work directory.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bpy
import mathutils

from harness.blender import helpers, live, queries
from harness.blender.context import SceneContext
from harness.protocol import MAX_MESSAGE_BYTES, PREFIX, integer, string

MAX_PRINTED = 8000


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=str)


def emit(identifier, result=None, error=None):
    value = {"id": identifier, ("error" if error is not None else "result"): error if error is not None else result}
    try:
        payload = _json(value)
        if len(payload.encode()) > MAX_MESSAGE_BYTES:
            payload = _json({"id": identifier, "error": {"type": "ValueError",
                             "message": "Response exceeds 128 KiB; request fewer entities/frames"}})
    except (ValueError, TypeError):
        payload = _json({"id": identifier, "error": {"type": "ValueError", "message": "non-finite or unsupported value"}})
    print("\n" + PREFIX + payload, flush=True)


class Session:
    def __init__(self, original):
        self.original = original
        self.load()

    def load(self):
        self.ctx = SceneContext()
        self.namespace = {"__name__": "update", "bpy": bpy, "mathutils": mathutils, "math": math, "S": helpers}
        self.reference = live.reference()

    def reset(self):
        bpy.ops.wm.open_mainfile(filepath=self.original)
        self.load()
        return {"ok": True, "reset": True, "summary": self.ctx.summary()}

    def run_python(self, code):
        string(code, "code", 60_000)
        scene = bpy.context.scene
        start = (scene.frame_current, scene.frame_subframe)
        live.evaluate(*start)
        before = live.snapshot()
        printed = io.StringIO()
        error = trace = None
        try:
            with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
                exec(compile(code, "<run_python>", "exec"), self.namespace)  # noqa: S102 - the model's edit, sandboxed
        except BaseException as exc:  # noqa: BLE001 - SystemExit included: a failed snippet is a result
            error = f"{type(exc).__name__}: {exc}"
            lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
            trace = "".join(line for line in lines if "harness/blender/worker.py" not in line)[-3000:]
        index_error = None
        try:
            self.ctx.refresh()
        except Exception as exc:  # noqa: BLE001
            index_error = f"{type(exc).__name__}: {exc}"
        text = printed.getvalue()
        result = {"ok": True, "executed": error is None, "error": error, "traceback": trace,
                  "printed": text[:MAX_PRINTED] + ("\n... (output truncated)" if len(text) > MAX_PRINTED else ""),
                  "changes": self._changes(before, start)}
        if self.ctx.duplicate_ids:
            result["duplicate_instance_ids"] = self.ctx.duplicate_ids[:10]
        if index_error:
            result["scene_index_error"] = index_error
        return result

    @staticmethod
    def _changes(before, start):
        """What the code changed, as the scene renders at the frame the code started on: Blender
        evaluates that frame (keyframes, drivers, physics cache) before the comparison. Code that only
        moves the current frame is not reported as moving objects; the scene is left on the frame the
        code chose, evaluated there, and the move gets its own line."""
        scene = bpy.context.scene
        end = (scene.frame_current, scene.frame_subframe)
        live.evaluate(*start)
        after = live.snapshot()
        if end != start:
            live.evaluate(*end)
        changes = live.diff(before, after)          # its frame_now is the frame the scene is left on
        if end != start:
            changes["frame_change"] = (f"your code moved the current frame from {start[0]} to {end[0]}; the changes "
                                       f"above are compared at frame {start[0]}; the scene stays at frame {end[0]}")
        return changes

    def handle(self, method, arguments):
        if method in queries.HANDLERS:
            return queries.execute(self.ctx, method, arguments)
        if method == "run_python":
            return self.run_python(arguments["code"])
        if method == "reset":
            return self.reset()
        if method == "overview":
            return {"ok": True, **live.overview()}
        if method == "check":
            failures = live.check(arguments["family"], arguments.get("spec", {}), self.reference)
            return {"ok": True, "pass": not failures, "failures": failures}
        if method == "save":
            path = arguments["path"]
            bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
            return {"ok": True, "path": path}
        if method == "render":
            return {"ok": True, **live.render(arguments)}
        raise ValueError("Unknown method")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    original = argv[argv.index("--original") + 1] if "--original" in argv else bpy.data.filepath
    try:
        session = Session(original)
        emit(0, {"ok": True, "summary": session.ctx.summary()})
    except Exception as exc:  # noqa: BLE001
        emit(0, error={"type": type(exc).__name__, "message": str(exc)[:1000]})
        return 1
    while True:
        line = sys.stdin.readline(MAX_MESSAGE_BYTES + 1)
        if not line:
            break
        identifier = None
        try:
            if len(line.encode()) > MAX_MESSAGE_BYTES:
                raise ValueError("Request exceeds 128 KiB")
            request = json.loads(line)
            if not isinstance(request, dict) or set(request) != {"id", "method", "arguments"}:
                raise ValueError("Expected id, method, and arguments")
            identifier = integer(request["id"], "id", 1, 2 ** 53 - 1)
            method = string(request["method"], "method", 64)
            emit(identifier, session.handle(method, request["arguments"]))
        except Exception as exc:  # noqa: BLE001
            message = str(exc)[:1000] if isinstance(exc, (ValueError, KeyError, TypeError)) else \
                "Blender query failed (" + type(exc).__name__ + ")"
            emit(identifier, error={"type": type(exc).__name__, "message": message})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
