"""What a tool call does on the host, for the two kinds of task.

QueryTools is VideoCoS engine/harness.py VSIHarness, ported unchanged in behaviour: the same context,
the same six tools, the same validation, defaults, size limits and answer acceptance.
UpdateTools adds the live edit loop: run_python on the session's scene, reset, and a submission that
must pass the task contract. Every executed snippet is kept in order (work/edits.py), so the
submitted scene can be rebuilt from the example's scene.blend, and the session can be restored if
Blender crashes or a snippet runs too long.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .answers import answer_format, normalize_answer
from .prompts import update_brief
from .protocol import TOOL_NAMES, UPDATE_TOOL_NAMES, tool_schemas, validate_arguments
from .session import QueryError, SessionError

DEFAULTS = {"scene_inventory": {"kind": "collection"}, "inspect_entities": {"detail": "basic"},
            "camera_history": {"detail": "summary"}}
FRAME_TOOLS = {"inspect_entities", "measure_pair", "viewpoint_relation"}
MAX_ARGUMENT_BYTES, MAX_RESULT_BYTES = 65536, 131072
# Run once when a session starts and after every reset_scene; recorded in the edit log.
PREPARE = {"T6": "S.freeze_physics()  # harness: T6 starts with every rigid body on its keys"}


class QueryTools:
    """One public VSI question over one live scene (VideoCoS VSIHarness)."""

    def __init__(self, session, task):
        self.session, self.task = session, task
        self.public = {**task.public, "kind": "understanding", "question_format": "vsi"}
        if self.public.get("options") is None:
            self.public["options"] = []

    def tools(self):
        return tool_schemas("query")

    def context(self):
        summary = {"evidence_mode": "codified_scene_only", "backend": "live_blender",
                   "object_count": self.session.summary["object_count"],
                   "permitted_frame_range": self.session.frame_range, "fps": float(self.session.summary["fps"]),
                   "units": "meters, Blender Z up, quaternion wxyz",
                   "query_time": "Any permitted integer frame, evaluated live; no scene.json snapshots.",
                   "limitations": "Authored geometry/names may be approximate. Bounds and sampled visibility/collisions do not certify mesh contact, full pixel visibility, or physics."}
        catalog = self.session.query("scene_inventory", {"kind": "collection", "limit": 60})
        return {"task": copy.deepcopy(self.public), "scene": summary, "catalog": catalog,
                "tool_capabilities": "Fixed live queries and declared numeric algorithms. No arbitrary Python, source files, source images, or evaluator feedback.",
                "answer_format": answer_format(self.public)}

    def _query(self, name, args):
        args = {**DEFAULTS.get(name, {}), **args}
        first, last = self.session.frame_range
        if name in FRAME_TOOLS:
            args.setdefault("frame", first)
        frames = [args[k] for k in ("frame", "start_frame", "end_frame") if k in args] + args.get("frames", [])
        if any(not first <= frame <= last for frame in frames):
            raise ValueError("Frame is outside the permitted scene frame range")
        if name == "measure_pair" and args["metric"] == "center":
            return self.session.distance(args["a"], args["b"], args["frame"])
        return self.session.query(name, args)

    def execute(self, name, args):
        if name not in TOOL_NAMES:
            return {"ok": False, "error": "Unknown or unavailable task tool"}
        try:
            validate_arguments(name, args)
            if len(json.dumps(args, allow_nan=False).encode()) > MAX_ARGUMENT_BYTES:
                raise ValueError("Tool arguments exceed 64 KB")
            if name == "submit_answer":
                submission = {k: args[k] for k in ("explanation",) if k in args}
                submission.update(text=normalize_answer(args["text"], self.public), raw_text=args["text"])
                return {"ok": True, "accepted": True, "submission": submission}
            result = self._query(name, args)
            if not isinstance(result, dict):
                raise ValueError("Backend returned an invalid result")
            result = {"ok": True, **{k: v for k, v in result.items() if k != "ok"}}
            if len(json.dumps(result, allow_nan=False).encode()) > MAX_RESULT_BYTES:
                raise ValueError("Tool response exceeds 128 KB; reduce selected objects, samples, or page size")
            return result
        except (ValueError, KeyError, TypeError, IndexError, RuntimeError) as exc:
            return {"ok": False, "error": str(exc)}


class UpdateTools(QueryTools):
    """An update task: the same queries, plus editing the live scene and submitting it."""

    def __init__(self, session, task):
        self.session, self.task = session, task
        self.public = copy.deepcopy(task.public)
        self.log = Path(session.workdir) / "edits.py"
        self.applied = []            # code applied since the last reset, replayed after a crash
        self.calls = []              # every snippet with its outcome, in order
        self._log_header()
        self._prepare()

    def _log_header(self):
        self.log.write_text(f"# Edits for task {self.task.id} ({self.task.family}), in the order they ran on\n"
                            f"# {self.task.scene_blend}. Replay: open that scene in Blender, bind S (harness/blender/helpers.py),\n"
                            f"# exec each block in order. Blocks that raised are kept, commented out.\n")

    def _append_log(self, code, outcome):
        with self.log.open("a") as handle:
            if outcome == "ok":
                handle.write(f"\n# --- block {len(self.calls)} ---\n{code}\n")
            else:
                commented = "\n".join("# " + line for line in code.splitlines())
                handle.write(f"\n# --- block {len(self.calls)} ({outcome}) ---\n{commented}\n")

    def _prepare(self):
        code = PREPARE.get(self.task.family)
        if code:
            result = self.session.request("run_python", {"code": code})
            if not result.get("executed"):
                raise SessionError(f"family preparation failed: {result.get('error')}")
            self.applied.append(code)
            self.calls.append({"code": code, "outcome": "ok", "by": "harness"})
            self._append_log(code, "ok")
        self.session.invalidate()

    def tools(self):
        return tool_schemas("update")

    def context(self):
        overview = self.session.request("overview", {})
        return {"task": {"id": self.task.id, "family": self.task.family, "inputs": copy.deepcopy(self.public)},
                "brief": update_brief(self.task.family, {"spec": self.task.spec}),
                "scene": overview["scene"], "scene_objects": overview["objects"],
                "environment_meshes": overview["environment"],
                "scene_code": {"note": "The program that built this scene with the codify library (not importable "
                                       "here; use bpy and S). Read it to see how each object, part, joint, "
                                       "physics setting and keyframe was made.",
                               "text": self.task.scene_code}}

    def _restore(self):
        """Blender died or hung: a fresh session on the original scene, with every applied edit replayed."""
        self.session.restart()
        for code in self.applied:
            self.session.request("run_python", {"code": code})
        self.session.invalidate()

    def execute(self, name, args):
        if name not in UPDATE_TOOL_NAMES and name not in TOOL_NAMES - {"submit_answer"}:
            return {"ok": False, "error": "Unknown or unavailable task tool"}
        try:
            validate_arguments(name, args)
            if name == "run_python":
                return self._run(args["code"])
            if name == "reset_scene":
                result = self.session.request("reset", {})
                self.applied.clear()
                self.calls.append({"code": "# reset_scene", "outcome": "reset", "by": "model"})
                with self.log.open("a") as handle:
                    handle.write("\n# --- reset_scene: everything above is discarded ---\n")
                self._prepare()
                return {"ok": True, "reset": True, "scene": self.session.summary["frame_range"]}
            if name == "submit_update":
                return self._submit(args)
            if name == "track_motion":
                result = self.session.query("track_motion", args)
                return {"ok": True, **{k: v for k, v in result.items() if k != "ok"}}
            return QueryTools.execute(self, name, args)
        except SessionError as exc:
            try:
                self._restore()
                note = "The scene was restored to its state before this call."
            except SessionError as again:
                note = f"Restoring the scene failed too ({again}); further calls may fail."
            return {"ok": False, "error": f"Blender failed: {exc}. {note}"}
        except (QueryError, ValueError, KeyError, TypeError, IndexError, RuntimeError) as exc:
            return {"ok": False, "error": str(exc)}

    def _run(self, code):
        try:
            result = self.session.request("run_python", {"code": code})
        except SessionError as exc:
            self.calls.append({"code": code, "outcome": "crashed", "by": "model"})
            self._append_log(code, "crashed or timed out")
            self._restore()
            return {"ok": False, "error": f"Blender failed while running this code ({exc}); the scene was restored "
                                          "to its state before the call. Avoid long loops and unbounded work."}
        outcome = "ok" if result.get("executed") else "raised"
        if result.get("executed") or result.get("changes", {}).keys() - {"frame_now"}:
            # A snippet that raised half-way may still have changed the scene: keep it for replay.
            self.applied.append(code)
        self.calls.append({"code": code, "outcome": outcome, "by": "model"})
        self._append_log(code, outcome)
        self.session.invalidate()
        return result

    def _submit(self, args):
        check = self.session.request("check", {"family": self.task.family, "spec": self.task.spec})
        if not check.get("pass"):
            return {"ok": False, "error": "The task contract is not met; nothing was submitted.",
                    "failures": check.get("failures", [])}
        saved = self.session.request("save", {"path": "/work/final.blend" if self.session.docker
                                              else str(Path(self.session.workdir) / "final.blend")})
        if saved.get("ok") is False:
            raise SessionError(saved["error"])
        return {"ok": True, "accepted": True,
                "submission": {"summary": args["summary"], "limitations": list(args.get("limitations", [])),
                               "final_blend": str(Path(self.session.workdir) / "final.blend"),
                               "edits": str(self.log), "blocks": len(self.calls)}}
