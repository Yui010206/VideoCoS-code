"""The run's modeling tools over MCP (stdio, JSON-RPC 2.0, one JSON object per line), for an agentic coding CLI (Codex)
that writes the scene in place of the API loop (codex_agent.py).

The tools are tools.Workspace's, unchanged: inspect_video, find_objects, execute_blender, render_scene; plus
execute_blender_file (the same build from a scene.py in the CLI's working directory, so it edits files instead of
sending the code as an argument), find_objects_result (waits for a running search) and submit (the final revision,
accepted under the API loop's rule: eligible and covering every certain object). Images come back as image content
and as file paths. A render counts as reviewed once the model makes its next call after seeing it.

State stays in the run directory (revisions.json, images/, evidence/, codex/submission.json), where the harness reads
it back when the CLI exits. stdout carries only the protocol; everything else goes to stderr.
A blind session (codex_agent: no perception for the agent) builds against evidence/blind and has no perception tools;
a raw one (the perception's layers as files, no fused objects) builds against evidence/raw with the same tools.
usage: python -m codify.mcp_server --session <run>/codex/session.json
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

from . import consistency, contract, judge, objects3d, perception, sandbox, tools
from .perception import now

PROTOCOL = "2025-06-18"
SUBMIT = {"name": "submit", "description": (
    "Hand over the final scene: the revision that holds every certain evidence object, with each uncertain one built or "
    "listed in left_out with the reason the frames show; an evidence object that is the same physical object as a "
    "registered one (a duplicate track, or a part of it) goes in left_out with same_as: that object's id, and one that "
    "is a piece of the room's structure (wall, floor, ceiling, window) with same_as: \"environment\". Accepted "
    "only if that revision built, passed its check and its source|render images were shown to you; otherwise the "
    "reasons come back and you continue."),
    "inputSchema": {"type": "object", "additionalProperties": False, "required": ["selected_revision", "summary"],
                    "properties": {"selected_revision": {"type": "string"}, "summary": {"type": "string"},
                                   "left_out": {"type": "array", "items": {
                                       "type": "object", "required": ["id", "reason"],
                                       "properties": {"id": {"type": "string"}, "reason": {"type": "string"},
                                                      "same_as": {"type": "string", "description": (
                                                          "the registered object this one is the same physical "
                                                          "object as (a duplicate track or a part of it), or "
                                                          "\"environment\" for a piece of the room built as "
                                                          "structure (wall, floor, ceiling, window)")}}}},
                                   "limitations": {"type": "array", "items": {"type": "string"}}}}}
BUILD_FILE = {"name": "execute_blender_file", "description": (
    "execute_blender with the scene.py read from a file (a path in your working directory): build it as a new "
    "revision, with the same result. Edit the file and build again for the next revision."),
    "inputSchema": {"type": "object", "additionalProperties": False, "required": ["path"],
                    "properties": {"path": {"type": "string"}}}}
PYTHON = {"name": "run_python", "description": (
    "Run a Python script (Python 3.11 with numpy only: no PIL, cv2 or scipy; no network) in the isolated sandbox and get what it prints: the "
    "evidence folder is read-only at /evidence (raw layers: sys.path.insert(0, '/evidence'); import raw) and /out is a "
    "scratch folder. For measuring the evidence (masks, depth, points, sizes, positions) when shell commands do not "
    "run here."),
    "inputSchema": {"type": "object", "additionalProperties": False, "required": ["code"],
                    "properties": {"code": {"type": "string", "maxLength": 100000},
                                   "timeout_s": {"type": "integer", "minimum": 10, "maximum": 900}}}}
WAIT = {"name": "find_objects_result", "description": (
    "Wait for the running find_objects search (up to `minutes`, default 10) and return what it found; the new evidence "
    "version is then used by every later build."),
    "inputSchema": {"type": "object", "additionalProperties": False,
                    "properties": {"minutes": {"type": "number", "minimum": 0.5, "maximum": 30}}}}


class Server:
    def __init__(self, session_path):
        self.session_path = Path(session_path)
        session = contract.read_json(self.session_path)
        self.work = self.session_path.parent
        settings = session["settings"]
        box = sandbox.Sandbox(settings["blender_dir"], cpus=settings.get("sandbox_cpus", 8),
                              memory=settings.get("sandbox_memory", "8g"), timeout=settings.get("sandbox_timeout", 900),
                              room=bool(settings.get("fast_room")))
        log = lambda *a: print(*a, file=sys.stderr, flush=True)
        self.workspace = tools.Workspace(session["run"], perception.Perception(Path(session["run"]), Path(session["video"]),
                                                                              session["config"], log), box, settings)
        self.mode = session.get("evidence_mode") or ("blind" if session.get("blind") else None)
        self.blind = self.mode is not None           # no fused objects for the agent: the input only, or raw layers
        if self.blind:                               # codex_agent.blind_evidence / raw_evidence
            self.workspace.evidence = self.workspace.run / "evidence" / self.mode
        else:
            self.workspace.prepare_evidence()        # cached stages: the version the harness published
        # Image ids continue after the harness's (the opening frames), and earlier revisions stay known.
        for path in sorted((self.workspace.run / "images").glob("img*.jpg")):
            self.workspace.images[path.stem] = {"path": str(path), "label": "opening image", "sha256": ""}
        if (self.workspace.run / "revisions.json").is_file():
            self.workspace.revisions = contract.read_json(self.workspace.run / "revisions.json")
        self.pending = []                            # renders shown to the model, reviewed by its next call
        self.started = time.time()                   # the session's clock, for settings["min_submit_minutes"]
        # The independent review at submit (judge.py) and the video check every build carries (consistency.py).
        self.judge = settings.get("judge") or None
        self.reviews = contract.read_json(self.work / "reviews.json") if (self.work / "reviews.json").is_file() else []
        if (self.judge or settings.get("video_check")) and self.workspace.evidence.name != "blind":
            import threading
            counts = [(settings.get("video_check") or {}).get("frames", 6)] + ([self.judge.get("frames", 12)] if self.judge else [])
            threading.Thread(target=self._prepare, args=(counts,), daemon=True).start()
        self.tools = [{"name": t["function"]["name"], "description": t["function"]["description"],
                       "inputSchema": t["function"]["parameters"]} for t in tools.tool_schemas(settings)]
        self.tools += [BUILD_FILE, WAIT, SUBMIT, PYTHON]
        # Complete objects from the frames (objects3d.py): the SAM 3D worker loads while the agent looks around.
        self.generator = None
        self.inventoried = set()                     # categories the inventory tool laid out (fused evidence)
        if settings.get("objects3d") and self.mode != "blind":
            self.generator = objects3d.Generator(settings["objects3d"], self.workspace.run / "objects3d", log)
            self.generator.start()
            self.tools += [objects3d.SCHEMA, objects3d.TRACK_SCHEMA]
            if self.mode is None:                     # fused evidence: the reconstruction's depth, cached once
                self.tools.append(objects3d.INVENTORY_SCHEMA)
                import threading
                threading.Thread(target=lambda: objects3d.FusedLayers(self.workspace).prepare_depth(),
                                 daemon=True).start()
        if self.blind:
            self.tools = [t for t in self.tools if t["name"] not in ("find_objects", "find_objects_result")]
            for t in self.tools:
                if t["name"] == "inspect_video":
                    properties = {k: v for k, v in t["inputSchema"]["properties"].items() if k in ("frame_ids", "zoom")}
                    description = ("Look at source frames (canonical frame indices), or a region of a frame zoomed at the "
                                   "source video's full resolution, for details too small to read.")
                    if self.mode == "raw":                # the raw tracks' masks drawn on the frames
                        properties["tracks"] = {"type": "array", "items": {"type": "string"}, "description": (
                            "track ids (tracks.json) whose masks are outlined and labelled on the frames; [\"all\"] for "
                            "every track detected there")}
                        description += " With tracks, those tracks' masks are outlined and labelled on the frames."
                    t.update(description=description, inputSchema={**t["inputSchema"], "properties": properties})

    def _prepare(self, counts):
        try:
            consistency.prepare(self.workspace, counts)
        except Exception as exc:                      # the check reports its own failure when it runs
            print(f"video check preparation failed: {exc}", file=sys.stderr, flush=True)

    def review(self, rid):
        """The video check on more frames with cards, then the independent reviewer; recorded in reviews.json."""
        log = lambda *a: print(*a, file=sys.stderr, flush=True)
        try:
            report = consistency.check(self.workspace, rid, count=self.judge.get("frames", 12),
                                       cards=self.judge.get("cards", 6))
            verdict = judge.review(self.workspace, rid, report, self.judge, log)
        except Exception as exc:
            verdict = {"revision": rid, "verdict": "error", "error": f"{type(exc).__name__}: {exc}"[:400],
                       "fixes": [], "unreliable": [], "score": None}
        self.reviews.append(verdict)
        contract.write_json(self.work / "reviews.json", self.reviews)
        self.trace({"event": "review", "revision": rid, "verdict": verdict["verdict"], "score": verdict.get("score"),
                    "fixes": len(verdict.get("fixes", []))})
        return verdict

    def trace(self, entry):
        with (self.work / "tool_trace.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": now(), **entry}, default=str) + "\n")

    def state(self, **extra):
        contract.write_json(self.work / "state.json", {"evidence": self.workspace.evidence.name, "updated_utc": now(),
                                                       "revisions": sorted(self.workspace.revisions), **extra})

    def call(self, name, arguments):
        ws = self.workspace
        for render in self.pending:                  # the model acted after seeing these
            render["reviewed"] = True
        self.pending.clear()
        ws._save_revisions()
        if name == "submit":
            return self.submit(arguments)
        if name == "find_objects_result":
            ws.wait_for_search(60 * float(arguments.get("minutes", 10)))
            finished = ws.search_result()
            outcome = tools.result("success" if finished else "error",
                                   summary={"find_objects_finished": finished} if finished else {},
                                   error=None if finished else "no finished search to report (none running, already "
                                                               "reported, or still running)")
        elif self.blind and name in ("find_objects", "find_objects_result"):
            outcome = tools.result("error", error="no perception in this run" if self.mode == "blind" else
                                   "no fused objects in this run: the raw layers are files (README.md in the evidence folder)")
        elif self.mode == "raw" and name == "inspect_video" and arguments.get("tracks"):
            outcome = self.track_overlays(arguments)
        elif self.blind and name == "inspect_video" and (arguments.get("layers") or arguments.get("object_ids")):
            outcome = tools.result("error", error="no perception in this run: frames and zooms only" if self.mode == "blind"
                                   else "frames and zooms only here: the masks are files (raw.overlay draws them)")
        elif self.blind and name == "inspect_video" and not (arguments.get("frame_ids") or arguments.get("zoom")):
            canonical = ws.video.get("canonical", {})
            outcome = tools.result("success", summary={"width": canonical.get("width"), "height": canonical.get("height"),
                                                       "frames": len(ws.timeline), "comparison_frames": ws.comparison})
        elif name == "make_objects" and self.generator is not None:
            import numpy as np
            rows, cards = objects3d.make_objects(ws, self.generator, arguments.get("objects") or [],
                                                 lambda *a: print(*a, file=sys.stderr, flush=True))
            images = [ws.register_image(np.vstack(cards[i:i + 3]), "make_objects cards: " + ", ".join(
                r["id"] for r in rows if "error" not in r)[:200], size=1200) for i in range(0, len(cards), 3)]
            failed = [r for r in rows if "error" in r]
            outcome = tools.result("success" if len(failed) < len(rows) else "error", summary={"objects": rows},
                                   images=images, error="; ".join(f"{r['id']}: {r['error']}" for r in failed) or None)
        elif name == "run_python":
            folder = ws.run / "python" / f"{len(list((ws.run / 'python').glob('[0-9]*'))) + 1:04d}" \
                if (ws.run / "python").is_dir() else ws.run / "python" / "0001"
            folder.mkdir(parents=True)
            (folder / "script.py").write_text(arguments["code"])
            r = ws.sandbox.python("script.py", folder, evidence=ws.evidence, timeout=int(arguments.get("timeout_s", 300)))
            text = r.get("stdout_tail", "") + (f"\n--- stderr ---\n{r['stderr_tail']}" if r["stderr_tail"].strip() else "")
            outcome = tools.result("success" if r["status"] == "ok" else "error",
                                   summary={"output": text[-20000:], "seconds": r["seconds"]},
                                   error=None if r["status"] == "ok" else f"{r['status']} (exit {r['returncode']})")
        elif name == "inventory" and self.generator is not None:
            import numpy as np
            rows, cards = objects3d.inventory(ws, self.generator, arguments.get("categories"),
                                              int(arguments.get("max_objects", 64)),
                                              lambda *a: print(*a, file=sys.stderr, flush=True))
            images = [ws.register_image(card, f"inventory: {row.get('category')}", size=1400)
                      for row, card in zip([r for r in rows if "error" not in r], cards)]
            self.inventoried |= {r["category"] for r in rows if "category" in r and "error" not in r}
            outcome = tools.result("success" if cards else "error", summary={"categories": rows}, images=images,
                                   error=None if cards else "; ".join(str(r.get("error")) for r in rows))
        elif name == "track_objects" and self.generator is not None:
            rows = objects3d.track_objects(ws, [str(i) for i in arguments.get("ids") or []],
                                           lambda *a: print(*a, file=sys.stderr, flush=True))
            failed = [r for r in rows if "error" in r]
            outcome = tools.result("success" if len(failed) < len(rows) else "error", summary={"objects": rows},
                                   error="; ".join(f"{r['id']}: {r['error']}" for r in failed) or None)
        elif name == "execute_blender_file":
            path = (self.work / arguments["path"]).resolve()
            if not path.is_file():
                outcome = tools.result("error", error=f"no file {path}")
            else:
                outcome = ws.dispatch("execute_blender", {"scene_code": path.read_text()})
        else:
            outcome = ws.dispatch(name, arguments)
        images = outcome.get("images", [])
        for record in ws.revisions.values():
            for render in record["renders"]:
                if set(render["images"]) & set(images):
                    render["sent"] = True
                    self.pending.append(render)
        ws._save_revisions()
        self.trace({"event": "tool", "name": name, "status": outcome["status"], "summary": outcome.get("summary"),
                    "error": outcome.get("error"), "images": images})
        self.state()
        visible = {k: v for k, v in outcome.items() if k != "images"}
        visible["images"] = [{"id": i, "label": ws.images[i]["label"], "path": ws.images[i]["path"]} for i in images]
        text = json.dumps(visible, default=str)
        text = text if len(text) <= 60000 else text[:60000] + " ... [result truncated at 60000 characters]"
        content = [{"type": "text", "text": text}]
        for i in images:
            content.append({"type": "image", "mimeType": "image/jpeg",
                            "data": base64.b64encode(Path(ws.images[i]["path"]).read_bytes()).decode()})
        return {"content": content, "isError": outcome["status"] != "success"}

    def track_overlays(self, arguments):
        """Raw mode: the given frames with the tracks' masks outlined and labelled (raw.py's overlay)."""
        import importlib.util
        import cv2
        ws = self.workspace
        spec = importlib.util.spec_from_file_location("raw_layers", ws.evidence / "raw.py")
        raw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(raw)
        known = raw.tracks()
        wanted = [str(t) for t in arguments["tracks"]]
        unknown = [t for t in wanted if t != "all" and t not in known]
        if unknown:
            return tools.result("error", error=f"no such tracks: {unknown[:20]}")
        images = []
        frames = arguments.get("frame_ids") or []
        for f in frames[: ws.settings.get("max_images_per_call", 6)]:
            if not 0 <= int(f) < len(ws.timeline):
                return tools.result("error", error=f"frame {f} is outside 0..{len(ws.timeline) - 1}")
            ids = None if "all" in wanted else wanted
            path = ws.run / "images" / f"tracks_{int(f):06d}.jpg"
            raw.overlay(int(f), path, ids)
            images.append(ws.register_image(cv2.imread(str(path)), f"source frame {f} with tracks "
                                                                  f"{'all' if ids is None else ', '.join(ids)}"))
        return tools.result("success" if images else "error", images=images,
                            error=None if images else "give frame_ids with tracks")

    VAGUE = {"item", "object", "thing", "stuff", "fixture", "part", "piece", "surface"}

    def uninventoried(self):
        """A room's four most numerous categories of sizeable things (two or more evidence objects, median extent at
        least 0.25 m, no vague names) not yet laid out by the inventory tool: their counts are where fused evidence
        goes wrong (fragments, repeats)."""
        if self.generator is None or self.mode is not None or not self.workspace.settings.get("inventory_gate", True):
            return []
        from .fusion import head_noun
        sizes = {}
        for o in contract.read_json(self.workspace.evidence / "objects.json").values():
            b = o.get("bounds") or [0] * 6
            sizes.setdefault(head_noun(o.get("category") or ""), []).append(max(b[1] - b[0], b[3] - b[2], b[5] - b[4]))
        import statistics
        top = [c for c, v in sorted(sizes.items(), key=lambda kv: -len(kv[1]))
               if c and c not in self.VAGUE and len(v) >= 2 and statistics.median(v) >= 0.25][:4]
        return [c for c in top if c not in self.inventoried]

    def open_repairs(self, rid):
        """The repairs of a revision's last check (empty when it has none, or no check)."""
        record = self.workspace.revisions.get(rid) or {}
        if not record.get("checks"):
            return []
        path = self.workspace.run / "revisions" / rid / record["checks"][-1]["file"]
        return [r.get("task", str(r)) for r in contract.read_json(path).get("repairs", [])] if path.is_file() else []

    def submit(self, arguments):
        ws = self.workspace
        rid = arguments.get("selected_revision")
        opens = float(ws.settings.get("min_submit_minutes") or 0)
        minute = (time.time() - self.started) / 60
        repairs = self.open_repairs(rid) if opens and rid else []
        if opens and minute < opens and repairs:
            self.trace({"event": "final_proposal", "selected_revision": rid, "eligible": False,
                        "reasons": [f"minute {minute:.1f}: submit opens at minute {opens:g} while repairs remain"]})
            lines = "\n".join(f"- {r}" for r in repairs[:8])
            return {"content": [{"type": "text", "text": (
                f"Not accepted yet: it is minute {minute:.0f} of the run and submit opens at minute {opens:g} while "
                f"{rid} still has repairs. Keep improving the scene: fix these repairs of {rid}:\n{lines}\n"
                "and look at other frames spread over the video (render_scene with frame_ids) to fix the objects that "
                "differ (wrong kind or shape, missing, duplicated, misplaced); build again, then submit.")}],
                "isError": True}
        missing_inventory = self.uninventoried()
        if missing_inventory:
            self.trace({"event": "final_proposal", "selected_revision": rid, "eligible": False,
                        "reasons": [f"inventory missing: {missing_inventory}"]})
            return {"content": [{"type": "text", "text": (
                "Not accepted yet: the room's most numerous categories have not been through the inventory tool: "
                f"{missing_inventory}. Run inventory on them, decide per category which evidence objects are one "
                "physical object (same_as), something else, or not an object, rebuild, and submit again.")}],
                "isError": True}
        ok, why = ws.eligible(rid) if rid else (False, ["no selected_revision"])
        if ok:
            missing, undecided = ws.coverage(rid, arguments.get("left_out"))
            if missing:
                ok, why = False, [f"certain evidence objects not in the scene: {missing[:40]}"]
            if undecided:
                ok, why = False, why + [f"uncertain objects neither built nor in left_out with a reason: {undecided[:40]}"]
        self.trace({"event": "final_proposal", "selected_revision": rid, "eligible": ok, "reasons": why})
        rounds = self.judge.get("rounds", 3) if self.judge else 0
        if ok and len(self.reviews) < rounds:
            verdict = self.review(rid)
            if verdict["verdict"] == "revise" and len(self.reviews) < rounds:
                lines = [f"- {f.get('object')}: {f.get('problem')} -> {f.get('change')}" for f in verdict["fixes"]]
                doubtful = [f"- {u.get('item')}: {u.get('reason')}" for u in verdict["unreliable"]]
                text = (f"Not accepted after the independent review against the video (review {len(self.reviews)} of "
                        f"{rounds}; score {verdict.get('score')}). Change:\n" + "\n".join(lines) +
                        ("\nObservations the reviewer found unreliable (do not fit the scene to them):\n" +
                         "\n".join(doubtful) if doubtful else "") + "\nBuild again and submit.")
                return {"content": [{"type": "text", "text": text}], "isError": True}
        if ok:
            contract.write_json(self.work / "submission.json", {**arguments, "accepted": True, "time": now()})
            self.state(submitted=rid)
            return {"content": [{"type": "text", "text": f"Accepted: {rid} is the final scene. You are done."}],
                    "isError": False}
        return {"content": [{"type": "text", "text": "Not accepted as final: " + "; ".join(why) +
                             ". Continue, then submit again."}], "isError": True}


def serve(session_path):
    protocol = os.fdopen(os.dup(1), "w", buffering=1)      # the real stdout, for JSON-RPC only
    os.dup2(2, 1)                                          # prints and child processes go to stderr
    sys.stdout = sys.stderr
    server = None

    def reply(message_id, result=None, error=None):
        message = {"jsonrpc": "2.0", "id": message_id}
        message.update({"error": error} if error else {"result": result})
        protocol.write(json.dumps(message) + "\n")
        protocol.flush()

    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        method, message_id = message.get("method"), message.get("id")
        if message_id is None:                            # notifications need no answer
            continue
        try:
            if method == "initialize":
                server = server or Server(session_path)
                reply(message_id, {"protocolVersion": message.get("params", {}).get("protocolVersion", PROTOCOL),
                                   "capabilities": {"tools": {"listChanged": False}},
                                   "serverInfo": {"name": "codify", "version": "1"}})
            elif method == "tools/list":
                reply(message_id, {"tools": server.tools})
            elif method == "tools/call":
                params = message.get("params", {})
                reply(message_id, server.call(params["name"], params.get("arguments") or {}))
            elif method == "ping":
                reply(message_id, {})
            else:
                reply(message_id, error={"code": -32601, "message": f"method not found: {method}"})
        except Exception as exc:                          # a tool's own failure is a result, not a dead server
            import traceback
            traceback.print_exc()
            if method == "tools/call":
                reply(message_id, {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True})
            else:
                reply(message_id, error={"code": -32603, "message": f"{type(exc).__name__}: {exc}"})
    if server is not None:
        server.workspace.wait_for_search(600)
        if server.generator is not None:
            server.generator.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    serve(parser.parse_args().session)
