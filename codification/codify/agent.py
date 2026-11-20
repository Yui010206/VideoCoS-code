"""MLLM modeling loop: observe frames and evidence, write Blender Python, render, compare, revise.

All local perception runs first; the model starts with the full evidence (summary, frames,
annotated frames), sees real images (source frames, overlays, source|render pairs) as image
parts, calls four tools, and finishes by naming a revision. The runner
accepts it only if Workspace.eligible() holds; at a budget limit it keeps the most complete
eligible revision (fallback) or reports failure. Budgets cap model requests, revisions, wall time and
(for paid models) money; a paid client is refused without a configured monetary cap.
No ground truth reaches this loop.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import re
import time
from pathlib import Path

from . import contract
from .perception import now
from .tools import TOOL_SCHEMAS, tool_schemas

CODE = Path(__file__).resolve().parent


class BudgetExceeded(RuntimeError):
    pass


ROOM_HELPERS = ("layout_report",)       # added for the rooms in v2 (2026-11-07): listed only for rooms


def assets_reference(room=True):
    """Signatures and docstrings of codify/assets.py (parsed, so bpy is not needed here); the room helpers
    (ROOM_HELPERS) only when room (settings fast_room)."""
    tree = ast.parse((CODE / "assets.py").read_text())
    lines = [ast.get_docstring(tree)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_") and (room or node.name not in ROOM_HELPERS):
            args = ast.unparse(node.args)
            doc = (ast.get_docstring(node) or "").strip()
            lines.append(f"A.{node.name}({args})" + (f"\n    {doc}" if doc else ""))
    return "\n".join(lines)


SYSTEM = """You write the Blender 4.5 Python (scene.py -> scene.blend) of a semantic, editable scene that reproduces one
input, a video or an image: its objects and their motion, the environment and the camera. Other systems then query the
.blend, render it from new viewpoints, simulate it forward or edit it.

Input. {evidence} The first messages hold the evidence summary (objects with world_aabb = world min/max corners at the
first pose, evidence_level and its reasons; unresolved tracks; structure; scale), one raw object record as scene.py reads
it through A.load_evidence(), and frames covering the whole input with their frame index and time, some of them
annotated (yellow 'id:category' projected boxes; tinted SAM 3 masks, white 'track:prompt' for tracks without a box).

Tools. inspect_video: more frames, annotated frames, depth, object crops, or a region zoomed at full resolution.
find_objects: when the frames show physical objects the objects table lacks, ask perception for them by name, all in
one call; the search runs while you keep working, and a later tool result brings the new evidence objects.
execute_blender: builds a complete scene.py as a new revision and returns errors, printed output, the inventory, its
check (silhouettes against the evidence masks, contacts and penetrations, the plan, joint sweeps, one simulated second
after the last input frame, evidence objects not yet built) as an ordered list of repairs with numbers, and
source|render pairs at the comparison frames {comparison} and a top view. render_scene: other frames or views of a
built revision.

Work: look at the frames and compare them with the objects table; call find_objects once for every physical thing
the frames show that the table lacks (a bag, a towel, a cup ...), then build a complete scene. Fix the build's
`repairs` in order (they carry measurements: penetration depth, silhouette IoU against the SAM 3 masks, floating
gaps, parts off their declared size, joint collisions), compare every source|render pair, fix what differs, build
again.
Finish with only a JSON object {{"selected_revision": "rNNN", "summary": "...", "left_out": [{{"id": "...", "reason":
"..."}}], "limitations": ["..."]}}. The selected revision holds every certain object; each uncertain one is built or
listed in left_out with the reason the frames show (a duplicate of another id, not a physical object, ...).

A finished scene.py:
1. A.setup_scene(ev["video"]) and A.add_source_camera(ev["cameras"]): every input frame on the timeline, the source
   camera on every frame.
2. One registered mesh per physical object (A.add_object, or your own mesh with A.register_object): instance_id = its
   evidence id, category, bounds, location = pose[1:4] and rotation_wxyz = pose[4:8] of poses[0] (a dynamic object
   takes poses = all rows), color = appearance median_srgb. The mesh fills its bounds and has the shape the frames
   show (a laptop is a thin base with a lid, a bottle has a neck, a bowl is hollow): a plain box is right only for a
   box-like thing. On a fixed-camera input (one viewpoint) give every static object its real look: texture =
   A.best_frame(ev, uid) projects the frame where it is best seen onto its visible faces; a picture gets its crop.
   A moving object and every object of a moving-camera walkthrough keep the flat color (median_srgb): a texture
   projected from one frame is wrong from every other place. In a room-scale video build the room like a set: floor
   and walls, then the furniture the frames show (couch, bed, table, cabinets, shelves: boxes and box assemblies with
   the frames' proportions, found with find_objects when the evidence lacks them), then what rests on it; cushions,
   pillows, bags and clothes are boxes with beveled edges, never spheres or ellipsoids; clutter smaller than 20 cm is a
   plain box or cylinder in its color. End scene.py with A.separate_objects(): it moves the smaller of two interpenetrating objects out by the
   measured depth (up to 8 cm) and reports what moved; deeper penetrations and the repairs list are yours to fix.
   How a movable piece is connected decides how it is built; take it from what the frames show (zoom with
   inspect_video) and from what such objects are, by asking what happens when it is used. A piece that stays attached
   while it turns, along an edge (a laptop lid, a door, a flip-top cap, a box flap) or about its own axle (a knob, a
   dial, a wheel), is a hinge: A.add_part with a revolute joint whose pivot is that edge or axle. A piece that stays
   on rails or telescopes (a drawer, a sliding door, an antenna) is a slider: a prismatic joint. A piece that comes
   off completely (the lid of a food container, bucket, jar, pot or shoebox, a bottle cap, a cover, a plug) is
   separate, even when it closes the object: its own active body resting on the base, never a joint (lifting it off
   is not sliding), built with A.add_object under its evidence id or a new id "m1", "m2", ... from its measured box,
   the base keeping its opening. find_objects with a piece's name (lid, cap, door, drawer) measures it; a piece found
   on an object is reported with that object's id and its own box. Perception may list a piece as its own object (a
   lid, a knob); how it is connected still follows from what it is. Declare every object with pieces in
   A.declare_plan: each part with its size, material, connection ("hinge", "slider", "separate" with the id of its
   own object, or "fixed") and the evidence for that connection, plus a must/should checklist; the check compares the
   build with it.
   When the input comes with a goal the scene will be used for, every object the goal names or implies is its own
   body and every action it implies can be carried out in the scene: a hinged or sliding piece has joint limits
   reaching both states, a separate lid can be lifted off, what something goes into is hollow with its opening, and
   what is picked up, placed, inserted, stacked or attached is a free (active) body sized to fit where it goes.
   A physical object the frames show clearly that perception cannot provide (find_objects found nothing usable) is
   built as well, with a new id "m1", "m2", ..., placed from the frames, depth and its neighbors; its source text
   says so.
3. The environment the frames show, as environment meshes (A.add_floor, A.add_wall, A.box_mesh + A.add_environment):
   the supporting surface where resting objects stand (structure.json, A.load_points()), continued past the image
   edges wherever the frames show it continuing, so renders show no edge or gap; in a room, walls just behind
   wall-mounted objects (a flat mounted object's thin axis is the wall normal), openings, and fixed structure only where
   objects and frames support it; elsewhere the backdrop the frames show. Plain materials matched to the frames.
4. Lights and a world background that reproduce the frames' exposure and shadows.
5. Physics, so the scene can be simulated forward from the last input frame: every object gets a rigid body
   (add_object(physics={{...}}) or A.set_physics): role "active" for anything free to fall, roll, slide, tip or be
   pushed, and for anything a hand, clamp or tool is holding (it keeps its measured pose through the input and is
   released at the last frame; the holder itself is passive); "passive" for what stays put (the ground, fixed
   furniture, things mounted or fixed); mass_kg,
   friction, restitution and collision shape from what the object is made of and its size. Environment meshes are
   fixed colliders. Every object keeps its measured pose through the last input frame and an active one is then
   released with its last measured velocity, so a resting object must sit on its support without intersecting it
   (the check flags objects that move after release although they were at rest, and moving objects left passive).
   Say what each object does when things act on it, from what it is made of and what the frames and the text
   show (set_physics behavior=..., properties={{...}}): "rigid", "brittle" (breaks or crushes under load: ceramic,
   glass, styrofoam, an egg; break_load_n), "magnetic" (a magnet: pull_n, reach_m, attracts), "soft" (clay, dough, a
   pillow, a sponge, a balloon), "cloth" (paper, a napkin, fabric), "granular" (sand), "liquid", "heat_source" (a
   flame); the helper library says what each needs. Magnets pull in the simulation; soft, cloth, granular and liquid
   objects are absorbing colliders; every behavior and its properties are read by whoever simulates the scene next.

Rules. {evidence_rule} +Z is up, units are the evidence's (world.json says which; a simulator's scene units are
not real-world sizes). Sizes and positions are the evidence's measurements: never replace them with what such an
object usually measures; change one only where a repair or the frames show it is wrong, by the amount they show.
Never invent camera or object motion. One registered mesh per object
(A.add_object, or A.register_object after A.join of parts) with a unique instance_id; the environment has none; no
billboards or camera-attached planes. scene.py runs in an empty Blender without network and may use only /evidence,
bpy, numpy and codify.assets (import codify.assets as A). Do not use execute_blender to print or explore the evidence
(a build without objects is not a scene; at most 2 are answered); when a frame is unclear, look again with
inspect_video. At most {max_revisions} built scenes (failed builds do not count) and {max_requests} model turns;
the run also has a cost budget and stops when it is spent, so build a complete first scene within your first few
turns and refine it afterwards: a run that stops before any scene is built has no output.

Helper library (codify.assets):
{api}
"""

# What the evidence is, per input: local perception estimates, or a given perception (e.g. an example's reference).
EVIDENCE = {
    "local": ("Perception ran locally before you start: cameras and metric depth (Depth Anything 3), object names from a "
              "local vision-language model, SAM 3 tracks for them, and fused 3D objects (duplicates and revisits "
              "already merged). Certain objects are built as given; you decide each uncertain one from the frames. "
              "Sizes are visible-surface extents and may be extended where the frames show more of an object."),
    "given": ("The perception was given: identities, names, bounds, poses (measured tracks for dynamic objects), the "
              "camera on every frame, colors and artwork crops. Every object is certain and is built exactly as given."),
}
EVIDENCE_RULE = {
    "local": "Never edit evidence files; when you change an estimate, say why in that object's `source` text.",
    "given": "Never edit evidence files or change the given bounds, poses or cameras.",
}


def _text_part(text, style):
    return {"type": "input_text" if style == "responses" else "text", "text": text}


def _image_part(path, detail, style):
    """A JPEG as a Responses API input_image, or a Chat Completions image_url part (OpenAI-compatible
    endpoints such as Gemini's)."""
    url = "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()
    if style == "responses":
        return {"type": "input_image", "image_url": url, "detail": detail}
    return {"type": "image_url", "image_url": {"url": url, "detail": detail}}


class Budget:
    def __init__(self, cfg, paid):
        self.max_requests = int(cfg.get("max_model_requests", 40))
        self.max_wall = float(cfg.get("max_wall_seconds", 3600))
        self.max_usd = cfg.get("max_usd")
        self.prices = cfg.get("usd_per_million_tokens")
        if paid and (self.max_usd is None or not self.prices or not {"input", "output"} <= set(self.prices)):
            raise ValueError("A paid model needs budget.max_usd and budget.usd_per_million_tokens {input, output} "
                             "in the config before any request is sent")
        self.requests, self.usd, self.last_cost = 0, 0.0, 0.0
        self.tokens = {"input": 0, "output": 0}
        self.started = time.time()

    def before_request(self):
        if self.requests >= self.max_requests:
            raise BudgetExceeded(f"model request cap {self.max_requests} reached")
        if time.time() - self.started > self.max_wall:
            raise BudgetExceeded(f"wall-time cap {self.max_wall:.0f} s reached")
        if self.max_usd is not None and self.usd + 1.5 * self.last_cost > float(self.max_usd):
            raise BudgetExceeded(f"monetary cap ${self.max_usd} would be exceeded (spent ${self.usd:.4f})")

    def after_request(self, usage):
        self.requests += 1
        tokens_in = int(usage.get("prompt_tokens") or 0)
        # Thinking is billed as output; Gemini's OpenAI-compatible endpoint counts it only in total_tokens.
        tokens_out = max(int(usage.get("completion_tokens") or 0), int(usage.get("total_tokens") or 0) - tokens_in)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        self.tokens["input"] += tokens_in
        self.tokens["cached_input"] = self.tokens.get("cached_input", 0) + cached
        self.tokens["output"] += tokens_out
        if self.prices:
            # cached input is billed at the cached_input rate when the config gives one
            cached_rate = self.prices.get("cached_input", self.prices["input"])
            self.last_cost = ((tokens_in - cached) * self.prices["input"] + cached * cached_rate
                              + tokens_out * self.prices["output"]) / 1e6
            self.usd += self.last_cost

    def record(self):
        return {"requests": self.requests, "tokens": self.tokens, "api_usd": round(self.usd, 6),
                "max_usd": self.max_usd, "wall_seconds": round(time.time() - self.started, 1)}


def api_model(config):
    """VideoCoS's ChatClient: Responses API (OpenAI, the default) or Chat Completions (OpenAI-compatible
    endpoints such as Gemini's) per model.api_style. The key is taken out of its environment variable, so no
    child process (sandbox, host Blender) inherits it, and is never saved."""
    import os
    import sys
    sys.path.insert(0, str(Path(config["videocos_repo"]).resolve()))
    from videocos.infer.provider import ChatClient
    api = {k: v for k, v in config["model"].items() if k not in ("provider", "api_key_env")}
    api["api_key"] = os.environ.pop(config["model"].get("api_key_env", "OPENAI_API_KEY"), "")
    api.setdefault("api_style", "responses")
    return ChatClient(api)


def _complete(model, messages, trace, attempts=3, tools=TOOL_SCHEMAS):
    """One model turn; transient provider failures (429, 5xx, connection) are retried with backoff."""
    for attempt in range(attempts):
        try:
            return model.complete(messages, tools)
        except Exception as exc:
            if not getattr(exc, "retryable", False) or attempt == attempts - 1:
                raise
            trace({"event": "model_retry", "error": str(exc), "status_code": getattr(exc, "status_code", None)})
            time.sleep(min(120, max(10 * 2 ** attempt, getattr(exc, "retry_after", None) or 0)))


def _final_choice(text):
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None


def opening(workspace, settings, task=None):
    """What the model starts with: frames covering the whole input (by kind of input: one image, a short clip from a
    fixed camera, or a long moving-camera video in many small frames the model zooms into), some of them annotated,
    and the text: video metadata, the full evidence and the task's text. Returns (kind, frames, their image ids,
    annotated frames, their image ids, text)."""
    if workspace.evidence is None:
        workspace.prepare_evidence()
    count = len(workspace.timeline)
    kind = "single" if count == 1 else workspace.camera_motion()
    n = min(count, settings.get("initial_frames", {}).get(kind, {"single": 1, "static": 8, "moving": 256}[kind]))
    initial = sorted({round(i * (count - 1) / max(n - 1, 1)) for i in range(n)})
    small = settings.get("overview_image_size", 448) if kind == "moving" else None
    first = [workspace.register_image(workspace.frame(f), f"source frame {f}", small,
                                      settings.get("overview_detail") if kind == "moving" else None,
                                      72 if kind == "moving" else 80) for f in initial]
    k = min(len(initial), settings.get("annotated_frames", 16))
    marked = [initial[round(i * (len(initial) - 1) / max(k - 1, 1))] for i in range(k)]
    annotated = [workspace.register_image(workspace._layered(f, ["masks", "boxes"]), f"source frame {f} annotated")
                 for f in marked]
    meta = {k: v for k, v in workspace.inspect_video()["summary"].items() if k != "evidence"}
    text = (f"Video metadata: {json.dumps(meta)}\n\nThe full perception evidence:\n{workspace.evidence_text()}"
            f"\n\n{len(initial)} source frames follow, covering the whole input ({kind} input).")
    if task and task.get("text"):
        # Text the task hands over with the input (a scene description, the goal the scene is for): context for the
        # model, never perception.
        text += f"\n\n{task['text']}"
    return kind, initial, first, marked, annotated, text


def fallback(workspace, reasons):
    """The revision kept when the model named none: the most complete eligible one (most objects, the latest of
    equals; a later probe build with one object does not replace a full scene), else the most complete one that built
    and passed check_scene without the model reviewing its renders. Returns (status, revision) and adds the reason."""
    objects = lambda rid: workspace.revisions[rid].get("objects") or 0
    order = sorted(workspace.revisions, key=lambda rid: (objects(rid), rid), reverse=True)
    eligible = [rid for rid in order if workspace.eligible(rid)[0]]
    checked = [rid for rid in order if workspace.revisions[rid]["executed"] and objects(rid) and
               any(c["pass"] and c["blend_sha256"] == workspace.revisions[rid].get("blend_sha256")
                   for c in workspace.revisions[rid]["checks"])]
    if eligible:
        reasons.append(f"no revision named; kept the most complete eligible revision {eligible[0]} "
                       f"({objects(eligible[0])} objects)")
        return "partial", eligible[0]
    if checked:
        reasons.append(f"no revision named; kept {checked[0]}, which built and passed check_scene but whose renders the "
                       "model did not review")
        return "partial", checked[0]
    built = [rid for rid in order if workspace.revisions[rid]["executed"] and objects(rid)]
    if built and (getattr(workspace, "settings", None) or {}).get("keep_failed_build"):
        # No build passed check_scene (a strict layout gate, 2026-11-07): the run still leaves a scene, the latest of the
        # builds with the fewest check failures among those with at least half the median object count (not a probe).
        least = 0.5 * sorted(objects(rid) for rid in built)[len(built) // 2]

        def failures(rid):
            try:
                check = workspace.revisions[rid]["checks"][-1]["file"]
                return len(contract.read_json(workspace.run / "revisions" / rid / check).get("failures", []))
            except Exception:
                return 99
        pick = max((rid for rid in built if objects(rid) >= least), key=lambda rid: (-failures(rid), rid))
        reasons.append(f"no revision passed check_scene; kept {pick}, the latest build with the fewest check failures "
                       f"({failures(pick)}, {objects(pick)} objects)")
        return "partial", pick
    return "failed", None


def run_agent(workspace, model, settings, budget, log=print, task=None):
    """Drive the modeling loop; returns {status, selected_revision, final, reasons}."""
    run = workspace.run
    trace_path = run / "tool_trace.jsonl"
    detail = settings.get("image_detail", "high")
    style = settings.get("model", {}).get("api_style", "responses")
    image = lambda i: _image_part(workspace.images[i]["path"], workspace.images[i].get("detail", detail), style)
    max_images = settings.get("max_images_per_call", 6)
    source = "given" if workspace.perception.config.get("evidence", {}).get("given") else "local"
    system = SYSTEM.format(evidence=EVIDENCE[source], evidence_rule=EVIDENCE_RULE[source],
                           max_revisions=settings.get("max_revisions", 4), max_requests=budget.max_requests,
                           max_images=max_images, comparison=workspace.comparison,
                           api=assets_reference(room=bool(settings.get("fast_room"))))
    kind, initial, first, marked, annotated, first_text = opening(workspace, settings, task)
    stamp = lambda f: f"frame {f}, t = {workspace.timeline[f]['timestamp_ns'] / 1e9:.2f} s"
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": [_text_part(first_text, style)] +
                 [part for f, i in zip(initial, first) for part in (_text_part(stamp(f), style), image(i))]},
                {"role": "user", "content": [_text_part(f"{len(marked)} of these frames annotated (yellow: projected 3D "
                                                        "box candidates 'id:category'; tinted: SAM 3 masks, white "
                                                        "'track:prompt' for tracks without a box).", style)] +
                 [part for f, i in zip(marked, annotated) for part in (_text_part(stamp(f), style), image(i))]}]
    pending_review = []           # renders whose images were sent, awaiting a model turn
    sent_log = [{"message": 1, "images": first}, {"message": 2, "images": annotated}]
    # Every request carries all images so far; past this many megabytes (base64) of images new ones are withheld,
    # under the provider's request size limit (about 20 MB for Gemini).
    image_limit = settings.get("max_image_mb", 16) * 1e6
    encoded = lambda batch: sum(Path(workspace.images[i]["path"]).stat().st_size for i in batch) * 4 / 3
    image_bytes = encoded(first) + encoded(annotated)

    def trace(entry):
        with trace_path.open("a") as stream:
            stream.write(json.dumps({"time": now(), **entry}, default=str) + "\n")

    (run / "system_prompt.txt").write_text(system)
    (run / "first_message_text.txt").write_text(first_text)
    trace({"event": "start", "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
           "input_kind": kind, "task": task, "initial_frames": initial, "annotated_frames": marked,
           "initial_images": first + annotated, "evidence_version": workspace.evidence.name,
           "comparison_frames": workspace.comparison})
    status, selected, final, reasons = "failed", None, None, []
    built = lambda: any(r["executed"] and r.get("objects") for r in workspace.revisions.values())
    grace = False           # one last request past the budget when no scene has been built yet
    while True:
        try:
            budget.before_request()
        except BudgetExceeded as exc:
            if grace or built():
                reasons.append(str(exc))
                break
            grace = True
            reasons.append(f"{exc}; one last request to build the scene")
            messages.append({"role": "user", "content": [_text_part(
                "The budget is spent and no scene has been built: build the complete scene now with "
                "execute_blender. This is the last request.", style)]})
        try:
            payload = _complete(model, messages, trace, tools=tool_schemas(settings))
        except Exception as exc:
            if not hasattr(exc, "retryable"):     # only provider failures end the loop; code errors propagate
                raise
            reasons.append(f"model request failed: {exc}")
            break
        budget.after_request(payload.get("usage") or {})
        message = payload["choices"][0]["message"]
        for render in pending_review:
            render["reviewed"] = True
        pending_review.clear()
        workspace._save_revisions()
        calls = message.get("tool_calls") or []
        trace({"event": "model", "request": budget.requests, "usage": payload.get("usage"),
               "content": message.get("content"), "tool_calls": [
                   {"name": c["function"]["name"], "arguments": _redact(c["function"]["arguments"])} for c in calls]})
        # Replay what the provider returned, including reasoning state (OpenAI reasoning_items, Gemini thought
        # signatures in extra_content on the message and on each tool call).
        messages.append({k: v for k, v in message.items() if k in ("role", "content", "tool_calls", "reasoning_items",
                                                                   "extra_content")})
        if not calls:
            final = _final_choice(message.get("content"))
            if final is None and workspace.search and not workspace.search.get("reported"):
                # A turn without a tool call while a search runs waits for it.
                workspace.wait_for_search()
                finished = workspace.search_result()
                trace({"event": "search_delivered", "result": finished})
                messages.append({"role": "user", "content": [_text_part(
                    "find_objects finished: " + json.dumps(finished, default=str), style)]})
                continue
            rid = (final or {}).get("selected_revision")
            ok, why = workspace.eligible(rid) if rid else (False, ["no JSON object with selected_revision"])
            if ok:
                missing, undecided = workspace.coverage(rid, (final or {}).get("left_out"))
                if missing:
                    ok, why = False, [f"certain evidence objects not in the scene: {missing[:40]}"]
                if undecided:
                    ok, why = False, why + [f"uncertain objects neither built nor in left_out with a reason: "
                                            f"{undecided[:40]}"]
            trace({"event": "final_proposal", "selected_revision": rid, "eligible": ok, "reasons": why})
            if ok:
                status, selected = "success", rid
                break
            messages.append({"role": "user", "content": [_text_part(
                "Not accepted as final: " + "; ".join(why) + ". Continue, then propose again.", style)]})
            continue
        images = []
        for call in calls:
            name = call["function"]["name"]
            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                outcome = {"status": "error", "error": f"arguments are not valid JSON: {exc}"}
            else:
                outcome = workspace.dispatch(name, arguments)
            trace({"event": "tool", "name": name, "status": outcome["status"], "summary": outcome.get("summary"),
                   "error": outcome.get("error"), "images": outcome.get("images", [])})
            images += outcome.get("images", [])
            visible = {k: v for k, v in outcome.items() if k != "images"}
            visible["images_attached_next"] = outcome.get("images", [])
            text = json.dumps(visible, default=str)
            text = text if len(text) <= 60000 else text[:60000] + " ... [result truncated at 60000 characters]"
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
        if budget.max_usd is not None and budget.usd + 4 * budget.last_cost > float(budget.max_usd) and not built():
            messages.append({"role": "user", "content": [_text_part(
                "The cost budget allows about two more turns and no scene has been built yet: build the complete "
                "scene now with execute_blender, without further inspection or searches.", style)]})
        for start in range(0, len(images), max_images):
            batch = images[start:start + max_images]
            if image_bytes + encoded(batch) > image_limit:
                messages.append({"role": "user", "content": [_text_part(
                    f"{len(batch)} images withheld: the images of this conversation reached the request size limit "
                    f"({image_limit / 1e6:.0f} MB). Decide from what you have seen; the final revision must be one "
                    "whose renders you have reviewed.", style)]})
                continue
            image_bytes += encoded(batch)
            labels = "; ".join(f"{i}: {workspace.images[i]['label']}" for i in batch)
            messages.append({"role": "user", "content": [_text_part("Images from the tool calls above: " + labels, style)] +
                             [image(i) for i in batch]})
            sent_log.append({"message": len(messages) - 1, "images": batch})
            for record in workspace.revisions.values():
                for render in record["renders"]:
                    if set(render["images"]) & set(batch):
                        render["sent"] = True
                        pending_review.append(render)
    if status != "success":
        status, selected = fallback(workspace, reasons)
    contract.write_json(run / "images_sent.json", sent_log)
    workspace.wait_for_search()          # a search still running would leave its SAM 3 worker behind
    return {"status": status, "selected_revision": selected, "final": final, "reasons": reasons,
            "budget": budget.record()}


def _redact(arguments):
    """Keep the trace small: scene code is saved in its revision, so record its hash only."""
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return arguments[:2000]
    if "scene_code" in args:
        args["scene_code"] = {"sha256": hashlib.sha256(args["scene_code"].encode()).hexdigest(),
                              "characters": len(args["scene_code"])}
    return args


def finalize(workspace, outcome, settings, handoff_config=None, log=print):
    """Copy the selected revision to final/, reopen and re-check it, and hand it to the engine's smoke tests
    (`run_codify.py preview` makes the report video)."""
    import shutil
    run, rid = workspace.run, outcome.get("selected_revision")
    report = {**outcome, "finalized_utc": now()}
    if not rid:
        contract.write_json(run / "final_report.json", report)
        return report
    final = run / "final"
    shutil.rmtree(final, ignore_errors=True)
    final.mkdir()
    source = run / "revisions" / rid
    for name in ("scene.py", "scene.blend", "inventory.json"):
        shutil.copyfile(source / name, final / name)
    record = workspace.revisions[rid]
    evidence = run / "evidence" / record["evidence_version"]
    check = workspace.sandbox.blender(["check", "--evidence", "/evidence", "--out", "/out/final_check.json"], final,
                                      evidence=evidence, blend="scene.blend")
    final_check = contract.read_json(final / "final_check.json") if check["status"] == "ok" else {"pass": False}
    report["final_check_pass"] = final_check["pass"]
    if not final_check["pass"]:
        report["status"] = "failed"
        report["reasons"].append("reopened final .blend failed check_scene")
    if handoff_config and handoff_config.get("videocos_repo") and report["status"] in ("success", "partial"):
        from . import handoff
        world = evidence / "world.json"
        scale = contract.read_json(world)["scale_status"] if world.is_file() else "relative"
        report["handoff"] = handoff.handoff(final / "scene.blend", workspace.sandbox.blender_dir / "blender",
                                            final / "handoff", handoff_config["videocos_repo"], scale)
    contract.write_json(run / "final_report.json", report)
    return report
