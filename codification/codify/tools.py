"""The modeling tools over one run workspace.

All local perception runs before the model starts (prepare_evidence: cameras and depth, object names from a
local VLM, SAM 3 tracks, fused 3D objects, support surface and vertical planes) and is published as one immutable
evidence version that the model receives in full. The model then works with four tools, inspect_video,
find_objects, execute_blender and render_scene, which each return {status, artifacts, summary, warnings, error}.
Images are files registered under IDs; the agent turns the IDs a tool returns into real image inputs for the
next model turn. Held-out frames (chosen before modeling) are refused by inspect_video and render_scene.
Generated scene code only runs inside the sandbox, against that evidence version.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import numpy as np

from . import contract
from .perception import now

LAYERS = ("masks", "depth", "boxes")
INT_LIST = {"type": "array", "items": {"type": "integer", "minimum": 0}, "maxItems": 16}
ZOOM = {"type": "array", "maxItems": 8, "items": {
    "type": "object", "additionalProperties": False, "required": ["frame", "box"], "properties": {
        "frame": {"type": "integer", "minimum": 0},
        "box": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}, "minItems": 4, "maxItems": 4,
                "description": "x0, y0, x1, y1 as fractions of the frame width and height"}}}}
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "inspect_video",
        "description": "Look at actual source frames (canonical frame indices); with layers, the same frames annotated "
                       "(masks: SAM 3 tracks, 'track:prompt' for tracks without a box; boxes: projected 3D box "
                       "candidates, 'id:category') and/or depth maps; crops of evidence objects; or zoom: a region "
                       "of a frame at the source video's full resolution, for details too small to read. With no "
                       "frame_ids/object_ids/zoom, returns metadata and the evidence summary.",
        "parameters": {"type": "object", "additionalProperties": False, "properties": {
            "frame_ids": INT_LIST, "zoom": ZOOM,
            "object_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "layers": {"type": "array", "items": {"type": "string", "enum": list(LAYERS)}, "maxItems": 3}}}}},
    {"type": "function", "function": {
        "name": "find_objects",
        "description": "Ask perception for physical objects the frames show but the objects table lacks, by name (a "
                       "short noun each; give all missing names in one call). It starts a search and returns at "
                       "once: SAM 3 searches the whole input and the new tracks are fused like the others, which "
                       "takes minutes on a long video. Keep working meanwhile; a later tool result reports the tracks "
                       "found per name and the new evidence objects (a new evidence version; existing ids stay), and "
                       "later builds use them. With nothing else to do, end the turn without a tool call to wait. A name that finds nothing can be retried as a plainer visual "
                       "description (a color and a shape, e.g. 'red object').",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["names"], "properties": {
            "names": {"type": "array", "items": {"type": "string", "maxLength": 40}, "minItems": 1, "maxItems": 8}}}}},
    {"type": "function", "function": {
        "name": "execute_blender",
        "description": "Build a new candidate revision: run the complete scene.py in a fresh, isolated Blender (no network). "
                       "Returns errors, printed output, the inventory and, when it builds, its check (structure, camera "
                       "and projected boxes against the evidence, evidence objects still missing or undecided, one "
                       "simulated second past the last input frame) with source|render image pairs at the comparison "
                       "frames and a top-down view.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["scene_code"], "properties": {
            "scene_code": {"type": "string", "maxLength": 200000}}}}},
    {"type": "function", "function": {
        "name": "render_scene",
        "description": "Render a built revision at other frames. camera 'source' (default) uses the reconstructed source "
                       "camera and returns source|render image pairs. Views of the whole room, orthographic: 'top' from above "
                       "without the ceiling; 'side_x' looking along +Y and 'side_y' looking along +X, without walls or ceiling, "
                       "with a red line at floor level.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["revision"], "properties": {
            "revision": {"type": "string"}, "frame_ids": INT_LIST,
            "camera": {"type": "string", "enum": ["source", "top", "side_x", "side_y"]}}}}},
]

# render_scene before the room views of v2 (2026-11-07). Every task but the rooms (settings fast_room) gets this one, so
# T5-T7 see the tool as their paper runs did.
RENDER_SCENE_EARLIER = {"type": "function", "function": {
    "name": "render_scene",
    "description": "Render a built revision at other frames. camera 'source' (default) uses the reconstructed source "
                   "camera and returns source|render image pairs; 'top' is an orthographic top-down layout view.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["revision"], "properties": {
        "revision": {"type": "string"}, "frame_ids": INT_LIST,
        "camera": {"type": "string", "enum": ["source", "top"]}}}}}


def tool_schemas(settings):
    """The tools as the agent sees them: render_scene with the room views for rooms (settings fast_room), the earlier
    render_scene for every other task."""
    if (settings or {}).get("fast_room"):
        return TOOL_SCHEMAS
    return [RENDER_SCENE_EARLIER if t["function"]["name"] == "render_scene" else t for t in TOOL_SCHEMAS]


def result(status, artifacts=None, summary=None, warnings=(), error=None, images=()):
    return {"status": status, "artifacts": {k: str(v) for k, v in (artifacts or {}).items()}, "summary": summary or {},
            "warnings": list(warnings), "error": error, "images": list(images)}


class Workspace:
    """Run directory, evidence versions, revisions and image artifacts for one modeling run."""

    def __init__(self, run_dir, perception, sandbox, settings):
        self.run = Path(run_dir)
        self.perception, self.sandbox, self.settings = perception, sandbox, settings
        self.images, self.revisions, self.evidence, self.search = {}, {}, None, None
        self.decode_dir = perception.decode()
        self.timeline = contract.read_jsonl(self.decode_dir / "timeline.jsonl")
        self.video = contract.read_json(self.decode_dir / "video_info.json")
        self.comparison = contract.comparison_frames(len(self.timeline), settings.get("comparison_frames"))
        (self.run / "images").mkdir(parents=True, exist_ok=True)
        (self.run / "revisions").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def _frames(self, frame_ids):
        frames = list(dict.fromkeys(int(f) for f in frame_ids))
        bad = [f for f in frames if not 0 <= f < len(self.timeline)]
        if bad:
            raise ValueError(f"frame_ids out of range 0..{len(self.timeline) - 1}: {bad}")
        if len(frames) > self.settings.get("max_images_per_call", 6):
            raise ValueError(f"at most {self.settings.get('max_images_per_call', 6)} frames per call")
        return frames

    def frame(self, f):
        import cv2
        return cv2.imread(str(self.decode_dir / f"frames/{f:06d}.jpg"))

    def register_image(self, image, label, size=None, detail=None, quality=80):
        import cv2
        size = size or self.settings.get("image_size", 512)
        scale = size / max(image.shape[:2])
        if scale < 1:
            image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        image_id = f"img{len(self.images) + 1:04d}"
        path = self.run / "images" / f"{image_id}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        self.images[image_id] = {"path": str(path), "label": label, "sha256": contract.sha256(path),
                                 **({"detail": detail} if detail else {})}
        return image_id

    def camera_motion(self):
        """'static' or 'moving', as the camera_depth evidence measured it (moving when not measured)."""
        world = self.evidence / "world.json" if self.evidence is not None else None
        motion = contract.read_json(world).get("camera_motion") if world is not None and world.is_file() else None
        return "static" if motion in ("static", "single") else "moving"

    def source_frame(self, f):
        """Frame f at the source video's resolution (BGR). The decoded frame when it was rectified or not
        downscaled; otherwise the source frame itself, decoded again by its presentation time."""
        import av
        transforms = [t["type"] for t in self.video.get("transforms", [])]
        if transforms != ["resize"]:
            return self.frame(f)
        target = self.timeline[f]["source_pts_ns"]
        with av.open(self.video["source_video"]) as container:
            stream = container.streams.video[0]
            container.seek(int(target / 1e9 / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(stream):
                if int(frame.pts * frame.time_base * 1_000_000_000) >= target - 1000:
                    return frame.to_ndarray(format="bgr24")
        return self.frame(f)

    def _timing(self, f):
        row = self.timeline[f]
        return {"frame_index": f, "blender_frame": f + 1, "timestamp_ns": row["timestamp_ns"],
                "source_frame_index": row["source_frame_index"], "source_pts_ns": row["source_pts_ns"]}

    def _labels(self, prompt_index, f):
        """One frame of a prompt's SAM 3 labels, or None without masks there. A label file holds the whole video
        in one compressed array, so on first use it is read once and its frames with masks are cached as uint16
        PNGs, per file (tracks joined by find_objects link the earlier files and reuse their cache)."""
        import cv2
        source = (self.perception.parts["tracks"] / f"labels_{prompt_index:03d}.npz").resolve()
        cache = self.run / "cache" / "labels" / hashlib.sha256(str(source).encode()).hexdigest()[:16]
        if not (cache / "complete").is_file():
            cache.mkdir(parents=True, exist_ok=True)
            labels = contract.read_labels(source)
            for frame in np.flatnonzero(labels.reshape(len(labels), -1).any(1)):
                cv2.imwrite(str(cache / f"{frame:06d}.png"), labels[frame])
            (cache / "complete").write_text("")
        path = cache / f"{f:06d}.png"
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None

    def _layered(self, f, layers):
        """The frame itself, or the frame annotated (masks: SAM 3 tracks, white 'track:prompt' labels for tracks
        without a 3D box; boxes: projected 3D box candidates, yellow 'id:category' labels) and/or its depth map."""
        import cv2
        image = self.frame(f)
        annotated, panels, ev = image.copy(), [], self.evidence
        font = 0.9 * image.shape[1] / 960

        def label(text, x, y, color):
            x, y = int(np.clip(x, 0, image.shape[1] - 60)), int(np.clip(y, 20 * font, image.shape[0] - 4))
            cv2.putText(annotated, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), max(3, round(5 * font)), cv2.LINE_AA)
            cv2.putText(annotated, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font, color, max(1, round(2 * font)), cv2.LINE_AA)

        objects = contract.read_json(ev / "objects.json") if ev is not None and (ev / "objects.json").is_file() else {}
        boxed = set(objects) if "boxes" in layers else set()
        if "masks" in layers and "tracks" in self.perception.parts:
            tracks = contract.read_json(self.perception.parts["tracks"] / "tracks.json")
            for prompt_index in sorted({t["prompt_index"] for t in tracks["tracks"]}):
                labels = self._labels(prompt_index, f)
                if labels is None:
                    continue
                for t in tracks["tracks"]:
                    if t["prompt_index"] == prompt_index and (labels == t["label_value"]).any():
                        mask = labels == t["label_value"]
                        color = np.random.default_rng(int(t["track_id"])).integers(60, 255, 3)
                        annotated[mask] = (0.5 * color + 0.5 * annotated[mask]).astype(np.uint8)
                        if t["track_id"] not in boxed:
                            ys, xs = np.nonzero(mask)
                            label(f"{t['track_id']}:{t['prompt']}", xs.min(), ys.min() + 20 * font, (255, 255, 255))
        if "boxes" in layers and objects and (ev / "cameras.json").is_file():
            from .contract import EDGES, box_corners, project
            row = contract.read_json(ev / "cameras.json")[f]
            for uid, item in objects.items():
                uv = project(row, box_corners(item))
                if uv is not None:
                    for a, b in EDGES:
                        cv2.line(annotated, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), (0, 255, 255), 2)
                    label(f"{uid}:{item['category']}", *uv.min(0), (0, 255, 255))
        if {"masks", "boxes"} & set(layers):
            panels.append(annotated)
        if "depth" in layers and "camera_depth" in self.perception.parts:
            geometry = contract.read_npz(self.perception.work / "geometry/geometry.npz", ["keyframes", "depth"])
            keys = [int(k) for k in geometry["keyframes"]]
            if f in keys:
                depth = geometry["depth"][keys.index(f)]
                norm = np.clip((depth - 0.3) / 6.0, 0, 1)
                colored = cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)
                panels.append(cv2.resize(colored, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST))
        return np.hstack(panels) if panels else image

    def _evidence_status(self):
        ev = self.evidence
        status = {"version": ev.name if ev else None,
                  "tasks_done": sorted(k for k in ("camera_depth", "tracks", "fuse") if k in self.perception.parts)}
        if ev and (ev / "objects.json").is_file():
            from .contract import box_corners
            objects = contract.read_json(ev / "objects.json")
            # world_aabb: world-space min/max corners of the local bounds at the first pose
            def motion(o):
                """For a dynamic object: where and when it moves (first and last pose)."""
                first, last = o["poses"][0], o["poses"][-1]
                at = lambda row: f"{np.round(row[1:4], 2).tolist()} at {row[0] / 1e9:.2f} s"
                return f"dynamic, {len(o['poses'])} poses: {at(first)} -> {at(last)}"
            status["objects"] = {uid: {"instance_name": o["instance_name"], "category": o["category"],
                                       "motion_type": o["motion_type"],
                                       **({"motion": motion(o)} if o["motion_type"] == "dynamic" else {}),
                                       "world_aabb": [np.round(box_corners(o).min(0), 2).tolist(),
                                                      np.round(box_corners(o).max(0), 2).tolist()],
                                       **{k: round(v, 2) if isinstance(v, float) else v
                                          for k, v in ((k, o["evidence"].get(k)) for k in (
                                              "evidence_level", "uncertain_because", "confidence", "views_used",
                                              "dimension_status", "low_depth_confidence"))
                                          if v is not None}}
                                 for uid, o in objects.items()}
            unresolved = contract.read_json(ev / "unresolved.json") if (ev / "unresolved.json").is_file() else []
            status["unresolved"] = [{k: u[k] for k in ("id", "category", "reason", "own_box") if k in u} for u in unresolved]
        if ev and (ev / "world.json").is_file():
            world = contract.read_json(ev / "world.json")
            status["world"] = {k: world[k] for k in ("up_status", "scale_status", "camera_height_above_floor_m") if k in world}
        if ev and (ev / "structure.json").is_file():
            status["structure"] = contract.read_json(ev / "structure.json")
        return status

    def evidence_text(self):
        """The evidence summary as readable lines for the model's first message, with the raw records of one
        object as scene.py reads them."""
        status = self._evidence_status()
        objects, unresolved = status.get("objects", {}), status.get("unresolved", [])
        rounded = lambda x: round(x, 2) if isinstance(x, float) else ({k: rounded(v) for k, v in x.items()}
                            if isinstance(x, dict) else [rounded(v) for v in x] if isinstance(x, list) else x)
        lines = [f"evidence version {status['version']}"] + [f"{k}: {json.dumps(rounded(status[k]))}"
                                                             for k in ("world", "structure") if k in status]
        lines.append(f"objects ({len(objects)}): id | category | evidence_level (why uncertain) | motion | "
                     "world_aabb min .. max | detection score")
        lines += [self._object_line(uid, o) for uid, o in objects.items()]
        lines.append(f"unresolved tracks ({len(unresolved)}): " +
                     "; ".join(f"{u['id']} {u['category']} ({u['reason']})" for u in unresolved))
        if objects:
            uid = next(iter(objects))
            appearance = contract.read_json(self.evidence / "appearance.json") if (self.evidence / "appearance.json").is_file() else {}
            lines.append(f"raw records in scene.py: ev['objects']['{uid}'] = "
                         f"{json.dumps(contract.read_json(self.evidence / 'objects.json')[uid])}; "
                         f"ev['appearance']['{uid}'] = {json.dumps(appearance.get(uid))}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ tools
    def inspect_video(self, frame_ids=None, object_ids=None, layers=None, zoom=None):
        if not frame_ids and not object_ids and not zoom:
            canonical = self.video["canonical"]
            status = self._evidence_status()
            return result("success", summary={
                "video": {**canonical, "projection": self.video.get("projection"), "transforms": self.video["transforms"]},
                "comparison_frames": self.comparison,
                "evidence": {"version": status["version"], "objects": len(status.get("objects", {})),
                             "unresolved_tracks": len(status.get("unresolved", [])),
                             **{k: status[k] for k in ("world", "structure") if k in status},
                             "objects_table": "in the first message; find_objects results list objects added later"}})
        images, timings = [], []
        for f in self._frames(frame_ids or []):
            images.append(self.register_image(self._layered(f, layers or []), f"source frame {f}" +
                                              (f" with {', '.join(layers)}" if layers else "")))
            timings.append(self._timing(f))
        if object_ids and self.evidence is not None and (self.evidence / "observations.jsonl").is_file():
            observations = contract.read_jsonl(self.evidence / "observations.jsonl")
            for uid in object_ids[: self.settings.get("max_images_per_call", 6)]:
                rows = [r for r in observations if r["object_id"] == uid and r["state"] == "detected"]
                if not rows:
                    continue
                best = max(rows, key=lambda r: r["mask_area_px"])
                x0, y0, x1, y1 = best["bbox_xyxy"]
                pad = 20
                crop = self.frame(best["frame_index"])[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
                images.append(self.register_image(crop, f"object {uid} at frame {best['frame_index']}"))
        for region in (zoom or [])[: self.settings.get("max_images_per_call", 6)]:
            f = self._frames([region["frame"]])[0]
            image = self.source_frame(f)
            h, w = image.shape[:2]
            x0, y0, x1, y1 = region["box"]
            x0, x1 = sorted((int(x0 * w), max(int(x1 * w), int(x0 * w) + 2)))
            y0, y1 = sorted((int(y0 * h), max(int(y1 * h), int(y0 * h) + 2)))
            images.append(self.register_image(image[y0:y1, x0:x1], f"frame {f} zoom {region['box']} "
                                                                    f"({x1 - x0}x{y1 - y0} source pixels)"))
            timings.append(self._timing(f))
        return result("success", summary={"frames": timings}, images=images)

    def prepare_evidence(self):
        """Publish one evidence version before the model starts: all local perception stages (cached), or the
        given perception of config evidence.given (e.g. an example's reference perception)."""
        self.evidence = self.perception.perceive()
        return self.evidence

    def find_objects(self, names):
        """Starts a search and returns at once. SAM 3 over a long video takes minutes, and the conversation goes on
        meanwhile (a paused conversation loses the provider's prompt cache); the result comes with a later tool
        result, and from then on builds use the new evidence version."""
        if self.perception.config.get("evidence", {}).get("given"):
            return result("error", error="the perception was given; there is no local perception to search")
        if self.search and not self.search["done"].is_set():
            return result("error", error=f"the search for {self.search['names']} is still running; its objects come "
                                         "with a later tool result")
        names = [n for n in dict.fromkeys(str(n).strip().lower() for n in names) if n]
        if not names:
            return result("error", error="no names to search")
        search = {"names": names, "done": threading.Event(),
                  "before": set(contract.read_json(self.evidence / "objects.json")) if self.evidence is not None else set()}

        def run():
            try:
                search["tracks_found"] = self.perception.add_names(names)
                self.perception.fuse_objects()
                search["evidence"] = self.perception.publish()
            except Exception as exc:          # reported with the next tool result
                search["error"] = f"{type(exc).__name__}: {exc}"
            search["done"].set()
        self.search = search
        threading.Thread(target=run, daemon=True).start()
        return result("success", summary={"search_started": names, "note": (
            "SAM 3 searches the whole input for these names and the new tracks are fused (minutes on a long video). "
            "Keep working; the new objects come with a later tool result and later builds use them.")})

    def search_result(self):
        """A finished search, once: switches to its evidence version and returns what it found (else None)."""
        search = self.search
        if not search or not search["done"].is_set() or search.get("reported"):
            return None
        search["reported"] = True
        if "error" in search:
            return {"names": search["names"], "error": search["error"]}
        self.evidence = search["evidence"]
        status = self._evidence_status()
        new = {u: o for u, o in status.get("objects", {}).items() if u not in search["before"]}
        # searched names that gave no object of their own: why (a part merged into its object keeps its own box)
        folded = [u for u in status.get("unresolved", []) if u["category"] in search["names"]]
        return {"evidence_version": self.evidence.name, "tracks_found": search["tracks_found"],
                "new_objects": [self._object_line(u, o) for u, o in new.items()],
                "not_new_objects": folded, "objects_total": len(status.get("objects", {}))}

    def wait_for_search(self, timeout=3600):
        if self.search:
            self.search["done"].wait(timeout)

    @staticmethod
    def _object_line(uid, o):
        why = f" ({', '.join(o['uncertain_because'])})" if o.get("uncertain_because") else ""
        lo, hi = o["world_aabb"]
        return (f"{uid} | {o['category']} | {o.get('evidence_level', 'certain')}{why} | {o.get('motion', o['motion_type'])} | "
                f"{lo} .. {hi} | {o.get('confidence', '-')}")

    def execute_blender(self, scene_code):
        # only a build that registers objects is a candidate scene and uses up the revision budget; failed and probe
        # builds do not (the request cap bounds them)
        candidates = [r for r in self.revisions.values() if r["executed"] and r.get("objects")]
        if len(candidates) >= self.settings.get("max_revisions", 4):
            return result("error", error=f"revision budget exhausted ({len(candidates)} revisions)")
        if self.evidence is None:
            self.evidence = self.perception.publish()
        rid = f"r{len(self.revisions) + 1:03d}"
        directory = self.run / "revisions" / rid
        directory.mkdir(parents=True)
        (directory / "scene.py").write_text(scene_code)
        run = self.sandbox.blender(["exec", "--code", "/out/scene.py", "--out", "/out"], directory, evidence=self.evidence)
        record = {"revision": rid, "created_utc": now(), "evidence_version": self.evidence.name,
                  "code_sha256": contract.sha256(directory / "scene.py"), "executed": run["status"] == "ok",
                  "exec": run, "checks": [], "renders": []}
        error = None
        if (directory / "exec_error.txt").is_file():
            error = (directory / "exec_error.txt").read_text()[-3000:]
        elif run["status"] != "ok":
            error = f"{run['status']}: {run['stderr_tail'][-1500:]}"
        summary = {"revision": rid, "executed": record["executed"]}
        if (directory / "exec_stdout.txt").is_file() and (directory / "exec_stdout.txt").stat().st_size:
            summary["printed"] = (directory / "exec_stdout.txt").read_text()[-3000:]
        if record["executed"]:
            record["blend_sha256"] = contract.sha256(directory / "scene.blend")
            inventory = contract.read_json(directory / "inventory.json")
            record["objects"] = len(inventory["semantic_objects"])
            summary["inventory"] = {"objects": [f"{o['instance_id']}:{o['name']}:{o['category']}" for o in inventory["semantic_objects"]],
                                    "environment_meshes": inventory["environment_meshes"],
                                    "unregistered_meshes": inventory["unregistered_meshes"],
                                    "camera_type": inventory["camera_type"], "frame_range": inventory["frame_range"],
                                    "fps": inventory["fps"], "resolution": inventory["resolution"]}
        self.revisions[rid] = record
        self._save_revisions()
        probes = sum(r["executed"] and not r.get("objects") for r in self.revisions.values())
        if record["executed"] and not record["objects"] and probes > self.settings.get("max_probe_builds", 2):
            return result("error", {"scene_code": directory / "scene.py"}, {"revision": rid, "executed": True},
                          error=f"this build registered no object, and builds without objects are limited to "
                                f"{self.settings.get('max_probe_builds', 2)}: its output is withheld. The evidence "
                                "summary (with each dynamic object's motion) and a raw record are in the first message, "
                                "the helper API in the instructions; look at frames with inspect_video.")
        images = []
        if record["executed"] and record["objects"]:
            # Every candidate scene comes back checked and rendered, so each build is one model turn.
            check = self.check_scene(rid)
            summary["check"] = check["summary"] if check["status"] == "success" else {"error": check["error"]}
            missing, undecided = self.coverage(rid, None)
            summary["check"]["evidence_objects_not_built"] = {"certain": missing, "uncertain": undecided}
            if missing:
                summary["check"]["repairs"] = [f"certain evidence objects not built: {missing[:20]}"] + summary["check"].get("repairs", [])
            overview = ("top", "side_x", "side_y") if self.settings.get("fast_room") else ("top",)
            for camera, extra in (("source", {"frame_ids": self.comparison}), *((c, {"frame_ids": self.comparison[:1]}) for c in overview)):
                render = self.render_scene(rid, camera=camera, **extra)
                images += render["images"]
                if render["status"] != "success":
                    summary.setdefault("render_errors", []).append(render["error"])
            if self.settings.get("video_check") and self.evidence.name != "blind":
                # The scene against the video: each object's silhouette and depth from the input's own camera.
                from . import consistency
                try:
                    report = consistency.check(self, rid, count=self.settings["video_check"].get("frames", 6))
                    summary["video_check"] = consistency.summary(report)
                except Exception as exc:              # a failed check never fails the build
                    summary["video_check"] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
        return result("success" if record["executed"] else "error", {"scene_code": directory / "scene.py"},
                      summary, error=error, images=images)

    def render_scene(self, revision, frame_ids=None, camera="source"):
        record = self._revision(revision)
        frames = self._frames(frame_ids or self.comparison)
        blender = [f + 1 for f in frames]
        tag = f"{camera}_{len(record['renders']) + 1:02d}"
        args = ["render", "--frames", ",".join(str(b) for b in blender), "--size", str(self.settings.get("render_size", 512)),
                "--samples", str(self.settings.get("render_samples", 16)), "--threads", str(self.sandbox.cpus),
                "--out", f"/out/renders/{tag}"]
        if camera != "source":            # rooms: the views of overview_camera; other tasks: the earlier top view
            args += ["--view", camera] if self.settings.get("fast_room") else (["--top"] if camera == "top" else [])
        run = self.sandbox.blender(args, self.run / "revisions" / revision, blend="scene.blend")
        if run["status"] != "ok":
            return result("error", error=f"render failed: {run['stderr_tail'][-1500:]}")
        import cv2
        images = []
        for f in frames:
            render = cv2.imread(str(self.run / "revisions" / revision / f"renders/{tag}/render_{f + 1:06d}.png"))
            if camera == "source":
                source = cv2.resize(self.frame(f), (render.shape[1], render.shape[0]), interpolation=cv2.INTER_AREA)
                images.append(self.register_image(np.hstack([source, render]),
                                                  f"{revision} frame {f}: source (left) | render (right)"))
            else:
                images.append(self.register_image(render, f"{revision} {camera} view at frame {f}"))
        record["renders"].append({"tag": tag, "camera": camera, "frames": frames, "images": images,
                                  "blend_sha256": record["blend_sha256"], "sent": False, "reviewed": False})
        self._save_revisions()
        return result("success", {"renders": self.run / "revisions" / revision / "renders" / tag},
                      {"revision": revision, "camera": camera, "frames": [self._timing(f) for f in frames],
                       "render": {"size": self.settings.get("render_size", 512),
                                  "samples": self.settings.get("render_samples", 16), "engine": "CYCLES"}},
                      images=images)

    def check_scene(self, revision):
        record = self._revision(revision)
        directory = self.run / "revisions" / revision
        name = f"check_{len(record['checks']) + 1:02d}.json"
        layout = self.settings.get("arrange")       # a room run's layout gate (arrange_room), with its thresholds
        run = self.sandbox.blender(["check", "--evidence", "/evidence", "--out", f"/out/{name}"]
                                   + (["--arrange", json.dumps(layout if isinstance(layout, dict) else {})] if layout else []),
                                   directory, evidence=self.run / "evidence" / record["evidence_version"], blend="scene.blend")
        if run["status"] != "ok":
            return result("error", error=f"check failed to run: {run['stderr_tail'][-1500:]}")
        check = contract.read_json(directory / name)
        record["checks"].append({"file": name, "pass": check["pass"], "blend_sha256": record["blend_sha256"]})
        self._save_revisions()
        metrics = check["metrics"]
        physics = metrics.get("physics")
        # Long scenes can flag hundreds of objects; the model gets the worst of them and the counts.
        moving = {u: o for u, o in (physics or {}).get("per_active_object", {}).items()
                  if o["release_speed_mps"] > 0.05 or o.get("flags")}
        moving = dict(sorted(moving.items(), key=lambda kv: -kv[1]["moved_m"])[:20]) | (
            {"...": f"{len(moving) - 20} more objects moved after release"} if len(moving) > 20 else {})
        warnings = check["warnings"][:12] + ([f"... {len(check['warnings']) - 12} more warnings"]
                                             if len(check["warnings"]) > 12 else [])
        repairs = check.get("repairs", [])
        sil = metrics.get("silhouette_vs_mask") or {}
        summary = {"revision": revision, "pass": check["pass"], "failures": check["failures"][:12], "warnings": warnings,
                   "repairs": [r["task"] for r in repairs[:6]] + ([f"... {len(repairs) - 6} more"] if len(repairs) > 6 else []),
                   "silhouette_vs_mask": {k: sil[k] for k in ("median_best_iou", "median_mean_iou", "below_min", "frames") if k in sil} or None,
                   "contacts": {"penetrations": (metrics.get("contacts") or {}).get("penetrations", [])[:8],
                                "floating": (metrics.get("contacts") or {}).get("floating", [])[:8]},
                   "parts": {o["instance_id"]: [f"{p['name']} ({p['joint']['type']})" for p in o["parts"]]
                             for o in check["inventory"]["semantic_objects"] if o["parts"]},
                   "plan": {k: v for k, v in (metrics.get("plan") or {}).items()
                            if k in ("declared_objects", "missing_parts", "size_off", "connection_off")},
                   "separations": len(check["inventory"].get("separations", [])),
                   "camera_vs_evidence": metrics.get("camera_vs_evidence", {}) and
                   {k: metrics["camera_vs_evidence"][k] for k in ("max_position_m", "max_rotation_deg", "max_K_difference_px")},
                   "object_reprojection_vs_tracker": {k: v for k, v in metrics.get("object_reprojection_vs_tracker", {}).items()
                                                      if k != "per_object_mean_iou"},
                   "physics": physics and {**{k: physics[k] for k in ("seconds", "active", "passive") if k in physics},
                                           "moving_after_release": moving,      # released moving, or flagged
                                           "flagged_objects": len(physics.get("problems", []))},
                   **({"dynamic_pose_deviation_m": metrics["dynamic_pose_deviation_m"]}
                      if "dynamic_pose_deviation_m" in metrics else {}),
                   "objects": len(check["inventory"]["semantic_objects"])}
        return result("success", {"check": directory / name}, summary)

    # ------------------------------------------------------------------ bookkeeping
    def _revision(self, revision):
        record = self.revisions.get(revision)
        if record is None:
            raise ValueError(f"unknown revision {revision}; known: {sorted(self.revisions)}")
        if not record["executed"]:
            raise ValueError(f"revision {revision} did not build; fix the code and execute a new revision")
        return record

    def eligible(self, revision):
        """A revision can be final only if it built, passed check_scene and its source renders were
        shown to the model and followed by a model turn."""
        record = self.revisions.get(revision)
        if record is None:
            return False, [f"unknown revision {revision}"]
        reasons = []
        if not record["executed"]:
            reasons.append("did not build")
        if not any(c["pass"] and c["blend_sha256"] == record.get("blend_sha256") for c in record["checks"]):
            reasons.append("no passing check_scene for this exact .blend")
        if not any(r["camera"] == "source" and r["sent"] and r["reviewed"] for r in record["renders"]):
            reasons.append("no source-camera render of this revision was shown to the model and reviewed")
        return not reasons, reasons

    def coverage(self, revision, left_out):
        """What a final revision misses: certain evidence objects not registered, and uncertain ones neither
        registered nor left out with a reason. Objects without a level (a given perception) count as certain. An
        object left out as the same physical object as a registered one (same_as: its id; a duplicate track or a
        part of it), or as a piece of the room the scene builds as structure (same_as: "environment"; a patch of wall,
        floor, ceiling or window), counts as registered."""
        objects = contract.read_json(self.evidence / "objects.json") if self.evidence is not None else {}
        path = self.run / "revisions" / revision / "inventory.json"
        built = {str(o["instance_id"]) for o in contract.read_json(path)["semantic_objects"]} if path.is_file() else set()
        built |= {str(item.get("id")) for item in left_out or [] if isinstance(item, dict) and item.get("reason")
                  and (str(item.get("same_as")) in built or item.get("same_as") == "environment")}
        explained = {str(item.get("id")) for item in left_out or [] if isinstance(item, dict) and item.get("reason")}
        level = lambda o: o.get("evidence", {}).get("evidence_level", "certain")
        missing = sorted((u for u, o in objects.items() if level(o) == "certain" and u not in built), key=str)
        undecided = sorted((u for u, o in objects.items() if level(o) == "uncertain" and u not in built | explained),
                           key=str)
        return missing, undecided

    def _save_revisions(self):
        contract.write_json(self.run / "revisions.json", self.revisions)

    def dispatch(self, name, arguments):
        handlers = {"inspect_video": self.inspect_video, "find_objects": self.find_objects,
                    "execute_blender": self.execute_blender, "render_scene": self.render_scene}
        if name not in handlers:
            return result("error", error=f"unknown tool {name}")
        finished = self.search_result()           # before the tool runs, so it already uses the new evidence
        try:
            outcome = handlers[name](**arguments)
        except (ValueError, TypeError, RuntimeError, KeyError) as exc:
            outcome = result("error", error=f"{type(exc).__name__}: {exc}")
        if finished:
            outcome["summary"] = {**outcome["summary"], "find_objects_finished": finished}
        return outcome
