"""Complete objects from the input's own frames (the make_objects tool): SAM 3D Objects turns the union of an object's
masks on one frame into a whole mesh, its hidden sides completed, with its pose in that frame's camera, conditioned on
the frame's measured depth; that frame's camera puts it in the world. A worker in the SAM 3D environment
(sam3d_worker.py) loads the model once per run. Each object goes to <evidence>/assets/<id>/ (mesh.npz, asset.json),
where scene code places it (assets.add_asset), and is checked against its masks on other frames (silhouette IoU) and
the measured depth (the mesh's front surface against it).

The layers come from the raw evidence (raw.py: tracks, masks, depth, cameras) or, for fused evidence, from the
perception's tracks and reconstruction behind its objects (an object id stands for its kept and merged tracks).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import queue
import re
import select
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import contract

WORKER = Path(__file__).with_name("sam3d_worker.py")
ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
CHECK_FRAMES = 5           # frames besides the source frame whose masks the mesh's silhouette is compared with
MAX_CANDIDATES = 40        # frames scored when choosing where an object is taken from
CARD = 300                 # card panel size (pixels)
_DEPTH_LOCK = threading.Lock()   # the depth cache is built once (the server's background thread or the first call)
SCHEMA = {"name": "make_objects", "description": (
    "Turn physical objects in the frames into complete 3D meshes (SAM 3D Objects). For each object give the tracks "
    "that make it up (every part of it: one track can be a part, several tracks one object) and optionally the frame "
    "to take it from (default: where its masks are largest and not cut by the image border). The mask on that frame, "
    "with the frame's measured depth, becomes a whole mesh with its hidden sides completed and vertex colours, placed "
    "in the world with that frame's camera. Returns per object its size, position and lowest point, how its "
    "silhouette matches its masks on that frame and on others (iou) and its front surface the measured depth "
    "(depth_error), and cards: the frame crop with the masks in green and the mesh's outline in magenta, the mesh "
    "seen from the side and from above. Build it with A.add_asset(id, ...). About 10 s per object; the model loads "
    "once per run (about 2 minutes, started with the run). Calling again with an id replaces that object."),
    "inputSchema": {"type": "object", "additionalProperties": False, "required": ["objects"], "properties": {
        "objects": {"type": "array", "minItems": 1, "maxItems": 16, "items": {
            "type": "object", "additionalProperties": False, "required": ["id", "tracks"], "properties": {
                "id": {"type": "string", "pattern": ID.pattern,
                       "description": "the object's id (its instance_id in the scene)"},
                "name": {"type": "string", "maxLength": 60, "description": "what the object is"},
                "tracks": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 12,
                           "description": "raw layers: track ids (tracks.json); fused evidence: object ids"},
                "frame": {"type": "integer", "minimum": 0}}}}}}}


TRACK_SCHEMA = {"name": "track_objects", "description": (
    "Follow made objects (make_objects ids) through the input: on every frame where an object's tracks are detected "
    "and there is depth, its mesh (rigid: the shape and size stay) is fitted to the frame's measured points inside its "
    "masks, starting from the neighbouring frame's pose (ICP on the side facing the camera), and checked by its "
    "silhouette. Returns per object the frames followed, how far it moved and turned over the input, and the frames "
    "where the fit failed (pose held there); writes the poses, which A.add_asset(id, animate=True) keys. For objects "
    "that move: a still object needs no track."),
    "inputSchema": {"type": "object", "additionalProperties": False, "required": ["ids"], "properties": {
        "ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 16}}}}


INVENTORY_SCHEMA = {"name": "inventory", "description": (
    "For counting the objects of a room (fused evidence): complete every evidence object of the given categories into "
    "a mesh (make_objects with the evidence ids, so A.add_asset(id) builds them) and lay them out: per object its "
    "completed size, lowest point above the floor, position, views and evidence level; the pairs whose completed shapes "
    "overlap and are never seen side by side (grouped: likely one object seen twice, or a part of one), the close pairs "
    "that do not (near_but_apart: look at them); and per category a card "
    "with each object's crop (green: its masks, magenta: the mesh) and a top-down map of the completed footprints. "
    "Then decide per category: one id per physical object, the others same_as it (or renamed when they are something "
    "else). Default categories: the ones with two or more evidence objects, most first; up to max_objects at a call."),
    "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
        "categories": {"type": "array", "items": {"type": "string"}, "maxItems": 24},
        "max_objects": {"type": "integer", "minimum": 1, "maximum": 96}}}}


class Generator:
    """SAM 3D workers (cfg["workers"], default 1, about 20 GB of GPU memory each): started once, in the background at
    the run's start; each takes one request at a time, so several objects are made at once."""

    def __init__(self, cfg, folder, log=print):
        self.cfg, self.folder, self.log = cfg, Path(folder), log
        self.count = max(1, int(cfg.get("workers", 1)))
        self.processes, self.idle = [], queue.Queue()
        self.ready, self.failure, self.lock = threading.Event(), None, threading.Lock()

    def start(self):
        with self.lock:
            if self.processes:
                return
            self.folder.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            if self.cfg.get("gpu") is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(self.cfg["gpu"])
            for i in range(self.count):
                self.processes.append(subprocess.Popen(
                    [self.cfg["python"], str(WORKER), "--repo", self.cfg["repo"], "--config", self.cfg["config"]],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(self.folder / f"worker{i}.log", "a"),
                    text=True, env=env, start_new_session=True))
        threading.Thread(target=self._wait_ready, daemon=True).start()

    def _wait_ready(self):
        for process in self.processes:
            line = self._readline(process, self.cfg.get("load_timeout", 900))
            if line and json.loads(line).get("ready"):
                self.idle.put(process)
                self.log(f"SAM 3D worker ready: {line.strip()}")
        if self.idle.empty():
            self.failure = f"SAM 3D worker did not start (see {self.folder}/worker*.log)"
        self.ready.set()

    @staticmethod
    def _readline(process, timeout):
        ready, _, _ = select.select([process.stdout], [], [], timeout)
        return process.stdout.readline() if ready else None

    def generate(self, request, timeout=600):
        self.start()
        self.ready.wait(self.cfg.get("load_timeout", 900))
        if self.failure or not self.ready.is_set():
            return {"ok": False, "error": self.failure or "SAM 3D worker still loading"}
        try:
            process = self.idle.get(timeout=timeout)
        except queue.Empty:
            return {"ok": False, "error": f"no SAM 3D worker free within {timeout} s"}
        try:
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            line = self._readline(process, timeout)
        finally:
            if process.poll() is None:
                self.idle.put(process)
        if not line:
            return {"ok": False, "error": f"no answer from the SAM 3D worker within {timeout} s"}
        return json.loads(line)

    def stop(self):
        for process in self.processes:
            if process.poll() is None:
                process.stdin.close()
                try:
                    process.wait(30)
                except subprocess.TimeoutExpired:
                    process.kill()


class RawLayers:
    """Masks, depth and cameras of the raw evidence (its raw.py)."""

    def __init__(self, workspace):
        spec = importlib.util.spec_from_file_location("raw_layers_objects3d", workspace.evidence / "raw.py")
        self.raw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.raw)
        self.frames_dir = workspace.evidence / "frames"

    def known(self, ref):
        return str(ref) in self.raw.tracks()

    def name(self, ref):
        return self.raw.tracks()[str(ref)]["name"]

    def frames(self, ref):
        return list(self.raw.tracks()[str(ref)]["frames"])

    def mask(self, ref, frame):
        return self.raw.mask(str(ref), frame) if frame in self.raw.tracks()[str(ref)]["frames"] else None

    def depth(self, frame):
        return self.raw.depth(frame)

    def depth_frames(self):
        return set(self.raw.depth_frames())

    def camera(self, frame):
        return self.raw.cameras()[frame]

    def image(self, frame):
        return self.frames_dir / f"{frame:06d}.jpg"


class FusedLayers:
    """Masks, depth and cameras behind fused evidence: an object id stands for its kept and merged SAM 3 tracks; depth
    is the reconstruction's valid depth on its keyframes (kept as one float16 array per run in cache/objects3d)."""

    def __init__(self, workspace):
        self.ws = workspace
        self.objects = contract.read_json(workspace.evidence / "objects.json")
        # A given perception (evidence.given) carries its track masks (track_masks, or gt_tracks in the ground-truth
        # evidences) and its depth in the evidence itself.
        tracks_dir = workspace.perception.parts.get("tracks") or next(
            (workspace.evidence / n for n in ("track_masks", "gt_tracks") if (workspace.evidence / n).is_dir()), None)
        table = {t["track_id"]: t for t in contract.read_json(tracks_dir / "tracks.json")["tracks"]}
        self.members = {}
        for oid, o in self.objects.items():
            ids = [oid, *[str(m) for m in (o.get("evidence") or {}).get("merged_track_ids") or []]]
            self.members[oid] = [(tracks_dir / f"labels_{table[m]['prompt_index']:03d}.npz", table[m]["label_value"],
                                  table[m]) for m in ids if m in table]
        self.cameras = {row["frame"] - 1: row for row in contract.read_json(workspace.evidence / "cameras.json")}
        self.size = (workspace.video["canonical"]["height"], workspace.video["canonical"]["width"])
        self._masks, self._depth = {}, None

    def known(self, ref):
        return str(ref) in self.members

    def name(self, ref):
        return self.objects[str(ref)].get("category", "")

    def detections(self, ref):
        """{frame: [(mask area, bbox), ...]} of the object's tracks."""
        out = {}
        for _, _, track in self.members[str(ref)]:
            for d in track.get("detections") or []:
                out.setdefault(int(d["frame_index"]), []).append((d.get("mask_area_px") or 0, d.get("bbox_xyxy")))
        return out

    def frames(self, ref):
        return sorted(self.detections(ref))

    def prefetch(self, pairs):
        """Read the masks of these (object, frame) pairs, one pass over each label file (long videos keep them
        compressed)."""
        from .consistency import _label_frames
        wanted = {}
        for ref, f in pairs:
            if (str(ref), f) not in self._masks:
                for path, value, _ in self.members[str(ref)]:
                    wanted.setdefault(path, set()).add(f)
        for path, frames in wanted.items():
            labels = _label_frames(path, sorted(frames))
            for ref, f in pairs:
                for p, value, _ in self.members[str(ref)]:
                    if p == path and f in labels:
                        hit = labels[f] == value
                        old = self._masks.get((str(ref), f))
                        self._masks[(str(ref), f)] = hit if old is None else old | hit
        for ref, f in pairs:
            self._masks.setdefault((str(ref), f), None)

    def mask(self, ref, frame):
        if (str(ref), frame) not in self._masks:
            self.prefetch([(ref, frame)])
        return self._masks[(str(ref), frame)]

    def prepare_depth(self):
        if self._depth is None:
            given = self.ws.evidence / "depth"
            folder = given if (given / "depth_frames.json").is_file() else self.ws.run / "cache" / "objects3d"
            path, index_path = folder / "depth.npy", folder / "depth_frames.json"
            with _DEPTH_LOCK:
                if not index_path.is_file():
                    self._build_depth(folder, path, index_path)
            self._depth = ({f: i for i, f in enumerate(contract.read_json(index_path))}, np.load(path, mmap_mode="r"))
        return self._depth

    def _build_depth(self, folder, path, index_path):
        from . import fusion
        cfg = {**fusion.DEFAULTS, **self.ws.perception.config.get("fuse", {})}
        rec = fusion.reconstruction(self.ws.perception.work / "geometry", self.ws.decode_dir, cfg,
                                    np.random.default_rng(cfg["seed"]))
        folder.mkdir(parents=True, exist_ok=True)
        stack = np.lib.format.open_memmap(path, "w+", np.float16, (len(rec.keyframes), *rec.depth[0].shape))
        for i in range(len(rec.keyframes)):
            stack[i] = np.where(rec.valid[i], rec.depth[i], 0)
        stack.flush()
        keyframes = [int(k) for k in rec.keyframes]
        del stack, rec
        contract.write_json(index_path, keyframes)      # written last: its presence means the array is whole

    def depth(self, frame):
        index, stack = self.prepare_depth()
        if frame not in index:
            return None
        return cv2.resize(np.asarray(stack[index[frame]], np.float32), self.size[::-1], interpolation=cv2.INTER_NEAREST)

    def depth_frames(self):
        return set(self.prepare_depth()[0])

    def camera(self, frame):
        return self.cameras[frame]

    def image(self, frame):
        return self.ws.decode_dir / "frames" / f"{frame:06d}.jpg"


def layers_of(workspace):
    return RawLayers(workspace) if workspace.evidence.name == "raw" else FusedLayers(workspace)


def _union(layers, refs, frame):
    out = None
    for r in refs:
        m = layers.mask(r, frame)
        if m is not None:
            out = m if out is None else out | m
    return out


def _border_share(mask, margin=3):
    """Share of the mask's pixels within margin pixels of the image border: the object is cut there."""
    edge = np.zeros_like(mask)
    edge[:margin], edge[-margin:], edge[:, :margin], edge[:, -margin:] = True, True, True, True
    return (mask & edge).sum() / max(mask.sum(), 1) * 50          # a few border pixels already mean a cut


def choose_frame(layers, refs):
    """The frame where the object's masks together are largest and least cut by the border, preferring frames with
    depth. With a camera that does not move, what changes between frames is the objects, and the scene continues from
    the input's end: the latest frame where the object is seen nearly as well (80 %) as anywhere."""
    frames = sorted(set().union(*(layers.frames(r) for r in refs)))
    with_depth = [f for f in frames if f in layers.depth_frames()]
    candidates = with_depth or frames
    if len(candidates) > MAX_CANDIDATES:
        candidates = [candidates[i] for i in np.linspace(0, len(candidates) - 1, MAX_CANDIDATES).round().astype(int)]
    scores = {}
    if hasattr(layers, "detections"):          # long videos: the detections' areas and boxes, no masks read
        H, W = layers.size
        found = [layers.detections(r) for r in refs]
        for f in candidates:
            hits = [d for det in found for d in det.get(f, [])]
            cut = any(b and (b[0] <= 2 or b[1] <= 2 or b[2] >= W - 3 or b[3] >= H - 3) for _, b in hits)
            scores[f] = sum(a for a, _ in hits) * (0.25 if cut else 1.0)
    else:
        for f in candidates:
            m = _union(layers, refs, f)
            if m is not None and m.any():
                scores[f] = m.sum() * max(0.05, 1 - _border_share(m)) ** 2
    if not scores:
        return None
    best = max(scores, key=scores.get)
    if _static_camera(layers, frames):
        best = max(f for f, v in scores.items() if v >= 0.8 * scores[best])
    return best


def _static_camera(layers, frames):
    """The camera stays put over these frames (the perception holds its pose when it does not move)."""
    rows = [layers.camera(f) for f in (frames[0], frames[-1])]
    return (np.allclose(rows[0]["position_world"], rows[1]["position_world"], atol=1e-3)
            and np.allclose(rows[0]["R_world_rgb"], rows[1]["R_world_rgb"], atol=1e-3))


def _check_frames(layers, refs, frame):
    """Frames besides the source frame where the object is compared with its masks, spread over its detections."""
    others = [f for f in sorted(set().union(*(layers.frames(r) for r in refs))) if f != frame]
    if len(others) > CHECK_FRAMES:
        others = [others[i] for i in np.linspace(0, len(others) - 1, CHECK_FRAMES).round().astype(int)]
    return others


def _camera(row):
    return (np.asarray(row["projection"]["K"], float), np.asarray(row["R_world_rgb"], float),
            np.asarray(row["position_world"], float))


def surface_samples(vertices, faces, colors, count=60000, seed=0):
    """Points spread over the mesh by area, with their interpolated colours."""
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(faces), count, p=area / area.sum())
    a, b = rng.random(count), rng.random(count)
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    w = np.stack([1 - a - b, a, b], 1)
    points = np.einsum("nk,nkd->nd", w, tri[pick])
    rgb = np.einsum("nk,nkd->nd", w, colors[faces[pick], :3].astype(float))
    return points, rgb


def project(points, row, shape):
    """Silhouette (bool) and front-surface depth (0 = none) of world points seen by the camera row, at shape."""
    K, R, C = _camera(row)
    H, W = shape
    sx, sy = W / row["projection"]["width"], H / row["projection"]["height"]
    cam = (points - C) @ R
    front = cam[:, 2] > 1e-3
    cam = cam[front]
    u = np.round((cam[:, 0] / cam[:, 2] * K[0, 0] + K[0, 2]) * sx).astype(int)
    v = np.round((cam[:, 1] / cam[:, 2] * K[1, 1] + K[1, 2]) * sy).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    depth = np.full((H, W), np.inf, np.float32)
    np.minimum.at(depth, (v[ok], u[ok]), cam[ok, 2].astype(np.float32))
    seen = np.isfinite(depth)
    kernel = np.ones((5, 5), np.uint8)
    sil = cv2.morphologyEx(seen.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    depth[~seen] = 0
    return sil, depth


def _splat_view(points, rgb, axis_u, axis_v, size=CARD):
    """Orthographic view of coloured points: image right = axis_u, image up = axis_v, nearest-to-viewer on top."""
    view = np.cross(axis_u, axis_v)                     # towards the viewer
    pu, pv, pd = points @ axis_u, points @ axis_v, points @ view
    span = max(np.ptp(pu), np.ptp(pv), 1e-6) * 1.1
    cu, cv_ = (pu.max() + pu.min()) / 2, (pv.max() + pv.min()) / 2
    x = ((pu - cu) / span + 0.5) * (size - 1)
    y = (0.5 - (pv - cv_) / span) * (size - 1)
    order = np.argsort(pd)                              # far first, near last
    image = np.full((size, size, 3), 235, np.uint8)
    xi, yi = np.clip(x[order].round().astype(int), 0, size - 1), np.clip(y[order].round().astype(int), 0, size - 1)
    image[yi, xi] = rgb[order][:, ::-1].clip(0, 255).astype(np.uint8)       # RGB -> BGR
    return image, span / 1.1


def _card(frame_bgr, mask, sil, points, rgb, row, title):
    """Frame crop (green: the masks, magenta: the mesh's outline), the mesh from the side and from above."""
    H, W = mask.shape
    panels = []
    ys, xs = np.nonzero(mask | sil)
    pad = int(0.25 * max(np.ptp(xs), np.ptp(ys), 20))
    x0, x1, y0, y1 = max(0, xs.min() - pad), min(W, xs.max() + pad), max(0, ys.min() - pad), min(H, ys.max() + pad)
    crop = frame_bgr.copy()
    for m, colour in ((mask, (60, 220, 60)), (sil, (220, 60, 220))):
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, contours, -1, colour, 2)
    crop = crop[y0:y1, x0:x1]
    scale = CARD / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
    canvas = np.full((CARD, CARD, 3), 255, np.uint8)
    canvas[:crop.shape[0], :crop.shape[1]] = crop
    panels.append(canvas)
    _, R, _ = _camera(row)
    forward = R[:, 2] * [1, 1, 0]
    forward = forward / max(np.linalg.norm(forward), 1e-6)
    right = np.cross(forward, [0, 0, 1.0])
    side, span_side = _splat_view(points, rgb, -forward, np.array([0, 0, 1.0]))    # seen from the camera's right
    top, span_top = _splat_view(points, rgb, right, forward)                        # from above, camera at the bottom
    for image, label in ((side, f"side ({span_side:.2f} m across)"), (top, f"above ({span_top:.2f} m across)")):
        cv2.putText(image, label, (6, CARD - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        panels.append(image)
    strip = np.hstack(panels)
    bar = np.full((26, strip.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title[:110], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, strip])


def _depth_scale(world, mesh, row, mask, depth, min_pixels=50):
    """Scale the posed mesh about the camera centre so its front surface meets the measured depth on the mask (the
    median ratio over the mask's inner pixels). The silhouette stays as it is: SAM 3D's pose can place a mesh that
    looks right too far and too large (or too near and too small) when the depth on the frame is coarse or the mask is
    cut, and only depth tells these apart. Returns the new world_from_local and the factor (None: too little depth)."""
    _, _, C = _camera(row)
    vertices = mesh["vertices"] @ world[:3, :3].T + world[:3, 3]
    points, _ = surface_samples(vertices, mesh["faces"], mesh["colors"], count=40000)
    half = (mask.shape[0] // 2, mask.shape[1] // 2)
    _, front = project(points, row, half)
    inner = cv2.erode(cv2.resize(mask.astype(np.uint8), half[::-1], interpolation=cv2.INTER_NEAREST),
                      np.ones((5, 5), np.uint8)).astype(bool)
    measured = cv2.resize(depth, half[::-1], interpolation=cv2.INTER_NEAREST)
    both = inner & (front > 0) & (measured > 0)
    if both.sum() < min_pixels:
        return world, None
    k = float(np.median(measured[both] / front[both]))
    if not 0.2 < k < 5:
        return world, None
    scaled = world.copy()
    scaled[:3, :3] *= k
    scaled[:3, 3] = C + k * (world[:3, 3] - C)
    return scaled, round(k, 3)


def _generate(workspace, generator, layers, plan):
    """One object's mesh and world pose (asset.json), made or reused; returns (meta, mask, depth) or an error text."""
    spec, oid, refs, frame = plan
    mask = _union(layers, refs, frame) if frame is not None else None
    if mask is None or mask.sum() < 30:
        return f"the tracks have no mask on frame {frame}" if frame is not None else "the tracks are detected on no frame"
    depth = layers.depth(frame)
    faces = generator.cfg.get("faces", 20000)
    folder = workspace.evidence / "assets" / oid
    key = hashlib.sha256(json.dumps([refs, frame, faces, contract.sha256(WORKER), "depth_scale",
                                     generator.cfg.get("stage1_shortcut"), generator.cfg.get("stage2_steps")]).encode()
                         ).hexdigest()[:16]
    meta_path = folder / "asset.json"
    if meta_path.is_file() and contract.read_json(meta_path).get("key") == key:
        return contract.read_json(meta_path), mask, depth
    (folder / "inputs").mkdir(parents=True, exist_ok=True)
    np.save(folder / "inputs" / "mask.npy", mask)
    if depth is not None:
        np.save(folder / "inputs" / "depth.npy", depth.astype(np.float32))
    cam_row = layers.camera(frame)
    K, Rw, C = _camera(cam_row)
    answer = generator.generate({"image": str(layers.image(frame)), "mask": str(folder / "inputs" / "mask.npy"),
                                 "depth": str(folder / "inputs" / "depth.npy") if depth is not None else None,
                                 "K": K.tolist(), "out": str(folder), "faces": faces,
                                 "stage1_shortcut": generator.cfg.get("stage1_shortcut", False),
                                 "stage2_steps": generator.cfg.get("stage2_steps")})
    if not answer.get("ok"):
        return answer.get("error") or "the worker failed"
    A = np.asarray(answer["camera_from_local"])
    world = np.eye(4)
    world[:3, :3], world[:3, 3] = Rw @ A[:3, :3], Rw @ A[:3, 3] + C
    scale = None
    if depth is not None:
        world, scale = _depth_scale(world, np.load(folder / "mesh.npz"), cam_row, mask, depth)
    meta = {"id": oid, "name": spec.get("name") or layers.name(refs[0]), "tracks": refs, "frame": frame, "key": key,
            "world_from_local": world.tolist(), "depth_used": depth is not None, "depth_scale": scale,
            "layout_iou": answer.get("layout_iou"), "seconds": answer.get("seconds"), "faces": answer["faces"]}
    contract.write_json(meta_path, meta)
    return meta, mask, depth


def _check(workspace, layers, meta, mask, depth):
    """The placed mesh against the object's masks (source frame and others) and the measured depth; the card."""
    oid, refs, frame = meta["id"], meta["tracks"], meta["frame"]
    world = np.asarray(meta["world_from_local"])
    mesh = np.load(workspace.evidence / "assets" / oid / "mesh.npz")
    vertices = mesh["vertices"] @ world[:3, :3].T + world[:3, 3]
    points, rgb = surface_samples(vertices, mesh["faces"], mesh["colors"])
    # The silhouette against the masks on the source frame and on frames spread over the rest (for an object that
    # does not move; a moving one only matches its source frame).
    ious, depth_error, source_sil = {}, None, None
    for f in [frame, *_check_frames(layers, refs, frame)]:
        m = _union(layers, refs, f)
        if m is None or not m.any():
            continue
        half = (m.shape[0] // 2, m.shape[1] // 2)
        sil, front = project(points, layers.camera(f), half)
        small = cv2.resize(m.astype(np.uint8), half[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        ious[f] = round(float((sil & small).sum() / max((sil | small).sum(), 1)), 3)
        if f == frame:
            source_sil = cv2.resize(sil.astype(np.uint8), m.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            if depth is not None:
                measured = cv2.resize(depth, half[::-1], interpolation=cv2.INTER_NEAREST)
                both = small & (front > 0) & (measured > 0)
                if both.sum() >= 20:
                    depth_error = round(float(np.median((front[both] - measured[both]) / measured[both])), 3)
    low, high = vertices.min(0), vertices.max(0)
    row = {"id": oid, "name": meta["name"], "frame": frame, "tracks": refs, "size_m": np.round(high - low, 3).tolist(),
           "center": np.round((high + low) / 2, 3).tolist(), "lowest_z": round(float(low[2]), 3),
           "iou": {"source": ious.get(frame), "others": {str(f): v for f, v in ious.items() if f != frame}},
           "depth_error": depth_error, "depth": "measured" if meta["depth_used"] else "SAM 3D's own estimate",
           "depth_scale": meta.get("depth_scale")}
    title = (f"{oid} ({meta['name']}) frame {frame}: iou {ious.get(frame)}, size {row['size_m']} m, "
             f"depth error {depth_error}")
    card = _card(cv2.imread(str(layers.image(frame))), mask, source_sil if source_sil is not None else mask, points,
                 rgb, layers.camera(frame), title)
    return row, card


def make_objects(workspace, generator, specs, log=print):
    """Generate, place and check the objects (as many at once as there are workers); returns (rows, card images as
    BGR arrays) in the order of specs."""
    from concurrent.futures import ThreadPoolExecutor
    layers = layers_of(workspace)
    plans, rows = [], {}
    for spec in specs:                         # the frames first, so a long video's masks are read in one pass
        oid, refs = str(spec["id"]), [str(r) for r in spec["tracks"]]
        name = spec.get("name") or (layers.name(refs[0]) if refs and layers.known(refs[0]) else "")
        unknown = [r for r in refs if not layers.known(r)]
        if not ID.match(oid) or unknown or not refs:
            rows[oid] = {"id": oid, "name": name,
                         "error": f"unknown tracks {unknown}" if unknown else "id: letters, digits, _ and -"}
            continue
        frame = choose_frame(layers, refs) if spec.get("frame") is None else int(spec["frame"])
        plans.append((spec, oid, refs, frame))
    if hasattr(layers, "prefetch"):
        layers.prefetch([(r, f) for _, _, refs, frame in plans if frame is not None
                         for f in [frame, *_check_frames(layers, refs, frame)] for r in refs])
    with ThreadPoolExecutor(max_workers=generator.count) as pool:
        made = list(pool.map(lambda plan: _generate(workspace, generator, layers, plan), plans))
    cards = {}
    for plan, result in zip(plans, made):
        spec, oid, refs, frame = plan
        if isinstance(result, str):
            rows[oid] = {"id": oid, "name": spec.get("name", ""), "frame": frame, "error": result}
            continue
        rows[oid], cards[oid] = _check(workspace, layers, *result)
        log(f"make_objects {oid}: frame {frame}, iou {rows[oid]['iou']['source']}, {result[0].get('seconds')} s")
    order = [str(s["id"]) for s in specs]
    return [rows[o] for o in order if o in rows], [cards[o] for o in order if o in cards]


def _kabsch(source, target):
    """The rotation R and translation t that move source points onto target points (least squares)."""
    cs, ct = source.mean(0), target.mean(0)
    U, _, Vt = np.linalg.svd((source - cs).T @ (target - ct))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, ct - R @ cs


def _fit(pose, local, normals, observed, C, size, iterations=25):
    """ICP of the mesh's camera-facing surface samples to the observed points from pose; returns the new pose and the
    median distance of the matched points (metres)."""
    from scipy.spatial import cKDTree
    pose = pose.copy()
    residual = None
    for _ in range(iterations):
        world = local @ pose[:3, :3].T + pose[:3, 3]
        facing = ((normals @ pose[:3, :3].T) * (C - world)).sum(1) > 0
        if facing.sum() < 30:
            break
        tree = cKDTree(world[facing])
        dist, index = tree.query(observed)
        keep = dist < max(3 * np.median(dist), 0.01 * size)
        if keep.sum() < 20:
            break
        R, t = _kabsch(world[facing][index[keep]], observed[keep])
        pose[:3, :3], pose[:3, 3] = R @ pose[:3, :3], R @ pose[:3, 3] + t
        residual = float(np.median(dist[keep]))
        if np.linalg.norm(t) < 1e-4 * size and np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) < 1e-3:
            break
    return pose, residual


def _smooth(poses, frames):
    """Moving average over three frames: translations, and rotations as sign-aligned mean quaternions."""
    from scipy.spatial.transform import Rotation
    out = {}
    for i, f in enumerate(frames):
        near = [frames[j] for j in (i - 1, i, i + 1) if 0 <= j < len(frames)]
        M = [poses[g] for g in near]
        scale = np.linalg.norm(M[0][:3, :3], axis=0).mean()
        q = Rotation.from_matrix([m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0).mean() for m in M]).as_quat()
        q[(q @ q[len(q) // 2]) < 0] *= -1
        mean = q.mean(0)
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(mean / np.linalg.norm(mean)).as_matrix() * scale
        pose[:3, 3] = np.mean([m[:3, 3] for m in M], 0)
        out[f] = pose
    return out


def track_objects(workspace, ids, log=print):
    """Follow each made object over the frames with detections and depth; writes assets/<id>/track.json."""
    layers = layers_of(workspace)
    rows = []
    for oid in ids:
        folder = workspace.evidence / "assets" / str(oid)
        if not (folder / "asset.json").is_file():
            rows.append({"id": oid, "error": "no such made object (make_objects first)"})
            continue
        meta = contract.read_json(folder / "asset.json")
        mesh = np.load(folder / "mesh.npz")
        tri = mesh["vertices"][mesh["faces"]]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = np.linalg.norm(cross, axis=1)
        rng = np.random.default_rng(0)
        pick = rng.choice(len(tri), 6000, p=area / area.sum())
        a, b = rng.random(6000), rng.random(6000)
        flip = a + b > 1
        a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
        local = tri[pick, 0] + a[:, None] * (tri[pick, 1] - tri[pick, 0]) + b[:, None] * (tri[pick, 2] - tri[pick, 0])
        normals = cross[pick] / np.maximum(area[pick, None], 1e-12)
        start = np.asarray(meta["world_from_local"])
        size = float(np.ptp(local @ start[:3, :3].T, 0).max())
        refs, source = meta["tracks"], meta["frame"]
        frames = [f for f in sorted(set().union(*(layers.frames(r) for r in refs))) if f in layers.depth_frames()]
        if hasattr(layers, "prefetch"):
            layers.prefetch([(r, f) for f in frames for r in refs])
        poses, fits, failed = {source: start}, {}, []
        for direction in (1, -1):
            pose = start
            for f in [g for g in frames if (g - source) * direction > 0][::direction]:
                mask = _union(layers, refs, f)
                depth = layers.depth(f)
                row = layers.camera(f)
                K, R, C = _camera(row)
                inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) if mask is not None else None
                if inner is None or depth is None or (inner & (depth > 0)).sum() < 50:
                    failed.append(f)
                    poses[f] = pose
                    continue
                v, u = np.nonzero(inner & (depth > 0))
                if len(u) > 3000:
                    keep = rng.choice(len(u), 3000, replace=False)
                    u, v = u[keep], v[keep]
                z = depth[v, u]
                observed = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1) @ R.T + C
                new, residual = _fit(pose, local, normals, observed, C, size)
                half = (mask.shape[0] // 2, mask.shape[1] // 2)
                sil, _ = project(local @ new[:3, :3].T + new[:3, 3], row, half)
                small = cv2.resize(mask.astype(np.uint8), half[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
                iou = float((sil & small).sum() / max((sil | small).sum(), 1))
                if residual is None or iou < 0.3:
                    failed.append(f)
                    poses[f] = pose                 # held: the next frame starts from the last good pose
                    continue
                pose = new
                poses[f] = pose
                fits[f] = {"iou": round(iou, 3), "residual_m": round(residual, 4)}
        followed = sorted(poses)
        smooth = _smooth(poses, followed)
        first, last = smooth[followed[0]], smooth[followed[-1]]
        turn = np.degrees(np.arccos(np.clip((np.trace((last[:3, :3] / np.linalg.norm(last[:3, :3], axis=0).mean()) @
                                                      (first[:3, :3] / np.linalg.norm(first[:3, :3], axis=0).mean()).T)
                                             - 1) / 2, -1, 1)))
        contract.write_json(folder / "track.json", {
            "id": oid, "source_frame": source, "frames": followed, "failed": sorted(failed),
            "poses": {str(f): smooth[f].tolist() for f in followed}, "fits": {str(f): v for f, v in fits.items()}})
        ious = [v["iou"] for v in fits.values()]
        rows.append({"id": oid, "frames": [followed[0], followed[-1]], "followed": len(followed) - len(failed),
                     "failed_frames": sorted(failed)[:30], "moved_m": round(float(np.linalg.norm(last[:3, 3] - first[:3, 3])), 3),
                     "turned_deg": round(float(turn), 1), "median_iou": round(float(np.median(ious)), 3) if ious else None,
                     "last_frame_position": np.round(last[:3, 3], 3).tolist()})
        log(f"track_objects {oid}: {rows[-1]['followed']} frames, moved {rows[-1]['moved_m']} m, "
            f"turned {rows[-1]['turned_deg']} deg, {len(failed)} failed")
    return rows


def _inside_share(points, hull_points):
    """Share of points inside the convex hull of hull_points."""
    from scipy.spatial import Delaunay
    try:
        return float((Delaunay(hull_points).find_simplex(points) >= 0).mean())
    except Exception:                          # flat or degenerate hull
        return 0.0


def _side_by_side(det_a, det_b):
    """Seen as two separate things at once: a frame where both are detected with boxes that barely overlap and neither
    lies inside the other."""
    for f in set(det_a) & set(det_b):
        for _, a in det_a[f]:
            for _, b in det_b[f]:
                if not a or not b:
                    continue
                ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                area = lambda r: max(1, (r[2] - r[0]) * (r[3] - r[1]))
                if ix / (area(a) + area(b) - ix) < 0.2 and ix / min(area(a), area(b)) < 0.6:
                    return True
    return False


def _map_card(rows, groups, size=CARD * 2):
    """Top-down map of the completed footprints (convex hulls of the meshes' xy), coloured by group, labelled."""
    import colorsys
    hulls = {r["id"]: r["_xy"] for r in rows}
    allxy = np.concatenate(list(hulls.values()))
    lo, hi = allxy.min(0), allxy.max(0)
    span = max(np.ptp(allxy, 0).max(), 0.5) * 1.08
    centre = (lo + hi) / 2
    to_px = lambda xy: np.stack([((xy[:, 0] - centre[0]) / span + 0.5) * (size - 1),
                                 (0.5 - (xy[:, 1] - centre[1]) / span) * (size - 1)], 1).astype(np.int32)
    image = np.full((size, size, 3), 250, np.uint8)
    for k, group in enumerate(groups):
        colour = tuple(int(255 * c) for c in colorsys.hsv_to_rgb((k * 0.618) % 1, 0.65, 0.85))[::-1]
        for oid in group:
            pts = to_px(hulls[oid])
            cv2.fillPoly(image, [pts], tuple(int(c * 0.35 + 255 * 0.65) for c in colour))
            cv2.polylines(image, [pts], True, colour, 2)
    for r in rows:
        x, y = to_px(r["_xy"].mean(0, keepdims=True))[0]
        cv2.putText(image, r["id"], (int(x) - 12, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(image, f"top view, {span / 1.08:.1f} m across (+x right, +y up)", (6, size - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    return image


def inventory(workspace, generator, categories=None, max_objects=64, log=print):
    """Complete the evidence objects of the categories and group the ones whose completions overlap; returns (rows per
    category, card images)."""
    from scipy.spatial import ConvexHull
    from .fusion import head_noun
    layers = layers_of(workspace)
    if not isinstance(layers, FusedLayers):
        return [{"error": "inventory works on fused evidence (a room's objects); with raw layers group the tracks"}], []
    by_category = {}
    for oid, o in layers.objects.items():
        by_category.setdefault(head_noun(o.get("category") or ""), []).append(oid)
    wanted = [head_noun(c) for c in categories] if categories else \
        [c for c, ids in sorted(by_category.items(), key=lambda kv: -len(kv[1])) if len(ids) >= 2 and c]
    chosen = []
    for c in wanted:
        ids = sorted(by_category.get(c, []), key=lambda i: -(layers.objects[i].get("evidence") or {}).get("views_used", 0))
        chosen += [(c, i) for i in ids][:max(0, max_objects - len(chosen))]
    if not chosen:
        return [{"error": f"no evidence objects of {wanted}"}], []
    made, cards = make_objects(workspace, generator, [{"id": i, "name": layers.objects[i].get("category", ""),
                                                       "tracks": [i]} for _, i in chosen], log)
    by_id = {r["id"]: r for r in made}
    card_of = dict(zip([r["id"] for r in made if "error" not in r], cards))    # cards come for the made ones only
    out_rows, out_cards = [], []
    for c in dict.fromkeys(c for c, _ in chosen):
        rows = []
        for _, i in [x for x in chosen if x[0] == c]:
            r = by_id.get(i, {"id": i, "error": "not made"})
            ev = layers.objects[i].get("evidence") or {}
            row = {"id": i, "category": layers.objects[i].get("category"), "views": ev.get("views_used"),
                   "level": ev.get("evidence_level", "certain")}
            if "error" in r:
                rows.append({**row, "error": r["error"]})
                continue
            meta = contract.read_json(workspace.evidence / "assets" / i / "asset.json")
            mesh = np.load(workspace.evidence / "assets" / i / "mesh.npz")
            M = np.asarray(meta["world_from_local"])
            world = mesh["vertices"] @ M[:3, :3].T + M[:3, 3]
            samples = surface_samples(world, mesh["faces"], mesh["colors"], count=3000)[0]
            xy = world[:, :2]
            try:
                xy = xy[ConvexHull(xy).vertices]
            except Exception:
                pass
            rows.append({**row, "size_m": r["size_m"], "lowest_z": r["lowest_z"], "center": r["center"],
                         "iou": r["iou"]["source"], "_samples": samples, "_world": world, "_xy": xy})
        ok = [r for r in rows if "error" not in r]
        det = {r["id"]: layers.detections(r["id"]) for r in ok}
        parent = {r["id"]: r["id"] for r in ok}
        find = lambda a: a if parent[a] == a else find(parent[a])
        overlaps, near = [], []
        for i, a in enumerate(ok):
            for b in ok[i + 1:]:
                gap = float(np.linalg.norm(np.subtract(a["center"], b["center"])[:2]))
                if gap > (max(a["size_m"]) + max(b["size_m"])) / 2:      # footprints apart: nothing to compare
                    continue
                # Two completions of one object overlap only partly (each is posed from its own frame); 0.15 of
                # either's surface inside the other's hull separates fragments from neighbours on the test rooms.
                share = max(_inside_share(a["_samples"], b["_world"]), _inside_share(b["_samples"], a["_world"]))
                apart = _side_by_side(det[a["id"]], det[b["id"]])
                pair = {"ids": [a["id"], b["id"]], "overlap": round(share, 2), "gap_m": round(gap, 2),
                        "seen_side_by_side": apart}
                if share >= 0.15 and not apart:
                    overlaps.append(pair)
                    parent[find(b["id"])] = find(a["id"])
                else:
                    near.append(pair)
        groups = {}
        for r in ok:
            groups.setdefault(find(r["id"]), []).append(r["id"])
        groups = sorted(groups.values(), key=lambda g: (-len(g), g))
        out_rows.append({"category": c, "evidence_objects": len(rows), "groups_by_overlap": len(groups),
                         "groups": groups, "overlaps": overlaps, "near_but_apart": near,
                         "objects": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]})
        heads = []
        for r in ok:
            card = card_of.get(r["id"])
            if card is None:
                continue
            tile = cv2.resize(card[26:, :CARD], (180, 180))
            label = np.full((22, 180, 3), 255, np.uint8)
            cv2.putText(label, f"{r['id']} {max(r['size_m']):.2f}m z{r['lowest_z']:.2f}", (3, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
            heads.append(np.vstack([label, tile]))
        if heads:
            cols = 4
            while len(heads) % cols:
                heads.append(np.full_like(heads[0], 255))
            grid = np.vstack([np.hstack(heads[k:k + cols]) for k in range(0, len(heads), cols)])
            top = _map_card(ok, groups)
            h = max(grid.shape[0], top.shape[0])
            pad = lambda im: np.vstack([im, np.full((h - im.shape[0], im.shape[1], 3), 255, np.uint8)])
            sheet = np.hstack([pad(grid), pad(top)])
            bar = np.full((26, sheet.shape[1], 3), 255, np.uint8)
            cv2.putText(bar, f"{c}: {len(rows)} evidence objects, {len(groups)} groups by overlap", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            out_cards.append(np.vstack([bar, sheet]))
        log(f"inventory {c}: {len(rows)} objects, {len(groups)} groups")
    return out_rows, out_cards


def prepare_meshes(workspace, cfg, log=print, min_size=0.25, limit=200):
    """A room (fused evidence), before the agent starts: complete every evidence object at least min_size m across
    (largest first, up to limit) and list in assets/default.json the ones whose mesh agrees with its fused box
    (source-frame IoU >= 0.4, at most 1.6 times its size, centre within half its size or 0.3 m): A.add_object builds
    those meshes in place of a proxy. Returns that record, or None for raw layers."""
    from .fusion import head_noun
    layers = layers_of(workspace)
    if not isinstance(layers, FusedLayers):
        return None
    boxes = {}
    for oid, o in layers.objects.items():
        bounds, pose = np.asarray(o.get("bounds") or [0] * 6, float), (o.get("poses") or [None])[0]
        if pose is None or head_noun(o.get("category") or "") in ("wall", "floor", "ceiling", "ground"):
            continue
        size = bounds[1::2] - bounds[0::2]
        if size.max() >= min_size:
            centre = np.asarray(pose[1:4], float) + contract.quat_wxyz_to_matrix(pose[4:8]) @ (
                (bounds[0::2] + bounds[1::2]) / 2)
            boxes[oid] = (float(size.max()), centre)
    ids = sorted(boxes, key=lambda i: -boxes[i][0])[:limit]
    started = time.time()
    generator = Generator(cfg, workspace.run / "objects3d" / "prepare", log)
    try:
        made, _ = make_objects(workspace, generator, [{"id": i, "name": layers.objects[i].get("category", ""),
                                                       "tracks": [i]} for i in ids], log)
    finally:
        generator.stop()
    record = {"ids": [], "rejected": {}, "min_size_m": min_size, "candidates": len(boxes), "tried": len(ids),
              "seconds": round(time.time() - started, 1)}
    for r in made:
        size, centre = boxes[r["id"]]
        if "error" in r:
            reason = r["error"]
        elif r["iou"]["source"] < 0.4:
            reason = f"source-frame IoU {r['iou']['source']:.2f}"
        elif max(r["size_m"]) > 1.6 * size:
            reason = f"mesh {max(r['size_m']):.2f} m, box {size:.2f} m"
        elif np.linalg.norm(np.subtract(r["center"], centre)) > max(0.3, size / 2):
            reason = f"centre {np.linalg.norm(np.subtract(r['center'], centre)):.2f} m from the box's"
        else:
            record["ids"].append(r["id"])
            continue
        record["rejected"][r["id"]] = reason
    contract.write_json(workspace.evidence / "assets" / "default.json", record)
    log(f"meshes before the agent: {len(record['ids'])} of {len(ids)} agree with their boxes ({record['seconds']:.0f} s)")
    return record
