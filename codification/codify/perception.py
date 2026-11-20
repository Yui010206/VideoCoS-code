"""Local perception before the model loop: cached stages and immutable, versioned evidence.

Stages: decode -> geometry (DA3) -> camera_depth (cameras, world frame, support surface) -> vocabulary
(local VLM) -> tracks (SAM 3) -> fuse (3D objects). A stage is reused only when its key (inputs, config,
upstream keys and the source of the code it runs) matches, so repeated identical requests never rerun
models. publish() writes an evidence version: a directory with a manifest of file hashes that is never
modified afterwards; generated scene code reads one version mounted read-only, and evaluation verifies
the hashes before reading it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import contract

CODE = Path(__file__).resolve().parent


def mode_of(config):
    """What the perception had beside the video: the dataset's cameras, a declared lens, or nothing."""
    if config.get("fuse", {}).get("given_cameras"):
        return "rgb_plus_cameras"
    if config.get("decode", {}).get("rectify"):
        return "rgb_plus_calibration"
    return "rgb_only"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_hash(spec):
    """sha256 of a module file, or of one function's source for "file.py:function"."""
    if ":" not in spec:
        return contract.sha256(CODE / spec)
    import importlib
    import inspect
    module, function = spec.split(":")
    source = inspect.getsource(getattr(importlib.import_module("codify." + module[:-3]), function))
    return hashlib.sha256(source.encode()).hexdigest()


def stage_key(name, section, upstream, code_specs, extra=None):
    payload = {"stage": name, "config": section, "upstream": upstream,
               "code": {spec: code_hash(spec) for spec in code_specs}, **(extra or {})}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_stage(work, name, key, func, log):
    """Reuse a stage only with a success marker for the same key; otherwise recompute it."""
    directory = work / name
    marker = work / f"{name}.stage.json"
    if marker.is_file():
        record = contract.read_json(marker)
        if record.get("status") == "success" and record.get("key") == key and directory.exists():
            log(f"[{name}] resume: inputs, config and code unchanged")
            return record
        marker.unlink()
    if directory.exists():
        shutil.rmtree(directory)
    started = time.time()
    log(f"[{name}] running")
    try:
        result = func(directory)
    except Exception as error:
        contract.write_json(marker.with_suffix(".failed.json"), {"status": "failed", "key": key,
                                                                 "error": repr(error), "finished_utc": now()})
        raise
    record = {"status": "success", "key": key, "finished_utc": now(),
              "wall_seconds": round(time.time() - started, 2), "result": result}
    contract.write_json(marker, record)
    marker.with_suffix(".failed.json").unlink(missing_ok=True)
    log(f"[{name}] done in {record['wall_seconds']} s: {json.dumps(result['summary'])[:300]}")
    for warning in result.get("warnings", []):
        log(f"[{name}] warning: {warning}")
    return record


def room_structure(points, rng, max_planes=8, inlier_m=0.03, min_fraction=0.03):
    """Floor extent and vertical plane candidates from gravity-aligned world points (floor at z=0).

    Candidates are evidence only: a vertical plane can be a wall, a door, a cabinet front or
    a window. Each has a horizontal unit normal n and offset d (n . p = d), the 2nd-98th
    percentile segment along the plane and its z range.
    """
    xy = np.percentile(points[:, :2], [2, 98], axis=0)
    floor = {"z": 0.0, "x_range": xy[:, 0].tolist(), "y_range": xy[:, 1].tolist(),
             "highest_surface_z_p99": float(np.percentile(points[:, 2], 99))}
    rest = points[points[:, 2] > 0.15]
    planes, total = [], len(rest)
    while len(planes) < max_planes and len(rest) > 100:
        best, best_count = None, 0
        for _ in range(400):
            a, b = rest[rng.choice(len(rest), 2, replace=False)]
            direction = (b - a)[:2]
            if np.linalg.norm(direction) < 0.3:
                continue
            n = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
            count = int((np.abs(rest[:, :2] @ n - a[:2] @ n) < inlier_m).sum())
            if count > best_count:
                best, best_count = (n, a[:2] @ n), count
        if best is None or best_count < min_fraction * total:
            break
        n, d = best
        inliers = np.abs(rest[:, :2] @ n - d) < inlier_m
        pts = rest[inliers]
        centered = pts[:, :2] - pts[:, :2].mean(0)
        direction = np.linalg.svd(centered, full_matrices=False)[2][0]
        n = np.array([-direction[1], direction[0]])
        d = float(pts[:, :2].mean(0) @ n)
        t = pts[:, :2] @ direction
        t0, t1 = np.percentile(t, [2, 98])
        z0, z1 = np.percentile(pts[:, 2], [2, 98])
        foot = n * d
        planes.append({"normal_xy": n.tolist(), "offset": d, "p0": (foot + direction * t0).tolist(),
                       "p1": (foot + direction * t1).tolist(), "z_range": [float(z0), float(z1)],
                       "support_fraction": float(inliers.sum() / total)})
        rest = rest[~inliers]
    return {"floor": floor, "vertical_planes": planes,
            "note": "Estimated from DA3 depth; vertical planes may be walls, doors, windows or furniture faces."}


def join_tracks(first, second, out):
    """One tracks directory from two SAM 3 runs over the same frames: the second run's prompts and track ids
    follow the first's (label files are linked, not copied), so existing track ids stay as they are."""
    first, second, out = Path(first), Path(second), Path(out)
    out.mkdir(parents=True)
    a, b = contract.read_json(first / "tracks.json"), contract.read_json(second / "tracks.json")
    if a["evaluated_frames"] != b["evaluated_frames"]:
        raise ValueError("the two track sets cover different frames")
    shift_prompt, shift_id = len(a["prompts"]), max((int(t["track_id"]) for t in a["tracks"]), default=0)
    for j in range(len(a["prompts"])):
        (out / f"labels_{j:03d}.npz").symlink_to((first / f"labels_{j:03d}.npz").resolve())
    for j in range(len(b["prompts"])):
        (out / f"labels_{shift_prompt + j:03d}.npz").symlink_to((second / f"labels_{j:03d}.npz").resolve())
    tracks = a["tracks"] + [{**t, "track_id": str(int(t["track_id"]) + shift_id),
                             "prompt_index": t["prompt_index"] + shift_prompt} for t in b["tracks"]]
    contract.write_json(out / "tracks.json", {**a, "prompts": a["prompts"] + b["prompts"], "tracks": tracks,
                                              "joined": {"first": str(first), "second": str(second),
                                                         "second_track_ids_after": shift_id}})
    return contract.tool_result("success", {"tracks": out / "tracks.json"},
                                {"prompts": len(a["prompts"]) + len(b["prompts"]), "track_count": len(tracks),
                                 "new_tracks": len(b["tracks"])})


class Perception:
    """Runs and caches perception tasks for one video inside run_dir/work."""

    def __init__(self, run_dir, video_path, config, log=print):
        self.run, self.video, self.config, self.log = Path(run_dir), Path(video_path), config, log
        self.work = self.run / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.keys, self.parts = {}, {}
        self.video_sha = contract.sha256(self.video)

    def _stage(self, name, section, upstream, code, func, extra=None):
        key = stage_key(name, section, {u: self.keys[u] for u in upstream}, code, extra)
        record = run_stage(self.work, name, key, func, self.log)
        self.keys[name] = key
        return record

    def decode(self):
        from . import video
        if "decode" not in self.keys:
            self._stage("decode", self.config["decode"], (), ("video.py",),
                        lambda d: video.decode_video(self.video, d, self.config["decode"]), {"video": self.video_sha})
        return self.work / "decode"

    def _model_section(self, name):
        return {k: v for k, v in self.config[name].items() if k not in ("gpu", "cpu_threads")}

    def camera_depth(self):
        from . import fusion, models
        decode = self.decode()
        geometry, code = models.geometry_stage(self.config["geometry"])
        self._stage("geometry", self._model_section("geometry"), ("decode",), code,
                    lambda d: geometry(decode / "frames", decode / "timeline.jsonl", d, self.config["geometry"]))
        def post(out):
            import cv2
            cfg = {**fusion.DEFAULTS, **self.config.get("fuse", {})}
            rng = np.random.default_rng(cfg["seed"])
            out.mkdir(parents=True)
            rec = fusion.reconstruction(self.work / "geometry", decode, cfg, rng)
            rows = fusion.camera_rows({"keyframes": rec.keyframes, "R_wc": rec.R_wc, "C": rec.C,
                                       "processed_hw": [rec.h, rec.w]}, rec.world, rec.video_info,
                                      contract.read_jsonl(decode / "timeline.jsonl"))
            structure = room_structure(rec.scene_points, np.random.default_rng(cfg["seed"] + 1),
                                       inlier_m=0.03 * rec.scene_scale)
            # Colors for a coarse colored point cloud (depth grid -> canonical pixels).
            colors = []
            for i, f in enumerate(rec.keyframes):
                image = cv2.imread(str(decode / f"frames/{f:06d}.jpg"))[:, :, ::-1]
                idx = np.flatnonzero(rec.valid[i])
                idx = np.random.default_rng(f).choice(idx, min(600, len(idx)), replace=False)
                u = np.clip(((rec.uu.ravel()[idx] + 0.5) * rec.W / rec.w).astype(int), 0, rec.W - 1)
                v = np.clip(((rec.vv.ravel()[idx] + 0.5) * rec.H / rec.h).astype(int), 0, rec.H - 1)
                cam = fusion.unproject(rec.K_inv, rec.uu.ravel()[idx], rec.vv.ravel()[idx], rec.depth[i].ravel()[idx])
                colors.append((rec.to_world(cam @ rec.R_wc[i].T + rec.C[i]), image[v, u]))
            np.savez_compressed(out / "points.npz", xyz=np.concatenate([c[0] for c in colors]).astype(np.float32),
                                rgb=np.concatenate([c[1] for c in colors]).astype(np.uint8))
            contract.write_json(out / "cameras.json", rows)
            contract.write_json(out / "world.json", rec.world)
            contract.write_json(out / "structure.json", structure)
            return contract.tool_result("success", {"cameras": out / "cameras.json", "world": out / "world.json",
                                                    "structure": out / "structure.json", "points": out / "points.npz"},
                                        {"frames": len(rows), "up_status": rec.up_status,
                                         "scale_status": rec.world["scale_status"],
                                         "vertical_planes": len(structure["vertical_planes"])}, rec.warnings)
        self._stage("camera_depth", self.config.get("fuse", {}), ("geometry",),
                    ("fusion.py:reconstruction", "fusion.py:estimate_world", "fusion.py:camera_rows",
                     "perception.py:room_structure"), post)
        self.parts["camera_depth"] = self.work / "camera_depth"
        return contract.read_json(self.work / "camera_depth.stage.json")["result"]

    def vocabulary(self):
        """Object names from the local VLM (config "vocabulary"); same stage name and key as the offline run."""
        from . import models
        decode = self.decode()
        # The text given with the input (T6's goal, T7's description) names objects the frames may show.
        context = (self.config.get("task") or {}).get("text")
        section = {**self._model_section("vocabulary"), **({"context": context} if context else {})}
        self._stage("vocabulary", section, ("decode",), models.VOCABULARY_CODE,
                    lambda d: models.discover_vocabulary(decode / "frames", decode / "timeline.jsonl", d,
                                                         {**self.config["vocabulary"], **({"context": context} if context else {})}))
        return contract.read_json(self.work / "vocabulary" / "vocabulary.json")["names"]

    def segment_track(self, prompts=None):
        """SAM 3 tracks for the object names: the local VLM's vocabulary (config "vocabulary"), an explicit list
        (config tracks.prompts), or the names given here (a find_objects search, its own stage)."""
        from . import models
        decode = self.decode()
        extra = None
        if prompts is None:
            if "vocabulary" in self.config:
                prompts = self.vocabulary()
                extra = {"prompts": prompts}
            elif self.config["tracks"].get("prompts"):
                prompts = list(self.config["tracks"]["prompts"])
            else:
                raise ValueError("segment_track needs object names: config 'vocabulary' (local VLM) or tracks.prompts")
            tag, section = "tracks", self._model_section("tracks")
        else:
            tag = "tracks_" + hashlib.sha256(json.dumps(prompts).encode()).hexdigest()[:10]
            section = {**self._model_section("tracks"), "prompts": prompts}
        self._stage(tag, section, ("decode",), models.TRACKS_CODE,
                    lambda d: models.segment_track(decode / "frames", decode / "timeline.jsonl", d,
                                                   {**self.config["tracks"], "prompts": prompts}), extra)
        self.keys["tracks"] = self.keys[tag]
        self.parts["tracks"] = self.work / tag
        return contract.read_json(self.work / f"{tag}.stage.json")["result"]

    def add_names(self, names):
        """SAM 3 tracks for more object names, joined after the current tracks; returns {name: tracks found}."""
        from . import models
        decode = self.decode()
        prompts = [n for n in dict.fromkeys(str(n).strip().lower() for n in names) if n]
        if not prompts or "tracks" not in self.parts:
            return {}
        tag = "tracks_" + hashlib.sha256(json.dumps(prompts).encode()).hexdigest()[:10]
        self._stage(tag, {**self._model_section("tracks"), "prompts": prompts}, ("decode",), models.TRACKS_CODE,
                    lambda d: models.segment_track(decode / "frames", decode / "timeline.jsonl", d,
                                                   {**self.config["tracks"], "prompts": prompts}))
        base = self.parts["tracks"]
        joined = "tracks_joined_" + hashlib.sha256((self.keys["tracks"] + self.keys[tag]).encode()).hexdigest()[:10]
        self.keys["previous_tracks"] = self.keys["tracks"]
        self._stage(joined, {}, ("previous_tracks", tag), ("perception.py:join_tracks",),
                    lambda d: join_tracks(base, self.work / tag, d))
        self.keys["tracks"], self.parts["tracks"] = self.keys[joined], self.work / joined
        found = contract.read_json(self.work / tag / "tracks.json")["tracks"]
        return {name: sum(t["prompt"] == name for t in found) for name in prompts}

    def fuse_objects(self):
        from . import fusion
        if "camera_depth" not in self.parts or "tracks" not in self.parts:
            raise RuntimeError("fuse_objects needs camera_depth and segment_track results first")
        decode = self.decode()
        tag = "fuse_" + self.keys["tracks"][:10]
        joined = contract.read_json(self.parts["tracks"] / "tracks.json").get("joined")
        # After a join, the earlier tracks keep their ids as kept members of any merged group.
        cfg = {**self.config.get("fuse", {}), **({"new_track_ids_after": joined["second_track_ids_after"]} if joined else {})}
        previous = self.parts.get("fuse")        # its per-track records are reused for the tracks it shares
        self._stage(tag, cfg, ("geometry", "tracks"), ("fusion.py", "contract.py"),
                    lambda d: fusion.fuse_objects(self.work / "geometry", self.parts["tracks"], decode / "frames", d, cfg,
                                                  reuse=previous))
        self.parts["fuse"] = self.work / tag
        return contract.read_json(self.work / f"{tag}.stage.json")["result"]

    def proposals(self):
        """Objects the vocabulary missed: class-agnostic SAM 3 proposals on a few frames, the ones no track explains
        named by the VLM from their crops, the named ones tracked from their masks and joined to the tracks
        (config "proposals"); returns the joined tracks directory, or None when nothing new was found."""
        from . import models
        if "proposals" not in self.config or "tracks" not in self.parts:
            return []
        decode = self.decode()
        tracks_dir = self.parts["tracks"]
        sam_cfg = {**self._model_section("tracks"), **{k: v for k, v in self.config["proposals"].items()}}
        sam_cfg = {k: v for k, v in sam_cfg.items() if k not in ("window", "window_overlap", "link_iou", "prompts")}
        self._stage("proposals", sam_cfg, ("decode", "tracks"), models.PROPOSAL_CODE,
                    lambda d: models.propose_objects(decode / "frames", decode / "timeline.jsonl", tracks_dir, d,
                                                     {**sam_cfg, "gpu": self.config["tracks"].get("gpu", 0),
                                                      "python": self.config["tracks"]["python"]}))
        context = (self.config.get("task") or {}).get("text")
        vlm_cfg = {**{k: v for k, v in self._model_section("vocabulary").items()
                      if k not in ("frames", "names_per_frame", "max_names", "max_new_tokens", "repetition_penalty")},
                   **({"context": context} if context else {})}
        self._stage("proposal_names", vlm_cfg, ("proposals",), models.PROPOSAL_NAME_CODE,
                    lambda d: models.name_proposals(self.work / "proposals", d,
                                                    {**vlm_cfg, "gpu": self.config["vocabulary"].get("gpu", 0),
                                                     "python": self.config["vocabulary"]["python"]}))
        # Named proposals are tracked from their own masks (text prompts for such names found nothing reliable);
        # the tracks that no existing track covers are joined to the others, their names as categories.
        track_cfg = {**sam_cfg, "group_size": self.config["proposals"].get("group_size", 16),
                     "track_window": self.config["proposals"].get("track_window", 0),
                     "dedupe_iou": self.config["proposals"].get("dedupe_iou", 0.5)}
        self._stage("proposal_tracks", track_cfg, ("proposal_names", "tracks"), models.PROPOSAL_TRACK_CODE,
                    lambda d: models.track_proposals(decode / "frames", self.work / "proposals",
                                                     self.work / "proposal_names" / "crop_names.json", tracks_dir, d,
                                                     {**track_cfg, "gpu": self.config["tracks"].get("gpu", 0),
                                                      "python": self.config["tracks"]["python"]}))
        new_tracks = self.work / "proposal_tracks"
        if not contract.read_json(new_tracks / "tracks.json")["tracks"]:
            return None
        joined = "tracks_joined_" + hashlib.sha256((self.keys["tracks"] + self.keys["proposal_tracks"]).encode()).hexdigest()[:10]
        self.keys["previous_tracks"] = self.keys["tracks"]
        self._stage(joined, {}, ("previous_tracks", "proposal_tracks"), ("perception.py:join_tracks",),
                    lambda d: join_tracks(tracks_dir, new_tracks, d))
        self.keys["tracks"], self.parts["tracks"] = self.keys[joined], self.work / joined
        return self.work / joined

    def perceive(self):
        """All local perception (cached stages), or the given perception of config evidence.given, published as
        one evidence version. After the named tracks, class-agnostic proposals add the names the vocabulary missed
        (their SAM 3 pass joins the existing tracks)."""
        given = self.config.get("evidence", {}).get("given")
        if not given:
            self.camera_depth()
            self.segment_track()
            self.proposals()
            self.fuse_objects()
        return self.publish(given)

    def _comparison_material(self, temp, decode):
        """The frames the scene is compared on, and on each of them a map of which object every pixel's SAM 3 mask
        belongs to (uint16 object ids): the checks' references for silhouette IoU, and the frames the model may
        project as textures."""
        import cv2
        timeline = contract.read_jsonl(decode / "timeline.jsonl")
        comparison = contract.comparison_frames(len(timeline), self.config.get("agent", {}).get("comparison_frames"))
        (temp / "frames").mkdir()
        (temp / "masks").mkdir()
        for f in comparison:
            shutil.copyfile(decode / f"frames/{f:06d}.jpg", temp / f"frames/{f:06d}.jpg")
        wanted = set(comparison)
        tracks = contract.read_json(self.parts["tracks"] / "tracks.json")
        by_track = {t["track_id"]: t for t in tracks["tracks"]}
        rows = [r for r in contract.read_jsonl(self.parts["fuse"] / "observations.jsonl")
                if r["frame_index"] in wanted and r["state"] == "detected" and r["object_id"].isdigit()]
        maps = {}
        for prompt_index in sorted({by_track[r.get("track_id", r["object_id"])]["prompt_index"] for r in rows}):
            labels = contract.read_labels(self.parts["tracks"] / f"labels_{prompt_index:03d}.npz")
            for r in rows:
                track = by_track[r.get("track_id", r["object_id"])]
                if track["prompt_index"] != prompt_index:
                    continue
                f = r["frame_index"]
                maps.setdefault(f, np.zeros(labels.shape[1:], np.uint16))
                mask = labels[f] == track["label_value"]
                maps[f][mask & (maps[f] == 0)] = int(r["object_id"])
            del labels
        for f in sorted(wanted):
            cv2.imwrite(str(temp / f"masks/{f:06d}.png"), maps.get(f, np.zeros((1, 1), np.uint16)))
        contract.write_json(temp / "frames.json", {"comparison": comparison,
                                                   "masks": "masks/<frame>.png: uint16 object id per pixel (0 = none)",
                                                   "frames": "frames/<frame>.jpg: the comparison frames"})

    def publish(self, given=None):
        """Write a new immutable evidence version when the evidence changed; return its directory. given: a
        directory already in evidence form (e.g. an example's reference perception) used instead of the local
        stages; its files are copied and the decoded video's timeline must match."""
        decode = self.decode()
        timeline = contract.read_jsonl(decode / "timeline.jsonl")
        if given:
            given = Path(given).resolve()
            given_manifest = contract.verify_evidence(given)
            mode = given_manifest["mode"]
            sources = {"decode": self.keys["decode"], "given": str(given), "given_digest": given_manifest["digest"]}
        else:
            mode = mode_of(self.config)
            sources = {"decode": self.keys["decode"], **{k: self.keys.get(k) for k in ("geometry", "tracks")},
                       **{f"part:{k}": str(v.name) for k, v in self.parts.items()}}
        digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        root = self.run / "evidence"
        versions = sorted(root.glob("v[0-9][0-9][0-9]")) if root.exists() else []
        if versions and contract.read_json(versions[-1] / "manifest.json")["digest"] == digest:
            return versions[-1]
        target = root / f"v{len(versions) + 1:03d}"
        temp = root / (target.name + ".tmp")
        shutil.rmtree(temp, ignore_errors=True)
        temp.mkdir(parents=True)
        info = contract.read_json(decode / "video_info.json")
        contract.write_json(temp / "video.json", {**info["canonical"], "projection": info.get("projection"),
                                                  "transforms": info["transforms"], "source_video_sha256": info["source_video_sha256"]})
        shutil.copyfile(decode / "timeline.jsonl", temp / "timeline.jsonl")
        if given:
            if len(contract.read_jsonl(given / "timeline.jsonl")) != len(timeline):
                raise RuntimeError("the given evidence's timeline does not match the decoded video")
            for item in given.iterdir():             # hard links: a given evidence is immutable (its manifest)
                if item.name not in ("manifest.json", "video.json", "timeline.jsonl"):
                    try:
                        shutil.copytree(item, temp / item.name, copy_function=os.link) if item.is_dir() \
                            else os.link(item, temp / item.name)
                    except OSError:
                        (shutil.copytree if item.is_dir() else shutil.copyfile)(item, temp / item.name)
        if "camera_depth" in self.parts:
            for name in ("cameras.json", "world.json", "structure.json", "points.npz"):
                shutil.copyfile(self.parts["camera_depth"] / name, temp / name)
        if "tracks" in self.parts:
            tracks_dir = self.parts["tracks"]
            tracks = contract.read_json(tracks_dir / "tracks.json")
            contract.write_json(temp / "tracks.json", {
                "prompts": tracks["prompts"], "evaluated_frames": tracks["evaluated_frames"],
                "tracks": [{"track_id": t["track_id"], "prompt": t["prompt"], "detections": len(t["detections"]),
                            "first_frame": t["detections"][0]["frame_index"], "last_frame": t["detections"][-1]["frame_index"],
                            "mean_score": round(float(np.mean([d["score"] for d in t["detections"]])), 3)}
                           for t in tracks["tracks"]]})
            # The masks stay in work/; their hashes let an evaluation check they were not edited afterwards.
            sources["artifacts"] = {"tracks_dir": str(tracks_dir.relative_to(self.run)),
                                    "tracks_json": contract.sha256(tracks_dir / "tracks.json"),
                                    "labels": {p.name: contract.sha256(p) for p in sorted(tracks_dir.glob("labels_*.npz"))}}
        if "fuse" in self.parts:
            fuse = self.parts["fuse"]
            for source, name in (("objects.json", "objects.json"), ("unresolved_tracks.json", "unresolved.json"),
                                 ("object_appearance.json", "appearance.json"), ("observations.jsonl", "observations.jsonl")):
                shutil.copyfile(fuse / source, temp / name)
            if (fuse / "appearance").is_dir():
                shutil.copytree(fuse / "appearance", temp / "appearance")
            self._comparison_material(temp, decode)
        contract.write_manifest(temp, mode, sources, digest=digest)
        temp.rename(target)
        return target
