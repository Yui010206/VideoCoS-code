"""Video consistency: a built scene against the video it was built from.

The scene is seen from the video's own camera on frames spread over the whole input (blender_ops ids: one ray per
cell of a pixel grid, the registered object hit and its camera depth) and compared, object by object, with what the
perception observed on those frames: SAM 3 masks (with fused evidence, each evidence object's merged tracks; with raw
layers, every track) and the measured depth.

Per scene object: the silhouette IoU with its observed mask (the same id with fused evidence, else the observation it
overlaps most), and the depth error where both agree (built surface minus measured, + = too far). Also the observed
objects nothing built explains, and the built objects the video shows nothing at. One score, computed in code: the
share of observed object pixels the right scene object covers (an IoU weighted by observed area). Cards show the worst
objects on their source frame, the observed mask outlined in green and the built silhouette in magenta.

Observed masks and depth at the sampled frames are cached per run (cache/video_check/); each check renders only the
scene. Report only: nothing here changes a scene.
"""
from __future__ import annotations

import hashlib
import json
import threading
import zipfile
from pathlib import Path

import numpy as np

from . import contract

MIN_CELLS = 6            # an observed mask smaller than this many grid cells on a frame is not scored
MAX_FRAME_SHARE = 0.4    # a mask covering more of the frame is background (a floor, a wall), not an object
TARGET_RAYS = 40000      # grid cells per frame: the stride follows the resolution
_LOCK = threading.Lock()  # one reading of the label stacks at a time (prepare() in the background, check() waits)


def sample_frames(workspace, count):
    """Canonical frame indices: the comparison frames, then evenly spread ones, at most `count`."""
    total = len(workspace.timeline)
    picks = list(dict.fromkeys(int(f) for f in workspace.comparison))
    for f in np.linspace(0, total - 1, num=max(count, 1)).round().astype(int):
        if len(picks) >= count:
            break
        if int(f) not in picks:
            picks.append(int(f))
    return sorted(picks[:count])


def grid_of(width, height):
    stride = max(1, int(np.ceil(np.sqrt(width * height / TARGET_RAYS))))
    return stride, np.arange(stride / 2, width, stride), np.arange(stride / 2, height, stride)


def _label_frames(path, frames):
    """{frame: label map} of a SAM 3 label stack (labels_NNN.npz, frames x H x W), read as a stream so that only the
    wanted frames are kept in memory."""
    wanted, out = set(frames), {}
    with zipfile.ZipFile(path) as archive, archive.open("labels.npy") as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                                 else np.lib.format.read_array_header_2_0)(stream)
        if fortran:
            raise ValueError(f"{path}: Fortran-ordered labels")
        size = int(np.prod(shape[1:])) * dtype.itemsize
        for f in range(shape[0]):
            if f > max(wanted):
                break
            chunk = stream.read(size)
            if f in wanted:
                out[f] = np.frombuffer(chunk, dtype).reshape(shape[1:])
    return out


def prepare(workspace, counts):
    """Read the observations for checks on these frame counts ahead of the first build (long videos take minutes)."""
    W, H = workspace.video["canonical"]["width"], workspace.video["canonical"]["height"]
    _, us, vs = grid_of(W, H)
    for count in counts:
        _observations(workspace, sample_frames(workspace, count), us, vs)


def _observations(workspace, frames, us, vs):
    with _LOCK:
        return _read_observations(workspace, frames, us, vs)


def _read_observations(workspace, frames, us, vs):
    """Observed instances at the sampled frames on the grid: {name: str}, {frame: {id: bool grid}}, {frame: depth
    grid or None}, and the mode ('fused' or 'raw'). Cached per run and inputs."""
    evidence = workspace.evidence
    raw = evidence.name == "raw"
    key = hashlib.sha256(json.dumps([str(evidence), frames, us.tolist(), vs.tolist()]).encode()).hexdigest()[:16]
    cache = workspace.run / "cache" / "video_check" / f"observed_{key}.npz"
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        return z["names"].item(), z["masks"].item(), z["depth"].item(), str(z["mode"])
    W, H = workspace.video["canonical"]["width"], workspace.video["canonical"]["height"]
    rows, cols = np.clip(vs.astype(int), 0, H - 1), np.clip(us.astype(int), 0, W - 1)
    if raw:
        tracks = contract.read_json(evidence / "tracks.json")["tracks"]
        groups = {t["track"]: [(t["labels"], t["value"])] for t in tracks}
        names = {t["track"]: t["name"] for t in tracks}
        root = evidence
    else:
        objects = contract.read_json(evidence / "objects.json")
        tracks_dir = workspace.perception.parts.get("tracks") or next(      # a given perception's masks
            (evidence / n for n in ("track_masks", "gt_tracks") if (evidence / n).is_dir()), None)
        table = {t["track_id"]: t for t in contract.read_json(tracks_dir / "tracks.json")["tracks"]}
        groups, names = {}, {}
        for oid, o in objects.items():
            members = [oid, *[str(m) for m in (o.get("evidence") or {}).get("merged_track_ids") or []]]
            groups[oid] = [(f"labels_{table[m]['prompt_index']:03d}.npz", table[m]["label_value"]) for m in members
                           if m in table]
            names[oid] = o.get("category", "")
        root = tracks_dir
    masks = {f: {} for f in frames}
    by_file = {}
    for oid, members in groups.items():
        for file, value in members:
            by_file.setdefault(file, []).append((oid, value))
    for file, users in by_file.items():
        labels = _label_frames(root / file, frames)
        for f, grid in labels.items():
            sampled = grid[np.ix_(rows, cols)]
            for oid, value in users:
                hit = sampled == value
                if hit.any():
                    masks[f][oid] = masks[f][oid] | hit if oid in masks[f] else hit
    depth = _observed_depth(workspace, frames, rows, cols, raw)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, names=np.array(names, dtype=object), masks=np.array(masks, dtype=object),
                        depth=np.array(depth, dtype=object), mode="raw" if raw else "fused")
    return names, masks, depth, "raw" if raw else "fused"


def _observed_depth(workspace, frames, rows, cols, raw):
    """{frame: measured camera depth on the grid (0 = none)} where the perception has depth for that frame."""
    import cv2
    W, H = workspace.video["canonical"]["width"], workspace.video["canonical"]["height"]
    if raw:
        z = np.load(workspace.evidence / "depth.npz")
        index = {int(f): i for i, f in enumerate(z["frames"])}
        return {f: z["depth"][index[f]][np.ix_(rows, cols)].astype(np.float32) if f in index else None for f in frames}
    from . import fusion
    geometry = workspace.perception.work / "geometry" / "geometry.npz"
    if not geometry.is_file() or geometry.stat().st_size > 1.5e9:      # a thousand-frame video: masks only
        return {f: None for f in frames}
    cfg = {**fusion.DEFAULTS, **workspace.perception.config.get("fuse", {})}
    rec = fusion.reconstruction(workspace.perception.work / "geometry", workspace.decode_dir, cfg,
                                np.random.default_rng(cfg["seed"]))
    index = {int(f): i for i, f in enumerate(rec.keyframes)}
    out = {}
    for f in frames:
        if f not in index:
            out[f] = None
            continue
        d = np.where(rec.valid[index[f]], rec.depth[index[f]], 0).astype(np.float32)
        out[f] = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)[np.ix_(rows, cols)]
    return out


def _distinct(masks):
    """One frame's observed masks with the near-identical ones merged (IoU above 0.5: one object under two names or
    two tracks), the larger keeping its id; the others stay as they are."""
    kept = {}
    for oid in sorted(masks, key=lambda k: -int(masks[k].sum())):
        m = masks[oid]
        same = next((k for k, x in kept.items() if (x & m).sum() > 0.5 * (x | m).sum()), None)
        if same is None:
            kept[oid] = m
        else:
            kept[same] = kept[same] | m
    return kept


def _match(observed, rendered, by_id):
    """{observed id: scene object} for one frame: the same id when the scene registered the evidence object (fused
    evidence), then one to one by the highest silhouette IoU above 0.05 among the rest."""
    targets = {oid: oid for oid in observed if oid in by_id}
    left_obs = [o for o in observed if o not in targets]
    left_scene = [n for n in rendered if n not in targets.values()]
    pairs = []
    for o in left_obs:
        for n in left_scene:
            inter = (observed[o] & rendered[n]).sum()
            if inter:
                pairs.append((inter / (observed[o] | rendered[n]).sum(), o, n))
    used_o, used_n = set(), set()
    for iou, o, n in sorted(pairs, reverse=True):
        if iou > 0.05 and o not in used_o and n not in used_n:
            targets[o] = n
            used_o.add(o)
            used_n.add(n)
    return targets


def _scene_view(workspace, rid, frames, stride):
    """The ids npz of a built revision at the sampled frames (scene frame = canonical frame + 1)."""
    directory = workspace.run / "revisions" / rid
    out = directory / f"video_check_s{stride}_{hashlib.sha256(json.dumps(frames).encode()).hexdigest()[:8]}.npz"
    if not out.is_file():
        result = workspace.sandbox.blender(["ids", "--frames", ",".join(str(f + 1) for f in frames), "--stride",
                                            str(stride), "--out", f"/out/{out.name}"], directory, blend="scene.blend",
                                           timeout=1800)
        if result.get("status") != "ok":
            raise RuntimeError("scene view failed: " + str(result.get("stderr_tail", result))[-800:])
    return np.load(out)


def check(workspace, rid, count=12, cards=0):
    """The video consistency report of revision rid on `count` sampled frames (and up to `cards` evidence cards)."""
    frames = sample_frames(workspace, count)
    W, H = workspace.video["canonical"]["width"], workspace.video["canonical"]["height"]
    stride, us, vs = grid_of(W, H)
    names, observed, measured, mode = _observations(workspace, frames, us, vs)
    view = _scene_view(workspace, rid, frames, stride)
    scene_names = [str(n) for n in view["names"]]
    inventory = contract.read_json(workspace.run / "revisions" / rid / "inventory.json")["semantic_objects"]
    categories = {str(o["instance_id"]): o["category"] for o in inventory}
    cells = len(us) * len(vs)
    rows = {n: {"ious": [], "weights": [], "depth": [], "frames": 0, "seen": False} for n in scene_names}
    obs_rows = {}
    explained = total = 0.0
    best_frame = {}
    for k, f in enumerate(frames):
        ids, depth = view["ids"][k], view["depth"][k]
        rendered = {scene_names[i]: ids == i for i in np.unique(ids) if i >= 0}
        for n in rendered:
            rows[n]["frames"] += 1
        meas = measured.get(f)
        scored = _distinct({oid: m for oid, m in observed[f].items() if MIN_CELLS <= m.sum() <= MAX_FRAME_SHARE * cells})
        targets = _match(scored, rendered, rows if mode == "fused" else {})
        paired = set()
        for oid, mask in scored.items():
            area = int(mask.sum())
            target = targets.get(oid)
            r = rendered.get(target, np.zeros_like(mask)) if target else np.zeros_like(mask)
            iou = float((r & mask).sum() / max((r | mask).sum(), 1))
            total += area
            explained += area * iou
            o = obs_rows.setdefault(oid, {"name": names.get(oid, ""), "area": 0, "best_iou": 0.0, "frames": 0,
                                          "largest": (0, None)})
            o["area"] += area
            o["frames"] += 1
            o["best_iou"] = max(o["best_iou"], iou)
            o["largest"] = max(o["largest"], (area, f), key=lambda x: x[0])
            if target is None:
                continue
            paired.add(target)
            row = rows[target]
            row["ious"].append(iou)
            row["weights"].append(area)
            row["seen"] = True
            if meas is not None:
                both = r & mask & (meas > 0) & (depth > 0)
                if both.sum() >= 3:
                    row["depth"].append(float(np.median(depth[both] - meas[both]) / max(np.median(meas[both]), 1e-6)))
            if area > best_frame.get(target, (0, None, None))[0]:
                best_frame[target] = (area, f, oid)
        for n, r in rendered.items():
            if n not in paired and observed[f] and any((r & m).any() for m in observed[f].values()):
                rows[n]["seen"] = True
    objects = []
    for n, row in rows.items():
        if not row["frames"] and not row["ious"]:
            continue
        w = np.array(row["weights"], float)
        iou = float(np.average(row["ious"], weights=w)) if len(w) else 0.0
        objects.append({"id": n, "category": categories.get(n, ""), "iou": round(iou, 3),
                        "depth_error": round(float(np.median(row["depth"])), 3) if row["depth"] else None,
                        "observed_area": int(w.sum()), "frames_visible": row["frames"],
                        "seen_in_video": row["seen"], "impact": round(float(w.sum()) * (1 - iou), 1)})
    objects.sort(key=lambda o: -o["impact"])
    missing = sorted(({"observed": oid, "name": o["name"], "area": o["area"], "frames": o["frames"],
                       "best_iou": round(o["best_iou"], 3), "frame": o["largest"][1]} for oid, o in obs_rows.items()
                      if o["best_iou"] < 0.1 and o["frames"] >= max(1, len(frames) // 4)), key=lambda m: -m["area"])
    unseen = [o for o in objects if not o["seen_in_video"] and o["frames_visible"]]
    report = {"revision": rid, "mode": mode, "frames": frames, "stride": stride,
              "score": round(explained / total, 3) if total else None,
              "objects": objects, "missing": missing[:20], "unseen": [o["id"] for o in unseen][:20],
              "depth_checked": any(v is not None for v in measured.values())}
    if cards:
        report["cards"] = _cards(workspace, rid, report, view, frames, observed, scene_names, stride, best_frame, cards)
    out = workspace.run / "revisions" / rid / f"video_check_{len(frames)}.json"
    contract.write_json(out, report)
    return report


def _cards(workspace, rid, report, view, frames, observed, scene_names, stride, best_frame, count):
    """Crops of the source frames for the worst objects: observed mask in green, built silhouette in magenta."""
    import cv2
    out_dir = workspace.run / "revisions" / rid / "video_check_cards"
    out_dir.mkdir(exist_ok=True)
    W, H = workspace.video["canonical"]["width"], workspace.video["canonical"]["height"]
    paths = []
    for o in report["objects"][:count]:
        if o["id"] not in best_frame:
            continue
        _, f, oid = best_frame[o["id"]]
        k = frames.index(f)
        image = workspace.frame(f).copy() if hasattr(workspace, "frame") else cv2.imread(
            str(workspace.decode_dir / f"frames/{f:06d}.jpg"))
        up = lambda m: cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        obs = up(observed[f][oid])
        built = up(view["ids"][k] == scene_names.index(o["id"]))
        for mask, colour in ((obs, (60, 220, 60)), (built, (230, 60, 230))):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, colour, 2)
        ys, xs = np.nonzero(obs | built)
        pad = max(20, int(0.25 * max(np.ptp(xs), np.ptp(ys))))
        x0, y0, x1, y1 = max(0, xs.min() - pad), max(0, ys.min() - pad), min(W, xs.max() + pad), min(H, ys.max() + pad)
        crop = image[y0:y1, x0:x1]
        scale = 360 / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
        label = f"{o['id']} {o['category']} IoU {o['iou']:.2f}" + (
            f" depth {o['depth_error']:+.0%}" if o["depth_error"] is not None else "") + f" frame {f}"
        bar = np.full((24, crop.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, label[:70], (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        path = out_dir / f"{o['id']}.jpg"
        cv2.imwrite(str(path), np.vstack([bar, crop]))
        paths.append({"id": o["id"], "frame": f, "path": str(path)})
    for m in report["missing"][:max(1, count // 2)]:        # seen, not built: the observed mask alone
        f = m["frame"]
        if f is None or m["observed"] not in observed[f]:
            continue
        image = workspace.frame(f).copy()
        obs = cv2.resize(observed[f][m["observed"]].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(obs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (60, 220, 60), 2)
        ys, xs = np.nonzero(obs)
        pad = max(20, int(0.25 * max(np.ptp(xs), np.ptp(ys))))
        crop = image[max(0, ys.min() - pad):min(H, ys.max() + pad), max(0, xs.min() - pad):min(W, xs.max() + pad)]
        scale = 360 / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
        bar = np.full((24, crop.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"seen, not built: {m['observed']} {m['name']} frame {f}"[:70], (4, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        path = out_dir / f"seen_{m['observed']}.jpg"
        cv2.imwrite(str(path), np.vstack([bar, crop]))
        paths.append({"id": f"seen {m['observed']}", "frame": f, "path": str(path)})
    return paths


def summary(report, worst=8):
    """What a build result carries: the score, the worst objects, what is seen but not built and built but unseen."""
    return {"score": report["score"], "frames": report["frames"],
            "meaning": ("score: share of the observed object pixels the right scene object covers, seen from the "
                        "video's camera on these frames (1 = every observed object's silhouette matched). iou: an "
                        "object's silhouette against its observed mask; depth_error: built surface minus measured "
                        "depth, relative (+ = too far, - = too near)"),
            "worst": [{k: o[k] for k in ("id", "category", "iou", "depth_error", "frames_visible")}
                      for o in report["objects"][:worst]],
            "seen_not_built": report["missing"][:6], "built_not_seen": report["unseen"][:6]}
