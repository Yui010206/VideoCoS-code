"""3D-first object layer for long videos (fused evidence), 2026-10-30.

Ground-truth perception (ScanNet) segments the room once in 3D and projects each instance into every frame, so one
physical object is one instance in every frame. Ours tracks objects in 2D and lifts each track on its own, so a track
that breaks becomes a duplicate and a part becomes its own object. These tools rebuild the object layer from the same
SAM 3 tracks, in 3D first. Each tool can be run and checked on its own:

    lift            a track's masks x depth -> world points (on up to `frames` keyframes, eroded masks)
    candidate_pairs pairs of tracks whose lifted points come near each other
    consensus       how much of one track's points lands inside the other's masks, on the other's frames
    distinct        seen as two things at once: one frame where both masks are there and barely overlap
    score_pairs     consensus both ways for every candidate pair (cached in <out>/lifted_scores.pkl)
    same_object     the merge rule on a scored pair: mutual consensus, or containment with some support back
    compatible      (names_rule, off by default: on scene0645 it lost more good merges than bad ones) containment
                    alone merges only one kind of thing (SYNONYMS) or a part and its whole (PARTS)
    cluster         tracks -> instances: merge one-object pairs, strongest first, not across a distinct pair
    instance_records an instance's tracks, points, gravity-aligned box and name (by its tracks' detections)
    instance_views  the k frames that show an instance best (largest masks, spread over time)
    check_instances a local vision-language model looks at those frames (instance outlined in red) and says what
                    it is: a whole object, a part of one (and of what), room structure, or nothing (vlm_check_worker.py)
    check_tracks    the same check on every track before clustering, without the detector's name; structure and
                    nothing are left out (bare walls that open detection named after a neighbour)
    apply_checks    structure and nothing dropped, parts merged into a nearby whole of that name, names replaced
    merge_revisits  one object left as several instances seen at different times: one kind of thing, each centre in
                    the other's box, never seen apart (seen_apart) on a frame holding both
    to_evidence     the instances as a given-evidence directory (fused format) for run_codify, masks = the tracks'
    evaluate        instances against a ground-truth evidence (oracle_scannet.py): found / split / missed / extra

    python -m codify.instances <perception run> [--gt <oracle evidence>] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from . import contract
from .objects3d import FusedLayers, _camera

KERNEL = np.ones((5, 5), np.uint8)


class TrackLayers(FusedLayers):
    """FusedLayers with every SAM 3 track as its own object: masks, detections, depth and cameras per track."""

    def __init__(self, run, cache=None):
        run = Path(run)
        evidence = sorted((run / "evidence").glob("v0[0-9][0-9]"))[-1]
        manifest = contract.read_json(evidence / "manifest.json")
        tracks_dir = run / manifest["sources"]["artifacts"]["tracks_dir"]
        ws = SimpleNamespace(evidence=evidence, run=Path(cache) if cache else run, decode_dir=run / "work" / "decode",
                             video=contract.read_json(run / "work" / "decode" / "video_info.json"),
                             perception=SimpleNamespace(parts={"tracks": tracks_dir}, work=run / "work",
                                                        config={}))
        super().__init__(ws)
        tracks = {t["track_id"]: t for t in contract.read_json(tracks_dir / "tracks.json")["tracks"]}
        self.objects = {tid: {"category": t["prompt"]} for tid, t in tracks.items()}
        self.members = {tid: [(tracks_dir / f"labels_{t['prompt_index']:03d}.npz", t["label_value"], t)]
                        for tid, t in tracks.items()}


def _views(layers, tid, frames):
    """Up to `frames` keyframes with depth where the track is detected, largest masks first, spread over time."""
    det, depth_frames = layers.detections(tid), layers.depth_frames()
    order = sorted((f for f in det if f in depth_frames), key=lambda f: -max(a for a, _ in det[f]))
    picks = []
    for f in order:
        if all(abs(f - g) > 15 for g in picks):
            picks.append(f)
        if len(picks) == frames:
            break
    return sorted(picks)


def lift(layers, tids, frames=8, per_view=300):
    """{track: {"frames": [...], "points": (N, 3) world points, "counts": points per frame}}: each chosen frame's
    eroded mask on valid depth,
    depths within the mask's 10-90th percentile (edges and what is seen behind are left out). A track that switched
    to another object keeps only the views at its main location: the per-view centres are grouped (within half the
    median view size, at least 0.3 m) and the largest group stays, as in fusion's static record."""
    from .fusion import view_clusters
    chosen = {t: _views(layers, t, frames) for t in tids}
    layers.prefetch([(t, f) for t, fs in chosen.items() for f in fs])
    H, W = layers.size
    out = {}
    for t, fs in chosen.items():
        pts, used = [], []
        for f in fs:
            m, d = layers.mask(t, f), layers.depth(f)
            if m is None or d is None:
                continue
            m = cv2.erode(m.astype(np.uint8), KERNEL).astype(bool) & (d > 0)
            ys, xs = np.nonzero(m)
            if len(xs) < 30:
                continue
            z = d[ys, xs]
            lo, hi = np.percentile(z, [10, 90])
            keep = (z >= lo) & (z <= hi)
            ys, xs, z = ys[keep], xs[keep], z[keep]
            pick = np.random.default_rng(f).choice(len(xs), min(per_view, len(xs)), replace=False)
            K, R, C = _camera(layers.camera(f))
            row = layers.camera(f)["projection"]
            sx, sy = W / row["width"], H / row["height"]
            cam = np.stack([(xs[pick] / sx - K[0, 2]) * z[pick] / K[0, 0], (ys[pick] / sy - K[1, 2]) * z[pick] / K[1, 1], z[pick]], 1)
            pts.append(cam @ R.T + C)
            used.append(f)
        if len(pts) > 1:
            cents = np.array([np.median(v, 0) for v in pts])
            size = float(np.median([np.ptp(np.percentile(v, [5, 95], 0), 0).max() for v in pts]))
            keep = sorted(view_clusters(cents, max(0.3, 0.5 * size))[0])
            pts, used = [pts[i] for i in keep], [used[i] for i in keep]
        if pts:
            out[t] = {"frames": used, "points": np.concatenate(pts), "counts": [len(v) for v in pts]}
    return out


def _project_share(layers, points, mask_track, f, min_points=30):
    """Share of world points landing inside mask_track's mask on frame f, among the points frame f could see (in
    front of the depth it measured there); None when fewer than min_points are seen."""
    H, W = layers.size
    m, d = layers.mask(mask_track, f), layers.depth(f)
    if m is None or d is None:
        return None
    K, R, C = _camera(layers.camera(f))
    row = layers.camera(f)["projection"]
    sx, sy = W / row["width"], H / row["height"]
    cam = (points - C) @ R
    cam = cam[cam[:, 2] > 0.05]
    u = np.round((cam[:, 0] / cam[:, 2] * K[0, 0] + K[0, 2]) * sx).astype(int)
    v = np.round((cam[:, 1] / cam[:, 2] * K[1, 1] + K[1, 2]) * sy).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[ok], v[ok], cam[ok, 2]
    seen = (d[v, u] == 0) | (z <= d[v, u] + 0.1 + 0.05 * z)
    return float(m[v[seen], u[seen]].mean()) if seen.sum() >= min_points else None


def self_consensus(layers, lifted, t):
    """How well a track agrees with itself: the points lifted from each of its views, projected into its other
    views, inside its own mask there (median over view pairs; None with fewer than two views). A track that drifted
    from one wall to the next, or from an object onto its neighbour, scores low."""
    v = lifted[t]
    if len(v.get("counts", [])) < 2:
        return None
    ends = np.cumsum([0] + v["counts"])
    shares = [_project_share(layers, v["points"][ends[i]:ends[i + 1]], t, g)
              for i in range(len(v["frames"])) for g in v["frames"] if g != v["frames"][i]]
    shares = [x for x in shares if x is not None]
    return float(np.median(shares)) if shares else None


def candidate_pairs(lifted, margin=0.15):
    """Pairs of tracks whose points' 5-95th percentile boxes, grown by margin (m), overlap."""
    ids = sorted(lifted)
    boxes = np.array([np.concatenate([np.percentile(lifted[t]["points"], 5, 0) - margin,
                                      np.percentile(lifted[t]["points"], 95, 0) + margin]) for t in ids])
    pairs = []
    for i in range(len(ids)):
        lo, hi = boxes[i, :3], boxes[i, 3:]
        hit = np.all((boxes[i + 1:, :3] <= hi) & (boxes[i + 1:, 3:] >= lo), axis=1)
        pairs += [(ids[i], ids[i + 1 + j]) for j in np.flatnonzero(hit)]
    return pairs


def consensus(layers, lifted, a, b, min_points=30):
    """Mean share of a's points that land inside b's masks on b's frames (only points b's frame could see: in
    front of the depth it measured there), and on how many frames; (None, 0) when no frame of b sees enough."""
    H, W = layers.size
    shares = []
    for f in lifted[b]["frames"]:
        m, d = layers.mask(b, f), layers.depth(f)
        if m is None or d is None:
            continue
        K, R, C = _camera(layers.camera(f))
        row = layers.camera(f)["projection"]
        sx, sy = W / row["width"], H / row["height"]
        cam = (lifted[a]["points"] - C) @ R
        cam = cam[cam[:, 2] > 0.05]
        u = np.round((cam[:, 0] / cam[:, 2] * K[0, 0] + K[0, 2]) * sx).astype(int)
        v = np.round((cam[:, 1] / cam[:, 2] * K[1, 1] + K[1, 2]) * sy).astype(int)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, z = u[ok], v[ok], cam[ok, 2]
        seen = (d[v, u] == 0) | (z <= d[v, u] + 0.1 + 0.05 * z)
        if seen.sum() >= min_points:
            shares.append(float(m[v[seen], u[seen]].mean()))
    return (float(np.mean(shares)) if shares else None), len(shares)


def distinct(layers, a, b, max_iou=0.1, max_inside=0.5):
    """Seen as two things at once: on most frames both tracks are detected on (at least two), their boxes barely
    overlap and neither lies mostly inside the other. Most, not one: a track that once switched to a neighbour is
    not two things."""
    da, db = layers.detections(a), layers.detections(b)
    common = sorted(set(da) & set(db))
    if len(common) < 2:
        return False
    apart = 0
    for f in common:
        (_, x), (_, y) = max(da[f], key=lambda d: d[0]), max(db[f], key=lambda d: d[0])
        if not x or not y:
            continue
        ix = max(0, min(x[2], y[2]) - max(x[0], y[0])) * max(0, min(x[3], y[3]) - max(x[1], y[1]))
        area = lambda r: max(1, (r[2] - r[0]) * (r[3] - r[1]))
        apart += ix / (area(x) + area(y) - ix) < max_iou and ix / min(area(x), area(y)) < max_inside
    return apart >= len(common) / 2


def score_pairs(layers, lifted, pairs):
    """Consensus both ways for every candidate pair: [{"pair": [a, b], "ab", "ba", "frames": [n_ab, n_ba]}]."""
    scored = []
    for a, b in pairs:
        ab, nab = consensus(layers, lifted, a, b)
        ba, nba = consensus(layers, lifted, b, a)
        scored.append({"pair": [a, b], "ab": ab, "ba": ba, "frames": [nab, nba]})
    return scored


# Names of one kind of thing, and parts with the wholes they belong to (general knowledge, not one room's labels).
# A pillow is not a part of a bed or a sofa: it is counted on its own.
SYNONYMS = [{"sofa", "couch"}, {"trash can", "trash bin", "garbage can", "wastebasket", "bin", "recycling bin"},
            {"tv", "television"}, {"picture", "picture frame", "painting", "poster", "artwork"}, {"rug", "carpet", "mat"},
            {"cabinet", "cupboard"}, {"nightstand", "bedside table", "end table", "side table"}, {"phone", "telephone"},
            {"heater", "radiator"}, {"armchair", "chair"}, {"curtain", "drape", "drapes"}]
PARTS = {"cushion": {"sofa", "couch", "armchair", "chair", "ottoman", "bench"}, "seat cushion": {"sofa", "couch", "armchair", "chair"},
         "lampshade": {"lamp"}, "lamp shade": {"lamp"}, "light fixture": {"lamp"}, "bed skirt": {"bed"}, "headboard": {"bed"},
         "footboard": {"bed"}, "mattress": {"bed"}, "bedspread": {"bed"}, "drawer": {"cabinet", "dresser", "nightstand", "desk"},
         "door knob": {"door"}, "doorknob": {"door"}, "handle": {"door", "cabinet", "drawer"}, "faucet": {"sink"},
         "toilet seat": {"toilet"}, "toilet lid": {"toilet"}, "curtain rod": {"curtain"}, "valance": {"curtain"},
         "curtain valance": {"curtain"}, "chair arm": {"chair", "armchair"}, "arm": {"chair", "armchair", "sofa", "couch"},
         "table top": {"table", "desk"}, "tabletop": {"table", "desk"}, "shelf": {"bookshelf", "cabinet"}}


def _kind(n):
    """The names one kind of thing goes by (SYNONYMS), or just the head noun."""
    from .fusion import head_noun
    n = n.strip().lower()
    return next((frozenset(g) for g in SYNONYMS if n in g or head_noun(n) in g), frozenset({head_noun(n)}))


def same_kind(name, names):
    """The name is one kind of thing with at least one of the names."""
    return any(_kind(name) & _kind(n) for n in names)


def compatible(names_a, names_b):
    """One kind of thing, or a part and its whole, among two sets of names."""
    kind = _kind
    for x in names_a:
        for y in names_b:
            x, y = x.strip().lower(), y.strip().lower()
            if kind(x) & kind(y) or y in PARTS.get(x, ()) or x in PARTS.get(y, ()) or \
                    any(h in PARTS.get(x, ()) for h in kind(y)) or any(h in PARTS.get(y, ()) for h in kind(x)):
                return True
    return False


def same_object(s, mutual=0.5, contained=0.8, back=0.2, min_frames=1, sure=0.8):
    """One object, by consensus: both ways at least `mutual`, or one way at least `contained` with the other way at
    least `back` (two tracks of one object whose masks cover it to different extents; a thing lying on another one
    gives almost nothing back). min_frames: a consensus measured on fewer frames than this counts only when both
    ways are `sure`, and a containment only when the container's side was measured on that many frames (one frame of
    a loose mask said "inside" for a laptop, a door and a curtain on 27dd4da69e, and chained them into one object)."""
    if s["ab"] is None or s["ba"] is None:
        return False
    n_ab, n_ba = s.get("frames", [min_frames, min_frames])
    lo, hi = sorted((s["ab"], s["ba"]))
    if min(n_ab, n_ba) < min_frames and lo < sure:
        if lo >= mutual:
            return False
        inside_frames = n_ab if s["ab"] >= s["ba"] else n_ba     # the container's frames
        return hi >= contained and lo >= back and inside_frames >= min_frames
    return lo >= mutual or (hi >= contained and lo >= back)


def _extent(points):
    """Diagonal of the points' 5-95th percentile box (m)."""
    return float(np.linalg.norm(np.percentile(points, 95, 0) - np.percentile(points, 5, 0)))


def cluster(layers, lifted, scored, sure=0.6, grow=1.5, slack=0.2, names_rule=False, log=print, **rule):
    """Tracks -> instances (lists of track ids): pairs that are one object (same_object) are merged, strongest
    first. A merge is skipped when the two groups hold a distinct pair, unless this pair's consensus is at least
    `sure` both ways (boxes apart on some frames are then a track that switched objects), and when the merged group
    would be larger than `grow` x the larger group + `slack` m (a chain of containments across neighbours)."""
    pairs = [tuple(s["pair"]) for s in scored]
    group = {t: {t} for t in lifted}
    names = {t: {layers.objects[t]["category"]} for t in lifted}
    rng = np.random.default_rng(0)
    sample = lambda P: P[rng.choice(len(P), min(2000, len(P)), replace=False)]
    points = {t: sample(lifted[t]["points"]) for t in lifted}
    merged, too_big = 0, 0
    for s in sorted((s for s in scored if same_object(s, **rule)), key=lambda s: -(s["ab"] + s["ba"])):
        a, b = s["pair"]
        ga, gb = group[a], group[b]
        if ga is gb:
            continue
        if min(s["ab"], s["ba"]) < sure and any(distinct(layers, x, y) for x in ga for y in gb if (x, y) in pair_set(pairs)):
            s["vetoed"] = True
            continue
        if names_rule and min(s["ab"], s["ba"]) < rule.get("mutual", 0.5) and not compatible(names[min(ga)], names[min(gb)]):
            s["names_apart"] = True       # containment alone joins one kind of thing, or a part and its whole
            continue
        pa, pb = points[min(ga)], points[min(gb)]
        joined = sample(np.concatenate([pa, pb]))
        if _extent(joined) > grow * max(_extent(pa), _extent(pb)) + slack:
            s["too_big"] = True
            too_big += 1
            continue
        union = ga | gb
        for t in union:
            group[t] = union
        points[min(union)] = joined
        names[min(union)] = names[min(ga)] | names[min(gb)]
        merged += 1
    instances = {frozenset(g) for g in group.values()}
    log(f"cluster: {len(lifted)} tracks, {len(scored)} scored pairs, {merged} merges ({too_big} skipped as too big) "
        f"-> {len(instances)} instances")
    return sorted((sorted(g, key=int) for g in instances), key=lambda m: int(m[0]))   # the same order in every run


_PAIRS = {}


def pair_set(pairs):
    key = id(pairs)
    if key not in _PAIRS:
        _PAIRS[key] = {p for a, b in pairs for p in ((a, b), (b, a))}
    return _PAIRS[key]


def instance_records(layers, lifted, instances):
    """Per instance: its tracks, a name (the prompt with the most detections among them), points, a gravity-aligned
    box (minimum-area footprint, z from the 1-99th percentiles) and how many keyframes see it."""
    out = []
    for members in instances:
        votes = {}
        for t in members:
            votes[layers.objects[t]["category"]] = votes.get(layers.objects[t]["category"], 0) + len(layers.detections(t))
        name = max(votes, key=votes.get)
        P = np.concatenate([lifted[t]["points"] for t in members])
        out.append({"tracks": members, "name": name, "names": votes, "points": P, "box": _fit_box(P),
                    "frames": sorted({f for t in members for f in layers.detections(t)})})
    return out


def _fit_box(P):
    """Gravity-aligned box of world points: minimum-area footprint, z from the 1-99th percentiles."""
    (cx, cy), (w, h), angle = cv2.minAreaRect(P[:, :2].astype(np.float32))
    z0, z1 = np.percentile(P[:, 2], [1, 99])
    return {"center": [float(cx), float(cy), float((z0 + z1) / 2)], "size": [float(w), float(h), float(z1 - z0)],
            "yaw_deg": float(angle)}


def view_frames(layers, record, k=3, gap=30):
    """The k frames where the instance's tracks cover the most pixels (by their detections), at least `gap` apart."""
    area = {}
    for t in record["tracks"]:
        for f, dets in layers.detections(t).items():
            area[f] = area.get(f, 0) + max(a for a, _ in dets)
    picks = []
    for f in sorted(area, key=lambda f: -area[f]):
        if all(abs(f - g) >= gap for g in picks):
            picks.append(f)
        if len(picks) == k:
            break
    return picks


def instance_views(layers, record, k=3, gap=30):
    """[(frame, union mask)] of the k frames where the instance's tracks cover the most pixels, at least `gap`
    frames apart."""
    picks = view_frames(layers, record, k, gap)
    layers.prefetch([(t, f) for f in picks for t in record["tracks"] if f in layers.detections(t)])
    out = []
    for f in sorted(picks):
        m = None
        for t in record["tracks"]:
            mt = layers.mask(t, f) if f in layers.detections(t) else None
            if mt is not None:
                m = mt if m is None else (m | mt)
        if m is not None and m.any():
            out.append((f, m))
    return out


def render_views(layers, views, folder, tag, style="outline"):
    """Each view as the full frame with the instance's outline drawn in red; returns the image paths. style "tint":
    also the outside dimmed to half, so a large region reads as the region and not as the object inside it."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for f, m in views:
        image = cv2.imread(str(layers.image(f)))
        if style == "tint":
            image[~m] = (image[~m] * 0.45).astype(np.uint8)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
        path = folder / f"{tag}_{f:06d}.jpg"
        cv2.imwrite(str(path), image)
        paths.append(str(path))
    return paths


def check_instances(layers, records, folder, python, weights, gpu, hints=True, style="outline", log=print):
    """Ask the local vision-language model about every instance (vlm_check_worker.py, in its own environment on
    `gpu`); each record gets "check": {name, kind, part_of, raw}. hints: tell the model the detectors' names."""
    folder = Path(folder)
    # every record's view masks in one pass over each label file: a long video's label stack is read as a stream,
    # so one record at a time read it once per record (27dd4da69e: hours)
    layers.prefetch([(t, f) for r in records for f in view_frames(layers, r) for t in r["tracks"]
                     if f in layers.detections(t)])
    items = []
    for i, r in enumerate(records):
        views = instance_views(layers, r)
        if views:
            items.append({"id": i, "images": render_views(layers, views, folder / "views", f"i{i:04d}", style),
                          "hints": list(r["names"]) if hints else []})
    contract.write_json(folder / "job.json", {"weights": str(weights), "items": items})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with open(folder / "worker.log", "w") as stream:
        subprocess.run([python, str(Path(__file__).with_name("vlm_check_worker.py")), str(folder / "job.json"),
                        str(folder / "answers.json")], env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    answers = contract.read_json(folder / "answers.json")
    for i, r in enumerate(records):
        r["check"] = answers["answers"].get(str(i))
    log(f"check: {len(items)} instances in {answers['seconds']} s")
    return answers


def check_tracks(layers, lifted, folder, python, weights, gpu, style="tint", log=print):
    """The same check on every track before clustering, without the detector's name (open detection names a bare
    stretch of wall after whatever stands next to it: a "headboard", a "backpack", a "lamp"; 2026-10-30, scene0645).
    Rendered with the outside dimmed ("tint"): on scene0645 it judged as well as the outline alone overall and got
    two more of the six largest wall tracks right. Returns {track: check}; tracks it calls structure or nothing are
    left out of the clustering."""
    records = [{"tracks": [t], "names": {layers.objects[t]["category"]: 1}} for t in lifted]
    check_instances(layers, records, folder, python, weights, gpu, hints=False, style=style, log=log)
    out = {r["tracks"][0]: r.get("check") for r in records}
    log(f"check_tracks: {sum((c or {}).get('kind') in ('structure', 'none') for c in out.values())} of {len(out)} tracks "
        f"are structure or nothing")
    return out


def _inside_box(points, record, margin=0.1):
    """Share of the points inside the record's gravity-aligned box grown by margin."""
    b = record["box"]
    c, size, yaw = np.asarray(b["center"]), np.asarray(b["size"]) / 2 + margin, np.radians(b["yaw_deg"])
    d = points - c
    x = d[:, 0] * np.cos(yaw) + d[:, 1] * np.sin(yaw)
    y = -d[:, 0] * np.sin(yaw) + d[:, 1] * np.cos(yaw)
    return float(np.mean((np.abs(x) <= size[0]) & (np.abs(y) <= size[1]) & (np.abs(d[:, 2]) <= size[2])))


def apply_checks(records, rename="agree", log=print):
    """The model's answers applied: structure and nothing dropped; a part goes into the whole it names when such a
    whole holds most of its points (else it stays, as an object of its own); objects take the model's name when one
    of the detectors' names for it is the same kind of thing (rename="agree"), or always (rename="always").
    "agree" because the 4B model renames freely in a cluttered office (ScanNet++ d755b3d9d8: a shoe and a toy became
    "radiator", a cup and a box "plant"), while on scene0645 always taking its name was right on only 2 more of 90."""
    from .fusion import head_noun
    kept = [r for r in records if not (r.get("check") or {}).get("kind") in ("structure", "none")]
    wholes = [r for r in kept if (r.get("check") or {}).get("kind") != "part"]
    merged = 0
    for r in [r for r in kept if (r.get("check") or {}).get("kind") == "part"]:
        target = head_noun(r["check"]["part_of"] or "")
        hosts = [w for w in wholes if target and head_noun((w.get("check") or {}).get("name") or w["name"]) == target
                 and _inside_box(r["points"], w) >= 0.5]
        if hosts:
            host = max(hosts, key=lambda w: _inside_box(r["points"], w))
            host["tracks"] = sorted(set(host["tracks"]) | set(r["tracks"]), key=int)
            host["points"] = np.concatenate([host["points"], r["points"]])
            host.setdefault("parts", []).append(r["check"]["name"])
            r["merged_into"] = True
            merged += 1
    out = [r for r in kept if not r.get("merged_into")]
    renamed = 0
    for r in out:
        c = r.get("check") or {}
        if c.get("name") and c["kind"] in ("object", "part") and (rename == "always" or same_kind(c["name"], r["names"])):
            renamed += c["name"] != r["name"]
            r["name"] = c["name"]
    log(f"apply_checks: {len(records)} -> {len(out)} ({len(records) - len(kept)} structure or nothing, {merged} parts "
        f"merged, {renamed} renamed)")
    return out


def _union_mask(layers, record, f):
    m = None
    for t in record["tracks"]:
        if f in layers.detections(t):
            mt = layers.mask(t, f)
            if mt is not None:
                m = mt if m is None else (m | mt)
    return m


def seen_apart(layers, a, b, apart=0.2, frames=8):
    """Seen as two things: on most of (up to `frames` of) the frames holding both instances, their masks overlap by
    less than `apart` of the smaller one. False when no frame holds both."""
    common = sorted(set(a["frames"]) & set(b["frames"]))
    if not common:
        return False
    picks = [common[round(i * (len(common) - 1) / max(frames - 1, 1))] for i in range(min(frames, len(common)))]
    votes = []
    for f in dict.fromkeys(picks):
        ma, mb = _union_mask(layers, a, f), _union_mask(layers, b, f)
        if ma is not None and mb is not None and ma.any() and mb.any():
            votes.append((ma & mb).sum() < apart * min(ma.sum(), mb.sum()))
    return bool(votes) and sum(votes) > len(votes) / 2


def merge_revisits(layers, records, margin=0.1, log=print):
    """One object left as several instances, seen at different times: its tracks never met in 3D (the camera's
    estimate drifted between visits, or a thin open object such as a chair, whose lifted points scatter onto what is
    seen through it). Instances of one kind of thing (compatible names) whose centres lie each inside the other's
    box (grown by `margin`) and that are never seen as two things at once (seen_apart) become one, closest pair
    first; chairs around one table are seen together, side by side, so they stay apart (2026-10-30: ARKitScenes
    42898849 had nine "chairs" within 0.3 m of each other at a table with five)."""
    from .fusion import center_ratio
    # grown by the margin, but by at most a quarter of the box's own footprint: a pillow lifted from one side is a
    # strip a few centimetres deep, and 10 cm around it reaches the next pillow (scene0645, pillows 47 and 48)
    grow = lambda r: min(margin, 0.25 * float(min(r["box"]["size"][:2])))
    box = lambda r: {"center": r["box"]["center"], "size": np.asarray(r["box"]["size"], float) + 2 * grow(r),
                     "yaw": np.radians(r["box"]["yaw_deg"])}
    pairs = []
    for i, a in enumerate(records):
        for j in range(i + 1, len(records)):
            b = records[j]
            if compatible({a["name"]}, {b["name"]}):
                ratio = max(center_ratio(a["box"]["center"], box(b)), center_ratio(b["box"]["center"], box(a)))
                if ratio <= 1.0:
                    pairs.append((ratio, i, j))
    common_frames = lambda a, b: sorted(set(a["frames"]) & set(b["frames"]))
    wanted = set()
    for _, i, j in pairs:               # the frames seen_apart looks at, read in one pass (see check_instances)
        common = common_frames(records[i], records[j])
        for f in [common[round(k * (len(common) - 1) / 7)] for k in range(min(8, len(common)))]:
            wanted |= {(t, f) for t in records[i]["tracks"] + records[j]["tracks"] if f in layers.detections(t)}
    layers.prefetch(sorted(wanted))
    head = list(range(len(records)))
    find = lambda i: i if head[i] == i else find(head[i])
    merged = 0
    for ratio, i, j in sorted(pairs):
        gi, gj = find(i), find(j)
        if gi == gj or seen_apart(layers, records[gi], records[gj]):
            continue
        keep, gone = sorted((gi, gj), key=lambda k: -len(records[k]["points"]))
        a, b = records[keep], records[gone]
        a["tracks"] = sorted(set(a["tracks"]) | set(b["tracks"]), key=int)
        a["points"] = np.concatenate([a["points"], b["points"]])
        a["names"] = {n: a["names"].get(n, 0) + b["names"].get(n, 0) for n in set(a["names"]) | set(b["names"])}
        a["frames"] = sorted(set(a["frames"]) | set(b["frames"]))
        a["box"] = _fit_box(a["points"])
        head[gone] = keep
        merged += 1
    out = [r for k, r in enumerate(records) if find(k) == k]
    log(f"merge_revisits: {len(records)} -> {len(out)} ({len(pairs)} candidate pairs, {merged} merges)")
    return out


def to_evidence(layers, records, source_run, out, comparison=None, name_share=0.0):
    """The instances as a given-evidence directory in the fused format, for run_codify (config evidence.given): one
    object per instance (its id = its first track's, the other tracks as merged_track_ids, so the masks stay the
    tracks' own), boxes from its points, observations per frame, comparison frames with object-id masks, colours;
    cameras, world, structure, points and the timeline from the source run's evidence; the track masks
    (track_masks/) and the depth cache (depth/) go along, since a given evidence has no perception behind it."""
    import shutil
    from scipy.spatial.transform import Rotation
    source_run, out = Path(source_run), Path(out)
    if out.exists() and not (out / "manifest.json").is_file():   # an earlier attempt that stopped half way
        shutil.rmtree(out)
    if out.exists():
        raise FileExistsError(out)
    source = sorted((source_run / "evidence").glob("v0[0-9][0-9]"))[-1]
    out.mkdir(parents=True)
    for name in ("cameras.json", "world.json", "structure.json", "points.npz", "timeline.jsonl", "video.json"):
        shutil.copyfile(source / name, out / name)
    tracks_dir = layers.ws.perception.parts["tracks"]
    shutil.copytree(tracks_dir, out / "track_masks")
    index, stack = layers.prepare_depth()
    (out / "depth").mkdir()
    contract.write_json(out / "depth" / "depth_frames.json", sorted(index, key=index.get))
    np.save(out / "depth" / "depth.npy", np.asarray(stack))
    timeline = contract.read_jsonl(source / "timeline.jsonl")
    frames = comparison or contract.comparison_frames(len(timeline), None)
    layers.prefetch([(t, f) for r in records for t in r["tracks"] for f in frames if f in layers.detections(t)])
    objects, observations, appearance = {}, [], {}
    maps = {f: np.zeros(layers.size, np.uint16) for f in frames}
    for r in records:
        oid = str(r["tracks"][0])
        b = r["box"]
        if not np.all(np.isfinite([*b["center"], *b["size"], b["yaw_deg"]])):     # degenerate points (45261575)
            print(f"left out instance {oid} ({r['name']}): its box is not finite", flush=True)
            continue
        q = Rotation.from_euler("z", np.radians(b["yaw_deg"])).as_quat()
        views = sorted({f for t in r["tracks"] for f in layers.detections(t)} & set(layers.depth_frames()))
        objects[oid] = {
            "instance_id": oid, "instance_name": f"{r['name'].replace(' ', '_')}_{oid}", "category": r["name"],
            "prototype_name": r["name"], "proxy_recipe": r["name"], "motion_type": "static",
            "canonical_pose": {"up_vector": [0, 0, 1], "front_vector": [0, -1, 0]},
            "bounds": [-b["size"][0] / 2, b["size"][0] / 2, -b["size"][1] / 2, b["size"][1] / 2, -b["size"][2] / 2, b["size"][2] / 2],
            "poses": [[0, *b["center"], float(q[3]), float(q[0]), float(q[1]), float(q[2])]],
            "evidence": {"identity_source": "sam3_tracks_clustered_in_3d (codify.instances)",
                         "geometry_source": "masked_depth_points_of_the_instance", "pose_source": "multi_view_static",
                         "rotation_status": "gravity-aligned, minimum-area footprint", "dimension_status": "visible_surface_extent",
                         "confidence": None, "evidence_level": "certain" if len(views) >= 3 else "uncertain",
                         "uncertain_because": [] if len(views) >= 3 else ["seen on fewer than three keyframes with depth"],
                         "names": sorted({n for n, c in r["names"].items() if c >= name_share * sum(r["names"].values())}
                                         | set(r.get("track_names", ()))
                                         | ({r["check"]["name"]} if (r.get("check") or {}).get("name") else set())),
                         "merged_track_ids": [str(t) for t in r["tracks"][1:]], "parts": r.get("parts", []),
                         "check": {k: v for k, v in (r.get("check") or {}).items() if k != "raw"},
                         "views_used": len(views), "points_used": int(len(r["points"])), "detected_frames": len(r["frames"])}}
        per_frame = {}
        for t in r["tracks"]:
            for f, dets in layers.detections(t).items():
                for area, box in dets:
                    if box:
                        old = per_frame.get(f)
                        per_frame[f] = (area + (old[0] if old else 0), box if not old else
                                        [min(box[0], old[1][0]), min(box[1], old[1][1]), max(box[2], old[1][2]), max(box[3], old[1][3])])
        observations += [{"frame_index": f, "timestamp_ns": timeline[f]["timestamp_ns"], "object_id": oid, "state": "detected",
                          "bbox_xyxy": box, "mask_area_px": int(area), "score": 1.0, "source": "sam3_tracks_clustered_in_3d"}
                         for f, (area, box) in sorted(per_frame.items())]
        colours = []
        for f in frames:
            for t in r["tracks"]:
                m = layers.mask(t, f) if f in layers.detections(t) else None
                if m is not None:
                    maps[f][m & (maps[f] == 0)] = int(oid)
                    colours.append(cv2.imread(str(layers.image(f)))[m][:, ::-1])
        pixels = np.concatenate(colours) if colours else np.zeros((0, 3))
        if len(pixels):                 # masks that are empty in every comparison frame give no colour (was NaN)
            appearance[oid] = {"median_srgb": (np.median(pixels, 0) / 255.0).tolist(), "sample_count": int(len(pixels)),
                               "source": "comparison frames inside the instance's masks"}
    observations.sort(key=lambda o: (o["frame_index"], int(o["object_id"])))
    (out / "frames").mkdir()
    (out / "masks").mkdir()
    for f in frames:
        shutil.copyfile(layers.image(f), out / "frames" / f"{f:06d}.jpg")
        cv2.imwrite(str(out / "masks" / f"{f:06d}.png"), maps[f])
    contract.write_json(out / "frames.json", {"comparison": frames, "masks": "masks/<frame>.png: uint16 object id per pixel (0 = none)",
                                             "frames": "frames/<frame>.jpg: the comparison frames"})
    contract.write_json(out / "objects.json", objects)
    contract.write_jsonl(out / "observations.jsonl", observations)
    contract.write_json(out / "appearance.json", appearance)
    contract.write_json(out / "unresolved.json", [])
    contract.write_json(out / "tracks.json", contract.read_json(tracks_dir / "tracks.json"))
    manifest = contract.write_manifest(out, contract.read_json(source / "manifest.json")["mode"],
                                       {"perception": str(source_run), "builder": "codify.instances.to_evidence"})
    return {"evidence": str(out), "objects": len(objects), "digest": manifest["digest"]}


def evaluate(layers, records, gt_evidence, step=10):
    """Instances against a ground-truth evidence by masks: an instance belongs to the GT object holding at least
    half of its (union of tracks') mask pixels over every `step`-th frame. Returns counts and per-GT-object lists."""
    gt_objects = contract.read_json(Path(gt_evidence) / "objects.json")
    labels = np.load(Path(gt_evidence) / "gt_tracks" / "labels_000.npz")["labels"]
    frames = range(0, len(labels), step)
    layers.prefetch([(t, f) for r in records for t in r["tracks"] for f in frames if f in layers.detections(t)])
    owner = []
    for r in records:
        acc = {}
        for f in frames:
            m = None
            for t in r["tracks"]:
                if f in layers.detections(t):
                    mt = layers.mask(t, f)
                    if mt is not None:
                        m = mt if m is None else (m | mt)
            if m is None:
                continue
            ids, counts = np.unique(labels[f][m], return_counts=True)
            for i, c in zip(ids.tolist(), counts.tolist()):
                acc[i] = acc.get(i, 0) + c
        total = sum(acc.values())
        best, n = max(((i, c) for i, c in acc.items() if i), key=lambda x: x[1], default=(0, 0))
        owner.append(str(best) if total and n / total >= 0.5 else None)
    members = {g: [k for k, w in enumerate(owner) if w == g] for g in gt_objects}
    status = {g: "found" if len(m) == 1 else "split" if m else "missed" for g, m in members.items()}
    return {"gt_objects": len(gt_objects), "instances": len(records),
            "found": sum(v == "found" for v in status.values()), "split": sum(v == "split" for v in status.values()),
            "pieces_in_splits": sum(len(members[g]) for g in gt_objects if status[g] == "split"),
            "missed": sum(v == "missed" for v in status.values()), "extra": sum(w is None for w in owner),
            "per_gt": {g: {"name": gt_objects[g]["category"], "status": status[g],
                           "ours": [records[k]["name"] for k in members[g]]} for g in gt_objects}}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", help="a perception run (evidence/<v>, work/ with the joined tracks)")
    p.add_argument("--gt", help="a ground-truth evidence to evaluate against")
    p.add_argument("--out", help="folder for instances.json and scores.json")
    p.add_argument("--same", type=float, default=0.5)
    p.add_argument("--cache", help="folder for the depth cache (default: the run)")
    p.add_argument("--check", action="store_true", help="run the vision-language check (needs --out)")
    p.add_argument("--vlm-python", help="Python of the vLLM environment (perception.py passes configs/machine.json's)")
    p.add_argument("--vlm-weights", help="the vision-language model folder (perception.py passes configs/machine.json's)")
    p.add_argument("--gpu", type=int, default=7)
    p.add_argument("--evidence", help="also write the instances as a given-evidence directory here")
    p.add_argument("--check-tracks", action="store_true",
                   help="check every track before clustering; structure and nothing are left out (needs --out)")
    p.add_argument("--instance-checks", help="reuse an earlier run's check/answers.json (same clustering only; runs "
                                             "before 2026-10-30 08:00 listed instances in a random order)")
    p.add_argument("--track-name-share", type=float, default=0.25,
                   help="a track name becomes another name of its instance when its tracks hold this share of the "
                        "instance's detections")
    p.add_argument("--name-share", type=float, default=0.0,
                   help="in the evidence, keep a detector's name for an instance only when it holds this share of the "
                        "instance's detections (a pan merged into a countertop made the countertop answer for the pan)")
    p.add_argument("--min-frames", type=int, default=1,
                   help="a consensus measured on fewer frames merges only when it is sure both ways (same_object)")
    p.add_argument("--min-views", type=int, default=0,
                   help="drop instances lifted from fewer keyframes than this (all their tracks together; fusion uses 3)")
    p.add_argument("--rename", choices=["agree", "always"], default="agree",
                   help="take the model's name when a detector's name agrees (default), or always")
    p.add_argument("--merge-revisits", action="store_true",
                   help="merge instances of one kind of thing that are one object seen at different times")
    p.add_argument("--track-checks", help="reuse the per-track checks in this JSON file (implies --check-tracks)")
    p.add_argument("--drop", help="a JSON list of track ids left out before clustering (e.g. a list made from ground "
                                  "truth, to measure what a track filter could gain)")
    a = p.parse_args()
    if (a.check or a.check_tracks) and not (a.vlm_python and a.vlm_weights):
        p.error("--check and --check-tracks need --vlm-python and --vlm-weights")
    layers = TrackLayers(a.run, a.cache)
    cache = Path(a.out) / "lifted_scores.pkl" if a.out else None
    if cache is not None and cache.is_file():
        with open(cache, "rb") as stream:
            lifted, scored = pickle.load(stream)
    else:
        lifted = lift(layers, list(layers.objects))
        scored = score_pairs(layers, lifted, candidate_pairs(lifted))
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "wb") as stream:
                pickle.dump((lifted, scored), stream)
    if a.drop:
        drop = set(contract.read_json(Path(a.drop)))
        lifted = {t: v for t, v in lifted.items() if t not in drop}
        scored = [s for s in scored if not drop & set(s["pair"])]
        print(f"dropped {len(drop)} tracks: {len(lifted)} left, {len(scored)} scored pairs")
    if a.check_tracks or a.track_checks:
        if a.track_checks:
            checks = contract.read_json(Path(a.track_checks))
        else:
            checks = check_tracks(layers, lifted, Path(a.out) / "check_tracks", a.vlm_python, a.vlm_weights, a.gpu)
        contract.write_json(Path(a.out) / "track_checks.json", checks)
        gone = {t for t, c in checks.items() if (c or {}).get("kind") in ("structure", "none")}
        lifted = {t: v for t, v in lifted.items() if t not in gone}
        scored = [s for s in scored if not gone & set(s["pair"])]
    instances = cluster(layers, lifted, scored, mutual=a.same, min_frames=a.min_frames)
    if a.min_views:     # fusion's rule: an object needs that many views with depth (here: lifted keyframes, all tracks)
        kept = [m for m in instances if len({f for t in m for f in lifted[t]["frames"]}) >= a.min_views]
        print(f"min_views {a.min_views}: {len(instances)} -> {len(kept)} instances")
        instances = kept
    records = instance_records(layers, lifted, instances)
    if a.gt:
        before = evaluate(layers, records, a.gt)
        print("clustered:", json.dumps({k: v for k, v in before.items() if k != "per_gt"}))
    if a.instance_checks:       # answers of an earlier run with the same clustering (same tracks, checks and rules)
        answers = contract.read_json(Path(a.instance_checks))["answers"]
        for i, r in enumerate(records):
            r["check"] = answers.get(str(i))
        records = apply_checks(records, a.rename)
    elif a.check:
        check_instances(layers, records, Path(a.out) / "check", a.vlm_python, a.vlm_weights, a.gpu)
        records = apply_checks(records, a.rename)
    if a.merge_revisits:
        records = merge_revisits(layers, records)
    if a.check_tracks or a.track_checks:
        # what the model called the tracks that make up most of an instance, kept as its other names; a name from a
        # track with a small share of the detections is left out (27dd4da69e: a "curtain" blob was matched as the
        # laptop through one small track the model called "laptop")
        for r in records:
            weight = {}
            for t in r["tracks"]:
                c = checks.get(t) or {}
                if c.get("name") and c.get("kind") in ("object", "part"):
                    weight[c["name"]] = weight.get(c["name"], 0) + len(layers.detections(t))
            total = sum(len(layers.detections(t)) for t in r["tracks"]) or 1
            r["track_names"] = sorted(n for n, w in weight.items() if w >= a.track_name_share * total)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        contract.write_json(out / "instances.json", [{k: v for k, v in r.items() if k not in ("points", "merged_into")}
                                                     for r in records])
        contract.write_json(out / "scores.json", scored)
    if a.evidence:
        print(json.dumps(to_evidence(layers, records, a.run, a.evidence, name_share=a.name_share)))
    if a.gt:
        result = evaluate(layers, records, a.gt)
        print(json.dumps({k: v for k, v in result.items() if k != "per_gt"}))
        if a.out:
            contract.write_json(Path(a.out) / "evaluation.json", result)


if __name__ == "__main__":
    main()
