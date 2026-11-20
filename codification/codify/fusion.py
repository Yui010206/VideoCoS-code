"""fuse_objects: masks + metric depth + cameras -> object boxes (static or moved), appearance, observations.

Inputs: one image, a fixed-camera clip or a moving-camera video (docs/offline_codification_implementation.md):
- Depth is camera-Z (DA3); masks are resampled to the depth grid with pixel-center alignment,
  eroded, and stripped of low-confidence and depth-discontinuity pixels before unprojection.
- Gravity/up comes from the dominant horizontal plane (RANSAC, normals near the mean camera
  up); z = 0 is the lowest well-supported horizontal surface. Both are recorded estimates.
- Per-view world centroids are clustered (single linkage). A dominant cluster (>= 60% of
  views) is kept and the other views are recorded as a suspected identity switch; without
  one, or when the kept views drift between the first and second half of the detections,
  the track stays in unresolved_tracks.json with the reason. A track becomes a static
  object only with >= min_views supporting views. Boxes are gravity-aligned with PCA yaw
  (a declared canonical assumption); extents are robust visible-surface extents.
- Poor depth is flagged (median DA3 confidence on the mask) and objects centered above an
  indoor height prior are kept unresolved as likely depth errors.
- One physical object, several candidates (group_same_objects, complete linkage): two prompts on the
  same object (detected on the same frames with overlapping boxes), or a revisit / window boundary
  (never on the same frame, each center inside the other's box or a moved object's rest box, similar
  colors). Names need not match; the best-supported candidate is kept with the voted name, a static
  group's box is refitted on all members' points, and the others are unresolved as merged_into with
  their detection rows moved to the kept id.
- Moving camera (track_motion): a track that rests at one place, is later seen elsewhere, and whose
  former place the depth maps then see through becomes dynamic, with a per-frame position track
  (rests + smoothed visible centroid, held while unobserved) and its yaw held.
- Fixed camera (fixed_camera_motion): a static object's surface stays put, so a centroid that keeps
  leaving its first place is motion; per-frame positions and rotations chained from surface features.
- Distances are scaled by the scene's median depth (reference_depth_m) and depth confidence is cut at
  the video's own lower quantile, so the same settings serve rooms, table tops and renders.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import contract
# Category -> procedural recipe (object_library) for the proxy of an open-vocabulary name; the same table is the
# legacy default prompt list inside the cached SAM 3 worker functions, so it lives in models.py.
from .models import VOCABULARY as RECIPE_ALIASES

DEFAULTS = {"intrinsics": "declared", "conf_min": 3.0, "edge_rel": 0.05, "erode_px": 2, "min_view_pixels": 40,
            "min_view_pixels_small": 12,
            "min_views": 3, "points_per_view": 1500, "percentiles": [2, 98], "min_extent_m": 0.02,
            "motion_tolerance_m": 0.2, "motion_tolerance_rel": 0.5, "dominant_cluster": 0.6, "duplicate_iou": 0.5,
            "low_depth_confidence": 4.0, "max_height_m": 3.0,
            "dynamic": True, "rest_views": 5, "rest_radius_rel": 0.5, "free_space_points": 2000,
            "free_space_gap_frames": 10, "free_space_margin_m": 0.05, "free_space_margin_rel": 0.2,
            "free_space_min_frames": 3, "free_space_fraction": 0.7,
            "move_min_m": 0.5, "max_speed_mps": 1.5, "step_noise_m": 0.1, "max_gap_s": 1.5,
            "merge_same_objects": True, "codetect_contained": 0.8, "codetect_fraction": 0.6, "same_color_overlap": 0.5,
            "same_mask_iou": 0.7, "attached_rel": 0.15, "piece_containment": 0.4,
            "certain_min_detections": 3, "certain_min_score": 0.5,
            "fixed_motion_px": 3.0, "fixed_motion_rel": 0.1, "fixed_motion_mask_iou": 0.8, "fixed_motion_range_rel": 0.02, "fixed_motion_jitter": 4.0, "kabsch_inlier_rel": 0.05,
            "floor_ransac_iterations": 2000, "floor_inlier_m": 0.02, "max_up_tilt_deg": 70.0,
            "wall_dominance": 1.5, "wall_perpendicular_tolerance_deg": 20.0, "extend_down_min_deg": 30.0,
            "picture_recipes": ["picture", "wall artwork"], "seed": 0,
            "given_cameras": None, "fixed_camera": False, "static_camera_rel": 0.005, "static_camera_deg": 0.5,
            "conf_quantile": 0.25, "low_conf_quantile": 0.35, "reference_depth_m": 1.65, "support_area_fraction": 0.5,
            "revisit_margin_m": 0.3, "part_containment": 0.5}
# Distances (and the speed) above that scale with the scene: multiplied by median depth / reference_depth_m.
METRIC_KEYS = ("min_extent_m", "motion_tolerance_m", "free_space_margin_m", "move_min_m", "step_noise_m",
               "max_speed_mps", "revisit_margin_m")


def scaled_K(K, factor):
    """Intrinsics after resizing by `factor` (OpenCV integer pixel centers)."""
    K = np.array(K, float)
    K[:2, :2] *= factor
    K[:2, 2] = (K[:2, 2] + 0.5) * factor - 0.5
    return K


def grid_index(n_out, n_in):
    """Nearest source index for each output pixel center (center-aligned resampling)."""
    return np.clip(np.round((np.arange(n_out) + 0.5) * n_in / n_out - 0.5), 0, n_in - 1).astype(int)


def erode(mask, pixels):
    import cv2
    if pixels <= 0:
        return mask
    return cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=pixels).astype(bool)


def depth_edges(depth, rel):
    """Pixels next to a relative depth jump larger than rel (dilated by one pixel)."""
    jump = np.zeros(depth.shape, bool)
    dx = np.abs(np.diff(depth, axis=1)) / np.minimum(depth[:, 1:], depth[:, :-1])
    dy = np.abs(np.diff(depth, axis=0)) / np.minimum(depth[1:], depth[:-1])
    jump[:, 1:] |= dx > rel
    jump[:, :-1] |= dx > rel
    jump[1:] |= dy > rel
    jump[:-1] |= dy > rel
    return jump


def unproject(K_inv, u, v, d):
    return (K_inv @ np.vstack([u, v, np.ones_like(u)]) * d).T


def center_ratio(point, box):
    """Largest |offset| / half-size over the axes of a gravity-aligned box; <= 1 means inside."""
    c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
    d = np.asarray(point, float) - np.asarray(box["center"], float)
    local = np.array([d[0] * c + d[1] * s, -d[0] * s + d[1] * c, d[2]])
    return float(np.max(np.abs(local) / (np.asarray(box["size"], float) / 2)))


def codetected_overlap(masks_a, masks_b, min_contained):
    """On the frames holding both tracks (coarse masks {frame: bool array}): the share of them on which the smaller
    mask lies mostly inside the larger one (the same object under two prompts, or a part and its whole), and the
    masks' median IoU (near 1 for the same object, low for a part and its whole). Masks of one prompt never overlap,
    and an object in front of another covers it instead of overlapping it. None when no frame holds both."""
    shared, ious = [], []
    for f, a in masks_a.items():
        if f in masks_b:
            b = masks_b[f]
            small, inter = min(a.sum(), b.sum()), (a & b).sum()
            shared.append(small > 0 and inter >= min_contained * small)
            ious.append(inter / max((a | b).sum(), 1))
    return (float(np.mean(shared)), float(np.median(ious))) if shared else None


def box_containment(a, b):
    """Share of the smaller of two gravity-aligned boxes ({center, size, yaw}) that lies inside the other."""
    import cv2
    rect = lambda x: ((float(x["center"][0]), float(x["center"][1])), (float(x["size"][0]), float(x["size"][1])),
                      float(np.degrees(x["yaw"])))
    kind, corners = cv2.rotatedRectangleIntersection(rect(a), rect(b))
    if kind == cv2.INTERSECT_NONE or corners is None or len(corners) < 3:
        return 0.0
    bottom = max(a["center"][2] - a["size"][2] / 2, b["center"][2] - b["size"][2] / 2)
    top = min(a["center"][2] + a["size"][2] / 2, b["center"][2] + b["size"][2] / 2)
    shared = cv2.contourArea(cv2.convexHull(corners.astype(np.float32))) * max(0.0, float(top - bottom))
    return float(shared / max(min(np.prod(a["size"]), np.prod(b["size"])), 1e-12))


def color_histogram(rgb):
    """Hue-saturation histogram of sampled mask pixels (sRGB uint8), normalized to sum 1. Pixels with little
    saturation (white, grey, black: below 40 of 255) have no meaningful hue; they fill four brightness bins instead,
    so two views of a white door or a grey monitor compare by brightness, not by noise in their hue."""
    import cv2
    hsv = cv2.cvtColor(np.asarray(rgb, np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    chromatic = hsv[:, 1] >= 40
    hist = np.histogram2d(hsv[chromatic, 0], hsv[chromatic, 1], bins=(12, 4), range=((0, 180), (40, 256)))[0].ravel()
    grey = np.histogram(hsv[~chromatic, 2], bins=4, range=(0, 256))[0]
    hist = np.concatenate([hist, grey]).astype(float)
    return hist / max(hist.sum(), 1.0)


def head_noun(name):
    """The last word of a name, singular: what kind of thing it names ("office chair" and "chairs" -> "chair")."""
    words = name.lower().split()
    word = words[-1] if words else ""
    return word[:-1] if word.endswith("s") and not word.endswith("ss") and len(word) > 3 else word


DISTINCT = "distinct"


def group_same_objects(order, pair):
    """Groups of candidates that are one physical object. order: ids, best supported first; pair(a, b) -> a score
    (lower = more alike), None (nothing says they are one) or DISTINCT (seen together as two objects). Closest pairs
    join first; two groups join when their kept members (each group's best supported) are one object too and no pair
    across them is DISTINCT. A group's other members (a part, a thing it was confused with) neither block nor carry
    a join. Returns {member: kept id}; each group keeps its first-listed member."""
    rank = {t: i for i, t in enumerate(order)}
    scores, distinct = {}, set()
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            score = pair(a, b)
            if score is DISTINCT:
                distinct.update({(a, b), (b, a)})
            elif score is not None:
                scores[(a, b)] = scores[(b, a)] = score
    group = {t: [t] for t in order}
    for _, a, b in sorted((v, a, b) for (a, b), v in scores.items() if rank[a] < rank[b]):
        ga, gb = group[a], group[b]
        if ga is gb:
            continue
        ka, kb = min(ga, key=rank.get), min(gb, key=rank.get)
        if (ka, kb) in scores and not any((x, y) in distinct for x in ga for y in gb):
            ga.extend(gb)
            for t in gb:
                group[t] = ga
    merged = {}
    for members in {id(g): g for g in group.values()}.values():
        keep = min(members, key=rank.get)
        merged.update({t: keep for t in members if t != keep})
    return merged


def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_wxyz(R):
    """Unit quaternion [w, x, y, z] of a rotation matrix."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return (q / np.linalg.norm(q)).tolist()


def kabsch(P, Q):
    """Rotation R and translation t with Q ~ P @ R.T + t."""
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return R, cq - cp @ R.T


def kabsch_ransac(P, Q, threshold, rng, iterations=100, min_inliers=6):
    best = None
    for _ in range(iterations):
        idx = rng.choice(len(P), 3, replace=False)
        R, t = kabsch(P[idx], Q[idx])
        inliers = np.linalg.norm(P @ R.T + t - Q, axis=1) < threshold
        if best is None or inliers.sum() > best.sum():
            best = inliers
    return kabsch(P[best], Q[best])[0] if best.sum() >= min_inliers else None


def fixed_camera_motion(obs, views, lift, at_depth, depth_at, gray, masks, cfg, rng, support_z=0.0):
    """Motion seen by a camera that does not move. The masks give the image motion: a detection whose mask moved as
    a whole (its centroid, and the box edges on both sides of the move) beyond a share of the object's size has
    moved; a mask that only grows or shrinks on one side (a part leaving it, a jaw opening, an occluder) has not.
    Two such detections, or the visible-surface centroid leaving its first place on three frames where the mask is no
    longer the first one, lastingly (motion along the ray, a turn; depth noise leaves the mask alone), make the object
    dynamic.
    Its position on each detected frame is its mask centroid's ray at its depth,
    measured where the mask moves slowly on confident depth and held from the nearest such frame elsewhere (motion
    blur pulls depth toward the background). An object never measured that way (seen only moving, e.g. once released
    from what held it) starts at the depth its first mask's pixels had in the frames just before it, when that is
    nearer: the camera saw that spot still and sharp. Rotations are chained over consecutive frames from surface
    features (optical flow lifted to 3D, Kabsch with RANSAC; held where too few features survive). None for an object
    at rest. obs: per detection {frame_index, uv, bbox, clipped, area, z, conf} (z, conf None without depth);
    depth_at(f, uv) -> camera depths at canonical pixels of frame f (NaN where invalid)."""
    import cv2
    ref = obs[0]
    size_px = float(np.sqrt(ref["area"]))
    tau = max(cfg["fixed_motion_px"], cfg["fixed_motion_rel"] * size_px)

    def rigid(o, base):
        d = np.subtract(o["uv"], base["uv"])
        for axis in (0, 1):
            if abs(d[axis]) >= tau / 2:
                edges = [o["bbox"][e] - base["bbox"][e] for e in (axis, axis + 2)
                         if not (o["clipped"][e] or base["clipped"][e])]
                if any(e * np.sign(d[axis]) < 0.3 * abs(d[axis]) for e in edges):
                    return False
        return True
    shifted = [np.hypot(*np.subtract(o["uv"], ref["uv"])) > tau for o in obs]
    displaced = sum(m and rigid(o, ref) for m, o in zip(shifted, obs))
    along_ray = False
    if len(views) >= 3:
        cents = np.array([np.median(v["points"], axis=0) for v in views])
        extent = float(np.median([np.ptp(np.percentile(v["points"], [5, 95], axis=0), axis=0).max() for v in views]))
        steps = np.linalg.norm(np.diff(cents, axis=0), axis=1)
        tolerance3d = max(cfg["fixed_motion_rel"] * extent,
                          cfg["fixed_motion_range_rel"] * float(np.median([v["range"] for v in views])),
                          cfg["fixed_motion_jitter"] * float(np.median(steps)))
        away = np.linalg.norm(cents - np.median(cents[: max(3, len(cents) // 10)], axis=0), axis=1) > tolerance3d
        # Depth noise moves the visible surface while the mask stays put; an object that moved (along its ray,
        # turning, coming into view) left a different mask, to the end of the clip (a flicker or a passing
        # occluder does not last).
        start = masks[views[0]["frame_index"]]
        changed = np.array([(lambda m: (m & start).sum() < cfg["fixed_motion_mask_iou"] * max((m | start).sum(), 1))(
            masks[v["frame_index"]]) for v in views])
        moved = away & changed
        along_ray = int(moved.sum()) >= 3 and bool(moved[-max(3, len(views) // 3):].mean() >= 0.5)
    if displaced < 2 and not along_ray:
        return None
    # Image position: the centroid, except where the mask changed shape instead of moving (held there).
    uv = [np.asarray(ref["uv"], float)]
    for i in range(1, len(obs)):
        moved = np.hypot(*np.subtract(obs[i]["uv"], uv[-1])) > tau / 2
        uv.append(np.asarray(obs[i]["uv"], float) if not moved or rigid(obs[i], obs[i - 1]) else uv[-1])
    # Depth: measured on slow frames with confident depth, else held from the nearest measured frame in time.
    frames = [o["frame_index"] for o in obs]
    confs = [o["conf"] for o in obs if o["conf"] is not None]
    conf_ok = 0.8 * float(np.median(confs)) if confs else np.inf
    speed = [np.hypot(*(uv[min(i + 1, len(uv) - 1)] - uv[max(i - 1, 0)])) / max(1, frames[min(i + 1, len(uv) - 1)]
                                                                                    - frames[max(i - 1, 0)])
             for i in range(len(obs))]
    slow = max(cfg["fixed_motion_px"] / 1.5, 0.05 * size_px)
    measured = [i for i, o in enumerate(obs) if o["z"] is not None and o["conf"] >= conf_ok and speed[i] <= slow]
    first = next((o["z"] for o in obs if o["z"] is not None), None)
    if not measured and frames[0] >= 2:
        ys, xs = np.nonzero(masks[frames[0]])
        pick = rng.choice(len(xs), min(300, len(xs)), replace=False)
        before = np.concatenate([depth_at(f, np.column_stack([xs[pick], ys[pick]]).astype(float))
                                 for f in range(max(0, frames[0] - 6), frames[0] - 1)])
        before = before[np.isfinite(before)]
        if len(before) and (first is None or np.median(before) < first):
            first = float(np.median(before))
    if first is None:
        return None
    # The object's box is fitted on one 3D view at that view's measured depth: the one on a measured frame with the
    # most confident depth (an object at rest, sharp), else the first view.
    on_measured = [v for v in views if v["frame_index"] in {frames[j] for j in measured}]
    reference_view = max(on_measured, key=lambda v: v.get("confidence", 0.0)) if on_measured else views[0]
    extent = float(np.ptp(np.percentile(reference_view["points"], [5, 95], axis=0), axis=0).max())
    depth = []
    for i in range(len(obs)):
        near = sorted(measured, key=lambda j: (abs(frames[j] - frames[i]), j > i))[:5] if measured else []
        depth.append(float(np.median([obs[j]["z"] for j in near])) if near else first)
    # While it is in the air (its held position more than its own size above the support), a rigid
    # object's depth follows its apparent size instead (size constancy): the depth of nearby measured frames times
    # the ratio of the masks' sizes (square roots of their areas, masks not cut by the image border), smoothed over
    # neighbouring frames since orientation changes the area too. A falling or thrown object is then placed along
    # its ray where its size says it is, not at the depth it comes to rest at; on its support, depth is held.
    apparent = [float(np.sqrt(o["area"])) if o["area"] > 0 and not any(o["clipped"]) else None for o in obs]
    references = [j for j in measured if apparent[j]]
    scaled = []
    if references and extent > 0:
        raw = {}
        for i in range(len(obs)):
            if i in measured or not apparent[i]:
                continue
            if at_depth(frames[i], uv[i], depth[i])[2] - support_z <= extent:
                continue
            ref = sorted(references, key=lambda j: (abs(frames[j] - frames[i]), j > i))[:5]
            raw[i] = float(np.median([obs[j]["z"] * apparent[j] for j in ref])) / apparent[i]
        height, width = masks[frames[0]].shape
        grid = np.stack(np.meshgrid(np.linspace(0, width - 1, 32), np.linspace(0, height - 1, 24)), -1).reshape(-1, 2)
        for i in raw:
            z = float(np.median([raw[j] for j in range(i - 2, i + 3) if j in raw]))
            # Neither through the support nor behind the farthest surface the camera sees (a mask cut by an occluder
            # or flickering shrinks and would push the object away).
            if at_depth(frames[i], uv[i], z)[2] >= support_z and z <= np.nanpercentile(depth_at(frames[i], grid), 95):
                depth[i] = z
                scaled.append(i)
    positions = np.array([at_depth(f, u, z) for f, u, z in zip(frames, uv, depth)])
    tolerance = float(np.linalg.norm(at_depth(frames[0], uv[0] + [tau, 0], depth[0]) - positions[0]))
    # Where the box sits and how much nearer or farther the chosen depth puts it (it moves along its ray and scales
    # with it).
    k0 = frames.index(reference_view["frame_index"])
    anchor, scale = at_depth(frames[k0], uv[k0], obs[k0]["z"]), depth[k0] / obs[k0]["z"]
    rotations, held = [np.eye(3)], 0
    for a, b in zip(frames, frames[1:]):
        R = None
        p0 = cv2.goodFeaturesToTrack(gray[a], 200, 0.01, 3, mask=masks[a].astype(np.uint8) * 255)
        if p0 is not None and len(p0) >= 6:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(gray[a], gray[b], p0, None)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray[b], gray[a], p1, None)
            u0, u1 = p0.reshape(-1, 2), p1.reshape(-1, 2)
            inside = np.zeros(len(u1), bool)
            xy = np.round(u1).astype(int)
            on = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < masks[b].shape[1]) & (xy[:, 1] < masks[b].shape[0])
            inside[on] = masks[b][xy[on, 1], xy[on, 0]]
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & inside & (np.linalg.norm(back.reshape(-1, 2) - u0, axis=1) < 1.0)
            P, Q = lift(a, u0[ok]), lift(b, u1[ok])
            good = np.isfinite(P).all(1) & np.isfinite(Q).all(1)
            if good.sum() >= 6:
                extent = float(np.ptp(P[good], axis=0).max())
                R = kabsch_ransac(P[good], Q[good], cfg["kabsch_inlier_rel"] * extent, rng)
        if R is None:
            R, held = np.eye(3), held + 1
        rotations.append(R @ rotations[-1])
    return {"frames": frames, "positions": positions, "anchor": anchor, "scale": scale, "reference": k0,
            "measured_frames": [frames[j] for j in measured],
            "reference_view": reference_view, "rotations": rotations, "tolerance": tolerance, "depth_scaled": len(scaled),
            "displacement_m": float(np.linalg.norm(positions - positions[0], axis=1).max()), "held_steps": held,
            "steps": len(frames) - 1, "depth_held": len(obs) - len(measured) - len(scaled), "image_displaced": int(displaced),
            "along_ray": bool(along_ray)}


def estimate_world(points, camera_up, cfg, rng, bin_m=0.02):
    """Up from the best horizontal-ish plane; floor = lowest well-supported horizontal layer. A wall is vertical:
    when a near-vertical plane carries clearly more points than the best horizontal-ish one and that one is not
    perpendicular to it (a sloped surface, e.g. cushions hiding the floor), up is the camera's up made perpendicular
    to the wall instead."""
    sample = points[rng.choice(len(points), min(60000, len(points)), replace=False)]
    best, best_count, wall, wall_count = None, 0, None, 0
    cos_tilt = np.cos(np.radians(cfg["max_up_tilt_deg"]))
    for _ in range(cfg["floor_ransac_iterations"]):
        p = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        count = int((np.abs((sample - p[0]) @ n) < cfg["floor_inlier_m"]).sum())
        if abs(n @ camera_up) < cos_tilt:
            if count > wall_count:
                wall, wall_count = n, count
        elif count > best_count:
            best, best_count = (n, p[0]), count
    if best is None:
        return None
    n, p0 = best
    inliers = sample[np.abs((sample - p0) @ n) < cfg["floor_inlier_m"]]
    centered = inliers - inliers.mean(0)
    up = np.linalg.svd(centered, full_matrices=False)[2][2]
    up = up if up @ camera_up > 0 else -up
    source = "plane"
    if wall is not None and wall_count >= cfg["wall_dominance"] * best_count and \
            abs(up @ wall) > np.sin(np.radians(cfg["wall_perpendicular_tolerance_deg"])):
        up = camera_up - (camera_up @ wall) * wall
        up /= np.linalg.norm(up)
        source = "camera_up_perpendicular_to_wall"
    floor, method = lowest_layer(points @ up, bin_m)
    return {"up": up, "floor_height": floor, "plane_inlier_fraction": best_count / len(sample),
            "up_vs_camera_mean_deg": float(np.degrees(np.arccos(np.clip(up @ camera_up, -1, 1)))),
            "floor_method": method, "up_source": source}


def ground_affine(depth, valid, K_inv, uu, vv, R_wc, C, ground_z, rng, samples=4000, tolerance=0.01, iterations=200):
    """(a, b, inlier share) with exact = a * depth + b on the pixels that see the ground plane z = ground_z: a pixel's
    exact depth there is where its ray meets the plane. RANSAC over pairs of downward-looking pixels (the ground is
    most of what they see; objects stand in front of it), least squares on the inliers. None when the ground is not
    the larger part of them."""
    idx = np.flatnonzero(valid)
    idx = rng.choice(idx, min(samples, len(idx)), replace=False)
    ray_z = (unproject(K_inv, uu.ravel()[idx], vv.ravel()[idx], np.ones(len(idx))) @ R_wc.T)[:, 2]
    down = ray_z < -1e-3
    d, t = depth.ravel()[idx][down], (C[2] - ground_z) / -ray_z[down]
    if len(d) < 50:
        return None
    best = np.zeros(len(d), bool)
    for _ in range(iterations):
        i, j = rng.choice(len(d), 2, replace=False)
        if abs(d[i] - d[j]) < 1e-3 * max(d[i], d[j]):
            continue
        a = (t[i] - t[j]) / (d[i] - d[j])
        inliers = np.abs(a * d + t[i] - a * d[i] - t) <= tolerance * t
        if a > 0 and inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 0.3 * len(d):
        return None
    a, b = np.linalg.lstsq(np.column_stack([d[best], np.ones(int(best.sum()))]), t[best], rcond=None)[0]
    return (float(a), float(b), float(best.mean())) if a > 0 else None


def lowest_layer(heights, bin_m=0.02):
    """Height of the lowest well-supported horizontal layer (the floor, or the table top in a close-up)."""
    bins = np.arange(heights.min(), heights.max() + bin_m, bin_m)
    hist, edges = np.histogram(heights, bins)
    peaks = [i for i in range(len(hist)) if hist[i] >= 0.02 * len(heights)
             and hist[i] >= hist[max(i - 1, 0)] and hist[i] >= hist[min(i + 1, len(hist) - 1)]]
    if peaks:
        return float((edges[peaks[0]] + edges[peaks[0] + 1]) / 2), "lowest horizontal histogram peak"
    return float(np.percentile(heights, 1)), "1st height percentile (no clear peak)"


def camera_motion_of(R_wc, C, depth, cfg):
    """'single' (one frame), 'static' (the camera does not move) or 'moving'."""
    if len(C) < 2:
        return "single"
    sample = depth[:, ::4, ::4]            # a scale for the thresholds; every pixel is not needed
    scene_depth = float(np.median(sample[sample > 0])) if np.any(sample > 0) else 1.0
    shift = np.linalg.norm(C - C[0], axis=1).max() / scene_depth
    turn = max(np.degrees(np.arccos(np.clip((np.trace(R.T @ R_wc[0]) - 1) / 2, -1, 1))) for R in R_wc)
    return "static" if shift < cfg["static_camera_rel"] and turn < cfg["static_camera_deg"] else "moving"


def view_clusters(centroids, radius):
    """Single-linkage clusters of per-view world centroids, largest first."""
    labels = -np.ones(len(centroids), int)
    for seed in range(len(centroids)):
        if labels[seed] >= 0:
            continue
        labels[seed], stack = seed, [seed]
        while stack:
            j = stack.pop()
            near = np.flatnonzero((labels < 0) & (np.linalg.norm(centroids - centroids[j], axis=1) <= radius))
            labels[near] = seed
            stack.extend(near.tolist())
    return sorted((np.flatnonzero(labels == k) for k in np.unique(labels)), key=len, reverse=True)


def fit_box(points, cfg):
    """Gravity-aligned box: PCA yaw on the horizontal footprint, robust percentile extents."""
    xy = points[:, :2] - points[:, :2].mean(0)
    eigvals, eigvecs = np.linalg.eigh(np.cov(xy.T))
    major = eigvecs[:, int(np.argmax(eigvals))]
    yaw = float(np.arctan2(major[1], major[0]))
    if yaw <= -np.pi / 2:
        yaw += np.pi
    elif yaw > np.pi / 2:
        yaw -= np.pi
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.column_stack([points[:, 0] * c + points[:, 1] * s, -points[:, 0] * s + points[:, 1] * c, points[:, 2]])
    low, high = np.percentile(local, cfg["percentiles"], axis=0)
    size = high - low
    clamped = bool(np.any(size < cfg["min_extent_m"]))
    mid = (low + high) / 2
    size = np.maximum(size, cfg["min_extent_m"])
    center = np.array([mid[0] * c - mid[1] * s, mid[0] * s + mid[1] * c, mid[2]])
    return {"center": center, "yaw": yaw, "size": size, "clamped": clamped}


def rest_segments(centroids, radius, min_views, smooth=5):
    """Median-smoothed centroids and the runs of consecutive views (time order) that stay within
    radius of the run's first view, as [(first, end, median)]; shorter runs are motion or noise."""
    half = smooth // 2
    smoothed = np.array([np.median(centroids[max(0, i - half):i + half + 1], axis=0) for i in range(len(centroids))])
    segments, start = [], 0
    for i in range(1, len(smoothed) + 1):
        if i == len(smoothed) or np.linalg.norm(smoothed[i] - smoothed[start]) > radius:
            if i - start >= min_views:
                segments.append((start, i, np.median(smoothed[start:i], axis=0)))
            start = i
    return smoothed, segments


def free_space(points, keys, rec, margin, covered=None, min_points=30):
    """(vacated, informative) frame counts for world points of a former object surface.

    In each keyframe the points are projected onto the depth grid; depth beyond a point means the
    camera sees through where the surface was, depth at the point means it is still occupied, and
    closer depth is an occluder. A point on a same-category mask (covered(k), depth grid) counts as
    occupied, so a small pose error cannot turn a still-present object into free space. A frame
    counts when enough points are seen unoccluded.
    """
    vacated = informative = 0
    reconstruction_points = points @ rec.R_norm + rec.origin
    for k in keys:
        cam = (reconstruction_points - rec.C[k]) @ rec.R_wc[k]
        cam = cam[cam[:, 2] > 0.2]
        uvw = cam @ rec.K_used.T
        u, v = np.round(uvw[:, 0] / uvw[:, 2]).astype(int), np.round(uvw[:, 1] / uvw[:, 2]).astype(int)
        inside = (u >= 0) & (u < rec.w) & (v >= 0) & (v < rec.h)
        u, v, z = u[inside], v[inside], cam[inside, 2]
        ok = rec.valid[k][v, u]
        difference = rec.depth[k][v[ok], u[ok]] - z[ok]
        same = covered(k)[v[ok], u[ok]] if covered is not None else np.zeros(len(difference), bool)
        behind = int(((difference > margin) & ~same).sum())
        on = int(((np.abs(difference) <= margin) | ((difference > margin) & same)).sum())
        if behind + on >= min_points and behind + on >= 0.5 * len(difference):
            informative += 1
            vacated += behind >= 0.6 * (behind + on)
    return vacated, informative


def track_motion(views, rec, cfg, rng, covered=None):
    """Dynamic-object evidence for one track (views with world points, time order), or None.

    The earliest rest location (rest_segments) counts as left when the move is observed: from the
    last view there, the track is followed until the first jump a carried object cannot make (a gap
    longer than max_gap_s, or a step faster than max_speed_mps), which marks an identity switch or
    an unobserved transition and ends the object's track. At least three followed views must lie
    farther than the motion tolerance and move_min_m, and in the frames after leaving, the depth map
    must see through the object's former surface (free_space) in most informative frames. Rotation
    is not measured: the box keeps the yaw fitted at its best-supported rest; positions follow the
    rests and, between them, the smoothed visible centroid, held while the object is not observed.
    """
    centroids = np.array([np.median(v["points"], axis=0) for v in views])
    sizes = [np.ptp(np.percentile(v["points"], [5, 95], axis=0), axis=0).max() for v in views]
    size = float(np.median(sizes))
    tolerance = max(cfg["motion_tolerance_m"], cfg["motion_tolerance_rel"] * size)
    smoothed, segments = rest_segments(centroids, cfg["rest_radius_rel"] * tolerance, cfg["rest_views"])
    if not segments:
        return None
    first = segments[0][2]
    at_first = [s for s in segments if np.linalg.norm(s[2] - first) <= tolerance]
    left = end = at_first[-1][1]
    fps = rec.video_info["canonical"]["fps_num"] / rec.video_info["canonical"]["fps_den"]
    while end < len(views):
        dt = (views[end]["frame_index"] - views[end - 1]["frame_index"]) / fps
        if dt > cfg["max_gap_s"] or \
                np.linalg.norm(smoothed[end] - smoothed[end - 1]) > cfg["step_noise_m"] + cfg["max_speed_mps"] * dt:
            break
        end += 1
    distance = np.linalg.norm(smoothed[left:end] - first, axis=1)
    if int((distance > max(tolerance, cfg["move_min_m"])).sum()) < 3:
        return None
    views, smoothed = views[:end], smoothed[:end]
    segments = [s for s in segments if s[1] <= end]
    surface = np.concatenate([views[i]["points"] for s in at_first for i in range(s[0], s[1])])
    surface = surface[rng.choice(len(surface), min(cfg["free_space_points"], len(surface)), replace=False)]
    after = views[left]["frame_index"] + cfg["free_space_gap_frames"]
    keys = [k for k, f in enumerate(rec.keyframes) if f >= after]
    margin = max(cfg["free_space_margin_m"], cfg["free_space_margin_rel"] * size)
    vacated, informative = free_space(surface, keys, rec, margin, covered)
    if informative < cfg["free_space_min_frames"] or vacated < cfg["free_space_fraction"] * informative:
        return None
    reference = max(segments, key=lambda s: s[1] - s[0])
    box = fit_box(np.concatenate([views[i]["points"] for i in range(reference[0], reference[1])]), cfg)
    offset = box["center"] - reference[2]
    position = smoothed.copy()
    for first_view, end_view, median in segments:
        position[first_view:end_view] = median
    observed = {v["frame_index"]: position[i] + offset for i, v in enumerate(views)}
    track, last = [], None
    for f in range(views[0]["frame_index"], views[-1]["frame_index"] + 1):
        last = observed.get(f, last)
        track.append((f, last, "observed" if f in observed else "held"))
    places = []
    for _, _, median in segments:
        if all(np.linalg.norm(median - p) > tolerance for p in places):
            places.append(median)
    return {"box": box, "track": track, "tolerance": tolerance, "displacement_m": float(distance.max()),
            "rest_frames": [[views[start]["frame_index"], views[end - 1]["frame_index"]] for start, end, _ in segments],
            "placements": [{**box, "center": p + offset} for p in places],
            "free_space": {"vacated_frames": vacated, "informative_frames": informative, "margin_m": margin,
                           "after_frame": after, "former_place": (first + offset).tolist()}}


def reconstruction(geometry_dir, decode_dir, cfg, rng):
    """Cameras, valid depth and the gravity-aligned world frame from the geometry stage.

    Shared by fuse_objects and the agent's camera_depth evidence; rng is consumed in the
    same order as before (scene-point sampling, then the plane RANSAC).
    """
    geometry = contract.read_npz(Path(geometry_dir) / "geometry.npz")
    model_info = contract.read_json(Path(geometry_dir) / "model_info.json")
    video_info = contract.read_json(Path(decode_dir) / "video_info.json")
    keyframes = [int(k) for k in geometry["keyframes"]]
    depth, conf, w2c, K_model = geometry["depth"], geometry["conf"], geometry["w2c"], geometry["K"]
    h, w = (int(x) for x in geometry["processed_hw"])
    W, H = video_info["canonical"]["width"], video_info["canonical"]["height"]
    warnings = []

    given = None
    if cfg.get("given_cameras"):
        from . import given_cameras
        timeline = contract.read_jsonl(Path(decode_dir) / "timeline.jsonl")
        K_given, w2c_given, given_world = given_cameras.load(cfg["given_cameras"], timeline, (W, H))
        if np.abs(K_given - K_given[0]).max() > 1e-6:
            raise ValueError("Given cameras with varying intrinsics are not supported")
        if not np.allclose(given_world["up"], [0, 0, 1]):
            raise ValueError("Given cameras must declare a +Z-up world")
        video_info = {**video_info, "projection": {"model": "pinhole", "K": K_given[0].tolist(), "width": W,
                                                   "height": H, "distortion_model": "none", "distortion_coeffs": []}}
        given = SimpleNamespace(w2c=w2c_given[keyframes], world=given_world)

    # Intrinsics on the depth grid: the declared (calibrated, rectified, or given) K when present.
    K_declared = scaled_K(video_info["projection"]["K"], w / W) if video_info.get("projection") else None
    focal_ratio = float(np.median(K_model[:, 0, 0]) / K_declared[0, 0]) if K_declared is not None else None
    if given is not None:
        K_used, intrinsics_source = K_declared, "given cameras"
    elif cfg["intrinsics"] == "declared" and K_declared is not None:
        K_used, intrinsics_source = K_declared, "declared calibration (rectified pinhole)"
        if abs(focal_ratio - 1) > 0.05:
            warnings.append(f"DA3 focal differs from the declared calibration by {100 * (focal_ratio - 1):+.1f}%; "
                            "declared K used for unprojection and export")
    else:
        K_used, intrinsics_source = np.median(K_model, axis=0), "geometry model (median over keyframes)"
    K_inv = np.linalg.inv(K_used)
    R_wc = np.transpose(w2c[:, :, :3], (0, 2, 1))
    C = -np.einsum("nij,nj->ni", R_wc, w2c[:, :, 3])
    uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    # Depth confidence has no common scale across scenes (renders score far below real rooms): the cut is
    # the configured minimum or the video's own lower quantile, whichever is lower.
    conf_cut = min(cfg["conf_min"], float(np.quantile(conf[:, ::4, ::4], cfg["conf_quantile"])))
    low_cut = min(cfg["low_depth_confidence"], float(np.quantile(conf[:, ::4, ::4], cfg["low_conf_quantile"])))
    valid = [(conf[i] >= conf_cut) & ~depth_edges(depth[i], cfg["edge_rel"]) & (depth[i] > 0)
             for i in range(len(keyframes))]
    median_depth = lambda: float(np.median(np.concatenate([depth[i][::4, ::4][valid[i][::4, ::4]]
                                                           for i in range(len(keyframes))])))
    depth_scale, depth_shift, scale_source, camera_motion = 1.0, None, None, camera_motion_of(R_wc, C, depth, cfg)
    if cfg.get("fixed_camera") and given is None and camera_motion == "moving":
        # The input states a camera that does not move (a locked-off shot): the predicted pose drift is the
        # geometry model's, not the camera's.
        warnings.append("Camera stated fixed by the input; predicted camera motion ignored")
        camera_motion = "static"
    if given is not None:
        R_given = np.transpose(given.w2c[:, :, :3], (0, 2, 1))
        C_given = -np.einsum("nij,nj->ni", R_given, given.w2c[:, :, 3])
        if camera_motion_of(R_given, C_given, depth, cfg) == "moving":
            # One scale for depth: the predicted camera path fitted to the given one.
            a, b = C - C.mean(0), C_given - C_given.mean(0)
            depth_scale = float(np.sqrt((b ** 2).sum() / max((a ** 2).sum(), 1e-12)))
            scale_source = "given cameras (predicted camera path fitted to them)"
        elif given.world.get("ground_z") is not None:
            # A camera that does not move has no baseline; the declared ground fixes the depth. Monocular depth is
            # right up to a scale and a shift (its error grows with distance): on each keyframe both are fitted
            # where the ground is seen, each pixel's exact depth following from its ray meeting the ground plane.
            sampler = np.random.default_rng(cfg["seed"] + 1)
            fits = [ground_affine(depth[i], valid[i], K_inv, uu, vv, R_given[i], C_given[i], given.world["ground_z"],
                                  sampler) for i in range(len(keyframes))]
            found = [f for f in fits if f]
            if found:
                held = tuple(np.median(found, axis=0))      # a keyframe whose ground did not fit takes the median
                fits = [f or held for f in fits]
                depth = np.stack([np.where(d > 0, np.maximum(a * d + b, 0.0), 0.0) for d, (a, b, _) in zip(depth, fits)])
                valid = [v & (d > 0) for v, d in zip(valid, depth)]
                depth_scale, depth_shift = held[0], held[1]
                scale_source = (f"declared ground of the given cameras' world: depth scale and shift fitted on the "
                                f"ground of {len(found)}/{len(fits)} keyframes (median {held[0]:.4f} x + {held[1]:.3f} m)")
        if scale_source is None:
            warnings.append("Given camera does not move and no ground height is declared: depth keeps the "
                            "geometry model's metric scale")
        if depth_shift is None:
            depth = depth * depth_scale
        R_wc, C, camera_motion = R_given, C_given, camera_motion_of(R_given, C_given, depth, cfg)
    elif camera_motion == "static":
        R_wc, C = np.repeat(R_wc[:1], len(R_wc), 0), np.repeat(C[:1], len(C), 0)
    # Distances in the config are tuned at a room scale (reference_depth_m); this scene's are proportional.
    scene_scale = median_depth() / cfg["reference_depth_m"]

    # World normalization from scene points (all keyframes, sampled).
    scene_points = []
    for i in range(len(keyframes)):
        idx = np.flatnonzero(valid[i])
        idx = rng.choice(idx, min(3000, len(idx)), replace=False)
        cam = unproject(K_inv, uu.ravel()[idx], vv.ravel()[idx], depth[i].ravel()[idx])
        scene_points.append(cam @ R_wc[i].T + C[i])
    scene_points = np.concatenate(scene_points)
    camera_up = -R_wc[:, :, 1].mean(0)
    camera_up /= np.linalg.norm(camera_up)
    if given is not None:
        floor, method = lowest_layer(scene_points[:, 2], 0.02 * scene_scale)
        world = {"up": np.array([0.0, 0.0, 1.0]), "floor_height": floor, "floor_method": method}
    else:
        world = estimate_world(scene_points, camera_up, {**cfg, "floor_inlier_m": cfg["floor_inlier_m"] * scene_scale},
                               rng, 0.02 * scene_scale)
    if world is None:
        up_status, up, floor = "unresolved", camera_up, float(np.percentile(scene_points @ camera_up, 1))
        warnings.append("No horizontal plane found; gravity/up unresolved (camera-mean axis used only to orient the preview)")
        world = {"floor_method": "none"}
    else:
        up_status, up, floor = ("reference" if given is not None else "estimated"), world["up"], world["floor_height"]
    if given is not None:
        # The given world frame is kept as it is (its own origin and axes).
        R_norm, origin = np.eye(3), np.zeros(3)
        yaw_convention = "the given cameras' world frame"
    else:
        forward = R_wc[0][:, 2] - (R_wc[0][:, 2] @ up) * up
        y_axis = forward / np.linalg.norm(forward)
        x_axis = np.cross(y_axis, up)
        R_norm = np.vstack([x_axis, y_axis, up])           # rows: bundle axes in reconstruction coordinates
        origin = C[0] - (C[0] @ up - floor) * up          # first camera center dropped onto the floor plane
        yaw_convention = "bundle +Y = first camera forward projected on the floor plane"
    to_world = lambda p: (np.asarray(p) - origin) @ R_norm.T
    camera_heights = (C - origin) @ up - (floor if given is not None else 0.0)
    if scale_source:
        scale = {"scale_status": "metric_calibrated", "scale_source": scale_source}
    elif model_info["is_metric"]:
        scale = {"scale_status": "metric_estimated", "scale_source": f"{model_info['model']} metric branch"}
    else:
        scale = {"scale_status": "relative", "scale_source": "none"}
    world_record = {"R_world_from_reconstruction": R_norm.tolist(), "origin_in_reconstruction": origin.tolist(),
                    "up_status": up_status, "up_in_reconstruction": up.tolist(),
                    "camera_mean_up_in_reconstruction": camera_up.tolist(), "camera_motion": camera_motion,
                    "floor": {k: v for k, v in world.items() if k not in ("up",)},
                    "camera_height_above_floor_m": {"median": float(np.median(camera_heights)),
                                                    "min": float(camera_heights.min()), "max": float(camera_heights.max())},
                    "yaw_convention": yaw_convention,
                    "intrinsics_source": intrinsics_source, "K_depth_grid": K_used.tolist(),
                    "model_to_declared_focal_ratio": focal_ratio, **scale,
                    "scale_factor": depth_scale, "depth_shift_m": depth_shift, "scale_uncertainty": None,
                    "depth_confidence_cut": conf_cut,
                    "low_depth_confidence_cut": low_cut,
                    "scene_scale": scene_scale,
                    "model_internal_metric_scale_factor": model_info.get("metric_scale_factor")}
    if not model_info["is_metric"]:
        warnings.append("Relative scale: metric reasoning handoff will be refused")

    return SimpleNamespace(keyframes=keyframes, depth=depth, conf=conf, K_used=K_used, K_inv=K_inv, R_wc=R_wc, C=C,
                           uu=uu, vv=vv, valid=valid, world=world_record, R_norm=R_norm, origin=origin,
                           to_world=to_world, scene_points=to_world(scene_points), up_status=up_status,
                           scene_scale=scene_scale,
                           warnings=warnings, h=h, w=w, W=W, H=H, video_info=video_info, model_info=model_info)


def _slerp(R0, R1, t):
    q0, q1 = np.array(contract.matrix_to_quat_wxyz(R0)), np.array(contract.matrix_to_quat_wxyz(R1))
    if q0 @ q1 < 0:
        q1 = -q1
    angle = np.arccos(np.clip(q0 @ q1, -1, 1))
    q = q0 if angle < 1e-9 else (np.sin((1 - t) * angle) * q0 + np.sin(t * angle) * q1) / np.sin(angle)
    return contract.quat_wxyz_to_matrix(q / np.linalg.norm(q))


def camera_rows(cameras, world, video_info, timeline):
    """One camera row per canonical frame from the cameras the reconstruction used (cameras: keyframes, R_wc, C,
    processed_hw; given cameras when they were an input, else the geometry model's, held for a fixed camera):
    keyframe estimates, interpolation between keyframes (labelled), held poses outside them."""
    keyframes = [int(k) for k in cameras["keyframes"]]
    R_wc, C = np.asarray(cameras["R_wc"]), np.asarray(cameras["C"])
    given = world.get("intrinsics_source") == "given cameras"
    source, measured = ("given_cameras", "reference") if given else ("geometry_model", "estimated")
    R_norm, origin = np.array(world["R_world_from_reconstruction"]), np.array(world["origin_in_reconstruction"])
    W, H = video_info["canonical"]["width"], video_info["canonical"]["height"]
    if world["intrinsics_source"].startswith("declared"):
        K = np.array(video_info["projection"]["K"], float)
    else:
        K = scaled_K(world["K_depth_grid"], W / int(cameras["processed_hw"][1]))
    projection = {"model": "pinhole", "K": K.tolist(), "width": W, "height": H, "distortion_model": "none",
                  "distortion_coeffs": []}
    rotations = {f: R_norm @ R_wc[i] for i, f in enumerate(keyframes)}
    centers = {f: R_norm @ (C[i] - origin) for i, f in enumerate(keyframes)}
    rows = []
    for row in timeline:
        f = row["frame_index"]
        if f in rotations:
            R, c, status = rotations[f], centers[f], measured
        elif f < keyframes[0] or f > keyframes[-1]:
            k = keyframes[0] if f < keyframes[0] else keyframes[-1]
            R, c, status = rotations[k], centers[k], "held"
        else:
            j = int(np.searchsorted(keyframes, f))
            a, b = keyframes[j - 1], keyframes[j]
            t = (f - a) / (b - a)
            R, c, status = _slerp(rotations[a], rotations[b], t), (1 - t) * centers[a] + t * centers[b], "interpolated"
        rows.append({"frame": f + 1, "timestamp_us": row["timestamp_ns"] / 1000, "position_world": c.tolist(),
                     "R_world_rgb": R.tolist(), "projection": projection, "pose_source": source,
                     "pose_status": status, "confidence": None})
    return rows


class LazyFrames(dict):
    """Frames read when first needed: {frame_index: image} through load(frame_index)."""

    def __init__(self, load):
        super().__init__()
        self.load = load

    def __missing__(self, f):
        self[f] = self.load(f)
        return self[f]


def fuse_objects(geometry_dir, tracks_dir, frames_dir, output_dir, config, reuse=None):
    """Masks + depth + cameras -> the evidence's objects, in six steps: a 3D record per track (_track_record:
    its world points, appearance, motion or static box), names that tile the support set aside (_surface_names),
    one physical object's candidates merged
    (_merge_candidates), boxes seen from one viewpoint extended down to their support (_extend_down), then the
    objects and observations written as evidence (_evidence_objects, _observations).
    reuse: an earlier fuse output of the same geometry, frames and settings whose tracks come first here
    (find_objects joins new tracks after them); those records are read back instead of computed, same result."""
    started = time.time()
    out = Path(output_dir)
    (out / "appearance").mkdir(parents=True, exist_ok=False)
    geometry_dir, tracks_dir, frames_dir = Path(geometry_dir), Path(tracks_dir), Path(frames_dir)
    tracks = contract.read_json(tracks_dir / "tracks.json")
    ctx = _context(geometry_dir, frames_dir, tracks, config, out)
    candidates, reused = _track_records(ctx, tracks, tracks_dir, geometry_dir, frames_dir, reuse)
    _surface_names(candidates, ctx)
    merged = _merge_candidates(candidates, ctx.cfg)
    if ctx.one_view:
        _extend_down(candidates, ctx)
    timeline = {r["frame_index"]: r["timestamp_ns"] for r in contract.read_jsonl(frames_dir.parent / "timeline.jsonl")}
    evaluated = tracks["evaluated_frames"]
    objects, unresolved, appearance = _evidence_objects(candidates, ctx, timeline, evaluated)
    observations = _observations(tracks, candidates, merged, evaluated, timeline)
    for tid in list(appearance):
        crop = appearance[tid]["best_view"].get("crop")
        if crop and tid not in objects:
            (out / "appearance" / f"{tid}.jpg").unlink()
            appearance[tid]["best_view"].pop("crop")
    rec, world, cfg = ctx.rec, ctx.rec.world, ctx.cfg
    contract.write_json(out / "objects.json", objects)
    contract.write_json(out / "unresolved_tracks.json", unresolved)
    contract.write_json(out / "object_appearance.json", appearance)
    contract.write_jsonl(out / "observations.jsonl", observations)
    contract.write_json(out / "world.json", world)
    np.savez(out / "cameras.npz", keyframes=np.array(rec.keyframes), R_wc=rec.R_wc, C=rec.C, processed_hw=np.array([rec.h, rec.w]))
    universe = sorted(candidates, key=int)
    report = {"config": cfg, "tracks": len(candidates), "resolved": len(objects), "unresolved": len(unresolved),
              "unresolved_reasons": {r: sum(u["reason"].split(":")[0] == r for u in unresolved)
                                     for r in sorted({u["reason"].split(":")[0] for u in unresolved})},
              "candidate_ids": universe, "evaluated_frames": evaluated, "keyframes": rec.keyframes,
              "per_track": {t: {k: v for k, v in candidates[t].items()
                                if k not in ("appearance", "box", "motion") and not k.startswith("_")}
                               | ({"motion": {k: candidates[t]["motion"].get(k) for k in ("displacement_m", "rest_frames", "free_space",
                                                                                           "image_displaced", "along_ray")}}
                                  if "motion" in candidates[t] else {}) for t in universe},
              "reused_track_records": reused, "world": world, "wall_seconds": round(time.time() - started, 2)}
    contract.write_json(out / "fusion_report.json", report)
    summary = {"tracks": len(candidates), "resolved": len(objects), "unresolved": len(unresolved),
               "merged_candidates": len(merged), "camera_motion": world["camera_motion"],
               "up_status": rec.up_status, "scale_status": world["scale_status"],
               "camera_height_median_m": round(world["camera_height_above_floor_m"]["median"], 3)}
    return contract.tool_result("success", {name: out / name for name in
                                            ("objects.json", "unresolved_tracks.json", "object_appearance.json",
                                             "observations.jsonl", "world.json", "cameras.npz", "fusion_report.json")},
                                summary, ctx.warnings)


def _context(geometry_dir, frames_dir, tracks, config, out):
    """What every step reads: the settings scaled to the scene, the reconstruction, the frames, the support plane
    and the pixel <-> world helpers."""
    import cv2
    cfg = {**DEFAULTS, **(config or {})}
    rng = np.random.default_rng(cfg["seed"])
    rec = reconstruction(geometry_dir, frames_dir.parent, cfg, rng)
    # Thresholds that count views or bound heights follow the input: one image or a fixed camera gives one
    # viewpoint (more frames from it add no geometry; an object seen in few of them is marked uncertain), and a
    # scene filmed from high up (a raised camera) holds objects above an indoor ceiling height.
    cfg["min_views"] = min(cfg["min_views"], len(tracks["evaluated_frames"])) \
        if rec.world["camera_motion"] == "moving" else 1
    cfg.update({k: cfg[k] * rec.scene_scale for k in METRIC_KEYS})
    cfg["low_depth_confidence"] = rec.world["low_depth_confidence_cut"]
    cfg["max_height_m"] = max(cfg["max_height_m"], 2 * rec.world["camera_height_above_floor_m"]["median"])
    frames = LazyFrames(lambda f: cv2.imread(str(frames_dir / f"{f:06d}.jpg"))[:, :, ::-1])
    frame_to_key = {f: i for i, f in enumerate(rec.keyframes)}
    w, h, W, H = rec.w, rec.h, rec.W, rec.H
    # The support plane (z = 0 after normalization, or the given world's floor) and the area it covers.
    support_z = float(rec.world["floor"].get("floor_height", 0.0)) if rec.up_status == "reference" else 0.0
    near = rec.scene_points[np.abs(rec.scene_points[:, 2] - support_z) < 0.02 * rec.scene_scale]
    support_area = (float(np.prod(np.ptp(np.percentile(near[:, :2], [2, 98], axis=0), axis=0)))
                    if len(near) >= 100 else np.inf)

    def pixels(uv):
        """Canonical pixel coordinates -> (column, row) on the depth grid."""
        col = np.clip(np.round((uv[:, 0] + 0.5) * w / W - 0.5).astype(int), 0, w - 1)
        row = np.clip(np.round((uv[:, 1] + 0.5) * h / H - 0.5).astype(int), 0, h - 1)
        return col, row

    def lift(f, uv):
        """Canonical pixel coordinates of frame f -> world points through that frame's depth (NaN if invalid)."""
        out_points = np.full((len(uv), 3), np.nan)
        k = frame_to_key.get(f)
        if k is None or not len(uv):
            return out_points
        col, row = pixels(uv)
        ok = rec.valid[k][row, col]
        cam = unproject(rec.K_inv, col[ok].astype(float), row[ok].astype(float), rec.depth[k][row[ok], col[ok]])
        out_points[ok] = rec.to_world(cam @ rec.R_wc[k].T + rec.C[k])
        return out_points

    def at_depth(f, uv, z):
        """Canonical pixel uv (continuous) of frame f at camera depth z -> world point."""
        k = frame_to_key[f]
        col, row = (uv[0] + 0.5) * w / W - 0.5, (uv[1] + 0.5) * h / H - 0.5
        cam = unproject(rec.K_inv, np.array([col]), np.array([row]), np.array([z]))
        return rec.to_world(cam @ rec.R_wc[k].T + rec.C[k])[0]

    def depth_at(f, uv):
        """Raw camera depths of frame f at canonical pixels uv (NaN where there is none). Unfiltered: a small
        object's own pixels are often below the scene's confidence cut, and its median is what is wanted."""
        k = frame_to_key.get(f)
        if k is None:
            return np.full(len(uv), np.nan)
        col, row = pixels(uv)
        return np.where(rec.depth[k][row, col] > 0, rec.depth[k][row, col], np.nan)

    return SimpleNamespace(
        cfg=cfg, rng=rng, rec=rec, out=out, frames=frames,
        gray=LazyFrames(lambda f: cv2.cvtColor(frames[f], cv2.COLOR_RGB2GRAY)),
        frame_to_key=frame_to_key, iy=grid_index(h, H), ix=grid_index(w, W), W=W, H=H,
        mask_frames=set(tracks["evaluated_frames"][:: max(1, -(-len(tracks["evaluated_frames"]) // 30))]),
        support_z=support_z, support_area=support_area, lift=lift, at_depth=at_depth, depth_at=depth_at,
        fixed=rec.world["camera_motion"] == "static", one_view=rec.world["camera_motion"] in ("static", "single"),
        warnings=rec.warnings)


def _track_records(ctx, tracks, tracks_dir, geometry_dir, frames_dir, reuse):
    """{track_id: record} for every track (prompt by prompt, so one prompt's label maps are in memory at a time),
    the leading prompts read back from `reuse` when its records are of the same inputs; also returns how many
    track records were reused. The records are saved for the next reuse."""
    cfg, out = ctx.cfg, ctx.out
    by_prompt = {}
    for track in tracks["tracks"]:
        by_prompt.setdefault(track["prompt_index"], []).append(track)
    prompt_groups, reused, first_warning = sorted(by_prompt.items()), 0, len(ctx.warnings)
    inputs = hashlib.sha256()                # contents, so a copied perception (another run directory) still matches
    for path in [geometry_dir / "geometry.npz", geometry_dir / "model_info.json", frames_dir.parent / "video_info.json",
                 frames_dir.parent / "timeline.jsonl", *([Path(cfg["given_cameras"])] if cfg.get("given_cameras") else []),
                 *sorted(frames_dir.glob("*.jpg"))]:
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 24), b""):
                inputs.update(block)
    context = hashlib.sha256(json.dumps(
        {"cfg": {k: v for k, v in cfg.items() if k != "new_track_ids_after"}, "inputs": inputs.hexdigest(),
         "code": Path(__file__).read_text()}, sort_keys=True, default=str).encode()).hexdigest()
    hashes = {t["track_id"]: hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest() for t in tracks["tracks"]}
    candidates, decisions = {}, []
    if reuse and (Path(reuse) / "track_records.pkl.gz").is_file():
        with gzip.open(Path(reuse) / "track_records.pkl.gz", "rb") as stream:
            cache = pickle.load(stream)
        # Usable when its tracks are exactly the leading prompts here, unchanged (the random draws then continue
        # from where they stopped).
        while reused < len(prompt_groups) and all(hashes[t["track_id"]] == cache["hashes"].get(t["track_id"])
                                                  for t in prompt_groups[reused][1]):
            reused += 1
        if cache["context"] != context or {t["track_id"] for _, g in prompt_groups[:reused] for t in g} != set(cache["hashes"]):
            reused = 0
        if reused:
            candidates.update(cache["records"])
            decisions.extend(cache["decisions"])
            ctx.warnings.extend(cache["warnings"])
            ctx.rng.bit_generator.state = cache["rng_state"]
            for tid, record in candidates.items():
                crop = (record["appearance"] or {}).get("best_view", {}).get("crop")
                if crop and (Path(reuse) / "appearance" / f"{tid}.jpg").is_file():
                    shutil.copy2(Path(reuse) / "appearance" / f"{tid}.jpg", out / "appearance" / f"{tid}.jpg")
                elif crop:
                    record["appearance"]["best_view"].pop("crop")
    for prompt_index, prompt_tracks in prompt_groups[reused:]:
        labels = contract.read_labels(tracks_dir / f"labels_{prompt_index:03d}.npz")
        if labels.shape[1:] != (ctx.H, ctx.W):
            raise ValueError(f"Mask grid {labels.shape[1:]} differs from the canonical frames {(ctx.H, ctx.W)}")
        for track in prompt_tracks:
            candidates[track["track_id"]] = _track_record(ctx, track, labels)
            decisions.append(track["track_id"])
        del labels
    with gzip.open(out / "track_records.pkl.gz", "wb", compresslevel=1) as stream:
        pickle.dump({"context": context, "hashes": hashes, "records": candidates, "decisions": decisions,
                     "warnings": ctx.warnings[first_warning:], "rng_state": ctx.rng.bit_generator.state}, stream, protocol=5)
    return candidates, sum(len(g) for _, g in prompt_groups[:reused])


def _track_record(ctx, track, labels):
    """One track -> its record: world points per detected keyframe (views), the masks' image positions on a fixed
    camera (obs), appearance, coarse masks for merging, then motion (a moving or a fixed camera) or a static box,
    or the reason it stays unresolved."""
    import cv2
    cfg, rng, rec = ctx.cfg, ctx.rng, ctx.rec
    world = rec.world
    tid, value = track["track_id"], track["label_value"]
    views, colors, best, obs = [], [], None, []
    for det in track["detections"]:
        f = det["frame_index"]
        if ctx.fixed and f in ctx.frame_to_key:
            # Where the whole mask is on this frame (the image motion of a fixed camera).
            ys, xs = np.nonzero(labels[f] == value)
            if len(xs):
                box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                obs.append({"frame_index": f, "uv": (float(xs.mean()), float(ys.mean())), "bbox": box,
                            "clipped": [box[0] <= 1, box[1] <= 1, box[2] >= ctx.W - 1, box[3] >= ctx.H - 1],
                            "area": int(len(xs)), "z": None, "conf": None})
        mask = erode(labels[f] == value, cfg["erode_px"])
        if mask.sum() >= 20:
            ys, xs = np.nonzero(mask)
            pick = rng.choice(len(xs), min(300, len(xs)), replace=False)
            colors.append(ctx.frames[f][ys[pick], xs[pick]])
            x0, y0, x1, y1 = det["bbox_xyxy"]
            border = x0 <= 1 or y0 <= 1 or x1 >= ctx.W - 1 or y1 >= ctx.H - 1
            score = len(xs) * (0.25 if border else 1.0)
            if best is None or score > best["score"]:
                best = {"score": score, "frame_index": f, "bbox_xyxy": det["bbox_xyxy"]}
        if f not in ctx.frame_to_key:
            continue
        k = ctx.frame_to_key[f]
        small = mask[np.ix_(ctx.iy, ctx.ix)] & rec.valid[k]
        idx = np.flatnonzero(small)
        if len(idx) < cfg["min_view_pixels"]:
            # A small or thin object loses its pixels to erosion and depth edges: use its raw mask on
            # confident depth and keep the nearer half (the object, not what is seen around its edges).
            raw = (labels[f] == value)[np.ix_(ctx.iy, ctx.ix)] & (rec.conf[k] >= world["depth_confidence_cut"]) & \
                (rec.depth[k] > 0)
            idx = np.flatnonzero(raw)
            if len(idx) < cfg["min_view_pixels_small"]:
                continue
            near = rec.depth[k].ravel()[idx]
            idx = idx[near <= np.median(near)]
        confidence = float(np.median(rec.conf[k].ravel()[idx]))
        d = rec.depth[k].ravel()
        if ctx.one_view:
            # One viewpoint: the whole mask is the object's silhouette. Its rim, which erosion and depth
            # edges removed above, is lifted too, at depths kept inside the interior's range (depth at a rim
            # mixes in the background), so small objects keep their full width and height.
            lo, hi = np.percentile(d[idx], [2, 98])
            idx = np.flatnonzero((labels[f] == value)[np.ix_(ctx.iy, ctx.ix)] & (rec.depth[k] > 0))
            d = np.clip(d, lo, hi)
        idx = rng.choice(idx, min(cfg["points_per_view"], len(idx)), replace=False)
        cam = unproject(rec.K_inv, rec.uu.ravel()[idx], rec.vv.ravel()[idx], d[idx])
        views.append({"frame_index": f, "points": rec.to_world(cam @ rec.R_wc[k].T + rec.C[k]),
                      "confidence": confidence, "range": float(np.median(np.linalg.norm(cam, axis=1)))})
        if obs and obs[-1]["frame_index"] == f:
            obs[-1].update(z=float(np.median(cam[:, 2])), conf=views[-1]["confidence"])
    record = {"track_id": tid, "prompt": track["prompt"], "detections": len(track["detections"]),
              "first_detected_frame": track["detections"][0]["frame_index"], "views_with_depth": len(views),
              "mean_score": float(np.mean([d["score"] for d in track["detections"]]))}
    appearance = None
    if colors:
        c = np.concatenate(colors)
        appearance = {"median_srgb": (np.median(c, axis=0) / 255).tolist(), "sample_count": int(len(c)),
                      "segmentation_views": len(colors), "source": "masked_video_median (eroded SAM 3 masks)",
                      "best_view": {"frame_index": best["frame_index"], "bbox_xyxy": best["bbox_xyxy"]}}
        if RECIPE_ALIASES.get(track["prompt"]) in cfg["picture_recipes"]:
            f = best["frame_index"]
            x0, y0, x1, y1 = best["bbox_xyxy"]
            crop = ctx.frames[f][y0:y1, x0:x1].copy()
            crop[~(labels[f][y0:y1, x0:x1] == value)] = np.median(c, axis=0).astype(np.uint8)
            cv2.imwrite(str(ctx.out / "appearance" / f"{tid}.jpg"), crop[:, :, ::-1])
            appearance["best_view"]["crop"] = f"appearance/{tid}.jpg"
    record["appearance"] = appearance
    # Coarse masks for telling whether two tracks are one object, on frames of a grid every track shares (so two
    # tracks are compared on the same frames; a short track off the grid keeps its own), and where each mask is
    # (its box center and size) on every detected frame.
    step, detected = max(1, round(max(ctx.H, ctx.W) / 160)), [d["frame_index"] for d in track["detections"]]
    picks = [f for f in detected if f in ctx.mask_frames] or detected[:: max(1, len(detected) // 3)]
    record["_masks"] = {f: (labels[f] == value)[::step, ::step] for f in picks}
    record["_where"] = {d["frame_index"]: ((d["bbox_xyxy"][0] + d["bbox_xyxy"][2]) / 2,
                                           (d["bbox_xyxy"][1] + d["bbox_xyxy"][3]) / 2, np.sqrt(d["mask_area_px"]))
                        for d in track["detections"]}
    record["_area"] = float(sum(d["mask_area_px"] for d in track["detections"]))
    record["_hist"] = color_histogram(np.concatenate(colors)) if colors else None
    motion = None
    if cfg["dynamic"] and len(views) >= cfg["min_views"] and world["camera_motion"] == "moving":
        motion = track_motion(views, rec, cfg, np.random.default_rng(cfg["seed"] + int(tid)),
                              lambda k: labels[rec.keyframes[k]][np.ix_(ctx.iy, ctx.ix)] > 0)
    elif cfg["dynamic"] and ctx.fixed and views and len(obs) >= 3:
        moved = fixed_camera_motion(obs, views, ctx.lift, ctx.at_depth, ctx.depth_at, ctx.gray,
                                    LazyFrames(lambda f, v=value, l=labels: l[f] == v), cfg,
                                    np.random.default_rng(cfg["seed"] + int(tid)), ctx.support_z)
        if moved:
            motion = _fixed_camera_track(moved, cfg)
    if motion:
        record.update(status="resolved", box=motion["box"], motion=motion,
                      points=int(sum(len(v["points"]) for v in views)), views_with_depth=len(views),
                      centroid_drift_m=motion["displacement_m"], drift_tolerance_m=motion["tolerance"],
                      depth_confidence_median=float(np.median([v["confidence"] for v in views])))
        return record
    return _static_record(ctx, record, views)


def _fixed_camera_track(moved, cfg):
    """fixed_camera_motion's result as a motion record: the box fitted on the reference frame, moved along the
    track and turned by the chained rotations."""
    box0 = fit_box(moved["reference_view"]["points"], cfg)
    box0["size"] = (np.asarray(box0["size"]) * moved["scale"]).tolist()
    offset = (np.asarray(box0["center"]) - moved["anchor"]) * moved["scale"]
    shift = lambda i: moved["positions"][i] + offset
    box0["center"] = shift(0).tolist()
    # The box was fitted on the reference frame: its orientation there is the fitted yaw.
    R_ref = moved["rotations"][moved["reference"]]
    quats = [rotation_wxyz(R @ R_ref.T @ yaw_matrix(box0["yaw"])) for R in moved["rotations"]]
    return {"box": box0, "kind": "fixed_camera", "reference": moved["reference"],
            "track": [(f, shift(i), "moving") for i, f in enumerate(moved["frames"])],
            "rotations_wxyz": quats,
            "placements": [box0, {**box0, "center": shift(len(moved["frames"]) - 1).tolist()}],
            "displacement_m": moved["displacement_m"], "tolerance": moved["tolerance"],
            "rest_frames": [], "free_space": None, "depth_held_frames": moved["depth_held"],
            "image_displaced": moved["image_displaced"], "along_ray": moved["along_ray"],
            "depth_from_apparent_size_frames": moved["depth_scaled"],
            "rotation_held_steps": moved["held_steps"], "rotation_steps": moved["steps"]}


def _static_record(ctx, record, views):
    """A track that did not move: its views' dominant location (views of another object set aside), the box of
    their points, or the reason it stays unresolved (no dominant location, too few views, drift, the support
    surface itself, an implausible height)."""
    cfg, rng = ctx.cfg, ctx.rng
    tid = record["track_id"]
    if len(views) >= cfg["min_views"]:
        cents = np.array([np.median(v["points"], axis=0) for v in views])
        sizes = [np.ptp(np.percentile(v["points"], [5, 95], axis=0), axis=0).max() for v in views]
        tolerance = max(cfg["motion_tolerance_m"], cfg["motion_tolerance_rel"] * float(np.median(sizes)))
        clusters = view_clusters(cents, tolerance)
        if len(clusters) > 1:
            record["excluded_view_frames"] = sorted(views[i]["frame_index"] for c in clusters[1:] for i in c)
        if len(clusters[0]) >= cfg["dominant_cluster"] * len(views):
            if len(clusters) > 1:
                # Detections nearer in time to a set-aside view than to a kept one show the other object
                # (an identity switch): they leave the masks that decide whether two tracks are one object.
                kept = np.array([views[i]["frame_index"] for i in clusters[0]])
                other = np.array(record["excluded_view_frames"])
                record["_masks"] = {f: m for f, m in record["_masks"].items()
                                    if np.abs(kept - f).min() <= np.abs(other - f).min()}
            views = [views[i] for i in sorted(clusters[0])]
            cents = cents[sorted(clusters[0])]
        else:
            views = None
    if views is None:
        return {**record, "status": "unresolved",
                "reason": "inconsistent_world_position (no dominant location: identity switch or motion)"}
    if len(views) < cfg["min_views"]:
        return {**record, "status": "unresolved", "reason": "insufficient_3d_support"}
    half = len(views) // 2
    drift = float(np.linalg.norm(np.median(cents[:half], 0) - np.median(cents[half:], 0))) if half else 0.0
    record.update(centroid_drift_m=drift, drift_tolerance_m=tolerance, views_with_depth=len(views),
                  depth_confidence_median=float(np.median([v["confidence"] for v in views])))
    points = np.concatenate([v["points"] for v in views])
    box = fit_box(points, cfg)
    # The points of the views kept here (not of views set aside as another object) complete the box of a merged group.
    record["_points"] = points[rng.choice(len(points), min(5000, len(points)), replace=False)]
    size = np.asarray(box["size"], float)
    on_support = abs(box["center"][2] - size[2] / 2 - ctx.support_z) < max(0.1 * size[:2].max(), cfg["min_extent_m"])
    if on_support:      # see _surface_names; a height at the minimum extent is no measurable height, and most
        # points near the support make a piece flat even when depth noise stretches its box (grass, a rug)
        height = min(size[2], float(np.percentile(points[:, 2], 75)) - ctx.support_z)
        record["_on_support"] = {"area": float(size[0] * size[1]),
                                 "flat": bool(height < max(0.1 * size[:2].max(), 1.001 * cfg["min_extent_m"]))}
        if record["_on_support"]["flat"] and size[2] <= 1.001 * cfg["min_extent_m"]:
            # Thinner than depth resolves (a marking on the support, a sheet lying on it): the box keeps the minimum
            # height standing on the support, not half through it.
            box["center"][2] = ctx.support_z + size[2] / 2
            record["_on_support"]["thin"] = True
    if drift > tolerance:
        record.update(status="unresolved", reason="inconsistent_world_position (possible motion)")
    elif box["center"][2] > cfg["max_height_m"]:
        record.update(status="unresolved", reason=f"implausible_height_above_floor "
                                                  f"({box['center'][2]:.2f} m; likely depth error)")
    else:
        record.update(status="resolved", box=box, points=int(sum(len(v["points"]) for v in views)))
        if record.get("excluded_view_frames"):
            ctx.warnings.append(f"track {tid} ({record['prompt']}): kept the dominant location; frames "
                                f"{record['excluded_view_frames']} look like a different object")
    return record


def _surface_names(candidates, ctx):
    """A name whose flat pieces lying on the support together cover most of it names the surface or what it is
    made of (a table top, a rug, grass, the pavers or tiles of a floor; a single piece is the case of one): none of
    its pieces on the support is an object (a tile under an object rises with it). Pieces off the support stay."""
    pieces = {}
    for tid, r in candidates.items():
        if r["status"] == "resolved" and "_on_support" in r:
            pieces.setdefault(r["prompt"], []).append(tid)
    for prompt, tids in pieces.items():
        flat = sum(candidates[t]["_on_support"]["area"] for t in tids if candidates[t]["_on_support"]["flat"])
        if flat > ctx.cfg["support_area_fraction"] * ctx.support_area:
            for t in tids:
                candidates[t].update(status="unresolved", reason=f"support_surface (flat pieces named '{prompt}' "
                                                                 "cover most of the floor or table they lie on)")


def _merge_candidates(candidates, cfg):
    """One physical object, several candidates: two prompts on the same object (detected on the same frames with
    overlapping masks, or a part that keeps its place on its whole and whose box lies in its whole's; pieces of one
    name whose boxes are deep in each
    other), a revisit or a window boundary (never on the same frame; each center inside the other's box, or inside a
    moved object's rest box, or, under names of one kind of thing, within revisit_margin_m of it; similar colors). Names do not need to match (open vocabularies say "cup" and "mug"), nor
    motion (one prompt's track can start once the object came to rest). Tracks of the vocabulary before those of proposals (whose scores are on another scale), then the most seen
    (mask area: the longer track, the whole over its parts); each group keeps the first, the others become unresolved
    as merged_into. Returns {merged track: kept track}."""
    later = cfg.get("new_track_ids_after")
    resolved = sorted((t for t in candidates if candidates[t]["status"] == "resolved"),
                      key=lambda t: (later is not None and int(t) > later, -candidates[t]["_area"], int(t)))
    placements = lambda r: r["motion"]["placements"] if "motion" in r else [r["box"]]

    def attached(ra, rb):
        """On every frame holding both, the smaller mask keeps its place on the larger one (offsets in units of the
        larger mask's size): one body (a print on a bag, a wheel on a toy), not a part moving in or out of its
        holder (a drawer, a released block)."""
        small, large = sorted((ra, rb), key=lambda r: r["_area"] / max(len(r["_where"]), 1))
        frames = sorted(set(small["_where"]) & set(large["_where"]))
        offsets = np.array([np.subtract(small["_where"][f][:2], large["_where"][f][:2]) / max(large["_where"][f][2], 1.0)
                            for f in frames])
        return len(frames) < 2 or float(np.percentile(np.linalg.norm(offsets - np.median(offsets, axis=0), axis=1),
                                                       90)) <= cfg["attached_rel"]

    def grown(box, small=False):
        """The box grown by the revisit margin on every side; for pieces (small) by at most half its own size."""
        margin = cfg["revisit_margin_m"]
        margin = min(margin, 0.5 * float(np.max(box["size"]))) if small else margin
        return {**box, "size": np.asarray(box["size"], float) + 2 * margin}

    def same_object(a, b):
        ra, rb = candidates[a], candidates[b]
        overlap = codetected_overlap(ra["_masks"], rb["_masks"], cfg["codetect_contained"])
        if overlap is not None:
            shared, iou = overlap
            if shared < cfg["codetect_fraction"]:
                # Masks apart on the same frames but boxes deep in each other, under one name: pieces of one object
                # (a medal's ribbon and disc, a label on a lid; one prompt's masks never overlap), where two
                # objects could not both be. Different names stay apart (a thing in a bowl is not the bowl).
                inside = box_containment(ra["box"], rb["box"]) if ra["prompt"] == rb["prompt"] and \
                    "motion" not in ra and "motion" not in rb else 0.0
                return 2.0 - inside if inside >= cfg["piece_containment"] else DISTINCT
            if ("motion" in ra) != ("motion" in rb) and iou < cfg["same_mask_iou"] and not attached(ra, rb):
                return DISTINCT     # seen together, one moving in or out of the other: a part leaving its holder
            if iou < cfg["same_mask_iou"] and "motion" not in ra and "motion" not in rb and \
                    box_containment(ra["box"], rb["box"]) < cfg["part_containment"]:
                # One mask inside the other in the image only: a thing in front of or behind another (a bottle
                # before a monitor, a window behind it), not a part, whose box lies in its whole's.
                return DISTINCT
            return 1 - shared
        if "motion" in ra and "motion" in rb:
            return None
        ratio = min(max(center_ratio(x["center"], y), center_ratio(y["center"], x))
                    for x in placements(ra) for y in placements(rb))
        if ratio > 1.0:
            # A revisit seen from elsewhere, or after the camera's estimate drifted: a thin box (a door, a picture,
            # a monitor) and the drift leave each center just outside the other's box. Under names of one kind of
            # thing ("monitor" and "computer monitor"), each center may lie within revisit_margin_m of the other's
            # box; such a pair ranks after every strict one.
            if not cfg["revisit_margin_m"] or head_noun(ra["prompt"]) != head_noun(rb["prompt"]):
                return None
            ratio = min(max(center_ratio(x["center"], grown(y)), center_ratio(y["center"], grown(x)))
                        for x in placements(ra) for y in placements(rb))
            if ratio > 1.0:
                # Pieces of one object seen at different times (a door's upper and lower panels): their boxes,
                # grown by the margin, meet. Two such objects side by side are seen together somewhere, and that
                # DISTINCT pair keeps their groups apart.
                if "motion" in ra or "motion" in rb or \
                        box_containment(grown(ra["box"], small=True), grown(rb["box"], small=True)) <= 0.0:
                    return None
                ratio = 2.0
            ratio += 1.0
        if ra["_hist"] is not None and rb["_hist"] is not None and \
                np.minimum(ra["_hist"], rb["_hist"]).sum() < cfg["same_color_overlap"]:
            return None
        return 1.0 + ratio
    merged = group_same_objects(resolved, same_object) if cfg["merge_same_objects"] else {}
    groups = {}
    for tid, keep in merged.items():
        groups.setdefault(keep, []).append(tid)
    for keep, members in groups.items():
        for tid in members:
            candidates[tid].update(status="unresolved", reason=f"merged_into:{keep}")
        candidates[keep]["merged_track_ids"] = sorted(members, key=int)
        weights = {}            # the name that covers the most of the object (a whole outweighs its parts)
        for t in [keep, *members]:
            weights[candidates[t]["prompt"]] = weights.get(candidates[t]["prompt"], 0.0) + \
                candidates[t]["_area"] * candidates[t]["mean_score"]
        candidates[keep]["names"] = sorted(weights, key=lambda n: -weights[n])
        if "motion" not in candidates[keep]:
            # Static: the members saw the same object from other times or prompts; their points complete the box.
            points = np.concatenate([candidates[t]["_points"] for t in [keep, *members]
                                     if "motion" not in candidates[t] and "_points" in candidates[t]])
            candidates[keep]["box"] = fit_box(points, cfg)
    return merged


def _extend_down(candidates, ctx):
    """One viewpoint looking down on an object sees its top, so its box ends at the visible surface and floats
    above what holds it. A static box seen from at least `extend_down_min_deg` above is extended down to what is
    below its footprint (the support plane, or another box's top) when the gap is plausible for a resting object:
    the center then sits at the object's middle, not its lid. Seen from the side, the silhouette already shows the
    object's full height, and what hangs or is held in the air keeps it. A moving object seen by a fixed camera is
    extended the same way where its box was measured (its reference frame, where it was still), on every frame."""
    cfg = ctx.cfg
    statics = {t: candidates[t]["box"] for t in candidates
               if candidates[t]["status"] == "resolved" and "motion" not in candidates[t]}
    camera = ctx.rec.to_world(ctx.rec.C[0])

    def gap_below(tid, center, size):
        """(gap, what is below) when the box at center would rest on something below it, else None."""
        cx, cy, cz = (float(v) for v in center)
        sx, sy, sz = (float(v) for v in size)
        ray = np.array([cx, cy, cz]) - camera
        if np.degrees(np.arcsin(-ray[2] / max(np.linalg.norm(ray), 1e-9))) < cfg["extend_down_min_deg"]:
            return None
        bottom, below, on = cz - sz / 2, ctx.support_z, "support surface"
        for other, ob in statics.items():
            if other == tid:
                continue
            ocx, ocy, ocz = (float(v) for v in ob["center"])
            osx, osy, osz = (float(v) for v in ob["size"])
            top = ocz + osz / 2
            if abs(ocx - cx) <= osx / 2 and abs(ocy - cy) <= osy / 2 and below < top <= bottom + 0.01:
                below, on = top, other
        gap = bottom - below
        return (gap, on) if 0.005 < gap <= max(3 * sz, max(sx, sy)) else None

    for tid, box in statics.items():
        found = gap_below(tid, box["center"], box["size"])
        if found:
            box["center"] = np.asarray(box["center"], float) - [0.0, 0.0, found[0] / 2]
            box["size"] = np.asarray(box["size"], float) + [0.0, 0.0, found[0]]
            box["rests_on"], box["extended_m"] = found[1], round(found[0], 4)
    for tid, r in candidates.items():
        m = r.get("motion")
        if r["status"] != "resolved" or not m or m.get("kind") != "fixed_camera":
            continue
        found = gap_below(tid, m["track"][m["reference"]][1], m["box"]["size"])
        if found:
            down, grow = np.array([0.0, 0.0, found[0] / 2]), np.array([0.0, 0.0, found[0]])
            m["box"].update(center=(np.asarray(m["box"]["center"]) - down).tolist(),
                            size=(np.asarray(m["box"]["size"]) + grow).tolist(), rests_on=found[1],
                            extended_m=round(found[0], 4))
            m["track"] = [(f, np.asarray(c) - down, state) for f, c, state in m["track"]]
            m["placements"] = [{**p, "center": (np.asarray(p["center"]) - down).tolist(), "size": m["box"]["size"]}
                               for p in m["placements"]]


def _evidence_objects(candidates, ctx, timeline, evaluated):
    """The evidence's objects (resolved candidates with their box or motion, evidence notes and level),
    unresolved candidates with their reason, and the appearance of every candidate."""
    cfg = ctx.cfg
    objects, unresolved, appearance = {}, [], {}
    for tid in sorted(candidates, key=int):
        rec = candidates[tid]
        group = [tid, *rec.get("merged_track_ids", [])]
        refs = {"first_detected_frame": min(candidates[t]["first_detected_frame"] for t in group),
                "detected_frames": sum(candidates[t]["detections"] for t in group),
                "views_with_depth": rec["views_with_depth"], "prompt": rec["prompt"]}
        if rec["appearance"]:
            appearance[tid] = rec["appearance"]
        if rec["status"] != "resolved":
            own = rec.get("box") if rec["reason"].startswith("merged_into") else None
            unresolved.append({"id": tid, "category": rec["prompt"], "reason": rec["reason"], "observation_refs": refs,
                               **({"duplicate_iou": rec["duplicate_iou"]} if "duplicate_iou" in rec else {}),
                               # a part found on an object (a lid, a door) keeps its own measured box
                               **({"own_box": {"center": np.round(own["center"], 3).tolist(),
                                               "size": np.round(own["size"], 3).tolist(),
                                               "yaw_deg": round(float(np.degrees(own["yaw"])), 1)}} if own else {})})
            continue
        box, names = rec["box"], rec.get("names", [rec["prompt"]])
        sx, sy, sz = (float(v) for v in box["size"])
        objects[tid] = {
            "instance_id": tid, "instance_name": f"{names[0].replace(' ', '_')}_{tid}", "category": names[0],
            "prototype_name": names[0], "proxy_recipe": RECIPE_ALIASES.get(names[0], names[0]),
            "motion_type": "static", "canonical_pose": {"up_vector": [0, 0, 1], "front_vector": [0, -1, 0]},
            "bounds": [-sx / 2, sx / 2, -sy / 2, sy / 2, -sz / 2, sz / 2],
            "poses": [[0, *(float(v) for v in box["center"]), float(np.cos(box["yaw"] / 2)), 0.0, 0.0,
                       float(np.sin(box["yaw"] / 2))]],
            "evidence": {"identity_source": "sam3_video_tracker", "geometry_source": "masked_depth_fusion",
                         "pose_source": "multi_view_static_fusion",
                         "rotation_status": "canonical_assumption (gravity-aligned, PCA yaw; front unobserved)",
                         "dimension_status": ("silhouette_extent (one viewpoint: the extent along the view is the "
                                              "visible surface's)" if ctx.one_view else "visible_surface_extent")
                                             + ("_min_clamped" if box["clamped"] else "")
                                             + (f"; bottom extended {box['extended_m']} m down to rest on {box['rests_on']}"
                                                if box.get("extended_m") else ""),
                         "confidence": rec["mean_score"], "confidence_kind": "mean SAM 3 detection score (uncalibrated)",
                         "views_used": rec["views_with_depth"], "points_used": rec["points"],
                         "centroid_drift_m": rec["centroid_drift_m"],
                         "depth_confidence_median": rec["depth_confidence_median"],
                         "low_depth_confidence": rec["depth_confidence_median"] < cfg["low_depth_confidence"],
                         "excluded_view_frames": rec.get("excluded_view_frames", []),
                         "merged_track_ids": rec.get("merged_track_ids", []), "names": names, **refs}}
        # Certain: seen often enough, confidently segmented, on trusted depth, at a measured size. Codification
        # builds every certain object as given and decides each uncertain one.
        doubts = [why for why, bad in (
            ("seen_in_few_frames", refs["detected_frames"] < min(cfg["certain_min_detections"], len(evaluated))),
            ("low_detection_score", rec["mean_score"] < cfg["certain_min_score"]),
            ("low_depth_confidence", objects[tid]["evidence"]["low_depth_confidence"]),
            ("size_at_minimum", box["clamped"]),
            ("flat_on_support (lies flat on the floor or table: a sheet, a mat, or a region or pattern of the surface "
             "itself)", rec.get("_on_support", {}).get("flat", False))) if bad]
        objects[tid]["evidence"].update(evidence_level="uncertain" if doubts else "certain", uncertain_because=doubts)
        if "motion" in rec:
            objects[tid].update(_motion_fields(rec["motion"], box, timeline, objects[tid]["evidence"]))
    return objects, unresolved, appearance


def _motion_fields(motion, box, timeline, evidence):
    """A moved object's poses on the timeline and its evidence notes (updated in place)."""
    rotation = [float(np.cos(box["yaw"] / 2)), 0.0, 0.0, float(np.sin(box["yaw"] / 2))]
    intervals = []
    for f, _, state in motion["track"]:
        if intervals and intervals[-1]["status"] == state:
            intervals[-1]["end_ns"] = timeline[f]
        else:
            intervals.append({"start_ns": timeline[f], "end_ns": timeline[f], "status": state})
    rotations = motion.get("rotations_wxyz") or [rotation] * len(motion["track"])
    if motion.get("kind") == "fixed_camera":
        evidence.update(
            pose_source=f"mask_centroid_track (fixed camera: per-frame mask centroid ray; depth measured on "
                        f"slow, confident frames, from the mask's apparent size on "
                        f"{motion.get('depth_from_apparent_size_frames', 0)} others and held on "
                        f"{motion['depth_held_frames']})",
            rotation_status=f"measured: chained from surface features tracked between frames "
                            f"({motion['rotation_held_steps']} of {motion['rotation_steps']} steps held)",
            pose_intervals=intervals, motion_tolerance_m=motion["tolerance"])
    else:
        evidence.update(
            pose_source="masked_depth_track (rests + free-space departure check)",
            rotation_status="held (yaw fitted at the best-supported rest; rotation not measured)",
            pose_intervals=intervals, rest_frames=motion["rest_frames"], free_space=motion["free_space"],
            motion_tolerance_m=motion["tolerance"])
    return {"motion_type": "dynamic",
            "poses": [[timeline[f], *(float(v) for v in center), *q] for (f, center, _), q in zip(motion["track"], rotations)]}


def _observations(tracks, candidates, merged, evaluated, timeline):
    """Detections only: a candidate without a row on an evaluated frame had no mask there. Rows of a merged track
    carry the kept id and their own track_id."""
    observations = []
    detected = {(t["track_id"], d["frame_index"]): d for t in tracks["tracks"] for d in t["detections"]}
    universe = sorted(candidates, key=int)
    for f in evaluated:
        for tid in universe:
            d = detected.get((tid, f))
            if d:
                observations.append({"frame_index": f, "timestamp_ns": timeline[f], "object_id": merged.get(tid, tid),
                                     "state": "detected", "bbox_xyxy": d["bbox_xyxy"], "mask_area_px": d["mask_area_px"],
                                     "score": d["score"], "source": "sam3_video_tracker",
                                     **({"track_id": tid} if tid in merged else {})})
    return observations
