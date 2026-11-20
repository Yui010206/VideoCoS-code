"""Deterministic gates on a built scene (inside Blender), and the repair tasks compiled from them.

Measurements first, with numbers, so a model's revision has something exact to fix (after 3DCodeVerse):
- silhouette: each object's rendered silhouette (a flat id render) against its SAM 3 mask on the comparison
  frames, as IoU and the width/height ratio of the two;
- contacts: surfaces of two objects (or an object and the environment) that cross, with the penetration depth,
  and active objects floating above whatever is below them;
- plan: the parts a build declared (A.declare_plan) exist and have the declared sizes;
- joints: each movable part swept through its limits, checked for collisions, for being attached and for
  sliding straight up off its object (a lid that lifts off is a separate body);
- proxies: plain boxes standing in for objects that are not boxes.
Repairs are ordered: 1 structural (penetration, unattached parts, parts hitting the environment or their root),
2 measured mismatches (silhouette, floating, sizes, passive movers, parts sweeping through other objects),
3 appearance and roles (proxies, loose objects left passive).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

PENETRATION_M = 0.008
FLOATING_M = 0.01
UNATTACHED_M = 0.03
SILHOUETTE_MIN_IOU = 0.5
LOOSE_MAX_M = 0.4           # a resting passive object smaller than this is probably loose
BOXY = ("box", "cube", "block", "brick", "book", "carton", "case", "crate", "cabinet", "drawer", "package",
        "container", "bin", "tray", "board", "shelf", "table", "desk", "counter", "label", "sticker", "bar", "mat")


def semantic_groups(scene):
    """{instance_id: [root mesh, its part meshes]} for the registered objects."""
    groups = {}
    for obj in scene.objects:
        if "instance_id" in obj and obj.type == "MESH":
            groups.setdefault(str(obj["instance_id"]), [obj])
    for obj in scene.objects:
        if "part_of" in obj and obj.type == "MESH" and str(obj["part_of"]) in groups:
            groups[str(obj["part_of"])].append(obj)
    return groups


def environment_meshes(scene):
    return [o for o in scene.objects if o.type == "MESH" and o.get("environment")]


def world_tree(objects, depsgraph):
    """One BVH tree over the world-space triangles of the given meshes, plus sample points on their surface
    (vertices and area-weighted points on the faces)."""
    verts, polys, offset = [], [], 0
    for obj in objects:
        mesh = obj.evaluated_get(depsgraph).to_mesh()
        M = obj.matrix_world
        verts += [M @ v.co for v in mesh.vertices]
        polys += [[offset + i for i in p.vertices] for p in mesh.polygons]
        offset += len(mesh.vertices)
        obj.evaluated_get(depsgraph).to_mesh_clear()
    if not polys:
        return None, np.zeros((0, 3))
    points = np.array([v[:] for v in verts])
    return BVHTree.FromPolygons(verts, polys, all_triangles=False), surface_samples(points, polys)


def world_tree_fast(objects, depsgraph, count=400, seed=0):
    """world_tree with the triangles read in bulk (loop_triangles) and the surface points drawn with numpy: the same
    tree and the same kind of samples, much faster on large meshes (completed SAM 3D objects)."""
    verts, tris, offset = [], [], 0
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        n = len(mesh.vertices)
        co = np.empty(3 * n, np.float32)
        mesh.vertices.foreach_get("co", co)
        m = np.array(obj.matrix_world)
        verts.append(co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3])
        t = np.empty(3 * len(mesh.loop_triangles), np.int32)
        mesh.loop_triangles.foreach_get("vertices", t)
        tris.append(t.reshape(-1, 3) + offset)
        offset += n
        evaluated.to_mesh_clear()
    if not offset or not sum(len(t) for t in tris):
        return None, np.zeros((0, 3))
    points, faces = np.vstack(verts).astype(np.float64), np.vstack(tris)
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    samples = points
    if area.sum() > 0:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(faces), count, p=area / area.sum())
        u, v = rng.random(count), rng.random(count)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        samples = np.vstack([points, a[pick] + (b[pick] - a[pick]) * u[:, None] + (c[pick] - a[pick]) * v[:, None]])
    return BVHTree.FromPolygons(points.tolist(), faces.tolist(), all_triangles=True), samples


def aabb(points):
    return points.min(0), points.max(0)


def _inside(tree, point, extent):
    """Ray parity along +X: inside a closed surface when the ray leaves it an odd number of times."""
    origin, direction, hits = Vector(point), Vector((1.0, 0.0, 0.0)), 0
    for _ in range(32):
        hit, _, _, dist = tree.ray_cast(origin, direction, extent)
        if hit is None:
            break
        hits += 1
        origin = hit + direction * 1e-5
    return hits % 2 == 1


def surface_samples(verts, polys, count=400, seed=0):
    """Points on a mesh's surface: its vertices plus points drawn on its triangles in proportion to area, so a
    thin part crossing a thin slab is caught even when no vertex lies inside the other body."""
    rng = np.random.default_rng(seed)
    tris = [(p[0], p[i], p[i + 1]) for p in polys for i in range(1, len(p) - 1)]
    if not tris:
        return verts
    a, b, c = (verts[np.array([t[k] for t in tris])] for k in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if area.sum() <= 0:
        return verts
    pick = rng.choice(len(tris), count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    points = a[pick] + (b[pick] - a[pick]) * u[:, None] + (c[pick] - a[pick]) * v[:, None]
    return np.vstack([verts, points])


def penetration(tree_a, verts_a, tree_b, verts_b, samples=300):
    """Depth (m) by which a's surface lies inside b, from a sample of a's surface points; 0 when none does."""
    lo_b, hi_b = aabb(verts_b)
    inside_box = np.all((verts_a >= lo_b - 1e-6) & (verts_a <= hi_b + 1e-6), axis=1)
    candidates = verts_a[inside_box]
    if not len(candidates) or not tree_a.overlap(tree_b):
        return 0.0
    if len(candidates) > samples:
        candidates = candidates[np.linspace(0, len(candidates) - 1, samples).astype(int)]
    extent = float(np.linalg.norm(hi_b - lo_b)) + 1.0
    depth = 0.0
    for p in candidates:
        if _inside(tree_b, p, extent):
            near = tree_b.find_nearest(Vector(p))
            if near[0] is not None:
                depth = max(depth, float(near[3]))
    return depth


def world_vertices(objects):
    """World positions of the meshes' vertices (codified meshes carry no modifiers)."""
    out = []
    for o in objects:
        co = np.empty(3 * len(o.data.vertices), np.float32)
        o.data.vertices.foreach_get("co", co)
        m = np.array(o.matrix_world)
        out.append(co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3])
    return np.vstack(out) if out else np.zeros((0, 3))


def footprint(verts, count=8):
    """Points (x, y) over the footprint of world vertices: its centre first, then its outline pulled halfway to the
    centre in `count` directions."""
    xy = np.asarray(verts)[:, :2]
    centre = xy.mean(0)
    if len(xy) > 4000:
        xy = xy[np.linspace(0, len(xy) - 1, 4000).astype(int)]
    angle = np.arctan2(*(xy - centre).T[::-1])
    reach = np.linalg.norm(xy - centre, axis=1)
    pts = [centre]
    for a in np.linspace(-np.pi, np.pi, count, endpoint=False):
        near = np.abs((angle - a + np.pi) % (2 * np.pi) - np.pi) < np.pi / count
        if near.any():
            i = np.flatnonzero(near)[np.argmax(reach[near])]
            pts.append(centre + 0.5 * (xy[i] - centre))
    return pts


def contacts(scene, groups, environment, active_ids, passive_ids=(), footprint_rays=False, fast=False):
    """Penetrating pairs, floating active objects and small loose objects left passive, at the first frame.
    passive_ids: passive objects without measured motion. Floating is measured under the centre of the bounding box;
    with footprint_rays under points spread over the footprint as well, and an object resting on any of them (an edge
    on a neighbour, a foot on a shelf) does not float. fast: the trees from world_tree_fast and a third of the
    surface points per pair."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    tree_of = world_tree_fast if fast else world_tree
    samples = 100 if fast else 300           # surface points tested per pair (fast: a third, for crowded rooms)
    trees = {uid: tree_of(objs, depsgraph) for uid, objs in groups.items()}
    env_tree = tree_of(environment, depsgraph) if environment else (None, None)
    boxes = {uid: aabb(v) for uid, (t, v) in trees.items() if t is not None}
    penetrations = []
    ids = sorted(boxes, key=lambda u: (not u.isdigit(), int(u) if u.isdigit() else 0, u))
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            (la, ha), (lb, hb) = boxes[a], boxes[b]
            if np.any(ha < lb - PENETRATION_M) or np.any(hb < la - PENETRATION_M):
                continue
            depth = max(penetration(trees[a][0], trees[a][1], trees[b][0], trees[b][1], samples),
                        penetration(trees[b][0], trees[b][1], trees[a][0], trees[a][1], samples))
            if depth >= PENETRATION_M:
                penetrations.append({"a": a, "b": b, "depth_m": round(depth, 4)})
        if env_tree[0] is not None:
            depth = penetration(trees[a][0], trees[a][1], env_tree[0], env_tree[1], samples)
            if depth >= PENETRATION_M:
                penetrations.append({"a": a, "b": "environment", "depth_m": round(depth, 4)})
    floating, loose_passive = [], []
    for uid in ids:
        lo, hi = boxes[uid]
        own = set(o.name for o in groups[uid])
        points = footprint(world_vertices(groups[uid])) if footprint_rays else [((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2)]
        gap, hit, obj = None, False, None
        for x, y in points:
            bottom = Vector((float(x), float(y), lo[2] - 0.001))
            ray_hit, location, _, _, ray_obj, _ = scene.ray_cast(depsgraph, bottom, Vector((0, 0, -1)), distance=10.0)
            ray_gap = float(bottom.z - location.z) if ray_hit and ray_obj.name not in own else None
            if ray_gap is not None and (gap is None or ray_gap < gap):
                gap, hit, obj = ray_gap, ray_hit, ray_obj
            elif obj is None:
                hit, obj = ray_hit, ray_obj
        # In a room (footprint_rays) furniture is passive, and it must not float either.
        if (uid in active_ids or (footprint_rays and uid in passive_ids)) and (gap is None or gap > FLOATING_M):
            floating.append({"id": uid, "gap_m": round(gap, 4) if gap is not None else None,
                             "above": obj.name if hit else "nothing"})
        # A small passive object resting on something else is probably loose (a cup, a key, a toy).
        supported = gap is not None and gap <= 0.02 and ("instance_id" in obj or "part_of" in obj or obj.get("environment"))
        if uid in passive_ids and supported and float(np.max(hi - lo)) < LOOSE_MAX_M:
            loose_passive.append({"id": uid, "size_m": round(float(np.max(hi - lo)), 3), "on": obj.name})
    return {"penetrations": penetrations, "floating": floating, "loose_passive": loose_passive,
            "threshold_m": PENETRATION_M}


def silhouettes(scene, groups, evidence, frames, out_dir, size=320):
    """Flat id renders at the comparison frames against the evidence masks: per-object IoU and the ratio of the
    rendered to the mask extents (width, height)."""
    masks_dir = Path(evidence) / "masks"
    frames = [f for f in frames if (masks_dir / f"{f:06d}.png").is_file()]
    if not frames or scene.camera is None:
        return None
    ids = [u for u in groups if u.isdigit()]
    index = {u: i + 1 for i, u in enumerate(ids)}
    r, d, vs = scene.render, scene.display.shading, scene.view_settings
    saved = (r.engine, r.resolution_percentage, r.film_transparent, r.image_settings.file_format,
             r.image_settings.color_mode, r.image_settings.color_depth, vs.view_transform,
             d.light, d.color_type, scene.display.render_aa, r.filepath, r.dither_intensity)
    # A scene's own look, exposure, gamma or compositor would change the id colours (every IoU then reads 0).
    saved_look = (vs.look, vs.exposure, vs.gamma, r.use_compositing, r.use_sequencer)
    colors = {o.name: tuple(o.color) for objs in groups.values() for o in objs}
    try:
        r.engine, r.film_transparent = "BLENDER_WORKBENCH", True
        r.resolution_percentage = max(1, min(100, round(100 * size / max(r.resolution_x, r.resolution_y))))
        r.image_settings.file_format, r.image_settings.color_mode, r.image_settings.color_depth = "PNG", "RGBA", "8"
        vs.view_transform = "Raw"
        vs.look, vs.exposure, vs.gamma, r.use_compositing, r.use_sequencer = "None", 0.0, 1.0, False, False
        d.light, d.color_type, scene.display.render_aa = "FLAT", "OBJECT", "OFF"
        r.dither_intensity = 0.0                       # exact 8-bit values: the id is in the red and green channels
        for u, objs in groups.items():
            for o in objs:
                o.color = ((index.get(u, 0) % 255) / 255.0, (index.get(u, 0) // 255) / 255.0, 0.0, 1.0)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        per_object, per_frame = {u: [] for u in ids}, {}
        for f in frames:
            scene.frame_set(f + 1)
            r.filepath = str(out_dir / f"id_{f:06d}.png")
            bpy.ops.render.render(write_still=True)
            img = bpy.data.images.load(r.filepath, check_existing=False)
            img.colorspace_settings.name = "Non-Color"          # raw values, no sRGB conversion on read
            w, h = img.size
            pix = np.array(img.pixels[:], np.float32).reshape(h, w, 4)[::-1]
            bpy.data.images.remove(img)
            rendered = np.where(pix[:, :, 3] > 0.5, np.round(pix[:, :, 0] * 255).astype(int)
                                + 255 * np.round(pix[:, :, 1] * 255).astype(int), 0)
            gt_img = bpy.data.images.load(str(masks_dir / f"{f:06d}.png"), check_existing=False)
            gt_img.colorspace_settings.name = "Non-Color"
            gw, gh = gt_img.size
            gt = np.round(np.array(gt_img.pixels[:], np.float32).reshape(gh, gw, 4)[::-1, :, 0] * 65535).astype(int)
            bpy.data.images.remove(gt_img)
            rows = np.clip(np.round((np.arange(h) + 0.5) * gh / h - 0.5), 0, gh - 1).astype(int)
            cols = np.clip(np.round((np.arange(w) + 0.5) * gw / w - 0.5), 0, gw - 1).astype(int)
            gt = gt[np.ix_(rows, cols)]
            per_frame[f] = {}
            for u in ids:
                mask, ours = gt == int(u), rendered == index[u]
                if not mask.any():
                    continue
                union = (mask | ours).sum()
                iou = float((mask & ours).sum() / union) if union else 0.0
                ratio = None
                if ours.any():
                    ys, xs = np.nonzero(ours)
                    gys, gxs = np.nonzero(mask)
                    ratio = [round(float(np.ptp(xs) + 1) / (np.ptp(gxs) + 1), 2), round(float(np.ptp(ys) + 1) / (np.ptp(gys) + 1), 2)]
                per_object[u].append(iou)
                per_frame[f][u] = {"iou": round(iou, 3), "size_ratio_wh": ratio}
        means = {u: round(float(np.mean(v)), 3) for u, v in per_object.items() if v}
        # An object placed right matches its mask on at least one comparison frame; on a long walkthrough it can be
        # occluded or off on another frame (camera drift), which is not the scene's fault, so the verdict uses the best.
        best = {u: round(float(np.max(v)), 3) for u, v in per_object.items() if v}
        return {"frames": frames, "per_object_mean_iou": means, "per_object_best_iou": best,
                "median_mean_iou": round(float(np.median(list(means.values()))), 3) if means else None,
                "median_best_iou": round(float(np.median(list(best.values()))), 3) if best else None,
                "below_min": sorted([u for u, v in best.items() if v < SILHOUETTE_MIN_IOU], key=int),
                "per_frame": per_frame, "min_iou": SILHOUETTE_MIN_IOU, "render_width_px": w}
    finally:
        (r.engine, r.resolution_percentage, r.film_transparent, r.image_settings.file_format,
         r.image_settings.color_mode, r.image_settings.color_depth, vs.view_transform,
         d.light, d.color_type, scene.display.render_aa, r.filepath, r.dither_intensity) = saved
        vs.look, vs.exposure, vs.gamma, r.use_compositing, r.use_sequencer = saved_look
        for objs in groups.values():
            for o in objs:
                o.color = colors[o.name]
        scene.frame_set(1)


def plan_contract(scene, groups):
    """Declared parts exist as declared (a hinge, slider or fixed part as a part of the object with that joint, a
    separate part as its own registered object) and have the declared sizes (max(1 cm, 10%) per axis)."""
    plan = json.loads(scene.get("codify_plan", "{}") or "{}")
    joint_of = {"hinge": "revolute", "slider": "prismatic", "fixed": "fixed"}
    missing, size_off, connection_off, checklist = [], [], [], {}
    for uid, item in plan.items():
        parts = {str(o["part_name"]): o for o in groups.get(uid, [])[1:]}
        for declared in item.get("parts", []):
            connection = declared.get("connection")
            if connection == "separate":
                own = groups.get(str(declared.get("object")), [])
                part = own[0] if own else None
                if part is None and declared["name"] in parts:
                    connection_off.append({"id": uid, "part": declared["name"], "declared": "separate",
                                           "built": "a part of the object"})
                    continue
            else:
                part = parts.get(declared["name"])
                if part is not None and connection in joint_of:
                    built = json.loads(part["joint"])["type"] if "joint" in part else "fixed"
                    if built != joint_of[connection]:
                        connection_off.append({"id": uid, "part": declared["name"], "declared": connection,
                                               "built": f"a {built} joint"})
            if part is None:
                missing.append({"id": uid, "part": declared["name"]})
                continue
            v = np.array([v.co[:] for v in part.data.vertices])
            built = np.ptp(v, axis=0)
            want = np.asarray(declared["size_m"], float)
            tolerance = np.maximum(0.01, 0.1 * want)
            if np.any(np.abs(np.sort(built) - np.sort(want)) > tolerance):
                size_off.append({"id": uid, "part": declared["name"], "declared_m": want.round(3).tolist(),
                                 "built_m": np.sort(built)[::-1].round(3).tolist()})
        if item.get("checklist"):
            checklist[uid] = item["checklist"]
    return {"declared_objects": len(plan), "missing_parts": missing, "size_off": size_off,
            "connection_off": connection_off, "checklist": checklist}


def joint_sweep(scene, groups, environment):
    """Every movable part at its lower, middle and upper limit: collisions with other objects and the environment
    (penetration >= threshold), and attachment at rest (nearest distance to the root <= 3 cm)."""
    from codify import assets
    depsgraph = bpy.context.evaluated_depsgraph_get()
    results = []
    others_cache = {}
    for uid, objs in groups.items():
        root, parts = objs[0], objs[1:]
        for part in parts:
            joint = json.loads(part["joint"])
            if joint["type"] == "fixed":
                continue
            rest = joint["value"]
            lo, hi = joint["limits"]
            record = {"id": uid, "part": str(part["part_name"]), "type": joint["type"], "limits": [lo, hi],
                      "collisions": [], "unattached": False, "lifts_off": False}
            root_tree, root_verts = world_tree([root], depsgraph)
            part_tree, part_verts = world_tree([part], depsgraph)
            if joint["type"] == "prismatic" and root_verts is not None and part_verts is not None:
                # Sliding straight up off the top of its object is lifting off: a lid or a cap, not a slider.
                axis = np.asarray(root.matrix_world.to_3x3() @ Vector(joint["axis"]))
                (_, _, part_low), (_, _, part_high) = aabb(part_verts)
                record["lifts_off"] = bool(abs(axis[2]) >= math.cos(math.radians(20)) * np.linalg.norm(axis) and
                                           part_low >= aabb(root_verts)[1][2] - max(0.01, 0.25 * (part_high - part_low)))
            if root_tree is not None and part_tree is not None:
                dist = min(float(root_tree.find_nearest(Vector(p))[3]) for p in part_verts[::max(1, len(part_verts) // 200)])
                record["rest_distance_to_root_m"] = round(dist, 4)
                record["unattached"] = dist > UNATTACHED_M
            if uid not in others_cache:
                others = {u: world_tree(g, depsgraph) for u, g in groups.items() if u != uid}
                if environment:
                    others["environment"] = world_tree(environment, depsgraph)
                others_cache[uid] = {u: t for u, t in others.items() if t[0] is not None}
            for value in sorted({lo, (lo + hi) / 2, hi}):
                assets.set_joint(part, value)
                bpy.context.view_layer.update()
                part_tree, part_verts = world_tree([part], depsgraph)
                if part_tree is None:
                    continue
                lo_p, hi_p = aabb(part_verts)
                for other, (other_tree, other_verts) in others_cache[uid].items():
                    lo_o, hi_o = aabb(other_verts)
                    if np.any(hi_p < lo_o - PENETRATION_M) or np.any(hi_o < lo_p - PENETRATION_M):
                        continue
                    depth = penetration(part_tree, part_verts, other_tree, other_verts)
                    if depth >= PENETRATION_M:
                        record["collisions"].append({"value": round(value, 3), "with": other, "depth_m": round(depth, 4)})
            assets.set_joint(part, rest)
            bpy.context.view_layer.update()
            results.append(record)
    return results


def box_proxies(groups, categories):
    """Objects whose only mesh is a plain box although their category is not box-like."""
    found = []
    for uid, objs in groups.items():
        if len(objs) > 1:
            continue
        mesh = objs[0].data
        category = categories.get(uid, "").lower()
        if len(mesh.vertices) == 8 and len(mesh.polygons) == 6 and not any(word in category for word in BOXY):
            found.append(uid)
    return found


def compile_repairs(check):
    """The ordered list of what to fix, with numbers; the first six go to the model."""
    m, repairs = check["metrics"], []
    for p in (m.get("contacts") or {}).get("penetrations", []):
        repairs.append({"priority": 1, "object": p["a"], "task": f"object {p['a']} penetrates {p['b']} by {p['depth_m'] * 1000:.0f} mm: "
                                                                  "move or shrink one until they only touch"})
    for j in m.get("joints") or []:
        if j.get("lifts_off"):
            repairs.append({"priority": 2, "object": j["id"], "task": f"part '{j['part']}' of object {j['id']} slides "
                            "straight up off its top: a piece that lifts off (a lid, a cap, a cover) is separate, its own "
                            "active body with its own id resting on the object, not a joint; keep the slider only if the "
                            "frames show a rail"})
        if j["unattached"]:
            repairs.append({"priority": 1, "object": j["id"], "task": f"part '{j['part']}' of object {j['id']} is "
                            f"{j['rest_distance_to_root_m'] * 1000:.0f} mm from its root at rest: attach it (pivot on the root)"})
        for c in j["collisions"]:
            if c["with"] == "environment" or c["with"] == j["id"]:
                repairs.append({"priority": 1, "object": j["id"], "task": f"part '{j['part']}' of object {j['id']} goes into "
                                f"{'the environment' if c['with'] == 'environment' else 'its own root'} at joint value "
                                f"{c['value']} ({c['depth_m'] * 1000:.0f} mm): fix the pivot, axis or limits"})
            else:
                repairs.append({"priority": 2, "object": j["id"], "task": f"part '{j['part']}' of object {j['id']} sweeps "
                                f"through object {c['with']} at joint value {c['value']} ({c['depth_m'] * 1000:.0f} mm): "
                                "expected if that object rests on the part (it must move first); otherwise fix the joint"})
    for p in (m.get("plan") or {}).get("missing_parts", []):
        repairs.append({"priority": 2, "object": p["id"], "task": f"plan declares part '{p['part']}' for object {p['id']} but it was not built"})
    for p in (m.get("plan") or {}).get("connection_off", []):
        repairs.append({"priority": 2, "object": p["id"], "task": f"plan says part '{p['part']}' of object {p['id']} "
                                                                  f"is {p['declared']} but it was built as {p['built']}: "
                                                                  "build it as declared, or correct the plan if the "
                                                                  "frames show another connection"})
    for p in (m.get("plan") or {}).get("size_off", []):
        repairs.append({"priority": 2, "object": p["id"], "task": f"part '{p['part']}' of object {p['id']} is {p['built_m']} m, "
                                                                  f"the plan says {p['declared_m']} m"})
    sil = m.get("silhouette_vs_mask") or {}
    for u in sil.get("below_min", []):
        detail = []
        for f, per in sil["per_frame"].items():
            if u in per and per[u]["size_ratio_wh"]:
                w, h = per[u]["size_ratio_wh"]
                detail.append(f"frame {f}: {w:.2f}x wider, {h:.2f}x taller than its mask")
        repairs.append({"priority": 2, "object": u, "task": f"object {u} silhouette IoU {sil['per_object_best_iou'][u]:.2f} at best vs "
                        "its SAM 3 mask (" + ("; ".join(detail[:2]) or "not visible in the render") + "): reshape or reposition it"})
    for f in (m.get("contacts") or {}).get("floating", []):
        gap = f"{f['gap_m'] * 100:.1f} cm above {f['above']}" if f["gap_m"] is not None else "with nothing below it"
        repairs.append({"priority": 2, "object": f["id"], "task": f"active object {f['id']} floats {gap}: lower it onto its support"})
    for u, d in (m.get("dynamic_pose_deviation_m") or {}).items():
        if d > 0.05:
            repairs.append({"priority": 1, "object": u, "task": f"moving object {u} leaves its measured poses by {d * 100:.0f} cm: "
                                                               f"key it on ev['objects']['{u}']['poses'] unchanged"})
    for u in m.get("passive_movers") or []:
        repairs.append({"priority": 2, "object": u, "task": f"moving object {u} has a passive body and would stop dead at the last frame: "
                                                           "make it active with a mass unless something holds it"})
    for item in (m.get("contacts") or {}).get("loose_passive", []):
        repairs.append({"priority": 3, "object": item["id"], "task": f"object {item['id']} ({item['size_m']} m, resting on "
                        f"{item['on']}) is passive: a loose object is active with a mass; keep it passive only if it is fixed in place"})
    for u in m.get("box_proxies") or []:
        repairs.append({"priority": 3, "object": u, "task": f"object {u} is a plain box: give it the shape the frames show "
                                                           "(parts, bevels, a profile), or say why a box is right"})
    repairs.sort(key=lambda r: (r["priority"], str(r["object"])))
    return repairs
