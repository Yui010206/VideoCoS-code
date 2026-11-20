"""Arranging a room's static objects the way a set dresser would (inside Blender): upright, on the floor or on what
holds them, apart. codify.assets.arrange_room calls arrange(); its report lists what moved and what is left to decide.

1 upright: an object that stands is turned onto the stable face of its hull nearest to facing down (rest_on), at most
  max_turn_deg; a mesh made from one view is often tilted about the line of sight;
2 floor: nothing reaches below the floor; a body that does is raised onto it;
3 seated, bottom up so a support is final before anything is seated on it: rays cast down over the footprint find
  the highest upward-facing surface below (the floor, the environment, another object) and the body is lowered onto
  it when the gap is at most max_drop. The surface under its centre may also raise a body sunk into it (by up to half
  its height, 10 cm at most); one under an edge only holds it, so a body overlapping a neighbour is not perched on
  it. Things that stand on the floor (chairs, tables, beds, shoes, bins, doors) are lowered whatever the gap; things
  that hang on a wall or a ceiling are kept where they are;
4 apart: separate_objects pushes interpenetrating pairs apart, then what moved is seated again.
A body still far above everything below it stands on something the scene lacks (a window sill, a shelf, a counter)
or is misplaced: the report lists it as unsupported.
"""
from __future__ import annotations

import json
import os
import re

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from codify import gates

ROOM = os.environ.get("CODIFY_ROOM") == "1"          # set by the sandbox in a room run (settings fast_room)

HANGS = {"painting", "picture", "artwork", "poster", "clock", "mirror", "curtain", "curtains", "blind", "blinds",
         "window", "hood", "light", "chandelier", "sconce", "switch", "socket", "outlet", "hook", "hanger", "calendar",
         "whiteboard", "rail", "vent", "thermostat", "conditioner", "handle", "knob", "doorway", "doorframe", "faucet",
         "tap"}
STANDS = {"chair", "stool", "sofa", "couch", "armchair", "bench", "ottoman", "table", "desk", "bed", "mattress",
          "dresser", "wardrobe", "refrigerator", "fridge", "washer", "washing", "dryer", "dishwasher", "trash",
          "garbage", "bin", "suitcase", "luggage", "mat", "rug", "carpet", "shoe", "shoes", "boot", "boots",
          "slipper", "slippers", "door", "radiator", "heater", "vacuum", "ladder", "nightstand", "bookcase",
          "bookshelf", "cart", "crib", "piano", "bicycle", "bike", "stroller", "toilet", "bathtub", "basket"}
DOWN = Vector((0.0, 0.0, -1.0))
HOLD_M = 0.02           # an edge of the footprint is held by a surface at most this far above the bottom
REST_GAP_M = 0.003      # a seated body is left this far above its support (the check's floating ray starts 1 mm under a body)


def _words(category):
    return set(re.findall(r"[a-z]+", str(category).lower()))


def _world(objs):
    """World-space vertices and triangles (n x 3 vertex indices) of the meshes, read in bulk (no modifiers in
    codified scenes)."""
    verts, tris, offset = [], [], 0
    for o in objs:
        mesh = o.data
        n = len(mesh.vertices)
        co = np.empty(3 * n, np.float32)
        mesh.vertices.foreach_get("co", co)
        m = np.array(o.matrix_world)
        verts.append(co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3])
        mesh.calc_loop_triangles()
        t = np.empty(3 * len(mesh.loop_triangles), np.int32)
        mesh.loop_triangles.foreach_get("vertices", t)
        tris.append(t.reshape(-1, 3) + offset)
        offset += n
    return ((np.vstack(verts) if verts else np.zeros((0, 3))),
            (np.vstack(tris) if tris else np.zeros((0, 3), np.int32)))


class Body:
    """One registered object (root and parts) or the environment, with a BVH tree in world space."""

    def __init__(self, uid, objs):
        self.uid, self.objs = uid, objs
        self.verts, self.tris = _world(objs)
        self.build()

    def build(self):
        self.lo, self.hi = self.verts.min(0), self.verts.max(0)
        self.tree = BVHTree.FromPolygons(self.verts.tolist(), self.tris.tolist(), all_triangles=True) \
            if len(self.tris) else None

    def lift(self, dz):
        self.objs[0].location.z += dz
        self.verts[:, 2] += dz
        self.build()

    def footprint(self):
        return gates.footprint(self.verts)


def _support(body, others, raise_limit):
    """(z, uid) of the surface the body rests on: the highest first hit on a level face of a ray cast down from above
    the body, at most raise_limit above its bottom under its centre and HOLD_M under the rest of its footprint. A body
    the ray starts inside (the body is in it or under it) holds nothing there."""
    best, best_uid = None, None
    for i, (x, y) in enumerate(body.footprint()):
        limit = raise_limit if i == 0 else HOLD_M
        origin = Vector((float(x), float(y), float(body.hi[2]) + 0.01))
        for other in others:
            if other.tree is None or other.lo[2] > body.hi[2] or not (
                    other.lo[0] <= x <= other.hi[0] and other.lo[1] <= y <= other.hi[1]):
                continue
            loc, normal = other.tree.ray_cast(origin, DOWN, 50.0)[:2]
            if loc is None or abs(normal.z) < 0.3:     # winding varies between meshes: a level face either way
                continue
            if other.hi[2] > origin.z and gates._inside(other.tree, origin, float(np.linalg.norm(other.hi - other.lo)) + 1):
                continue
            if loc.z <= body.lo[2] + limit and (best is None or loc.z > best):
                best, best_uid = float(loc.z), other.uid
    return best, best_uid


def _stand_together(bodies, gap=0.3, dz=0.1):
    """Groups of two or more bodies standing together: footprints less than gap apart in plan, bottoms within dz."""
    parent = list(range(len(bodies)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(bodies):
        for j in range(i + 1, len(bodies)):
            b = bodies[j]
            if abs(a.lo[2] - b.lo[2]) <= dz and max(b.lo[0] - a.hi[0], a.lo[0] - b.hi[0], b.lo[1] - a.hi[1],
                                                    a.lo[1] - b.hi[1]) < gap:
                parent[root(i)] = root(j)
    groups = {}
    for i, b in enumerate(bodies):
        groups.setdefault(root(i), []).append(b.uid)
    return [g for g in groups.values() if len(g) >= 2]


def arrange(max_drop=0.25, hang=(), stand=(), keep=(), turn=True, max_turn_deg=35.0, separate=True,
            auto_support=False):
    from codify import assets as A
    scene = bpy.context.scene
    groups = gates.semantic_groups(scene)
    moving = {u for u, objs in groups.items() if any(A._moves(o) for o in objs)}
    keep, hang, stand = {str(u) for u in keep}, {str(u) for u in hang}, {str(u) for u in stand}
    category = {u: str(objs[0].get("category", "")) for u, objs in groups.items()}
    hung = {u for u in groups if u in hang or (u not in stand and _words(category[u]) & HANGS)} - moving - keep
    standing = {u for u in groups if u in stand or (u not in hang and _words(category[u]) & STANDS)} - hung
    fixed = [u for u in groups if u not in moving and u not in keep]
    todo = [u for u in fixed if u not in hung]
    report = {"turned": [], "raised_to_floor": [], "seated": [], "kept_hanging": sorted(hung, key=str),
              "unsupported": [], "supports_built": [], "separated": [], "max_drop_m": max_drop}
    if turn:
        for u in todo:
            objs = groups[u]
            if len(objs) == 1:          # an object with parts keeps its pose (its joints are measured in it)
                bottom = float(_world(objs)[0][:, 2].min())
                angle = A.rest_on(objs[0], z=bottom, turn=True, max_turn_deg=max_turn_deg, max_points=2000)
                if angle:
                    report["turned"].append({"id": u, "deg": round(angle, 1)})
    bpy.context.view_layer.update()
    bodies = {u: Body(u, objs) for u, objs in groups.items()}
    env = [Body(o.name, [o]) for o in gates.environment_meshes(scene) if o.type == "MESH" and len(o.data.polygons)]
    floor = min((b.hi[2] for b in env), default=None)          # the lowest top of the environment: the floor
    for u in fixed:
        if floor is not None and bodies[u].lo[2] < floor - 0.002:
            report["raised_to_floor"].append({"id": u, "dz_m": round(float(floor - bodies[u].lo[2]), 3)})
            bodies[u].lift(float(floor - bodies[u].lo[2]) + REST_GAP_M)

    def seat(u, record=True):
        body = bodies[u]
        under, uid = _support(body, env + [b for v, b in bodies.items() if v != u],
                              min(0.10, 0.5 * float(body.hi[2] - body.lo[2])))
        if ROOM and under is None and u in standing and floor is not None:   # rooms (v2): past the floor's edge, to floor level
            under, uid = floor, "floor level"
        gap = None if under is None else float(body.lo[2] - under)
        if gap is not None and (gap <= max_drop or u in standing):
            if abs(gap - REST_GAP_M) > 0.001:
                body.lift(REST_GAP_M - gap)
                if record:
                    report["seated"].append({"id": u, "dz_m": round(REST_GAP_M - gap, 3), "on": uid})
            return None
        return {"id": u, "category": category[u], "gap_m": None if gap is None else round(gap, 3), "above": uid,
                "bottom_z": round(float(body.lo[2]), 3)}

    for u in sorted(todo, key=lambda v: bodies[v].lo[2]):
        left = seat(u)
        if left:
            report["unsupported"].append(left)
    if auto_support and report["unsupported"]:
        # Unsupported objects that stand together at one height stand on a surface the scene lacks (a sill, a shelf
        # board, a counter top): a slab under each group (add_support), then they are seated on it.
        loose = [bodies[x["id"]] for x in report["unsupported"] if x["id"] not in standing]
        for ids in _stand_together(loose):
            for slab in A.add_support(ids):
                env.append(Body(slab.name, [slab]))
                report["supports_built"].append({"ids": json.loads(slab["support_for"]),
                                                 "top_z": round(float(slab.matrix_world.translation.z
                                                                      + max(v.co.z for v in slab.data.vertices)), 3),
                                                 "size_m": [round(slab.dimensions.x, 2), round(slab.dimensions.y, 2)]})
            for u in ids:
                seat(u)
        placed = {u for g in report["supports_built"] for u in g["ids"]}
        report["unsupported"] = [x for x in report["unsupported"] if x["id"] not in placed]
    bpy.context.view_layer.update()
    if separate:
        report["separated"] = A.separate_objects(passes=1, fast=True)
        # What moved apart, and what rests on it, is seated again (bottom up); what that leaves without a support
        # (pushed off a small one) joins the unsupported.
        bodies = {u: Body(u, objs) for u, objs in groups.items()}
        unsupported = {item["id"] for item in report["unsupported"]}
        for u in sorted(todo, key=lambda v: bodies[v].lo[2]):
            if u not in unsupported:
                left = seat(u, record=False)
                if left:
                    report["unsupported"].append({**left, "pushed_off": True})
    bpy.context.view_layer.update()
    scene["codify_arrangement"] = json.dumps(report)
    return report


def layout_report():
    """The room as a whole, for a look beyond the source camera; prints LAYOUT: what should stand on the floor and does not,
    objects of one kind that overlap (possible duplicates), the support slabs (placeholders for what the scene lacks), the
    floor's area, the walls, the camera's height above the floor."""
    scene = bpy.context.scene
    bpy.context.view_layer.update()
    groups = gates.semantic_groups(scene)
    bodies = {u: Body(u, objs) for u, objs in groups.items()}
    env_objs = [o for o in gates.environment_meshes(scene) if o.type == "MESH" and len(o.data.polygons)]
    env = [Body(o.name, [o]) for o in env_objs]
    floor = min((b.hi[2] for b in env), default=0.0)
    category = {u: str(objs[0].get("category", "")) for u, objs in groups.items()}
    off = []
    for u, b in bodies.items():
        words = _words(category[u])
        if not words & STANDS or words & HANGS or b.lo[2] - floor <= 0.05:
            continue
        under, _ = _support(b, env + [x for v, x in bodies.items() if v != u], 0.0)
        if under is None or b.lo[2] - under > 0.05:
            off.append({"id": u, "category": category[u], "bottom_above_floor_m": round(float(b.lo[2] - floor), 2)})
    ids, repeats = list(bodies), []
    for i, a in enumerate(ids):
        for c in ids[i + 1:]:
            if category[a] != category[c]:
                continue
            A, C = bodies[a], bodies[c]
            inter = np.minimum(A.hi, C.hi) - np.maximum(A.lo, C.lo)
            if (inter > 0).all() and inter.prod() > 0.3 * min((A.hi - A.lo).prod(), (C.hi - C.lo).prod()):
                repeats.append({"ids": [a, c], "category": category[a]})
    slabs = [{"name": o.name, "above_floor_m": round(float(b.hi[2] - floor), 2), "under": json.loads(o["support_for"])}
             for o, b in zip(env_objs, env) if "support_for" in o]
    floors = [b for o, b in zip(env_objs, env) if "floor" in o.name.lower()]
    camera = scene.camera.matrix_world.translation.z - floor if scene.camera else None
    report = {"camera_above_floor_m": None if camera is None else round(float(camera), 2),
              "floor_area_m2": round(float(sum((b.hi[0] - b.lo[0]) * (b.hi[1] - b.lo[1]) for b in floors)), 1),
              "walls": sum("wall" in o.name.lower() for o in env_objs), "objects": len(bodies),
              "off_the_floor": sorted(off, key=lambda x: -x["bottom_above_floor_m"])[:40], "possible_duplicates": repeats[:40],
              "support_slabs": slabs[:40]}
    print("LAYOUT", json.dumps(report))
    return report
