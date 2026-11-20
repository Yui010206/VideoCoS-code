"""Evaluated mesh arrays, world bounds, poses, paged geometry and materials.

Every function takes the worker's ``SceneContext``. Vertices are read with
``foreach_get`` into local-space numpy arrays (meters) and mapped to world
space per frame, so frame-independent meshes are read once per query.
"""
from __future__ import annotations

import numpy as np

from ..protocol import MAX_VERTICES, MAX_TRIANGLES, QueryCapacityError


def transform(ctx, evaluated):
    matrix = evaluated.matrix_world
    rows = [[float(v) for v in row] for row in matrix]
    for axis in range(3):
        rows[axis][3] *= ctx.scale
    location, rotation, scale = matrix.decompose()
    return {"matrix_world": rows, "translation_m": list(location * ctx.scale),
            "rotation_quaternion_wxyz": list(rotation), "scale": list(scale),
            "matrix_convention": "Multiply local coordinates in meters; matrix includes shear if present. Quaternion/scale decomposition may omit shear."}


def _to_world(ctx, matrix, points):
    # Elementwise products avoid multithreaded BLAS, which is slow inside the worker.
    rotation = np.array(matrix.to_3x3())
    return sum(points[:, [k]] * rotation[:, k] for k in range(3)) + np.array(matrix.translation) * ctx.scale


def world(ctx, mid, points):
    """Map local-space meter coordinates of mesh ``mid`` to world meters at the current frame."""
    return _to_world(ctx, ctx.meshes[mid].evaluated_get(ctx.depsgraph).matrix_world, points)


def _static(obj):
    """True when evaluated geometry cannot change between frames (object motion is fine)."""
    return (not obj.modifiers and obj.data.shape_keys is None and obj.data.animation_data is None
            and obj.parent_type not in ("ARMATURE", "LATTICE"))


def _capacity(kind, limit):
    return QueryCapacityError(f"Entity exceeds the evaluated {kind} work limit ({limit:,}); request individual mesh members")


def mesh(ctx, mid, vertex_room=MAX_VERTICES, triangle_room=MAX_TRIANGLES):
    """Cached local vertices and triangles of one mesh; counts are checked before arrays are read."""
    cached = ctx.mesh_cache.get(mid)
    if cached is None:
        obj = ctx.meshes[mid]
        evaluated = obj.evaluated_get(ctx.depsgraph)
        data = evaluated.to_mesh()
        if data is None:
            raise ValueError("Entity has no evaluated mesh")
        try:
            if len(data.vertices) > vertex_room:
                raise _capacity("vertex", MAX_VERTICES)
            loops = np.empty(len(data.polygons), np.int32)
            data.polygons.foreach_get("loop_total", loops)
            if np.maximum(loops - 2, 0).sum() > triangle_room:
                raise _capacity("triangle", MAX_TRIANGLES)
            data.calc_loop_triangles()
            local = np.empty(len(data.vertices) * 3, np.float32)
            data.vertices.foreach_get("co", local)
            triangles = np.empty(len(data.loop_triangles) * 3, np.int32)
            data.loop_triangles.foreach_get("vertices", triangles)
            cached = {"local": local.reshape(-1, 3).astype(float) * ctx.scale,
                      "triangles": triangles.reshape(-1, 3), "polygon_count": len(data.polygons),
                      "static": _static(obj)}
        finally:
            evaluated.to_mesh_clear()
        ctx.mesh_cache[mid] = cached
    if len(cached["local"]) > vertex_room:
        raise _capacity("vertex", MAX_VERTICES)
    if len(cached["triangles"]) > triangle_room:
        raise _capacity("triangle", MAX_TRIANGLES)
    return cached


def meshes(ctx, mids):
    """``(mid, mesh)`` pairs; the group shares one vertex/triangle cap, independent of order."""
    parts, vertices, triangles = [], 0, 0
    for mid in mids:
        value = mesh(ctx, mid, MAX_VERTICES - vertices, MAX_TRIANGLES - triangles)
        vertices, triangles = vertices + len(value["local"]), triangles + len(value["triangles"])
        parts.append((mid, value))
    return parts


def bounds(ctx, mids):
    """World axis-aligned bounds of the evaluated vertices of these meshes."""
    points = [world(ctx, mid, value["local"]) for mid, value in meshes(ctx, mids) if len(value["local"])]
    if not points:
        raise ValueError("Entity contains no evaluated vertices")
    lo = np.min([p.min(0) for p in points], 0).tolist()
    hi = np.max([p.max(0) for p in points], 0).tolist()
    return {"bbox_min": lo, "bbox_max": hi,
            "center": [(a + b) / 2 for a, b in zip(lo, hi)],
            "dimensions": [b - a for a, b in zip(lo, hi)]}


def pose(ctx, uid, frame):
    """Evaluated object pose; ``local_box_*`` is the object-local vertex box mapped to world."""
    if uid.startswith("collection:") and uid in ctx.entities:
        box = bounds(ctx, ctx.members(uid))
        center, lo, hi = box["center"], box["bbox_min"], box["bbox_max"]
        return {"object_id": uid, "frame": frame, "position": center,
                "quaternion": [1., 0., 0., 0.],
                "matrix_world": [[1., 0., 0., center[0]], [0., 1., 0., center[1]],
                                 [0., 0., 1., center[2]], [0., 0., 0., 1.]],
                "local_box_center": center,
                "local_box_corners": [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
                "local_bounds": [[lo[i] - center[i] for i in range(3)], [hi[i] - center[i] for i in range(3)]],
                "geometry_available": True,
                "pose_note": "Collection has no transform; this is its world-AABB center with identity rotation."}
    obj = ctx.object(uid)
    evaluated = obj.evaluated_get(ctx.depsgraph)
    pose_transform = transform(ctx, evaluated)
    lo = hi = np.zeros(3)
    local = mesh(ctx, "mesh:" + obj.name)["local"] if obj.type == "MESH" else ()
    if len(local):
        lo, hi = local.min(0), local.max(0)
    corners = [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    mapped = _to_world(ctx, evaluated.matrix_world, np.array(corners + [(lo + hi) / 2]))
    return {"object_id": uid, "frame": frame, "position": pose_transform["translation_m"],
            "quaternion": pose_transform["rotation_quaternion_wxyz"],
            "matrix_world": pose_transform["matrix_world"], "local_box_center": mapped[8].tolist(),
            "local_box_corners": mapped[:8].tolist(), "local_bounds": [lo.tolist(), hi.tolist()],
            "geometry_available": bool(len(local)),
            "geometry_note": "local_box_* is the object-local box of evaluated vertices mapped to world (oriented, not the world AABB); non-mesh objects have zero bounds, without metadata fallback."}


def geometry(ctx, uid, offset, limit):
    vertices, triangles, members, base = [], [], [], 0
    for mid, value in meshes(ctx, ctx.members(uid)):
        vertices.append(world(ctx, mid, value["local"]))
        triangles.append(value["triangles"] + base)
        members.append({"id": mid, "vertex_start": base, "vertex_count": len(value["local"]),
                        "triangle_count": len(value["triangles"]),
                        "transform": transform(ctx, ctx.meshes[mid].evaluated_get(ctx.depsgraph))})
        base += len(value["local"])
    vertices, triangles, end = np.concatenate(vertices), np.concatenate(triangles), offset + limit
    def after(items):
        return end if end < len(items) else None
    return {"coordinate_space": "world meters", "vertex_count": len(vertices), "triangle_count": len(triangles),
            "vertices": vertices[offset:end].tolist(), "triangles": triangles[offset:end].tolist(),
            "vertex_offset": offset, "triangle_offset": offset,
            "next_vertex_offset": after(vertices), "next_triangle_offset": after(triangles),
            "members": members[offset:end], "member_offset": offset, "next_member_offset": after(members),
            "pagination": "geometry_offset applies independently to vertices, triangles, and member transforms. Triangle indices refer to the full vertex array; pages may require other vertex pages.",
            "collection_transform": "Collections have no single object transform; inspect member transforms." if uid.startswith("collection:") else None}


def materials(ctx, uid):
    output = []
    seen = set()
    slots_total = 0
    for mid in ctx.members(uid):
        evaluated = ctx.meshes[mid].evaluated_get(ctx.depsgraph)
        for slot in evaluated.material_slots:
            material = slot.material
            slots_total += 1
            if material is None or material.as_pointer() in seen:
                continue
            seen.add(material.as_pointer())
            if len(output) >= 64:
                continue
            value = {"name": material.name, "base_color": list(material.diffuse_color),
                     "roughness": float(material.roughness), "metallic": float(material.metallic),
                     "uses_nodes": bool(material.use_nodes), "texture_present": False,
                     "base_color_linked": False, "roughness_linked": False, "metallic_linked": False}
            stack = [material.node_tree] if material.node_tree else []
            checked = set()
            principled = []
            while stack:
                tree = stack.pop()
                if tree.as_pointer() in checked:
                    continue
                checked.add(tree.as_pointer())
                for node in tree.nodes:
                    if node.bl_idname.startswith("ShaderNodeTex"):
                        value["texture_present"] = True
                    if node.bl_idname == "ShaderNodeBsdfPrincipled":
                        principled.append(node)
                    if node.type == "GROUP" and node.node_tree:
                        stack.append(node.node_tree)
            value["principled_shader_count"] = len(principled)
            if len(principled) == 1:
                for field, socket in (("base_color", "Base Color"), ("roughness", "Roughness"), ("metallic", "Metallic")):
                    inp = principled[0].inputs.get(socket)
                    if inp is not None:
                        value[field + "_linked"] = bool(inp.is_linked)
                        value[field] = list(inp.default_value) if field == "base_color" else float(inp.default_value)
            output.append(value)
    return {"materials": output, "unique_material_count": len(seen), "material_slots": slots_total,
            "truncated": len(seen) > len(output),
            "interpretation": "Whitelisted viewport or single-Principled input defaults. Linked sockets, mixed shaders, textures and lighting require rendering; these values are not rendered colors."}
