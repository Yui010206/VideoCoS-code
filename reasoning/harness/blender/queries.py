"""Worker-side handlers: one function per method, dispatched by name. Runs inside Blender."""
from __future__ import annotations

import math
import time

import numpy as np

from .camera import camera_history
from .geometry import mesh_surface_distance
from .mesh import bounds, geometry, materials, mesh, meshes, pose, transform, world
from .context import semantic_metadata
from ..protocol import HEAVY_QUERY_SECONDS, compact_numbers, sample_frames, validate_arguments


def _page(items, args, default_limit):
    offset, limit = args.get("offset", 0), args.get("limit", default_limit)
    return items[offset:offset + limit], offset + limit if offset + limit < len(items) else None


def scene_metadata(ctx, args):
    objects = ctx.semantic_objects or ctx.meshes
    ids, next_offset = _page(sorted(objects), args, 100)
    return {"ok": True, "objects": {uid: semantic_metadata(uid, objects[uid]) for uid in ids},
            "next_offset": next_offset}


def scene_states(ctx, args):
    frame = ctx.set_frame(args["frame"])
    return {"ok": True, "frame": frame, "states": {uid: pose(ctx, uid, frame) for uid in args["ids"]}}


def scene_inventory(ctx, args):
    query, kind = args.get("query", ""), args.get("kind", "collection")
    matches = sorted((e for e in ctx.entities.values() if (kind == "all" or e["kind"] == kind)
                      and query.casefold() in e["name"].casefold()), key=lambda e: e["id"])
    page, next_offset = _page(matches, args, 80)
    return {"ok": True, "total_matches": len(matches), "offset": args.get("offset", 0),
            "entities": [{k: e[k] for k in ("id", "name", "kind")} | {"mesh_part_count": len(e["members"])}
                         for e in page],
            "next_offset": next_offset,
            "warning": "Names are scene-authored. Collections can contain multiple objects and meshes can be parts."}


def inspect_entities(ctx, args):
    frame = ctx.set_frame(args.get("frame", ctx.first))
    detail = args.get("detail", "basic")
    output = []
    for uid in args["ids"]:
        members = ctx.members(uid)
        entity = ctx.entities[uid]
        item = {**{k: v for k, v in entity.items() if k != "members"}, **bounds(ctx, members),
                "member_count": len(members), "member_preview": members[:12],
                "members_truncated": len(members) > 12, "geometry_available": True}
        if entity["kind"] == "mesh":
            value = mesh(ctx, uid)
            item.update(vertex_count=len(value["local"]), triangle_count=len(value["triangles"]),
                        polygon_count=value["polygon_count"])
            if detail in {"transform", "geometry"}:
                item["transform"] = transform(ctx, ctx.meshes[uid].evaluated_get(ctx.depsgraph))
        elif detail == "transform":
            item["transform"] = None
            item["transform_note"] = "Collections have no single transform; inspect their members."
        if detail == "geometry":
            item["geometry"] = geometry(ctx, uid, args.get("geometry_offset", 0), args.get("geometry_limit", 64))
        elif detail == "materials":
            item["appearance"] = materials(ctx, uid)
        output.append(compact_numbers(item) if detail == "basic" else item)
    return {"ok": True, "frame": frame, "detail": detail, "entities": output, "units": "meters",
            "geometry": "World axis-aligned bounds from active view-layer/viewport evaluated mesh vertices; geometry detail exposes paged evaluated triangles. Render-only settings can differ.",
            "member_lookup": "Use detail=geometry with geometry_offset to page all collection members and their transforms."}


def _surface(ctx, uid):
    parts = meshes(ctx, ctx.members(uid))
    starts = np.cumsum([0] + [len(value["local"]) for _, value in parts])
    return (np.concatenate([world(ctx, mid, value["local"]) for mid, value in parts]),
            np.concatenate([value["triangles"] + start for (_, value), start in zip(parts, starts)]))


def measure_pair(ctx, args):
    frame = ctx.set_frame(args.get("frame", ctx.first))
    a, b = bounds(ctx, ctx.members(args["a"])), bounds(ctx, ctx.members(args["b"]))
    delta = [b["center"][i] - a["center"][i] for i in range(3)]
    if args.get("metric") == "center":
        return {"ok": True, "object_a": args["a"], "object_b": args["b"], "frame": frame,
                "distance_m": math.sqrt(sum(v*v for v in delta)),
                "center_a": a["center"], "center_b": b["center"],
                "metric": "Euclidean distance between world bounding-box centers", "units": "meters"}
    gap = [max(a["bbox_min"][i] - b["bbox_max"][i], b["bbox_min"][i] - a["bbox_max"][i], 0) for i in range(3)]
    result = {"ok": True, "a": args["a"], "b": args["b"], "frame": frame, "units": "meters",
              "metric": args.get("metric", "bounds"), "center_distance": math.sqrt(sum(v*v for v in delta)),
              "center_delta_b_minus_a": delta, "aabb_surface_distance_proxy": math.sqrt(sum(v*v for v in gap)),
              "warning": "AABB separation is a lower-bound proxy for mesh surface distance; centers are AABB centers."}
    if args.get("metric") == "surface":
        result["surface"] = mesh_surface_distance(*_surface(ctx, args["a"]), *_surface(ctx, args["b"]),
                                                  max_pairs=200_000, deadline=time.monotonic() + HEAVY_QUERY_SECONDS)
    return result


def viewpoint_relation(ctx, args):
    frame = ctx.set_frame(args.get("frame", ctx.first))
    origin, facing, subject = (bounds(ctx, ctx.members(args[key]))["center"] for key in ("observer", "facing", "subject"))
    dx, dy = facing[0] - origin[0], facing[1] - origin[1]
    length = math.hypot(dx, dy)
    if length < 1e-8:
        raise ValueError("Observer and facing entity have identical horizontal centers")
    forward = [dx / length, dy / length]
    right = [forward[1], -forward[0]]
    relative = [subject[i] - origin[i] for i in range(2)]
    right_m = sum(a*b for a, b in zip(relative, right))
    forward_m = sum(a*b for a, b in zip(relative, forward))
    return {"ok": True, "frame": frame, "right_m": right_m, "forward_m": forward_m,
            "vertical_m": subject[2] - origin[2], "clockwise_angle_deg": math.degrees(math.atan2(right_m, forward_m)),
            "basis": "Horizontal evaluated AABB-center observer -> facing, +Z up"}


def track_motion(ctx, args):
    """World-AABB centres of a few entities over sampled frames, with speed summaries.

    Reads the scene as it is now, edits included. A rigid-body simulation must be baked first:
    stepping to an arbitrary unbaked frame does not run the solver.
    """
    request = dict(args)
    if "frames" not in request and "step" not in request:
        request.setdefault("limit", 32)
    frames = sample_frames(request, ctx.first, ctx.last)
    fps = ctx.scene.render.fps / ctx.scene.render.fps_base
    centres = {uid: [] for uid in args["ids"]}
    for uid in args["ids"]:
        ctx.members(uid)  # unknown ids fail before any frame is evaluated
    for frame in frames:
        ctx.set_frame(frame)
        for uid in args["ids"]:
            centres[uid].append(bounds(ctx, ctx.members(uid))["center"])
    tracks = []
    for uid, points in centres.items():
        steps = [math.dist(a, b) for a, b in zip(points, points[1:])]
        gaps = [(b - a) / fps for a, b in zip(frames, frames[1:])]
        speeds = [d / g for d, g in zip(steps, gaps) if g > 0]
        tracks.append({"id": uid, "frames": frames, "centers": [[round(v, 4) for v in p] for p in points],
                       "displacement_m": round(math.dist(points[0], points[-1]), 4),
                       "path_m": round(sum(steps), 4),
                       "peak_speed_mps": round(max(speeds), 4) if speeds else 0.0,
                       "final_speed_mps": round(speeds[-1], 4) if speeds else 0.0})
    return {"ok": True, "frame_range": [ctx.first, ctx.last], "fps": fps, "tracks": tracks,
            "note": "Speeds are between consecutive samples, so a coarse sampling under-reports peaks."}


HANDLERS = {
    "summary": lambda ctx, args: ctx.summary(),
    "_scene_metadata": scene_metadata,
    "_scene_states": scene_states,
    "scene_inventory": scene_inventory,
    "inspect_entities": inspect_entities,
    "measure_pair": measure_pair,
    "viewpoint_relation": viewpoint_relation,
    "camera_history": camera_history,
    "track_motion": track_motion,
}


def execute(ctx, method, arguments):
    """Validate and run one request; caches never outlive it and the frame is restored."""
    if method not in HANDLERS:
        raise ValueError("Unknown method")
    validate_arguments(method, arguments)
    with ctx.query():
        return HANDLERS[method](ctx, arguments)
