"""Source camera inspection, sampled visibility, and first-hit refinement.

Visibility uses geometric probes. It does not certify original video pixels.
"""
from __future__ import annotations

import time

from .mesh import bounds, meshes, transform, world
from ..protocol import (HEAVY_QUERY_SECONDS, MAX_REFINEMENT_FRAMES, QueryCapacityError,
                             compact_numbers, integer, sample_frames)


def refine_first_visible(frames, observations, evaluate, extra_limit=MAX_REFINEMENT_FRAMES):
    """Scan the first sampled false->true bracket; never assume monotonic visibility.

    ``observations`` caches coarse and newly evaluated visibility at integer
    frames. Only previously unqueried frames count toward the extra-work cap.
    ``evaluate`` returns None once the query's time budget is spent.
    The returned hit is local to this bracket, never a global first appearance.
    """
    integer(extra_limit, "extra_limit", 0, MAX_REFINEMENT_FRAMES)
    metadata = {
        "status": "no_sampled_hit", "bracket": None,
        "evaluated_frame_count": 0, "extra_frame_limit": extra_limit,
        "complete_within_bracket": False, "unresolved_range": None,
        "scope": "Only the first sampled invisible-to-visible bracket is refined, within the requested sample envelope. Earlier brief appearances between coarse samples may be missed; this is not original-video pixel visibility.",
    }
    first_index = next((i for i, frame in enumerate(frames)
                        if observations[frame].get("sampled_visible") is True), None)
    if first_index is None:
        if any(not observations[frame].get("available") for frame in frames):
            metadata.update(status="unavailable", unresolved_range=[frames[0], frames[-1]])
        return None, metadata
    if first_index == 0:
        metadata["status"] = "visible_at_start"
        return None, metadata
    low, high = frames[first_index - 1], frames[first_index]
    metadata["bracket"] = [low, high]
    previous = observations[low]
    if not previous.get("available") or previous.get("sampled_visible") is not False:
        metadata.update(status="unavailable", unresolved_range=[low, high])
        return None, metadata
    for frame in range(low + 1, high + 1):
        if frame not in observations:
            if metadata["evaluated_frame_count"] >= extra_limit:
                metadata.update(status="capped", unresolved_range=[frame, high])
                return None, metadata
            value = evaluate(frame)
            if value is None:
                metadata.update(status="time_limit", unresolved_range=[frame, high])
                return None, metadata
            observations[frame] = value
            metadata["evaluated_frame_count"] += 1
        value = observations[frame]
        if value.get("sampled_visible") is True:
            metadata.update(status="refined" if high > low + 1 else "adjacent_samples",
                            complete_within_bracket=True)
            return {"frame": frame, "visible_entity_ids": list(value.get("visible_entity_ids", []))}, metadata
        if not value.get("available") or value.get("sampled_visible") is not False:
            metadata.update(status="unavailable", unresolved_range=[frame, high])
            return None, metadata
    # The coarse high endpoint is cached as visible; reaching this is a broken
    # caller contract rather than evidence of absence.
    raise ValueError("Visibility refinement lost its sampled visible endpoint")



def camera(ctx):
    if ctx.scene.camera is None:
        return None, None
    cam = ctx.scene.camera.evaluated_get(ctx.depsgraph)
    data = cam.data
    return cam, {"type": data.type, "lens_mm": float(data.lens),
                 "sensor_width_mm": float(data.sensor_width), "sensor_height_mm": float(data.sensor_height),
                 "shift_x": float(data.shift_x), "shift_y": float(data.shift_y),
                 "clip_start_m": float(data.clip_start) * ctx.scale,
                 "clip_end_m": float(data.clip_end) * ctx.scale,
                 "ortho_scale_m": float(data.ortho_scale) * ctx.scale,
                 "position": list(cam.matrix_world.translation * ctx.scale), **transform(ctx, cam)}


def visibility(ctx, uid, cam):
    """Ray-probe the AABB center and up to 32 vertices of the entity's shown meshes."""
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view
    if cam is None:
        return {"available": False, "reason": "Scene has no active camera"}
    if cam.data.type not in ("PERSP", "ORTHO"):
        return {"available": False, "reason": "Visibility probes support PERSP and ORTHO cameras only"}
    shown = [mid for mid in ctx.members(uid) if ctx.meshes[mid].visible_get()]
    if not shown:
        return {"available": False, "sampled_visible": None, "visibility": "excluded", "error_type": "excluded",
                "reason": "Hidden or excluded from the active view layer; ray probes cannot see it."}
    parts = meshes(ctx, shown)
    total = sum(len(value["local"]) for _, value in parts)
    if not total:
        return {"available": False, "reason": "Entity has no evaluated vertices"}
    # Uniformly sample across all members rather than only the first mesh.
    indices = sorted({round((total - 1) * i / 31) for i in range(32)})
    points, start = [bounds(ctx, shown)["center"]], 0
    for mid, value in parts:
        picked = [i - start for i in indices if start <= i < start + len(value["local"])]
        if picked:
            points.extend(world(ctx, mid, value["local"][picked]).tolist())
        start += len(value["local"])
    member_names = {ctx.meshes[mid].name for mid in shown}
    frustum = visible = 0
    direction_camera = cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    direction_camera.normalize()
    camera_origin = cam.matrix_world.translation
    epsilon = max(1e-6, 1e-5 / ctx.scale)
    for point in points:
        endpoint = Vector(point) / ctx.scale
        projected = world_to_camera_view(ctx.scene, cam, endpoint)
        if not (cam.data.clip_start <= projected.z <= cam.data.clip_end and
                0 <= projected.x <= 1 and 0 <= projected.y <= 1):
            continue
        frustum += 1
        if cam.data.type == "ORTHO":
            depth = (endpoint - camera_origin).dot(direction_camera)
            origin = endpoint - direction_camera * (depth - cam.data.clip_start)
            direction = direction_camera.copy()
        else:
            direction = endpoint - camera_origin
            if direction.length < epsilon:
                continue
            direction.normalize()
            cosine = direction.dot(direction_camera)
            if cosine <= 0:
                continue
            origin = camera_origin + direction * (cam.data.clip_start / cosine)
        length = (endpoint - origin).length
        hit, _, _, _, obj, _ = ctx.scene.ray_cast(ctx.depsgraph, origin, direction, distance=length + epsilon)
        if hit and obj is not None and getattr(obj, "original", obj).name in member_names:
            visible += 1
    return {"available": True, "probe_count": len(points), "frustum_probe_count": frustum,
            "visible_probe_count": visible, "sampled_visible": visible > 0,
            "method": "Center plus up to 32 evaluated vertices; first ray hit must be a member of the target entity.",
            "limitation": "Sampled geometric probes, not rendered pixel visibility. Materials/transparency/render visibility and unsampled surfaces may differ."}


def history_visibility(ctx, ids, cam):
    """Union over separately evaluated entities; one entity's failure never hides another."""
    values = []
    for uid in ids:
        try:
            value = visibility(ctx, uid, cam)
        except ValueError as exc:
            value = {"available": False, "sampled_visible": None, "reason": str(exc),
                     "error_type": "capacity" if isinstance(exc, QueryCapacityError) else "invalid"}
        values.append((uid, value))
    visible = [uid for uid, value in values if value.get("sampled_visible") is True]
    if len(values) == 1:
        return values[0][1], visible
    available_count = sum(bool(value.get("available")) for _, value in values)
    all_available = available_count == len(values)
    result = {
        # A positive union is known even when another entity is unavailable;
        # a negative union is known only when every selected entity is known.
        "available": bool(visible) or all_available,
        "entity_count": len(ids), "available_entity_count": available_count,
        "unknown_entity_count": len(ids) - available_count,
        "visible_entity_count": len(visible),
        **{key: sum(value.get(key, 0) for _, value in values)
           for key in ("probe_count", "frustum_probe_count", "visible_probe_count")},
        "method": "Union of separately probed selected entities; each uses its own center and up to 32 vertices.",
    }
    if visible or all_available:
        result["sampled_visible"] = bool(visible)
    failures = [{"entity_id": uid, "error_type": value["error_type"], "reason": value["reason"]}
                for uid, value in values if "error_type" in value]
    if failures:
        result["unavailable_entities"] = failures
    if not all_available:
        result["reason"] = "Some selected entities have unavailable visibility; without a positive hit the union is unknown, not invisible."
    return result, visible


SAMPLE_KEYS = ("available", "sampled_visible", "visibility", "probe_count", "frustum_probe_count",
               "visible_probe_count", "reason", "entity_count", "available_entity_count",
               "unknown_entity_count", "visible_entity_count", "error_type", "unavailable_entities")


def camera_history(ctx, args):
    """Sample camera poses and visibility; stops early with truncated=true at the time budget."""
    deadline = time.monotonic() + HEAVY_QUERY_SECONDS
    frames = sample_frames(args, ctx.first, ctx.last)
    uid = args.get("entity_id")
    ids = args.get("entity_ids", [uid] if uid is not None else [])
    detail = args.get("detail", "summary")
    for entity_id in ids:
        ctx.members(entity_id)
    samples, hits, visibility_samples, observations, evaluated = [], [], [], {}, []
    fps = ctx.scene.render.fps / ctx.scene.render.fps_base
    for frame in frames:
        if time.monotonic() > deadline:
            break
        evaluated.append(frame)
        ctx.set_frame(frame)
        cam, value = camera(ctx)
        sample = {"frame": frame, "time_s": (frame - ctx.first) / fps, "camera": value}
        if ids:
            seen, visible_ids = history_visibility(ctx, ids, cam)
            observations[frame] = {**seen, "visible_entity_ids": visible_ids}
            visibility_samples.append({"frame": frame, "time_s": sample["time_s"],
                                       **{key: seen[key] for key in SAMPLE_KEYS if key in seen}})
            if seen.get("sampled_visible"):
                hits.append({"frame": frame, "time_s": sample["time_s"], **(seen if detail == "full" else {})})
            if detail == "full":
                sample["visibility"] = seen
        if detail == "full":
            samples.append(sample)
        elif detail == "poses" or not ids:
            # A pose does not repeat lens defaults and a full matrix at
            # every frame. Full camera parameters remain available on demand.
            sample["camera"] = ({key: value[key] for key in ("type", "position", "rotation_quaternion_wxyz")
                                 if key in value} if value is not None else None)
            samples.append(sample)
    result = {"ok": True, "detail": detail, "sampled_frames": frames,
              "sample_count": len(frames), "sampled_interval": [frames[0], frames[-1]],
              "visibility_method": "Center plus up to 32 evaluated vertices; the first ray hit must belong to the target. PERSP/ORTHO only.",
              "warning": "Queries evaluate the exact requested frames; temporal sampling can miss events between them. First sampled hit is not exact first video appearance."}
    if len(evaluated) < len(frames):
        result.update(truncated=True, unevaluated_frames=frames[len(evaluated):])
    if samples:
        result["camera_samples"] = samples
    if ids:
        def evaluate_visibility(frame):
            if time.monotonic() > deadline:
                return None
            ctx.set_frame(frame)
            value, visible_ids = history_visibility(ctx, ids, camera(ctx)[0])
            return {**value, "visible_entity_ids": visible_ids}
        refined, refinement = refine_first_visible(evaluated, observations, evaluate_visibility)
        if refined is not None:
            refined["time_s"] = (refined["frame"] - ctx.first) / fps
        if refinement["status"] == "time_limit":
            result["truncated"] = True
        result.update(first_sampled_hit=hits[0] if hits else None,
                      hit_samples=hits, total_hit_samples=len(hits),
                      available_samples=sum(bool(row.get("available")) for row in visibility_samples),
                      queried_entity_ids=list(ids),
                      first_sampled_visible_entity_ids=(observations[hits[0]["frame"]]["visible_entity_ids"]
                                                        if hits else []),
                      first_visible_hit=refined, refinement=refinement)
        if uid is not None:
            result["entity_id"] = uid
        else:
            result["entity_ids"] = list(ids)
            result["visibility_method"] = "Union of selected entity visibility. Each entity independently uses its center plus up to 32 vertices and occlusion-tested ray hits. PERSP/ORTHO only."
        if detail != "full":
            result["visibility_samples"] = visibility_samples
    if result.get("truncated"):
        result["truncation_note"] = (f"Stopped at the {HEAVY_QUERY_SECONDS:g} s worker time budget; results cover only "
                                     "evaluated frames. Request fewer frames or entities.")
    return result if detail == "full" else compact_numbers(result)
