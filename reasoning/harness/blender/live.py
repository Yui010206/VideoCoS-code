"""Live-edit support inside Blender: what an edit changed, the task contract, the overview, rendering.

Runs inside Blender only. The contract checks are the only thing that can refuse a submission,
and each one is a fact the task states (the camera, the frame range, frames that must not move);
nothing here guesses whether the physics or the plan is good.
"""
from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Quaternion, Vector

from . import helpers as S

POSITION_TOL_M = 1e-3
ANGLE_TOL_DEG = 0.1
MAX_LISTED = 20


def _scene_state():
    s = bpy.context.scene
    world = s.rigidbody_world
    cam = s.camera
    return {"frame_start": s.frame_start, "frame_end": s.frame_end, "fps": s.render.fps / s.render.fps_base,
            "camera": cam.name if cam else None,
            "camera_location": [round(v, 4) for v in cam.matrix_world.to_translation()] if cam else None,
            "resolution": [s.render.resolution_x, s.render.resolution_y],
            "physics_cache": None if world is None else
            {"frames": [world.point_cache.frame_start, world.point_cache.frame_end],
             "baked": bool(world.point_cache.is_baked), "enabled": bool(world.enabled)}}


def snapshot():
    """Everything an edit can change that the model should hear about, at the current frame."""
    objects = {}
    for o in bpy.data.objects:
        action = o.animation_data.action if o.animation_data else None
        q = o.matrix_world.to_quaternion()
        body = o.rigid_body
        objects[o.name] = {
            "id": None if o.get("instance_id") is None else str(o["instance_id"]), "type": o.type,
            "location": list(o.matrix_world.to_translation()), "rotation": [q.w, q.x, q.y, q.z],
            "keys": sum(len(fc.keyframe_points) for fc in action.fcurves) if action else 0,
            "physics": None if body is None else (body.type, bool(body.kinematic), round(body.mass, 4),
                                                  round(body.friction, 3), round(body.restitution, 3),
                                                  body.collision_shape, round(body.linear_damping, 3),
                                                  round(body.angular_damping, 3))}
    return {"scene": _scene_state(), "objects": objects}


def evaluate(frame, subframe=0.0):
    """Let Blender evaluate the scene at this frame: keyframes, drivers and the physics cache are
    applied, so the scene is what renders there. A location or rotation set without a keyframe on an
    animated object is replaced by its keyed value; edits to unanimated properties stay."""
    bpy.context.scene.frame_set(frame, subframe=subframe)
    bpy.context.view_layer.update()


def _angle_deg(a, b):
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return math.degrees(2 * math.acos(min(1.0, dot)))


def diff(before, after):
    """A compact account of what changed: objects added or removed, moved at the current frame,
    keyframes added, physics changed, and scene settings."""
    label = lambda name, row: f"{row['id']} ({name})" if row["id"] else name
    b, a = before["objects"], after["objects"]
    out = {"added": [label(n, a[n]) for n in sorted(set(a) - set(b))],
           "removed": [label(n, b[n]) for n in sorted(set(b) - set(a))], "moved": [], "keyframes": [],
           "physics": [], "scene": {}}
    for name in sorted(set(a) & set(b)):
        x, y = b[name], a[name]
        shift = math.dist(x["location"], y["location"])
        turn = _angle_deg(x["rotation"], y["rotation"])
        if shift > POSITION_TOL_M or turn > ANGLE_TOL_DEG:
            out["moved"].append({"object": label(name, y), "shift_m": round(shift, 4), "turn_deg": round(turn, 2),
                                 "now": [round(v, 4) for v in y["location"]]})
        if x["keys"] != y["keys"]:
            out["keyframes"].append({"object": label(name, y), "keys": [x["keys"], y["keys"]]})
        if x["physics"] != y["physics"]:
            out["physics"].append({"object": label(name, y), "from": x["physics"], "to": y["physics"]})
    for key, value in after["scene"].items():
        if before["scene"].get(key) != value:
            out["scene"][key] = [before["scene"].get(key), value]
    for key in ("added", "removed", "moved", "keyframes", "physics"):
        if len(out[key]) > MAX_LISTED:
            out[key] = out[key][:MAX_LISTED] + [f"... {len(out[key]) - MAX_LISTED} more"]
    out["frame_now"] = bpy.context.scene.frame_current
    return {k: v for k, v in out.items() if v or k == "frame_now"}


def _sample_frames(first, last, count=12):
    if last <= first:
        return [first]
    step = max(1, (last - first) // (count - 1))
    return sorted(set(range(first, last + 1, step)) | {last})


def reference():
    """The scene as loaded: what the contract compares against after editing."""
    s = bpy.context.scene
    held = s.frame_current
    frames = _sample_frames(s.frame_start, s.frame_end)
    poses = {}
    for frame in frames:
        s.frame_set(frame)
        for o in bpy.data.objects:
            if o.get("instance_id") is not None:
                poses.setdefault(str(o["instance_id"]), {})[frame] = (list(o.matrix_world.to_translation()),
                                                                      list(o.matrix_world.to_quaternion()))
    cam = s.camera
    s.frame_set(s.frame_start)
    camera = None if cam is None else {"name": cam.name, "location": list(cam.matrix_world.to_translation()),
                                       "rotation": list(cam.matrix_world.to_quaternion())}
    s.frame_set(held)
    return {"frame_start": s.frame_start, "frame_end": s.frame_end, "fps": s.render.fps / s.render.fps_base,
            "poses": poses, "camera": camera}


def _poses_kept(ref, up_to, failures, what):
    s = bpy.context.scene
    held, worst, worst_turn = s.frame_current, None, None
    for uid, frames in ref["poses"].items():
        try:
            o = S.obj(uid)
        except ValueError:
            failures.append(f"object {uid} was deleted; {what} must keep every object")
            continue
        for frame, (location, rotation) in frames.items():
            if frame > up_to:
                continue
            s.frame_set(frame)
            shift = math.dist(location, o.matrix_world.to_translation())
            turn = _angle_deg(rotation, list(o.matrix_world.to_quaternion()))
            if shift > POSITION_TOL_M and (worst is None or shift > worst[0]):
                worst = (shift, uid, frame)
            if turn > ANGLE_TOL_DEG and (worst_turn is None or turn > worst_turn[0]):
                worst_turn = (turn, uid, frame)
    s.frame_set(held)
    if worst:
        failures.append(f"object {worst[1]} moved {worst[0] * 1000:.1f} mm at frame {worst[2]}: {what}")
    if worst_turn:
        failures.append(f"object {worst_turn[1]} turned {worst_turn[0]:.1f} deg at frame {worst_turn[2]}: {what}")


def _camera_kept(ref, failures):
    s = bpy.context.scene
    if ref["camera"] is None:
        return
    if s.camera is None or s.camera.name != ref["camera"]["name"]:
        failures.append(f"the scene camera must stay {ref['camera']['name']!r} (the source camera)")
        return
    held = s.frame_current
    s.frame_set(ref["frame_start"])
    shift = math.dist(ref["camera"]["location"], s.camera.matrix_world.to_translation())
    turn = _angle_deg(ref["camera"]["rotation"], list(s.camera.matrix_world.to_quaternion()))
    s.frame_set(held)
    if shift > POSITION_TOL_M or turn > ANGLE_TOL_DEG:
        failures.append("the source camera was moved; keep it as reconstructed")


def check(family, spec, ref):
    """The task contract. Returns the list of failures; empty means the update may be submitted."""
    s = bpy.context.scene
    failures = []
    fps = s.render.fps / s.render.fps_base
    if abs(fps - ref["fps"]) > 1e-6:
        failures.append(f"keep the scene at {ref['fps']} fps (now {fps})")
    if family == "T5":
        target = spec["target_camera"]
        cam = s.camera
        if [s.frame_start, s.frame_end] != [ref["frame_start"], ref["frame_end"]]:
            failures.append(f"keep the frame range {ref['frame_start']}-{ref['frame_end']}: T5 renders the same instants")
        if cam is None or cam.type != "CAMERA":
            failures.append("there is no scene camera; make target_camera the scene camera (S.set_camera)")
        else:
            bpy.context.view_layer.update()
            shift = math.dist(cam.matrix_world.to_translation(), target["position"])
            turn = _angle_deg(list(cam.matrix_world.to_quaternion()), target["quaternion_wxyz"])
            if shift > POSITION_TOL_M or turn > ANGLE_TOL_DEG:
                failures.append(f"the scene camera is {shift:.3f} m / {turn:.2f} deg from target_camera")
            if abs(cam.data.lens - target["focal_length_mm"]) > 1e-3 or abs(cam.data.sensor_width - target["sensor_width_mm"]) > 1e-3:
                failures.append("the scene camera's focal length / sensor width differ from target_camera")
            if [s.render.resolution_x, s.render.resolution_y] != list(target["resolution_wh"]):
                failures.append(f"render resolution must be {target['resolution_wh']}")
    elif family == "T6":
        if s.frame_start != ref["frame_start"]:
            failures.append("keep frame_start; frame 1 is the given starting frame")
        if s.frame_end < s.frame_start + round(3 * fps):
            failures.append("the timeline is shorter than 3 s: extend it (S.timeline) and animate the manipulation")
        _poses_kept(ref, ref["frame_start"], failures, "frame 1 must stay the starting frame")
        _camera_kept(ref, failures)
    elif family == "T7":
        want = ref["frame_end"] + round(float(spec["future_seconds"]) * ref["fps"])
        if s.frame_start != ref["frame_start"] or s.frame_end != want:
            failures.append(f"the timeline must run {ref['frame_start']}-{want}: exactly the {spec['future_seconds']} s "
                            f"after the last observed frame {ref['frame_end']} (now {s.frame_start}-{s.frame_end})")
        _poses_kept(ref, ref["frame_end"], failures, "the observed frames must keep their reconstructed poses")
        _camera_kept(ref, failures)
        world = s.rigidbody_world
        if world is None or not world.point_cache.is_baked or world.point_cache.frame_end < s.frame_end:
            failures.append("bake the simulation to the end of the timeline (S.bake()) so the future is fixed")
    return failures


def overview():
    s = bpy.context.scene
    return {"scene": _scene_state(), "objects": S.objects(),
            "environment": sorted(o.name for o in bpy.data.objects if o.type == "MESH" and o.get("instance_id") is None
                                  and o.get("part_of") is None)[:60]}


def render(spec):
    """Write frames to spec['directory'] with the scene camera; settings come from the host, not the model."""
    s = bpy.context.scene
    out = Path(spec["directory"])
    out.mkdir(parents=True, exist_ok=True)
    s.render.engine = spec.get("engine", "CYCLES")
    if s.render.engine == "CYCLES":
        s.cycles.samples = int(spec.get("samples", 32))
        s.cycles.device = "CPU"
    s.render.resolution_percentage = int(spec.get("percentage", 100))
    s.render.image_settings.file_format = "PNG"
    s.render.use_persistent_data = True       # measured pixel-identical, ~1.7x faster across frames
    first, last = int(spec.get("frame_start", s.frame_start)), int(spec.get("frame_end", s.frame_end))
    written = 0
    for frame in range(first, last + 1):
        s.frame_set(frame)
        s.render.filepath = str(out / f"frame_{frame:06d}")
        bpy.ops.render.render(write_still=True)
        written += 1
    return {"directory": str(out), "frames": [first, last], "written": written,
            "fps": s.render.fps / s.render.fps_base, "camera": s.camera.name if s.camera else None,
            "resolution": [s.render.resolution_x, s.render.resolution_y]}
