"""The helper module bound as ``S`` inside run_python. Runs inside Blender only.

A small vocabulary on top of bpy for the edits update tasks need. It follows the conventions of
the codify library that wrote the scene (scene.py): an object is a mesh with an ``instance_id``
custom property; a movable part is a child with ``part_of``, ``part_name`` and a JSON ``joint``
{type, axis, pivot (root frame), limits, value, rest_value}; every object is a rigid body.
Anything not covered here is plain bpy, which the model may always use directly.
"""
from __future__ import annotations

import json
import math

import bpy
from mathutils import Quaternion, Vector

CAMERA_NAME = "TaskCamera"


def obj(ref):
    """The Blender object for an instance id ('m3'), a part ('m3/lid'), or an object name."""
    ref = str(ref)
    if "/" in ref and not bpy.data.objects.get(ref):
        root_id, part = ref.split("/", 1)
        root = obj(root_id)
        for child in root.children:
            if str(child.get("part_name")) == part or child.name.endswith("/" + part):
                return child
        raise ValueError(f"{root_id} has no part named {part!r}; S.objects() lists the parts")
    for candidate in bpy.data.objects:
        if str(candidate.get("instance_id")) == ref:
            return candidate
    found = bpy.data.objects.get(ref)
    if found is None:
        raise ValueError(f"No object with instance_id or name {ref!r}; S.objects() lists them")
    return found


def _joint(part):
    raw = part.get("joint")
    if raw is None:
        return None
    return json.loads(raw) if isinstance(raw, str) else dict(raw)


def _box(o):
    corners = [o.matrix_world @ Vector(c) for c in o.bound_box]
    low = [min(c[i] for c in corners) for i in range(3)]
    high = [max(c[i] for c in corners) for i in range(3)]
    return low, high


def _physics(o):
    body = o.rigid_body
    if body is None:
        return None
    return {"role": body.type.lower(), "kinematic": bool(body.kinematic), "mass_kg": round(body.mass, 4),
            "friction": round(body.friction, 3), "restitution": round(body.restitution, 3),
            "shape": body.collision_shape, "linear_damping": round(body.linear_damping, 3),
            "angular_damping": round(body.angular_damping, 3)}


def objects():
    """Every registered object: id, name, category, box at the current frame, physics, animation, parts."""
    table = []
    for o in sorted(bpy.data.objects, key=lambda x: str(x.get("instance_id", "~"))):
        if o.get("instance_id") is None:
            continue
        low, high = _box(o)
        parts = []
        for child in o.children:
            if child.get("part_of") is None:
                continue
            joint = _joint(child) or {}
            parts.append({"ref": f"{o['instance_id']}/{child.get('part_name')}", "joint": joint.get("type"),
                          "axis": joint.get("axis"), "limits": joint.get("limits"), "value": joint.get("value"),
                          "rest_value": joint.get("rest_value")})
        action = o.animation_data.action if o.animation_data else None
        table.append({"id": str(o["instance_id"]), "name": o.name, "category": o.get("category"),
                      "motion_type": o.get("motion_type"),
                      "center": [round((a + b) / 2, 4) for a, b in zip(low, high)],
                      "size": [round(b - a, 4) for a, b in zip(low, high)],
                      "physics": _physics(o),
                      "animated": sorted({fc.data_path for fc in action.fcurves}) if action else [],
                      "parts": parts})
    return table


def pose(ref, frame=None):
    """World location, rotation (quaternion wxyz) and box of an object at a frame (default: current)."""
    s = bpy.context.scene
    held = s.frame_current
    if frame is not None:
        s.frame_set(int(frame))
    try:
        o = obj(ref)
        m = o.matrix_world
        low, high = _box(o)
        q = m.to_quaternion()
        return {"frame": s.frame_current, "location": [round(v, 5) for v in m.to_translation()],
                "quaternion_wxyz": [round(v, 6) for v in (q.w, q.x, q.y, q.z)],
                "box_min": [round(v, 4) for v in low], "box_max": [round(v, 4) for v in high]}
    finally:
        if frame is not None:
            s.frame_set(held)


def _on_keys(o):
    """A rigid body follows its keyframes only when kinematic; an active body ignores them."""
    body = o.rigid_body
    if body is None:
        return False
    changed = not body.kinematic
    action = o.animation_data.action if o.animation_data else None
    if action:
        for fc in [fc for fc in action.fcurves if fc.data_path == "rigid_body.kinematic"]:
            action.fcurves.remove(fc)
            changed = True
    body.kinematic = True
    return changed


def key(ref, frame, location=None, quaternion_wxyz=None, interpolation=None):
    """Set and keyframe an object's location and/or rotation at a frame (rotation as quaternion wxyz).
    A rigid body is put on its keys (kinematic) so the solver does not override the motion."""
    o = obj(ref)
    frame = int(frame)
    took_from_solver = _on_keys(o)
    if location is not None:
        o.location = Vector(location)
        o.keyframe_insert("location", frame=frame)
    if quaternion_wxyz is not None:
        o.rotation_mode = "QUATERNION"
        o.rotation_quaternion = Quaternion(quaternion_wxyz)
        o.keyframe_insert("rotation_quaternion", frame=frame)
    if interpolation and o.animation_data and o.animation_data.action:
        for fc in o.animation_data.action.fcurves:
            for point in fc.keyframe_points:
                if int(point.co[0]) == frame:
                    point.interpolation = interpolation
    return {"id": ref, "frame": frame, "now_kinematic": took_from_solver}


def set_joint(ref, value, frame=None):
    """Move a part to joint position `value` (radians for revolute, meters for prismatic) relative to how
    it was built (rest_value), exactly as codify's set_joint does; keyframe it if `frame` is given."""
    part = obj(ref)
    joint = _joint(part)
    if joint is None or joint.get("type") not in ("revolute", "prismatic"):
        raise ValueError(f"{ref} is not a revolute or prismatic part")
    lo, hi = joint.get("limits", [-math.inf, math.inf])
    if lo != hi and not lo - 1e-6 <= float(value) <= hi + 1e-6:
        raise ValueError(f"{value} is outside the joint limits {joint['limits']}")
    delta = float(value) - float(joint.get("rest_value", 0.0))
    axis, pivot = Vector(joint["axis"]), Vector(joint["pivot"])
    part.rotation_mode = "QUATERNION"
    if joint["type"] == "revolute":
        part.rotation_quaternion = Quaternion(axis, delta)
        part.location = pivot
    else:
        part.rotation_quaternion = Quaternion((1, 0, 0, 0))
        part.location = pivot + axis * delta
    joint["value"] = float(value)
    part["joint"] = json.dumps(joint)
    _on_keys(part)
    if frame is not None:
        part.keyframe_insert("location", frame=int(frame))
        part.keyframe_insert("rotation_quaternion", frame=int(frame))
    return {"part": ref, "value": float(value), "frame": frame}


def timeline(end, start=None):
    """Set the frame range; the rigid-body cache is stretched to cover it."""
    s = bpy.context.scene
    if start is not None:
        s.frame_start = int(start)
    s.frame_end = int(end)
    world = s.rigidbody_world
    if world is not None:
        world.point_cache.frame_end = max(world.point_cache.frame_end, s.frame_end)
    return {"frame_start": s.frame_start, "frame_end": s.frame_end, "fps": s.render.fps / s.render.fps_base}


def set_camera(position, quaternion_wxyz, focal_length_mm, sensor_width_mm, resolution_wh, name=CAMERA_NAME, **_ignored):
    """Make a camera with this pose and intrinsics the scene camera (the render resolution follows).
    Extra keys of a task camera (look_at, fov_rad) are accepted and ignored: S.set_camera(**camera)."""
    s = bpy.context.scene
    cam = bpy.data.objects.get(name)
    if cam is None or cam.type != "CAMERA":
        cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
        s.collection.objects.link(cam)
    cam.location = Vector(position)
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = Quaternion(quaternion_wxyz)
    cam.data.lens = float(focal_length_mm)
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.sensor_width = float(sensor_width_mm)
    s.camera = cam
    s.render.resolution_x, s.render.resolution_y = int(resolution_wh[0]), int(resolution_wh[1])
    s.render.resolution_percentage = 100
    bpy.context.view_layer.update()       # a new object's matrix_world is stale until this
    return {"camera": cam.name, "location": list(cam.matrix_world.to_translation())}


def physics(ref, role=None, mass_kg=None, friction=None, restitution=None, shape=None,
            linear_damping=None, angular_damping=None):
    """Change a rigid body. role 'active' is simulated, 'passive' only collides. There is no rolling
    resistance in the solver: a rolling ball slows only through angular_damping."""
    o = obj(ref)
    body = o.rigid_body
    if body is None:
        raise ValueError(f"{ref} has no rigid body")
    if role is not None:
        body.type = str(role).upper()
    if shape is not None:
        body.collision_shape = str(shape).upper()
    for name, value in (("mass", mass_kg), ("friction", friction), ("restitution", restitution),
                        ("linear_damping", linear_damping), ("angular_damping", angular_damping)):
        if value is not None:
            setattr(body, name, float(value))
    return _physics(o)


def release(ref, frame):
    """On its keys up to and including `frame`, simulated from frame+1 (keys the kinematic flag)."""
    o = obj(ref)
    if o.rigid_body is None:
        raise ValueError(f"{ref} has no rigid body")
    o.rigid_body.type = "ACTIVE"
    action = o.animation_data.action if o.animation_data else None
    if action:
        for fc in [fc for fc in action.fcurves if fc.data_path == "rigid_body.kinematic"]:
            action.fcurves.remove(fc)
    for at, value in ((int(frame), True), (int(frame) + 1, False)):
        o.rigid_body.kinematic = value
        o.keyframe_insert("rigid_body.kinematic", frame=at)
    for fc in o.animation_data.action.fcurves:
        if fc.data_path == "rigid_body.kinematic":
            for point in fc.keyframe_points:
                point.interpolation = "CONSTANT"
    return {"id": ref, "released_after": int(frame)}


def hold(ref):
    """Keep a body on its keys for the whole timeline (held by a hand or fixed in place)."""
    o = obj(ref)
    _on_keys(o)
    return {"id": ref, "kinematic": True}


def freeze_physics():
    """Put every rigid body on its keys: nothing is simulated, only keyframes move objects."""
    frozen = []
    for o in bpy.data.objects:
        if o.rigid_body is not None and o.rigid_body.type == "ACTIVE":
            _on_keys(o)
            frozen.append(o.name)
    return {"kinematic": frozen}


def bake(end=None):
    """(Re)bake the rigid-body simulation to the end of the timeline. Query future frames only after this:
    jumping to an unbaked frame does not run the solver."""
    s = bpy.context.scene
    world = s.rigidbody_world
    if world is None:
        raise ValueError("The scene has no rigid-body world")
    cache = world.point_cache
    cache.frame_end = max(int(end or s.frame_end), s.frame_end)
    with bpy.context.temp_override(scene=s, point_cache=cache):
        if cache.is_baked:
            bpy.ops.ptcache.free_bake()
        bpy.ops.ptcache.bake(bake=True)
    return {"cache": [cache.frame_start, cache.frame_end], "baked": bool(cache.is_baked)}
