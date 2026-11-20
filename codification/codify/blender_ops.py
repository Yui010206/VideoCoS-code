"""Blender-side operations run inside the sandbox (Blender 4.5, background).

  exec    run generated scene.py in a fresh empty scene, pack resources, save scene.blend,
          write inventory.json (exit 3 with exec_error.txt on any exception)
  render  render the scene camera (or a given one) at given frames into PNGs plus render_info.json
  check   structural checks, evidence residuals and one simulated second into check.json
  dump    ids, local extents, world matrices and camera state at given frames (evaluation and tests)
  ids     the registered object and camera depth seen through each cell of a pixel grid at given frames
"""
import argparse
import contextlib
import io
import json
import math
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codify import contract  # noqa: E402

ENVIRONMENT = "Environment"


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=float))


def keyed(obj):
    """True when obj has keyed location or rotation (rigid-body switches alone are not motion)."""
    action = obj.animation_data.action if obj.animation_data else None
    return bool(action) and any(fc.data_path in ("location", "rotation_quaternion", "rotation_euler")
                                for fc in action.fcurves)


def physics(obj):
    body = obj.rigid_body
    if body is None:
        return None
    record = json.loads(obj["physics"]) if "physics" in obj else {}
    return {"role": body.type.lower(), "mass_kg": round(record.get("mass_kg") or body.mass, 4),
            "friction": round(body.friction, 3), "restitution": round(body.restitution, 3),
            "shape": body.collision_shape, "behavior": record.get("behavior", "rigid"),
            "behavior_declared": "behavior" in record, "properties": record.get("properties", {})}


def inventory():
    scene = bpy.context.scene
    world = scene.rigidbody_world
    own = {c for c in (getattr(world, "collection", None), getattr(world, "constraints", None)) if c}
    semantic, environment, other, colliders = [], [], [], 0
    parts = {}
    for obj in scene.objects:
        if "part_of" in obj and obj.type == "MESH":
            joint = json.loads(obj["joint"]) if "joint" in obj else {"type": "fixed"}
            parts.setdefault(str(obj["part_of"]), []).append({"name": str(obj["part_name"]), "vertices": len(obj.data.vertices),
                                                              "joint": {k: joint[k] for k in ("type", "axis", "pivot", "limits", "value") if k in joint},
                                                              "physics": physics(obj)})
    for obj in scene.objects:
        if "instance_id" in obj:
            semantic.append({"instance_id": str(obj["instance_id"]), "name": obj.name, "type": obj.type,
                             "category": str(obj.get("category", "")), "motion_type": str(obj.get("motion_type", "")),
                             "collections": [c.name for c in obj.users_collection if c not in own],
                             "vertices": len(obj.data.vertices) if obj.type == "MESH" else 0,
                             "animated": keyed(obj), "physics": physics(obj),
                             "parts": parts.get(str(obj["instance_id"]), []),
                             **({"frame_texture": json.loads(obj["frame_texture"])} if "frame_texture" in obj else {})})
        elif obj.type == "MESH" and "part_of" not in obj:
            in_env = any(c.name == ENVIRONMENT for c in obj.users_collection)
            (environment if in_env else other).append(obj.name)
            colliders += bool(in_env and obj.rigid_body)
    images = [{"name": i.name, "packed": i.packed_file is not None} for i in bpy.data.images
              if i.source not in ("GENERATED", "VIEWER")]
    return {"semantic_objects": semantic, "environment_meshes": environment, "unregistered_meshes": other,
            "environment_colliders": colliders, "plan": json.loads(scene.get("codify_plan", "{}") or "{}"),
            "separations": json.loads(scene.get("codify_separations", "[]") or "[]"),
            "camera": scene.camera.name if scene.camera else None,
            "camera_type": scene.camera.data.type if scene.camera else None,
            "frame_range": [scene.frame_start, scene.frame_end], "fps": scene.render.fps / scene.render.fps_base,
            "resolution": [scene.render.resolution_x, scene.render.resolution_y], "images": images,
            "lights": sum(o.type == "LIGHT" for o in scene.objects)}


def run_exec(args):
    out = Path(args.out)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    code = Path(args.code).read_text()
    printed = io.StringIO()       # the scene code's own prints go back to the model
    try:
        with contextlib.redirect_stdout(printed):
            exec(compile(code, "scene.py", "exec"), {"__name__": "__main__", "__file__": "scene.py"})
    except BaseException:
        (out / "exec_error.txt").write_text(traceback.format_exc(limit=8))
        sys.exit(3)
    finally:
        (out / "exec_stdout.txt").write_text(printed.getvalue())
    try:        # what the declared behaviors need in the simulation (magnet fields)
        from codify import assets
        assets.apply_behaviors()
    except BaseException:
        (out / "exec_error.txt").write_text("after scene.py, building the declared behaviors failed:\n"
                                            + traceback.format_exc(limit=8))
        sys.exit(3)
    if bpy.data.images:
        bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "scene.blend"))
    _write(out / "inventory.json", inventory())


def overview_camera(scene, view, frame):
    """A temporary orthographic view of the room as a whole (nothing is saved into the .blend), framed on the registered
    objects. top: from above, without what covers the room. side_x: looking along +Y (X left to right, Z up); side_y: looking
    along +X (Y right to left, Z up): without walls or ceiling, with a red line at floor level, so what stands on what and
    what hangs in the air shows."""
    scene.frame_set(frame)
    meshes = [o for o in scene.objects if o.type == "MESH"]
    registered = [o for o in meshes if "instance_id" in o or "part_of" in o]

    def box(o):
        c = np.array([v[:] for v in o.bound_box]) @ np.array(o.matrix_world)[:3, :3].T + np.array(o.matrix_world)[:3, 3]
        return c.min(0), c.max(0)
    boxes = [box(o) for o in (registered or meshes)]
    lo, hi = np.min([b[0] for b in boxes], 0), np.max([b[1] for b in boxes], 0)
    shell = [o for o in meshes if o not in registered]
    floor = min((box(o)[1][2] for o in shell), default=float(lo[2]))
    lo[2] = min(lo[2], floor)
    room = max(float(np.prod(hi[:2] - lo[:2])), 1e-6)
    for o in shell:
        a, b = box(o)
        name = o.name.lower()
        overhead = "ceiling" in name or "roof" in name or (a[2] > floor + 1.6 and float(np.prod(b[:2] - a[:2])) > 0.2 * room)
        tall = b[2] - a[2] > 0.5 or "wall" in name
        if overhead or (view != "top" and tall):
            o.hide_render = True
    data = bpy.data.cameras.new("Overview")
    data.type, data.clip_end = "ORTHO", float(np.max(hi - lo)) + 30.0
    cam = bpy.data.objects.new("Overview", data)
    scene.collection.objects.link(cam)
    centre = (lo + hi) / 2
    if view == "top":
        data.ortho_scale = float(max(hi[:2] - lo[:2]) * 1.1)
        cam.location = (centre[0], centre[1], hi[2] + 5)
    else:
        along_y = view == "side_x"
        data.ortho_scale = float(max(hi[0 if along_y else 1] - lo[0 if along_y else 1], hi[2] - lo[2]) * 1.1)
        cam.location = (centre[0], lo[1] - 8, centre[2]) if along_y else (lo[0] - 8, centre[1], centre[2])
        cam.rotation_euler = (math.pi / 2, 0.0, 0.0 if along_y else -math.pi / 2)
        mat = bpy.data.materials.new("FloorLine")
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.9, 0.05, 0.05, 1)
        mat.node_tree.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (0.9, 0.05, 0.05, 1)
        mat.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 1.0
        half = float(np.max(hi - lo)) * 2
        t = max(0.012, 0.004 * float(np.max(hi - lo)))
        z0, z1 = floor - 2 * t, floor
        v = [(x, y, z) for x in (centre[0] - half, centre[0] + half) for y in (centre[1] - half, centre[1] + half) for z in (z0, z1)]
        f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        line = bpy.data.meshes.new("FloorLine")
        line.from_pydata(v, [], f)
        line.update()
        line.materials.append(mat)
        obj = bpy.data.objects.new("FloorLine", line)
        scene.collection.objects.link(obj)
    scene.camera = cam
    for marker in scene.timeline_markers:
        marker.camera = None
    scene.render.resolution_x = scene.render.resolution_y = max(scene.render.resolution_x, scene.render.resolution_y)
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1


def run_render(args):
    scene = bpy.context.scene
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if scene.camera is None:
        raise SystemExit("Scene has no camera")
    if args.view:                     # a view of the whole room (rooms, v2)
        overview_camera(scene, args.view, int(args.frames.split(",")[0]))
    elif args.top:                    # every other task: the earlier top view
        # Temporary orthographic top-down view of the registered objects (not saved into the .blend); framed on them,
        # not on a floor that runs far past the image edges, which would leave the objects a few pixels each.
        scene.frame_set(int(args.frames.split(",")[0]))
        meshes = [o for o in scene.objects if o.type == "MESH"]
        framed = [o for o in meshes if "instance_id" in o or "part_of" in o] or meshes
        points = [np.array([c[:] for c in o.bound_box]) @ np.array(o.matrix_world)[:3, :3].T + np.array(o.matrix_world)[:3, 3]
                  for o in framed]
        lo, hi = np.concatenate(points).min(0), np.concatenate(points).max(0)
        data = bpy.data.cameras.new("TopView")
        data.type, data.ortho_scale, data.clip_end = "ORTHO", float(max(hi[:2] - lo[:2]) * 1.15), float(hi[2] - lo[2] + 10)
        cam = bpy.data.objects.new("TopView", data)
        scene.collection.objects.link(cam)
        cam.location = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] + 5)
        scene.camera = cam
        scene.render.resolution_y = scene.render.resolution_x
        scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    if args.camera:
        # Temporary static camera from a spec (a T5 target view: position, Blender-axis quaternion, lens), not saved.
        spec = json.loads(Path(args.camera).read_text())
        data = bpy.data.cameras.new("GivenView")
        data.lens, data.sensor_width, data.sensor_fit = spec["focal_length_mm"], spec["sensor_width_mm"], "HORIZONTAL"
        data.clip_end = 1000.0
        cam = bpy.data.objects.new("GivenView", data)
        scene.collection.objects.link(cam)
        cam.location = spec["position"]
        cam.rotation_mode, cam.rotation_quaternion = "QUATERNION", spec["quaternion_wxyz"]
        for marker in scene.timeline_markers:          # camera switches bound to frames would take the view back
            marker.camera = None
        scene.camera = cam
        scene.render.resolution_x, scene.render.resolution_y = spec["resolution_wh"]
        scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scale = args.size / max(scene.render.resolution_x, scene.render.resolution_y)
    scene.render.resolution_percentage = max(1, min(100, round(100 * scale)))
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.render.threads_mode = "FIXED"
    scene.render.threads = args.threads
    scene.render.image_settings.file_format = "PNG"
    info = {"frames": [], "percentage": scene.render.resolution_percentage, "samples": args.samples,
            "engine": scene.render.engine}
    frames = [int(f) for f in args.frames.split(",")]
    step_to(scene, max(frames))          # the rigid-body cache stepped in order up to the last frame rendered
    if args.sweep_joints:
        # Every movable part of the scene moved through its joint range from the source camera at the first frame,
        # one part after another (open, then back): sweep_<k>.png. Shows what can be acted on, for reports only.
        from codify import assets
        scene.frame_set(frames[0])
        parts = [o for o in scene.objects if "joint" in o and json.loads(o["joint"]).get("type") in ("revolute", "prismatic")]
        k, swept = 0, []
        for part in sorted(parts, key=lambda o: o.name):
            joint = json.loads(part["joint"])
            lo, hi = joint.get("limits", [joint["value"], joint["value"]])
            start_value = float(joint["value"])
            far = hi if abs(hi - start_value) >= abs(lo - start_value) else lo
            steps = max(2, args.sweep_joints)
            path = [start_value + (far - start_value) * i / (steps - 1) for i in range(steps)]
            for value in path + path[::-1]:
                assets.set_joint(part, value)
                scene.render.filepath = str(out / f"sweep_{k:06d}.png")
                bpy.ops.render.render(write_still=True)
                k += 1
            assets.set_joint(part, start_value)
            swept.append({"part": part.name, "part_of": str(part.get("part_of")), "type": joint["type"],
                          "from": start_value, "to": far, "frames": 2 * steps})
        info["sweep"] = swept
    for frame in frames:
        scene.frame_set(frame)
        scene.render.filepath = str(out / f"render_{frame:06d}.png")
        bpy.ops.render.render(write_still=True)
        cam = scene.camera
        info["frames"].append({"frame": frame, "file": f"render_{frame:06d}.png",
                               "camera_matrix_world": [list(r) for r in cam.matrix_world],
                               "camera_type": cam.data.type, "lens": cam.data.lens,
                               "shift": [cam.data.shift_x, cam.data.shift_y]})
    _write(out / "render_info.json", info)


def scene_K(scene):
    cam = scene.camera.data
    r = scene.render
    return contract.blender_to_pinhole(cam.lens, cam.sensor_width, cam.shift_x, cam.shift_y, r.pixel_aspect_x,
                                       r.pixel_aspect_y, r.resolution_x, r.resolution_y)


def projected_box(obj, vertices, scene, K):
    """Half-open pixel AABB of the mesh at the current frame, or None if behind the camera."""
    if not len(vertices):
        return None
    M = np.array(obj.matrix_world)
    world = vertices @ M[:3, :3].T + M[:3, 3]
    cw = np.array(scene.camera.matrix_world)
    R = cw[:3, :3] @ contract.FLIP_YZ
    cam = (world - cw[:3, 3]) @ R
    if np.any(cam[:, 2] < 0.05):
        return None
    uv = contract.project_pinhole(K, cam) + 0.5
    W, H = scene.render.resolution_x, scene.render.resolution_y
    box = [max(0.0, uv[:, 0].min()), max(0.0, uv[:, 1].min()), min(W, uv[:, 0].max()), min(H, uv[:, 1].max())]
    return box if box[2] > box[0] and box[3] > box[1] else None


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def step_to(scene, frame):
    """Rigid bodies are simulated frame by frame: step the world from its cache start up to frame."""
    world = scene.rigidbody_world
    if world is not None and frame > scene.frame_end:
        for f in range(world.point_cache.frame_start, frame + 1):
            scene.frame_set(f)


def simulate(scene, seconds=1.0):
    """The physics check: step the rigid-body world to `seconds` past the last input frame and report each
    active object's release speed (from its keys), how far it went and its top speed, with flags for what a
    plausible scene does not do."""
    world, last = scene.rigidbody_world, scene.frame_end
    if world is None:
        return {"simulated": False, "problems": []}
    fps = scene.render.fps / scene.render.fps_base
    end = last + max(1, round(seconds * fps))
    active = [o for o in scene.objects if "instance_id" in o and o.rigid_body and o.rigid_body.type == "ACTIVE"]
    lowest = [np.array([o.matrix_world @ v.co for v in o.data.vertices])[:, 2].min() for o in scene.objects
              if o.get("environment") and o.rigid_body and o.type == "MESH" and len(o.data.vertices)]
    track = {o.name: [] for o in active}
    for f in range(world.point_cache.frame_start, end + 1):
        scene.frame_set(f)
        for o in active:
            track[o.name].append(np.array(o.matrix_world.translation))
    start = world.point_cache.frame_start
    # What a magnet attracts is expected to move after release.
    attracted = {a for o in scene.objects if "physics" in o
                 for a in json.loads(o["physics"]).get("properties", {}).get("attracts", [])}
    per_object, problems = {}, []
    for o in active:
        P = np.array(track[o.name])
        i = last - start
        release = float(np.linalg.norm(P[i] - P[i - 1]) * fps) if i > 0 else 0.0
        future = P[i:]
        moved = float(np.linalg.norm(future - P[i], axis=1).max())
        top = float((np.linalg.norm(np.diff(future, axis=0), axis=1) * fps).max()) if len(future) > 1 else 0.0
        size = float(max(o.dimensions))
        uid = str(o["instance_id"])
        flags = []
        if release < 0.05 and moved > max(0.03, 0.25 * size) and uid not in attracted:
            flags.append("at rest in the input but moved after release (no support under it, or it intersects "
                         "another body)")
        if top > 15:
            flags.append(f"flew off at {top:.0f} m/s (it intersects another body at release)")
        if lowest and future[-1][2] < min(lowest) - 0.05:
            flags.append("fell below every collider")
        per_object[uid] = {"release_speed_mps": round(release, 3), "moved_m": round(moved, 3),
                           "top_speed_mps": round(top, 2), "position_end": np.round(future[-1], 3).tolist(),
                           **({"flags": flags} if flags else {})}
        problems += [f"{uid}: {flag}" for flag in flags]
    scene.frame_set(1)
    return {"simulated": True, "seconds": round((end - last) / fps, 2), "active": len(active),
            "passive": sum(bool(o.rigid_body) and o.rigid_body.type == "PASSIVE" for o in scene.objects),
            "per_active_object": per_object, "problems": problems}


def layout_failures(scene, contacts, count, layout):
    """The layout gate of a room run: arrange_room ran on the scene, few objects float above what is under them (hung
    ones aside) and few pairs cross deeper than separate_objects resolves."""
    arranged = json.loads(scene.get("codify_arrangement", "null") or "null")
    if arranged is None:
        return ["layout: end scene.py with A.arrange_room() (it seats, turns and separates the objects and reports "
                "what it could not place)"]
    out = []
    hung = set(arranged.get("kept_hanging", []))
    floating = [f for f in contacts["floating"] if f["id"] not in hung and (f["gap_m"] is None or f["gap_m"] > 0.03)]    # 1-3 cm is seating noise
    from codify import arrange, gates
    groups = gates.semantic_groups(scene)
    furniture = [f for f in floating if arrange._words(groups[f["id"]][0].get("category", "")) & arrange.STANDS
                 and (f["gap_m"] is None or f["gap_m"] > 0.05)]
    if len(furniture) > layout.get("max_floating_furniture", 2):
        listed = ", ".join(f"{f['id']} {groups[f['id']][0].get('category', '')} ({'nothing below' if f['gap_m'] is None else str(round(100 * f['gap_m'])) + ' cm'})"
                           for f in sorted(furniture, key=lambda f: -(f["gap_m"] or 9))[:25])
        out.append(f"layout: furniture must stand on the floor; these do not: {listed}. Put each on the floor (stand= in "
                   "arrange_room); give a table or counter that is only a top its legs or body; or leave it out in submit if it is not there")
    allowed = max(5, int(layout.get("max_floating_share", 0.1) * count))
    if len(floating) > allowed:
        gap = lambda f: "nothing below" if f["gap_m"] is None else f"{100 * f['gap_m']:.0f} cm"
        listed = ", ".join(f"{f['id']} ({gap(f)})" for f in sorted(floating, key=lambda f: -(f["gap_m"] or 9))[:25])
        out.append(f"layout: {len(floating)} of {count} objects float above what is under them (at most {allowed}): "
                   f"{listed}. For each, from the frames: build what it stands on (A.add_support, or the furniture), "
                   "pass it to arrange_room in hang= (on a wall or a ceiling) or stand= (on the floor), or leave it "
                   "out in submit (misplaced, a duplicate, not an object)")
    deep_m = layout.get("deep_m", 0.08)
    deep = [p for p in contacts["penetrations"] if p["depth_m"] > deep_m]
    allowed = max(5, int(layout.get("max_deep_share", 0.15) * count))
    if len(deep) > allowed:
        listed = ", ".join(f"{p['a']}/{p['b']} ({100 * p['depth_m']:.0f} cm)" for p in sorted(deep, key=lambda p: -p["depth_m"])[:25])
        out.append(f"layout: {len(deep)} pairs cross by more than {100 * deep_m:.0f} cm (at most {allowed}): {listed}. "
                   "Two that are one object: keep one (same_as in submit); a box too big for its object: give it the "
                   "size the frames show; a misplaced one: move it where the frames put it or leave it out")
    return out


def behavior_check(scene, inv, warnings):
    """What the objects' declared behaviors are; magnets that would also pull objects they do not attract (every
    released body within reach_m is pulled), and objects whose behavior was not declared (they are rigid)."""
    objects = inv["semantic_objects"]
    behaviors = {o["instance_id"]: o["physics"]["behavior"] for o in objects if o["physics"]}
    undeclared = [o["instance_id"] for o in objects if o["physics"] and not o["physics"]["behavior_declared"]]
    if undeclared:
        warnings.append(f"physics: no behavior declared (rigid assumed) for {undeclared[:20]}: say what each one does "
                        "when things act on it (set_physics behavior=...), rigid included")
    by_id = {str(o["instance_id"]): o for o in scene.objects if "instance_id" in o}
    scene.frame_set(scene.frame_end)
    stray = []
    for o in objects:
        p = o["physics"]
        if not p or p["behavior"] != "magnetic":
            continue
        magnet = by_id[o["instance_id"]].matrix_world.translation
        reach, attracts = float(p["properties"].get("reach_m", 0)), set(p["properties"].get("attracts", []))
        for q in objects:
            if q["instance_id"] == o["instance_id"] or q["instance_id"] in attracts or not q["physics"] \
                    or q["physics"]["role"] != "active":
                continue
            if (by_id[q["instance_id"]].matrix_world.translation - magnet).length < reach:
                stray.append({"magnet": o["instance_id"], "object": q["instance_id"]})
    if stray:
        warnings.append(f"physics: magnets would also pull objects they do not attract (within reach_m at the last "
                        f"frame): {stray[:10]}; shorten reach_m or say why")
    scene.frame_set(1)
    return {"declared": behaviors, "undeclared": undeclared, "magnet_strays": stray,
            "applied": json.loads(scene.get("codify_behaviors", "[]") or "[]")}


def run_check(args):
    scene = bpy.context.scene
    ev = Path(args.evidence) if args.evidence else None
    read = lambda name: json.loads((ev / name).read_text()) if ev and (ev / name).is_file() else None
    video, cameras = read("video.json"), read("cameras.json")
    inv = inventory()
    failures, warnings, metrics = [], [], {}
    # First, on a fresh cache: one simulated second past the input (the physics check).
    metrics["physics"] = simulate(scene)
    warnings += [f"physics: {p}" for p in metrics["physics"]["problems"]]
    missing = [o["instance_id"] for o in inv["semantic_objects"] if not o["physics"]]
    if missing:
        failures.append(f"objects without physics (A.set_physics, or add_object(physics=...)): {missing[:20]}")
    if inv["semantic_objects"] and not inv["environment_colliders"]:
        warnings.append("physics: no environment collider (released objects fall forever)")
    passive_movers = [o["instance_id"] for o in inv["semantic_objects"]
                      if o["animated"] and o["physics"] and o["physics"]["role"] == "passive"]
    if passive_movers:
        warnings.append(f"physics: moving objects with a passive body stop dead at the last input frame: {passive_movers[:20]} "
                        "(make them active with a mass unless something visibly carries or holds them)")
    metrics["passive_movers"] = passive_movers
    metrics["behaviors"] = behavior_check(scene, inv, warnings)
    # Deterministic gates (gates.py): contacts, silhouettes against the evidence masks, the plan, joints, proxies.
    from codify import gates
    groups = gates.semantic_groups(scene)
    environment = gates.environment_meshes(scene)
    # Floating is checked for active bodies at rest; a keyed (measured-moving) object is where it was seen.
    active = {o["instance_id"] for o in inv["semantic_objects"]
              if o["physics"] and o["physics"]["role"] == "active" and not o["animated"]}
    categories = {o["instance_id"]: o["category"] for o in inv["semantic_objects"]}
    passive = {o["instance_id"] for o in inv["semantic_objects"]
               if o["physics"] and o["physics"]["role"] == "passive" and not o["animated"]}
    scene.frame_set(1)
    layout = json.loads(args.arrange) if getattr(args, "arrange", None) else None
    metrics["contacts"] = gates.contacts(scene, groups, environment, active, passive, footprint_rays=layout is not None,
                                         fast=layout is not None)
    # Something resting in or on a soft, cloth, granular or liquid object sinks into it: not a penetration to fix.
    absorbing = {o["instance_id"] for o in inv["semantic_objects"]
                 if o["physics"] and o["physics"]["behavior"] in ("soft", "cloth", "granular", "liquid")}
    contacts = metrics["contacts"]
    contacts["into_absorbing"] = [p for p in contacts["penetrations"] if p["a"] in absorbing or p["b"] in absorbing]
    contacts["penetrations"] = [p for p in contacts["penetrations"] if p not in contacts["into_absorbing"]]
    frames_info = read("frames.json") or {}
    metrics["silhouette_vs_mask"] = gates.silhouettes(scene, groups, ev, frames_info.get("comparison", []),
                                                      Path(args.out).parent / "silhouettes") if ev else None
    metrics["plan"] = gates.plan_contract(scene, groups)
    metrics["joints"] = gates.joint_sweep(scene, groups, environment)
    metrics["box_proxies"] = gates.box_proxies(groups, categories)
    if any(j["unattached"] or j["collisions"] for j in metrics["joints"]):
        warnings.append("joints: a movable part is unattached or collides when swept through its limits (see repairs)")
    ids = [o["instance_id"] for o in inv["semantic_objects"]]
    if len(ids) != len(set(ids)):
        failures.append("duplicate instance_id values")
    for o in inv["semantic_objects"]:
        if o["type"] != "MESH":
            failures.append(f"{o['name']}: semantic object is not a mesh")
        cols = o["collections"]
        members = [m for m in bpy.data.collections[cols[0]].all_objects] if len(cols) == 1 and cols[0] in bpy.data.collections else []
        if not members or any(not ("instance_id" in m or str(m.get("part_of")) == o["instance_id"]) for m in members) \
                or sum("instance_id" in m for m in members) != 1:
            failures.append(f"{o['name']}: its collection must hold only this object and its parts")
        if (o["motion_type"] == "dynamic") != o["animated"]:
            failures.append(f"{o['name']}: motion_type {o['motion_type']} does not match its animation")
    if inv["unregistered_meshes"]:
        failures.append(f"meshes that are neither registered objects nor Environment: {inv['unregistered_meshes'][:10]}")
    if not ids:
        failures.append("no registered objects")
    if scene.camera is None:
        failures.append("no scene camera")
    if bpy.data.libraries or bpy.data.movieclips or any(not i["packed"] for i in inv["images"]):
        failures.append("external resources (libraries, movie clips or unpacked images)")
    if video:
        expected = [1, int(video["frame_count"])], video["fps_num"] / video["fps_den"], [video["width"], video["height"]]
        if inv["frame_range"] != expected[0] or abs(inv["fps"] - expected[1]) > 1e-9 or inv["resolution"] != expected[2]:
            failures.append(f"timeline/resolution {inv['frame_range']} @ {inv['fps']} fps {inv['resolution']} "
                            f"differs from the video {expected}")
    if scene.camera is not None and cameras and len(cameras) == scene.frame_end and scene.camera.data.type == "PERSP":
        K = scene_K(scene)
        pos_err = rot_err = 0.0
        for row in cameras:
            scene.frame_set(row["frame"])
            cw = np.array(scene.camera.matrix_world)
            pos_err = max(pos_err, float(np.abs(cw[:3, 3] - row["position_world"]).max()))
            R = cw[:3, :3] @ contract.FLIP_YZ
            rot_err = max(rot_err, math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R.T @ np.array(row["R_world_rgb"])) - 1) / 2)))))
        metrics["camera_vs_evidence"] = {"max_position_m": pos_err, "max_rotation_deg": rot_err,
                                         "K_scene": K.tolist(), "K_evidence": cameras[0]["projection"]["K"],
                                         "max_K_difference_px": float(np.abs(K - np.array(cameras[0]["projection"]["K"])).max())}
        observations = [json.loads(l) for l in (ev / "observations.jsonl").read_text().splitlines()] \
            if ev and (ev / "observations.jsonl").is_file() else []
        semantic = {str(o["instance_id"]): o for o in scene.objects if "instance_id" in o and o.type == "MESH"}
        vertices = {k: np.array([v.co[:] for v in o.data.vertices]) for k, o in semantic.items()}
        by_frame, per_object = {}, {}
        for row in observations:
            if row["state"] == "detected" and row["object_id"] in semantic:
                by_frame.setdefault(row["frame_index"], []).append(row)
        for frame, rows in sorted(by_frame.items()):
            scene.frame_set(frame + 1)
            for row in rows:
                box = projected_box(semantic[row["object_id"]], vertices[row["object_id"]], scene, K)
                per_object.setdefault(row["object_id"], []).append(iou(box, row["bbox_xyxy"]) if box else 0.0)
        means = {k: float(np.mean(v)) for k, v in per_object.items()}
        metrics["object_reprojection_vs_tracker"] = {
            "objects": len(means), "median_mean_iou": float(np.median(list(means.values()))) if means else None,
            "below_0.3": sorted(k for k, v in means.items() if v < 0.3), "per_object_mean_iou": means,
            "note": "objects whose instance_id equals an evidence track_id; projected mesh AABB vs tracker box"}
    elif cameras and scene.camera is not None:
        warnings.append("camera residuals skipped (non-PERSP camera or frame count differs from evidence)")
    # Measured motion is kept: where a moving evidence object's mesh is at its pose times, against the poses.
    fps = scene.render.fps / scene.render.fps_base
    built = {str(o["instance_id"]): o for o in scene.objects if "instance_id" in o}
    deviation = {}
    for uid, item in (read("objects.json") or {}).items():
        if item.get("motion_type") == "dynamic" and uid in built:
            rows = item["poses"][:: max(1, len(item["poses"]) // 30)] + item["poses"][-1:]
            worst = 0.0
            for row in rows:
                frame = row[0] * fps / 1e9 + 1
                scene.frame_set(int(frame), subframe=float(frame - int(frame)))
                worst = max(worst, float(np.linalg.norm(np.array(built[uid].matrix_world.translation) - row[1:4])))
            deviation[uid] = round(worst, 3)
    if deviation:
        metrics["dynamic_pose_deviation_m"] = deviation
        off = sorted(u for u, d in deviation.items() if d > 0.05)
        if off:
            warnings.append(f"moving objects whose animation leaves their measured poses by more than 5 cm: {off} "
                            "(follow the poses; change one only where the frames show it is wrong, and say why)")
    if layout is not None:
        failures += layout_failures(scene, metrics["contacts"], len(groups), layout)
    scene.frame_set(1)
    report = {"pass": not failures, "failures": failures, "warnings": warnings, "metrics": metrics, "inventory": inv,
              "scope": "Structure and consistency with perception evidence; not accuracy against independent reference data."}
    report["repairs"] = gates.compile_repairs(report)
    _write(args.out, report)


def run_ids(args):
    """What the scene shows where from the scene camera: one ray per cell of a pixel grid (every `stride` pixels of
    the render resolution) at the given frames, the registered object it hits (a part counts as its object) and the
    camera depth there (metres along the optical axis). For comparing a built scene with the source's masks and depth.
    Writes an npz: frames, ids (frames x rows x cols int16: index into names, -1 nothing, -2 environment, -3 an
    unregistered mesh), depth (float32, 0 where nothing is hit), names, stride."""
    scene = bpy.context.scene
    if scene.camera is None or scene.camera.data.type != "PERSP":
        raise SystemExit("ids needs a perspective scene camera")
    frames = [int(f) for f in args.frames.split(",")]
    step_to(scene, max(frames))
    W, H, s = scene.render.resolution_x, scene.render.resolution_y, max(1, int(args.stride))
    us, vs = np.arange(s / 2, W, s), np.arange(s / 2, H, s)
    names = sorted({str(o["instance_id"]) for o in scene.objects if "instance_id" in o},
                   key=lambda n: (not n.isdigit(), int(n) if n.isdigit() else 0, n))
    index = {n: i for i, n in enumerate(names)}

    def owner(obj):
        o = obj
        while o is not None:
            if "instance_id" in o:
                return index[str(o["instance_id"])]
            if "part_of" in o and str(o["part_of"]) in index:
                return index[str(o["part_of"])]
            o = o.parent
        return -2 if any(c.name == ENVIRONMENT for c in obj.users_collection) else -3

    ids = np.full((len(frames), len(vs), len(us)), -1, np.int16)
    depth = np.zeros((len(frames), len(vs), len(us)), np.float32)
    for k, frame in enumerate(frames):
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        cam = scene.camera
        M = np.array(cam.matrix_world)
        R, origin = M[:3, :3], M[:3, 3]
        forward = -R[:, 2] / np.linalg.norm(R[:, 2])
        tr, br, bl, tl = (np.array(v[:]) for v in cam.data.view_frame(scene=scene))
        from mathutils import Vector
        start = Vector(origin.tolist())
        for i, v in enumerate(vs):
            left = tl + (bl - tl) * (v / H)
            step = (tr - tl) * (1.0 / W)
            for j, u in enumerate(us):
                d = R @ (left + step * u)
                hit, loc, _, _, obj, _ = scene.ray_cast(depsgraph, start, Vector((d / np.linalg.norm(d)).tolist()))
                if hit and obj is not None:
                    ids[k, i, j] = owner(obj.original if hasattr(obj, "original") else obj)
                    depth[k, i, j] = float(np.dot(np.array(loc[:]) - origin, forward))
    np.savez_compressed(args.out, frames=np.array(frames), ids=ids, depth=depth, names=np.array(names), stride=s,
                        resolution=np.array([W, H]))


def run_dump(args):
    """IDs, local mesh extents, world matrices and camera state of the open .blend at the given Blender frames
    (the rigid-body cache stepped in order)."""
    scene = bpy.context.scene
    frames = [int(f) for f in args.frames.split(",")]
    objects = {str(o["instance_id"]): o for o in scene.objects if "instance_id" in o}
    cam = scene.camera
    dump = {"fps": scene.render.fps / scene.render.fps_base, "frame_range": [scene.frame_start, scene.frame_end],
            "objects": {}, "camera": {"type": cam.data.type if cam else None, "matrices": {},
                                      "polynomial": [getattr(cam.data, f"fisheye_polynomial_k{i}", None) for i in range(5)]
                                      if cam else None}}
    for uid, o in objects.items():
        item = {"type": o.type, "name": o.name, "category": str(o.get("category", "")), "matrices": {},
                "physics": physics(o), "parts": [str(p["part_name"]) for p in scene.objects
                                                 if str(p.get("part_of")) == str(o["instance_id"])]}
        if o.type == "MESH":
            v = np.array([vert.co[:] for vert in o.data.vertices])
            item["local_extents"] = np.column_stack((v.min(0), v.max(0))).ravel().tolist()
            item["vertex_count"] = len(v)
        dump["objects"][uid] = item
    step_to(scene, max(frames))
    for frame in frames:
        scene.frame_set(frame)
        for uid, o in objects.items():
            dump["objects"][uid]["matrices"][frame] = [list(row) for row in o.matrix_world]
        if cam:
            dump["camera"]["matrices"][frame] = [list(row) for row in cam.matrix_world]
    _write(args.out, dump)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("exec")
    p.add_argument("--code", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("render")
    p.add_argument("--frames", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--top", action="store_true")
    p.add_argument("--view", choices=["top", "side_x", "side_y"], help="a view of the whole room (overview_camera)")
    p.add_argument("--camera", help="JSON camera spec to render from instead of the scene camera (T5 target views)")
    p.add_argument("--sweep-joints", type=int, default=0, help="also render every movable part through its joint "
                                                                 "range, this many steps each way (report previews)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("check")
    p.add_argument("--evidence")
    p.add_argument("--out", required=True)
    p.add_argument("--arrange", help="JSON thresholds of the layout gate of a room run (arrange_room): "
                                      "max_floating_share, max_deep_share, deep_m")
    p = sub.add_parser("dump")
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("ids")
    p.add_argument("--frames", required=True)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"exec": run_exec, "render": run_render, "check": run_check, "dump": run_dump, "ids": run_ids}[args.mode](args)


if __name__ == "__main__":
    main()
