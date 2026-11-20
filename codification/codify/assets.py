"""Generic Blender helpers for generated scene code (Blender 4.5, inside the sandbox).

    import codify.assets as A
    ev = A.load_evidence()                       # read-only perception evidence (/evidence)
    A.setup_scene(ev["video"])                   # fps, frames 1..N, resolution, meters, preview light
    A.add_source_camera(ev["cameras"])           # reconstructed source camera, keyed every frame
    A.add_object("17", "sofa_17", "sofa", bounds=[-1, 1, -0.45, 0.45, -0.4, 0.4],
                 location=(2.0, 1.1, 0.4), yaw_deg=30, recipe="couch", color=(0.6, 0.6, 0.62))
    A.add_floor(x_range, y_range); A.add_wall(p0, p1, height=2.6)

World: right-handed, +Z up, meters. Object bounds are [xmin, xmax, ymin, ymax, zmin, zmax]
in the object's local frame (canonical up +Z, front -Y unless given); location/rotation map
local to world. Blender frame = canonical frame_index + 1. Colors are sRGB in [0, 1].
One physical object = one mesh with one instance_id in its own collection (register_object);
room geometry goes to the "Environment" collection and has no instance_id.
Physics: every object is a rigid body (set_physics, or add_object(physics=...)) with a declared behavior (rigid,
brittle, magnetic, soft, ...) and the environment is a fixed collider, so the scene can be simulated forward from the
input's last frame (meters, kg, gravity -Z).
Assemblies: an object with movable parts (a laptop lid, a door, a drawer, a cap) is its root mesh plus parts
(add_part) joined to it by typed joints (fixed, revolute, prismatic) that downstream systems read and move
(set_joint); declare_plan states the parts, sizes and checklist the build commits to; project_texture puts a
source frame on an object's visible faces.
Generated objects: add_asset places a complete mesh that the make_objects tool made from the input's own frames (SAM 3D
Objects: the whole shape with its hidden sides, vertex colours, at its estimated pose); rest_on lays an object on the
stable face closest to facing down and seats it on a support at height z.
"""
import json
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from . import contract
from .object_library import build_object

PICTURE_RECIPES = ("wall artwork", "picture")
ENVIRONMENT = "Environment"
COLLISION_SHAPES = ("SPHERE", "BOX", "CYLINDER", "CAPSULE", "CONE", "CONVEX_HULL", "MESH")
FUTURE_SECONDS = 5.0          # the rigid-body cache reaches this far past the last input frame
BEHAVIORS = ("rigid", "brittle", "magnetic", "soft", "cloth", "granular", "liquid", "heat_source")
ABSORBING = ("soft", "cloth", "granular", "liquid")     # absorbing colliders here; declared for later simulation


def load_evidence(root="/evidence"):
    """All perception evidence of this run (estimates, not ground truth); missing files give None or empty values.
    video {fps_num, fps_den, frame_count, width, height, projection}; timeline [{frame_index, timestamp_ns, ...}];
    cameras [one per frame: {frame, position_world, R_world_rgb, projection {K, width, height}}] for add_source_camera;
    world {up_status, scale_status, camera_height_above_floor_m}; structure {floor {z, x_range, y_range},
    vertical_planes [{p0, p1 (xy ends), z_range, normal_xy, support_fraction}]};
    objects {id: {instance_id, instance_name, category, proxy_recipe, motion_type,
    bounds [xmin, xmax, ymin, ymax, zmin, zmax] (meters, local frame; fused boxes cover visible surfaces),
    poses [[t_ns, tx, ty, tz, qw, qx, qy, qz], ...] (local frame -> world), evidence {confidence, views_used,
    dimension_status, rotation_status, low_depth_confidence, ...}}}; unresolved [{id, category, reason}] (tracks
    without a reliable 3D box); appearance {id: {median_srgb, best_view {frame_index, bbox_xyxy}}};
    observations [{object_id, frame_index, state, bbox_xyxy, mask_area_px}] (2D tracker boxes);
    frames {frame_index: path} the comparison frames as JPEG files (for project_texture)."""
    root = Path(root)
    read = lambda name, default=None: json.loads((root / name).read_text()) if (root / name).is_file() else default
    lines = lambda name: [json.loads(l) for l in (root / name).read_text().splitlines() if l.strip()] \
        if (root / name).is_file() else []
    frames = read("frames.json", {})
    return {"root": str(root), "video": read("video.json"), "timeline": lines("timeline.jsonl"),
            "cameras": read("cameras.json"), "world": read("world.json"), "structure": read("structure.json"),
            "tracks": read("tracks.json"), "objects": read("objects.json", {}), "unresolved": read("unresolved.json", []),
            "appearance": read("appearance.json", {}), "observations": lines("observations.jsonl"),
            "frames": {f: str(root / f"frames/{f:06d}.jpg") for f in frames.get("comparison", [])
                       if (root / f"frames/{f:06d}.jpg").is_file()}}


def load_points(root="/evidence"):
    """Coarse colored world point cloud from depth (xyz float32 N x 3, rgb uint8 N x 3)."""
    data = np.load(Path(root) / "points.npz")
    return data["xyz"], data["rgb"]


def setup_scene(video, name="codified_scene", samples=16):
    scene = bpy.context.scene
    scene.name = name
    scene.render.fps, scene.render.fps_base = int(video["fps_num"]), float(video["fps_den"])
    scene.frame_start, scene.frame_end = 1, int(video["frame_count"])
    scene.render.resolution_x, scene.render.resolution_y = int(video["width"]), int(video["height"])
    scene.render.resolution_percentage = 100
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.view_settings.view_transform = "AgX"
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    background.inputs[0].default_value = (0.8, 0.8, 0.8, 1)
    background.inputs[1].default_value = 0.9
    if "PreviewSun" not in bpy.data.objects:
        sun = bpy.data.objects.new("PreviewSun", bpy.data.lights.new("PreviewSun", "SUN"))
        sun.data.energy = 2.0
        sun.rotation_euler = (math.radians(35), 0, math.radians(30))
        scene.collection.objects.link(sun)
    return scene


def material(name, srgb, roughness=0.6, metallic=0.0):
    """Principled material from an sRGB color (converted to linear)."""
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb[:3]]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node = mat.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = (*linear, 1.0)
    node.inputs["Roughness"].default_value = roughness
    node.inputs["Metallic"].default_value = metallic
    mat.diffuse_color = (*linear, 1.0)
    return mat


def animate(obj, frames, loc, quat):
    """Bulk LINEAR keys with CONSTANT extrapolation; quaternion signs made continuous (wxyz)."""
    q = np.array(quat, float)
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] *= -1
    obj.rotation_mode = "QUATERNION"
    action = bpy.data.actions.new(obj.name + "_pose")
    obj.animation_data_create()
    obj.animation_data.action = action
    for name, values in [("location", np.asarray(loc, float)), ("rotation_quaternion", q)]:
        for j in range(values.shape[1]):
            fc = action.fcurves.new(data_path=name, index=j)
            fc.extrapolation = "CONSTANT"
            fc.keyframe_points.add(len(frames))
            fc.keyframe_points.foreach_set("co", np.column_stack((frames, values[:, j])).astype("float32").ravel())
            for key in fc.keyframe_points:
                key.interpolation = "LINEAR"
            fc.update()
    if len(action.slots):
        obj.animation_data.action_slot = action.slots[0]


_keys = animate              # add_asset's own `animate` argument hides the function there


def picture_texture(obj, image_path, bounds, uid):
    """Packed image on the faces normal to the thin axis of the local bounds."""
    bb = np.array(bounds).reshape(3, 2)
    img = bpy.data.images.load(str(image_path), check_existing=True)
    img.pack()
    mat = bpy.data.materials.new("video_crop_" + uid)
    mat.use_nodes = True
    tree = mat.node_tree
    tex = tree.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tree.links.new(tex.outputs["Color"], tree.nodes.get("Principled BSDF").inputs["Base Color"])
    slot = len(obj.data.materials)
    obj.data.materials.append(mat)
    uv = obj.data.uv_layers.new(name="VideoCrop")
    thin = int(np.argmin(bb[:, 1] - bb[:, 0]))
    vertical = 1 if thin != 1 else 2
    horizontal = next(a for a in range(3) if a not in (thin, vertical))
    for poly in obj.data.polygons:
        if abs(poly.normal[thin]) > 0.8:
            poly.material_index = slot
        for li in poly.loop_indices:
            v = obj.data.vertices[obj.data.loops[li].vertex_index].co
            uv.data[li].uv = ((v[horizontal] - bb[horizontal, 0]) / max(bb[horizontal, 1] - bb[horizontal, 0], 1e-6),
                              (v[vertical] - bb[vertical, 0]) / max(bb[vertical, 1] - bb[vertical, 0], 1e-6))


def add_source_camera(rows, name="SourceCamera"):
    """Camera from evidence camera rows: exact pinhole (PERSP) or the approximate ADT fisheye."""
    scene = bpy.context.scene
    projection = rows[0]["projection"]
    width, height = projection["width"], projection["height"]
    data = bpy.data.cameras.new(name)
    cam = bpy.data.objects.new(name, data)
    scene.collection.objects.link(cam)
    if projection["model"] == "pinhole":
        params = contract.pinhole_to_blender(projection["K"], width, height)
        data.type = "PERSP"
        data.sensor_fit, data.sensor_width, data.lens = params["sensor_fit"], params["sensor_width"], params["lens"]
        data.shift_x, data.shift_y = params["shift_x"], params["shift_y"]
        scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = params["pixel_aspect_x"], params["pixel_aspect_y"]
        info = {"blender_type": "PERSP", **params, "visibility_queries": "supported (PERSP)"}
    else:
        if width != height:
            raise ValueError("The fisheye polynomial path supports square images only")
        k = np.asarray(projection["distortion_coeffs"][:6], float)
        data.type = "PANO"
        data.panorama_type = "FISHEYE_LENS_POLYNOMIAL"
        data.sensor_width = data.sensor_height = 36
        data.fisheye_fov = math.pi
        theta = np.linspace(0, 1.15, 200)
        rp = projection["K"][0][0] * theta * (1 + sum(k[i] * theta ** (2 * i + 2) for i in range(6)))
        coef = np.polynomial.polynomial.polyfit(rp * 36 / width, -theta, 4)
        for i in range(5):
            setattr(data, f"fisheye_polynomial_k{i}", float(coef[i]))
        info = {"blender_type": "PANO", "panorama_type": "FISHEYE_LENS_POLYNOMIAL", "polynomial": coef.tolist(),
                "principal_point_offset": "ignored by Blender",
                "visibility_queries": "unsupported: camera_history supports PERSP/ORTHO only"}
    frames = np.array([row["frame"] for row in rows], float)
    loc = np.array([row["position_world"] for row in rows], float)
    quat = np.array([Matrix(contract.blender_camera_rotation(r["R_world_rgb"]).tolist()).to_quaternion()[:] for r in rows])
    animate(cam, frames, loc, quat)
    cam["pose_sources"] = sorted({row["pose_source"] for row in rows})
    cam["pose_status_counts"] = json.dumps({s: sum(r["pose_status"] == s for r in rows) for s in contract.CAMERA_POSE_STATUSES})
    scene.camera = cam
    cam.data["codify_camera_info"] = json.dumps({"projection_model": projection["model"], "K": projection["K"], **info})
    return cam


def _moves(obj):
    """True when obj has keyed location or rotation (a measured or authored motion)."""
    action = obj.animation_data.action if obj.animation_data else None
    return bool(action) and any(fc.data_path in ("location", "rotation_quaternion", "rotation_euler")
                                for fc in action.fcurves)


def set_physics(obj, role, mass_kg=None, friction=0.5, restitution=0.0, shape="CONVEX_HULL", behavior="rigid",
                properties=None):
    """Rigid body for simulating the scene forward past the input's last frame (meters, kg, gravity -Z).
    role "active": free to fall, roll, slide, tip or be pushed (needs mass_kg); "passive": a fixed collider (the
    ground, furniture that stays put, anything mounted, clamped or held in place). Through the last input frame every
    body follows its measured pose (its keys); then an active body is released and continues with its last measured
    velocity, and a passive one keeps its last pose. friction 0-1 (ice 0.05, wood 0.4, rubber 0.9); restitution 0-1
    (clay 0, wood 0.3, tennis ball 0.75); shape: SPHERE, BOX, CYLINDER, CAPSULE, CONE, CONVEX_HULL, or MESH (passive
    concave shapes only). Collision margins are 1 mm, so resting contact is where the meshes touch.
    behavior: what the object does when things act on it, from what it is made of:
      "rigid" (default): keeps its shape.
      "brittle": breaks or crushes under load (ceramic, glass, styrofoam, an egg, a cracker); properties
        {"break_load_n": the force it takes before it breaks (styrofoam cup ~50 N, egg ~40 N, mug ~2000 N)}.
        Simulated rigid here; declared for whoever simulates the scene next.
      "magnetic": a magnet; properties {"pull_n": its pull on iron at 5 cm, "reach_m": how far it acts (it pulls
        nothing farther), "attracts": [ids of the iron or steel objects it acts on]}. It pulls every released
        body within reach_m toward itself, so keep reach_m short enough that only the attracted objects are in it;
        with no attracts it pulls nothing in the simulation (a magnet that only holds itself to a board).
      "soft" (clay, dough, a pillow, a sponge, a balloon), "cloth" (paper, a napkin, fabric), "granular" (sand,
        kinetic sand, rice, salt), "liquid" (water, juice, paint): simulated as absorbing colliders (restitution 0,
        friction at least 0.8: what lands on them stays) and declared for whoever simulates the scene next;
        properties may say how ({"stiffness": "low" | "medium" | "high"}, {"viscosity": ...}).
      "heat_source": a flame, a candle, a hot plate (a declaration; properties {"temperature_c": ...}).
    behavior and properties are written to the object's physics record, which downstream systems read."""
    if role not in ("active", "passive"):
        raise ValueError("role is 'active' or 'passive'")
    if role == "active" and not (mass_kg and mass_kg > 0):
        raise ValueError("an active body needs mass_kg > 0")
    if shape not in COLLISION_SHAPES:
        raise ValueError(f"shape is one of {COLLISION_SHAPES}")
    properties = dict(properties or {})
    _check_behavior(behavior, properties)
    if behavior in ABSORBING:
        friction, restitution = max(float(friction), 0.8), 0.0
    scene = bpy.context.scene
    if obj.rigid_body is None:
        with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj],
                                       selected_editable_objects=[obj]):
            bpy.ops.rigidbody.object_add(type=role.upper())
    body = obj.rigid_body
    body.type, body.collision_shape = role.upper(), shape
    body.friction, body.restitution = float(friction), float(restitution)
    body.use_margin, body.collision_margin = True, 0.001
    if mass_kg:
        body.mass = float(mass_kg)
    last = scene.frame_end
    if role == "active":
        _steady_release(obj, last, scene.render.fps / scene.render.fps_base)
        # Kinematic (on its keys) through the last input frame, simulated from the next one.
        for frame, value in ((last, True), (last + 1, False)):
            body.kinematic = value
            obj.keyframe_insert("rigid_body.kinematic", frame=frame)
        for fc in obj.animation_data.action.fcurves:
            if fc.data_path == "rigid_body.kinematic":
                for key in fc.keyframe_points:
                    key.interpolation = "CONSTANT"
    else:
        body.kinematic = _moves(obj)
    world = scene.rigidbody_world
    fps = scene.render.fps / scene.render.fps_base
    # Every body is on its keys until the last input frame, so the simulation needs only a few frames before it
    # (they carry the release velocity).
    world.point_cache.frame_start = max(1, last - 10)
    world.point_cache.frame_end = last + round(FUTURE_SECONDS * fps)
    world.substeps_per_frame = max(10, math.ceil(300 / fps))
    obj["physics"] = json.dumps({"role": role, "mass_kg": mass_kg, "friction": friction, "restitution": restitution,
                                 "shape": shape, "released_after_frame": last if role == "active" else None,
                                 "behavior": behavior, "properties": properties})
    return obj


def _check_behavior(behavior, properties):
    if behavior not in BEHAVIORS:
        raise ValueError(f"behavior is one of {BEHAVIORS}")
    if behavior == "brittle" and not float(properties.get("break_load_n") or 0) > 0:
        raise ValueError("a brittle object needs properties={'break_load_n': N > 0}")
    if behavior == "magnetic":
        if not (float(properties.get("pull_n") or 0) > 0 and float(properties.get("reach_m") or 0) > 0):
            raise ValueError("a magnet needs properties={'pull_n': N > 0, 'reach_m': m > 0, 'attracts': [ids]}")
        properties["attracts"] = [str(i) for i in properties.get("attracts", [])]


def physics_record(obj):
    """The physics an object was given (set_physics), with its behavior ('rigid' when none was declared)."""
    record = json.loads(obj["physics"]) if "physics" in obj else {}
    record.setdefault("behavior", "rigid")
    record.setdefault("properties", {})
    return record


def apply_behaviors():
    """Build what the declared behaviors need in the simulation; run once after scene.py (the sandbox does it before
    saving): a magnet gets its force field."""
    scene = bpy.context.scene
    done = []
    for obj in [o for o in scene.objects if "instance_id" in o and o.type == "MESH" and o.rigid_body]:
        if obj.get("behavior_applied"):
            continue
        record = physics_record(obj)
        if record["behavior"] == "magnetic" and record["properties"].get("attracts"):   # nothing to pull: no field
            _magnet_field(obj, record["properties"])
            done.append({"object": str(obj["instance_id"]), "field": True})
        obj["behavior_applied"] = True
    scene["codify_behaviors"] = json.dumps(done)
    return done


# A Blender force field of strength S pulls a rigid body at distance d with F = MAGNET_K * S / (fps * (1 + d)^2)
# newtons (measured in Blender 4.5, sphere falloff power 2; independent of the body's mass and of substeps).
MAGNET_K = 1.35


def _magnet_field(obj, properties):
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    field = bpy.data.objects.new(f"{obj.name}/field", None)
    scene.collection.objects.link(field)
    field.parent = obj
    field.matrix_parent_inverse = Matrix.Identity(4)
    with bpy.context.temp_override(object=field, active_object=field):
        bpy.ops.object.forcefield_toggle()
    f = field.field
    f.type, f.shape, f.falloff_type, f.falloff_power = "FORCE", "POINT", "SPHERE", 2.0
    f.strength = -float(properties["pull_n"]) * fps * (1.05 ** 2) / MAGNET_K
    f.use_max_distance, f.distance_max = True, float(properties["reach_m"])
    field["field_of"] = str(obj["instance_id"])
    return field


def _steady_release(obj, last, fps, window_s=0.4, rest_mps=0.05):
    """The velocity a released body starts with is the one its keys show between the last two frames, so a
    centimetre of pose jitter there becomes half a metre per second. Re-key the last frames of an animated object on
    a straight-line fit of its last window_s seconds: the release velocity becomes the measured mean velocity, and
    an object that was at rest (fitted speed below rest_mps) is released from rest. Nothing moves more than the
    jitter the fit removes."""
    action = obj.animation_data.action if obj.animation_data else None
    curves = [fc for fc in (action.fcurves if action else []) if fc.data_path == "location"]
    if len(curves) != 3:
        return
    window = max(4, round(window_s * fps))
    frames = np.arange(last - window, last + 1, dtype=float)
    samples = np.array([[fc.evaluate(f) for fc in sorted(curves, key=lambda c: c.array_index)] for f in frames])
    slope, intercept = np.polyfit(frames, samples, 1)
    if np.linalg.norm(slope) * fps < rest_mps:
        slope = np.zeros(3)
        intercept = samples.mean(0)
    rekey = frames[window // 2:]
    for fc in sorted(curves, key=lambda c: c.array_index):
        for f in rekey:
            value = float(slope[fc.array_index] * f + intercept[fc.array_index])
            fc.keyframe_points.insert(f, value, options={"REPLACE"})
        fc.update()
    obj["release"] = json.dumps({"fit_frames": [int(frames[0]), int(last)], "re_keyed_from": int(rekey[0]),
                                 "velocity_mps": [round(float(v * fps), 4) for v in slope]})


def _scene_collections(obj):
    """obj's collections other than the rigid-body world's and the constraints' own (they only mark it as simulated)."""
    scene = bpy.context.scene
    own = {c for c in (getattr(scene.rigidbody_world, "collection", None),
                       getattr(scene.rigidbody_world, "constraints", None)) if c}
    return [c for c in obj.users_collection if c not in own]


def register_object(obj, uid, name, category, prototype=None, motion_type="static",
                    geometry_status="authored_proxy", source=None):
    """Make obj the single semantic mesh of one physical object, alone in its own collection."""
    if obj.type != "MESH":
        raise TypeError("A semantic object must be one mesh; join parts first (join)")
    uid = str(uid)
    if any(str(o.get("instance_id")) == uid for o in bpy.data.objects if o is not obj and "instance_id" in o):
        raise ValueError(f"instance_id {uid} is already used")
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    for c in _scene_collections(obj):
        c.objects.unlink(obj)
    collection.objects.link(obj)
    obj.name = collection.name
    obj["instance_id"] = uid
    obj["category"] = category
    obj["prototype_name"] = prototype or category
    obj["motion_type"] = motion_type
    obj["geometry_status"] = geometry_status
    if source:
        obj["codify_source"] = source
    return obj


def add_object(uid, name, category, bounds, location=(0, 0, 0), rotation_wxyz=None, yaw_deg=None, recipe=None,
               color=(0.48, 0.48, 0.46), up=(0, 0, 1), front=(0, -1, 0), poses=None, prototype=None,
               source=None, crop=None, geometry_status="procedural_proxy_in_estimated_bounds", physics=None,
               texture=None, mesh=True):
    """Category proxy fitted exactly to local bounds, registered and posed. In a room, an evidence object whose
    complete mesh was made before the agent started and agrees with its box (assets/default.json) is built as that
    mesh at its estimated pose instead (add_asset; the placement arguments are then not used); mesh=False keeps the
    proxy.

    recipe picks the proxy shape (default: category); a name without a recipe gives a plain box. Recipes:
    accessory, air purifier, armchair, baking pan, bar stool, bed frame, blender, book, bookcase, bottle, bowl, box,
    built in, cabinet, can, candle, chair, clock, clothes basket, coaster, coffee maker, coffee table, computer
    keyboard, computer mouse, console table, container, cooktop, couch, cup, curtain, cutting board, desk lamp,
    dining chair, dining table, dishrack, door, dressing table, figurine, food object, frying pan, hat, hook, house
    plant, jar, keys, kitchen utensil, laptop computer, lid, light, magazine, mattress, mouse pad, napkin, notebook, pillow,
    plate, pot, rack, refrigerator, remote control, router, scissors, shelf, shoe rack, side table, speaker, spoon,
    spray bottle, step stool, table, tea kettle, television, tissue box, toaster, toy, trash can, tray, tv stand,
    vase, wall artwork, wall clock, wine rack.
    location and rotation_wxyz place the local frame of bounds (for an evidence box: location=pose[1:4],
    rotation_wxyz=pose[4:8]); yaw_deg instead rotates about +Z. up/front: the proxy's canonical axes in that frame.
    poses (optional, dynamic): rows [t_ns, x, y, z, qw, qx, qy, qz] on the canonical timeline.
    physics (dict): the arguments of set_physics, e.g. {"role": "active", "mass_kg": 0.3, "friction": 0.5,
    "restitution": 0.2, "shape": "CONVEX_HULL"}.
    texture: (image_path, camera_row) of a comparison frame where the object is well seen, projected onto its faces
    that face that camera (project_texture); best_frame(ev, uid) picks the frame. Ignored for a dynamic object
    (poses), which keeps its flat color.
    """
    if mesh and not poses and str(uid) in _default_meshes():
        return add_asset(str(uid), name, category, physics=physics)
    instance = {"instance_name": name, "category": recipe or category,
                "canonical_pose": {"up_vector": list(up), "front_vector": list(front)}}
    obj = build_object(instance, list(bounds), tuple(color[:3]) + (1.0,))
    register_object(obj, uid, name, category, prototype, "dynamic" if poses else "static", geometry_status, source)
    if crop and (recipe or category) in PICTURE_RECIPES:
        picture_texture(obj, crop, bounds, str(uid))
    if poses:
        rows = np.asarray(poses, float)
        scene = bpy.context.scene
        fps = scene.render.fps / scene.render.fps_base
        quats = [Matrix(contract.quat_wxyz_to_matrix(r[4:8]).tolist()).to_quaternion()[:] for r in rows]
        animate(obj, rows[:, 0] * fps / 1e9 + 1, rows[:, 1:4], quats)
    else:
        obj.location = location
        obj.rotation_mode = "QUATERNION"
        if rotation_wxyz is not None:
            obj.rotation_quaternion = Quaternion(rotation_wxyz).normalized()
        elif yaw_deg is not None:
            obj.rotation_quaternion = Quaternion((0, 0, 1), math.radians(yaw_deg))
    if physics:
        set_physics(obj, **physics)
    if texture and not poses:       # a texture projected from one frame is wrong at every other pose of a moving object
        project_texture(obj, texture[0], texture[1], str(uid))
    return obj


JOINT_TYPES = ("fixed", "revolute", "prismatic")


def add_part(root, part, name, joint=None, color=None, physics=None):
    """A named part of a registered object: `part` is a mesh in the root's local frame (built like any mesh, e.g.
    box_mesh). It is parented to the root, put in the root's collection and tagged part_of = the root's instance_id.
    joint: how the part moves relative to the root: {"type": "fixed" | "revolute" | "prismatic", "axis": [x, y, z]
    and "pivot": [x, y, z] in the root's frame, "limits": [lo, hi] (radians for revolute, meters for prismatic),
    "value": the current position (default 0 = as built; the frames show it, e.g. a lid at 1.6 rad)}. Downstream
    systems read the joints and move parts with set_joint, so a laptop lid, a door, a drawer or a cap must be a
    part. The part's origin is moved to the pivot; a revolute or prismatic joint also gets a Blender rigid-body
    hinge or slider constraint with the limits. physics: as set_physics; default passive (kinematic: it holds
    its pose in the simulation); make a part active when it can fall or be pushed on its own."""
    if part.type != "MESH":
        raise TypeError("a part must be one mesh; join pieces first")
    if "instance_id" not in root:
        raise ValueError("add_part needs a registered object as root (add_object / register_object)")
    joint = {"type": "fixed", **(joint or {})}
    if joint["type"] not in JOINT_TYPES:
        raise ValueError(f"joint type is one of {JOINT_TYPES}")
    pivot = np.asarray(joint.get("pivot", [0.0, 0.0, 0.0]), float)
    axis = np.asarray(joint.get("axis", [0.0, 0.0, 1.0]), float)
    if joint["type"] != "fixed" and np.linalg.norm(axis) < 1e-9:
        raise ValueError("a revolute or prismatic joint needs an axis")
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    # The part's origin at the pivot, in the root's frame: a rotation or translation about the origin is the joint.
    for v in part.data.vertices:
        v.co = Vector(np.asarray(v.co) - pivot)
    part.parent = root
    part.matrix_parent_inverse = Matrix.Identity(4)
    part.location = Vector(pivot)
    part.rotation_mode = "QUATERNION"
    part.rotation_quaternion = Quaternion((1, 0, 0, 0))
    for c in _scene_collections(part):
        c.objects.unlink(part)
    root.users_collection[0].objects.link(part)
    part.name = f"{root.name}/{name}"
    part["part_of"] = str(root["instance_id"])
    part["part_name"] = name
    part["joint"] = json.dumps({"type": joint["type"], "axis": axis.tolist(), "pivot": pivot.tolist(),
                                "limits": [float(v) for v in joint.get("limits", [0.0, 0.0])],
                                "value": float(joint.get("value", 0.0)), "rest_value": float(joint.get("value", 0.0))})
    if color is not None and not part.data.materials:
        part.data.materials.append(material(f"{part.name}_mat", color))
    set_physics(part, **(physics or {"role": "passive"}))
    if joint["type"] != "fixed" and root.rigid_body is not None:
        _joint_constraint(root, part, joint["type"], axis, pivot, joint.get("limits"))
    return part


def _joint_constraint(root, part, kind, axis, pivot, limits):
    """A Blender rigid-body HINGE (revolute) or SLIDER (prismatic) between root and part at the pivot, its Z (hinge)
    or X (slider) axis along the joint axis, with the joint limits."""
    scene = bpy.context.scene
    empty = bpy.data.objects.new(f"{part.name}/joint", None)
    scene.collection.objects.link(empty)
    world_pivot = root.matrix_world @ Vector(pivot)
    world_axis = (root.matrix_world.to_3x3() @ Vector(axis)).normalized()
    empty.location = world_pivot
    empty.rotation_mode = "QUATERNION"
    empty.rotation_quaternion = world_axis.to_track_quat("Z" if kind == "revolute" else "X", "Y")
    # Linking the empty into the rigid-body world's constraints collection creates its constraint (the operator
    # for this crashes Blender in background mode).
    world = scene.rigidbody_world
    if world.constraints is None:
        world.constraints = bpy.data.collections.new("RigidBodyConstraints")
    world.constraints.objects.link(empty)
    bpy.context.view_layer.update()
    con = empty.rigid_body_constraint
    con.type = "HINGE" if kind == "revolute" else "SLIDER"
    con.object1, con.object2 = root, part
    con.disable_collisions = True
    if limits:
        lo, hi = (float(limits[0]), float(limits[1]))
        if kind == "revolute":
            con.use_limit_ang_z, con.limit_ang_z_lower, con.limit_ang_z_upper = True, lo, hi
        else:
            con.use_limit_lin_x, con.limit_lin_x_lower, con.limit_lin_x_upper = True, lo, hi
    empty["joint_of"] = part.name
    return empty


def set_joint(part, value):
    """Move a part to joint position `value` (radians about the axis through the pivot for a revolute joint, meters
    along the axis for a prismatic one), relative to how it was built (joint rest_value)."""
    joint = json.loads(part["joint"])
    delta = float(value) - joint["rest_value"]
    axis, pivot = Vector(joint["axis"]), Vector(joint["pivot"])
    if joint["type"] == "revolute":
        part.rotation_mode = "QUATERNION"
        part.rotation_quaternion = Quaternion(axis, delta)
        part.location = pivot
    elif joint["type"] == "prismatic":
        part.rotation_quaternion = Quaternion((1, 0, 0, 0))
        part.location = pivot + axis * delta
    joint["value"] = float(value)
    part["joint"] = json.dumps(joint)
    return part


CONNECTIONS = {"hinge": "revolute", "slider": "prismatic", "fixed": "fixed", "separate": None}


def declare_plan(plan):
    """What the build commits to, checked after it: {instance_id: {"parts": [{"name": str, "size_m": [x, y, z],
    "material": str, "connection": "hinge" | "slider" | "fixed" | "separate", "object": id (separate only),
    "evidence": str}], "checklist": [{"item": str, "priority": "must" | "should"}]}}. Declare it for every object
    built from several pieces: a hinge, slider or fixed part must exist as add_part(name) with that joint type, a
    separate part as its own registered object `object` (a lid resting on its container), each with extents within
    max(1 cm, 10%) of size_m; evidence says what in the frames shows the connection. Simple objects need no plan."""
    if not isinstance(plan, dict):
        raise TypeError("plan is a dict keyed by instance_id")
    for uid, item in plan.items():
        for part in item.get("parts", []):
            if not part.get("name") or len(part.get("size_m", [])) != 3:
                raise ValueError(f"plan[{uid}]: each part needs a name and size_m [x, y, z]")
            if part.get("connection", "hinge") not in CONNECTIONS:
                raise ValueError(f"plan[{uid}] part {part['name']}: connection is one of {sorted(CONNECTIONS)}")
            if part.get("connection") == "separate" and not part.get("object"):
                raise ValueError(f"plan[{uid}] part {part['name']}: a separate part names its own object id")
        for entry in item.get("checklist", []):
            if entry.get("priority", "must") not in ("must", "should") or not entry.get("item"):
                raise ValueError(f"plan[{uid}]: checklist items need item text and priority must|should")
    bpy.context.scene["codify_plan"] = json.dumps(plan)
    return plan


def project_texture(obj, image_path, camera, uid=None, mask=None, min_facing=0.3):
    """The source frame `image_path` projected from `camera` (the evidence camera row of that frame) onto the faces
    of obj that face the camera: per-vertex UVs from the pinhole projection, an image-texture material on those
    faces, the object's existing material on the rest. The image is packed into the .blend. Use it with
    ev["frames"][f] and ev["cameras"][f] at a frame where the object is well seen (appearance best_view).
    Only the object's own pixels keep their color, the rest of the frame takes their median color: by the evidence's
    id masks for a numbered uid, or by `mask` (a boolean H x W array of the frame, rows top to bottom, e.g. raw
    perception's raw.mask(track, frame), several tracks OR-ed). Without either the whole frame is projected, with
    whatever lies in front of or behind the object. Faces seen at a grazing angle (cosine to the view below
    min_facing) keep the object's material: a frame stretched over them smears."""
    projection = camera["projection"]
    if projection["model"] != "pinhole":
        raise ValueError("project_texture needs a pinhole camera row")
    uid = uid or str(obj.get("instance_id", obj.name))
    # A texture is one pose's view: an object that moves (animated) or a part that moves on a joint keeps its color.
    joint = json.loads(obj["joint"]).get("type") if "joint" in obj else None
    if joint in ("revolute", "prismatic") or (obj.animation_data and obj.animation_data.action and any(
            fc.data_path in ("location", "rotation_quaternion", "rotation_euler") for fc in obj.animation_data.action.fcurves)):
        obj["frame_texture"] = json.dumps({"skipped": "moves (animated or jointed): a one-frame texture would be wrong"})
        return 0
    K, W, H = np.asarray(projection["K"], float), projection["width"], projection["height"]
    R, C = np.asarray(camera["R_world_rgb"], float), np.asarray(camera["position_world"], float)
    img = _own_pixels(Path(image_path), uid, mask)
    mat = bpy.data.materials.new(f"frame_texture_{uid}")
    mat.use_nodes = True
    tree = mat.node_tree
    tex = tree.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = "CLIP"
    tree.links.new(tex.outputs["Color"], tree.nodes.get("Principled BSDF").inputs["Base Color"])
    if not obj.data.materials:
        obj.data.materials.append(material(f"{uid}_base", (0.5, 0.5, 0.5)))
    slot = len(obj.data.materials)
    obj.data.materials.append(mat)
    uv = obj.data.uv_layers.get("FrameProjection") or obj.data.uv_layers.new(name="FrameProjection")
    M = np.array(obj.matrix_world)
    world = np.array([v.co[:] for v in obj.data.vertices]) @ M[:3, :3].T + M[:3, 3]
    cam = (world - C) @ R                                  # camera axes: +X right, +Y down, +Z forward
    depth = np.clip(cam[:, 2], 1e-6, None)
    u = (K[0, 0] * cam[:, 0] / depth + K[0, 2] + 0.5) / W
    v = 1.0 - (K[1, 1] * cam[:, 1] / depth + K[1, 2] + 0.5) / H
    view_dir = (C - world)
    normals = np.array([poly.normal[:] for poly in obj.data.polygons]) @ M[:3, :3].T
    textured = 0
    for poly in obj.data.polygons:
        verts = [obj.data.loops[li].vertex_index for li in poly.loop_indices]
        to_camera = view_dir[verts].mean(0)
        facing = normals[poly.index] @ to_camera > min_facing * np.linalg.norm(normals[poly.index]) * np.linalg.norm(to_camera)
        inside = all(0 <= u[i] <= 1 and 0 <= v[i] <= 1 and cam[i, 2] > 0.05 for i in verts)
        if facing and inside:
            poly.material_index = slot
            textured += 1
        for li in poly.loop_indices:
            i = obj.data.loops[li].vertex_index
            uv.data[li].uv = (float(u[i]), float(v[i]))
    obj["frame_texture"] = json.dumps({"frame": int(camera["frame"]) - 1, "faces": textured})
    return textured


def _own_pixels(image_path, uid, mask=None):
    """The frame as a packed image in which only object uid's pixels (`mask`, else the evidence id mask of that
    frame) keep their color; the rest (what lies on it or in front of it, the background) takes the object's median
    color, so the projection paints the object and nothing else. Without a mask (an object the model added) the
    frame."""
    mask_path = image_path.parent.parent / "masks" / f"{image_path.stem}.png"
    img = bpy.data.images.load(str(image_path), check_existing=True)
    w, h = img.size
    if mask is not None:
        own = np.asarray(mask, bool)[::-1]                 # Blender's pixel rows run bottom to top
        if own.shape != (h, w):
            raise ValueError(f"project_texture mask is {own.shape[1]}x{own.shape[0]}, the frame {w}x{h}")
        uid = f"{uid}_masked"
    elif not (str(uid).isdigit() and mask_path.is_file()):
        img.pack()
        return img
    else:
        ids = bpy.data.images.load(str(mask_path), check_existing=False)
        ids.colorspace_settings.name = "Non-Color"
        if tuple(ids.size) != (w, h):
            bpy.data.images.remove(ids)
            img.pack()
            return img
        own = np.round(np.array(ids.pixels[:], np.float32).reshape(h, w, 4)[:, :, 0] * 65535).astype(int) == int(uid)
        bpy.data.images.remove(ids)
    if own.sum() < 16:
        img.pack()
        return img
    pix = np.array(img.pixels[:], np.float32).reshape(h, w, 4)
    pix[~own, :3] = np.median(pix[own, :3], axis=0)
    out = bpy.data.images.new(f"frame_{image_path.stem}_object_{uid}", w, h, alpha=True)
    out.colorspace_settings.name = img.colorspace_settings.name
    out.pixels.foreach_set(pix.ravel())
    out.pack()
    return out


def best_frame(ev, uid):
    """(image_path, camera_row) of the comparison frame where object uid's SAM 3 mask is largest, or None when it
    is not seen on any comparison frame. For add_object(texture=...) / project_texture."""
    frames = ev.get("frames") or {}
    best, area = None, 0
    for row in ev.get("observations", []):
        if str(row.get("object_id")) == str(uid) and row.get("state") == "detected" and row["frame_index"] in frames \
                and row.get("mask_area_px", 0) > area:
            best, area = row["frame_index"], row["mask_area_px"]
    return (frames[best], ev["cameras"][best]) if best is not None else None


def separate_objects(max_depth=0.08, passes=3, fast=False):
    """Resolve small penetrations between registered objects (and with the environment) deterministically: for each
    pair whose surfaces interpenetrate by up to max_depth, the smaller object is moved out along the axis of least
    overlap by the penetration depth plus 2 mm (an object inside the environment moves up). Call it at the end of
    scene.py, after every object is placed; it returns what moved so the source text can say so. Moving objects
    and parts are never moved; deeper penetrations are left for a real fix."""
    from codify import gates
    scene = bpy.context.scene
    moved = []
    for _ in range(passes):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        groups = gates.semantic_groups(scene)
        env = gates.environment_meshes(scene)
        report = gates.contacts(scene, groups, env, set(), fast=fast)
        if not report["penetrations"]:
            break
        trees = {u: (gates.world_tree_fast if fast else gates.world_tree)(objs, depsgraph) for u, objs in groups.items()}
        fixed = set()
        for p in report["penetrations"]:
            a, b, depth = p["a"], p["b"], p["depth_m"]
            if depth > max_depth or a in fixed or b in fixed:
                continue
            root_a, other = groups[a][0], b
            if _moves(root_a):
                continue
            lo_a, hi_a = gates.aabb(trees[a][1])
            if b == "environment":
                shift = np.array([0.0, 0.0, depth + 0.002])
            else:
                lo_b, hi_b = gates.aabb(trees[b][1])
                root_b = groups[b][0]
                if np.prod(hi_a - lo_a) > np.prod(hi_b - lo_b) and not _moves(root_b):      # the smaller one moves
                    a, other, root_a, lo_a, hi_a, lo_b, hi_b = b, a, root_b, lo_b, hi_b, lo_a, hi_a
                overlap = np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b)
                axis = int(np.argmin(overlap))
                direction = 1.0 if (lo_a + hi_a)[axis] >= (lo_b + hi_b)[axis] else -1.0
                shift = np.zeros(3)
                shift[axis] = direction * (depth + 0.002)
            root_a.location = Vector(np.asarray(root_a.location) + shift)
            fixed.add(a)
            moved.append({"object": a, "away_from": other, "shift_m": np.round(shift, 4).tolist()})
        bpy.context.view_layer.update()
        if not fixed:
            break
    scene["codify_separations"] = json.dumps(moved)
    return moved


def arrange_room(max_drop=0.25, hang=(), stand=(), keep=(), turn=True, max_turn_deg=35.0, separate=True,
                 auto_support=False):
    """Arrange a room's static objects the way a set dresser would; call it at the end of scene.py, after every object
    and support is built, in place of separate_objects (it runs it). Bottom up: an object that stands is turned onto
    its most stable face when that is at most max_turn_deg away (rest_on), anything below the floor is raised onto it,
    and each object is seated on the highest surface under it (the floor, the environment, another object) when it
    floats at most max_drop above it; the surface under its centre also raises an object sunk into it. Things that
    stand on the floor (chairs, tables, beds, shoes, bins, doors; more ids in stand) are lowered whatever the gap;
    things that hang on a wall or a ceiling (pictures, clocks, curtains, hoods, lights; more ids in hang) keep their
    place; ids in keep and moving objects are not touched. Then separate_objects, and what moved is seated again.
    Prints and returns the report: turned, raised_to_floor, seated (dz_m, on), kept_hanging, separated and
    unsupported, the objects still far above everything under them (gap_m, above, bottom_z): each stands on something
    the scene lacks (a window sill, a shelf, a counter top: add_support), hangs (hang=), stands on the floor (stand=),
    or is misplaced or not an object (leave it out). auto_support: unsupported objects that stand together (footprints
    less than 0.3 m apart, bottoms within 0.1 m) get a slab under them first (add_support, listed in supports_built)."""
    from codify import arrange
    report = arrange.arrange(max_drop=max_drop, hang=hang, stand=stand, keep=keep, turn=turn,
                             max_turn_deg=max_turn_deg, separate=separate, auto_support=auto_support)
    print("ARRANGEMENT", json.dumps({k: len(v) if isinstance(v, list) else v for k, v in report.items()}))
    print("UNSUPPORTED", json.dumps(report["unsupported"]))
    return report


def layout_report():
    """Print LAYOUT, the room as a whole: what should stand on the floor and does not, objects of one kind that overlap,
    the support slabs, the floor's area, the walls and the camera's height. Call it at the end of scene.py."""
    from codify import arrange
    return arrange.layout_report()


def add_support(ids, thickness=0.03, margin=0.04, name=None, color=(0.86, 0.85, 0.82), gap=0.3):
    """Level slabs the scene lacks (a window sill, a shelf board, a counter top) under the registered objects ids, as
    environment: the objects are grouped by their footprints (neighbours less than gap apart in plan join a group), and
    each group gets one slab, its top at the group's lowest bottom, covering the group's footprints plus margin, so a
    slab spans only objects that stand together. Build them before arrange_room, which then seats the objects on them.
    Returns the slabs."""
    from codify import gates
    groups = gates.semantic_groups(bpy.context.scene)
    ids = [str(u) for u in ids if str(u) in groups]
    verts = {u: gates.world_vertices(groups[u]) for u in ids}
    box = {u: (v.min(0), v.max(0)) for u, v in verts.items() if len(v)}
    ids = [u for u in ids if u in box]
    parent = {u: u for u in ids}

    def root(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            (la, ha), (lb, hb) = box[a], box[b]
            if max(lb[0] - ha[0], la[0] - hb[0], lb[1] - ha[1], la[1] - hb[1]) < gap:
                parent[root(a)] = root(b)
    clusters = {}
    for u in ids:
        clusters.setdefault(root(u), []).append(u)
    slabs = []
    for k, members in enumerate(clusters.values()):
        lo = np.min([box[u][0] for u in members], axis=0)
        hi = np.max([box[u][1] for u in members], axis=0)
        size = (float(hi[0] - lo[0]) + 2 * margin, float(hi[1] - lo[1]) + 2 * margin, thickness)
        label = (name or "Support_" + "_".join(members[:3])) + (f"_{k}" if len(clusters) > 1 else "")
        slab = box_mesh(label, size, ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] - thickness / 2), color)
        add_environment(slab)
        slab["support_for"] = json.dumps(members)
        slabs.append(slab)
    return slabs


def join(objects, name):
    """Join mesh parts into one mesh object (use before register_object)."""
    objects = list(objects)
    with bpy.context.temp_override(active_object=objects[0], selected_editable_objects=objects,
                                   selected_objects=objects):
        bpy.ops.object.join()
    objects[0].name = name
    return objects[0]


def box_mesh(name, size, center=(0, 0, 0), color=None):
    """Unregistered axis-aligned box mesh (local coordinates) for custom shapes; pass it to add_environment
    (architecture) or register_object (an object), or join parts first."""
    sx, sy, sz = (s / 2 for s in size)
    cx, cy, cz = center
    verts = [(cx + x * sx, cy + y * sy, cz + z * sz) for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if color is not None:
        mesh.materials.append(material(name + "_mat", color))
    return obj


def _environment_collection():
    if ENVIRONMENT not in bpy.data.collections:
        bpy.context.scene.collection.children.link(bpy.data.collections.new(ENVIRONMENT))
    return bpy.data.collections[ENVIRONMENT]


def add_environment(obj, friction=0.5, collider=True):
    """Move a mesh into the Environment collection: room architecture (floor, walls, windows, stairs, counters),
    not a semantic object. It becomes a fixed collider (passive rigid body, its own mesh as the shape) unless
    collider=False (a backdrop nothing can touch)."""
    for c in _scene_collections(obj):
        c.objects.unlink(obj)
    _environment_collection().objects.link(obj)
    obj["environment"] = True
    if collider:
        set_physics(obj, "passive", friction=friction, shape="MESH")
    return obj


def add_floor(x_range, y_range, z=0.0, color=(0.55, 0.45, 0.35), name="Floor", friction=0.5, collider=True):
    """A 2 cm slab whose top is at z (the support surface); a fixed collider unless collider=False."""
    (x0, x1), (y0, y1) = x_range, y_range
    obj = box_mesh(name, (x1 - x0, y1 - y0, 0.02), ((x0 + x1) / 2, (y0 + y1) / 2, z - 0.01), color)
    return add_environment(obj, friction=friction, collider=collider)


def add_wall(p0, p1, height=2.6, thickness=0.1, z0=0.0, color=(0.85, 0.85, 0.82), name="Wall", friction=0.5,
             collider=True):
    """Vertical slab from p0 to p1 (xy), standing on z0; a fixed collider unless collider=False (a backdrop)."""
    p0, p1 = np.asarray(p0[:2], float), np.asarray(p1[:2], float)
    length = float(np.linalg.norm(p1 - p0))
    obj = box_mesh(name, (length, thickness, height), (0, 0, height / 2), color)
    obj.location = (*((p0 + p1) / 2), z0)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Quaternion((0, 0, 1), math.atan2(*(p1 - p0)[::-1]))
    return add_environment(obj, friction=friction, collider=collider)


_DEFAULT_MESHES = None


def _default_meshes(root="/evidence"):
    """The evidence ids add_object builds as their complete mesh (assets/default.json; none outside a room)."""
    global _DEFAULT_MESHES
    if _DEFAULT_MESHES is None:
        path = Path(root) / "assets" / "default.json"
        _DEFAULT_MESHES = set(json.loads(path.read_text()).get("ids", [])) if path.is_file() else set()
    return _DEFAULT_MESHES


def add_asset(asset_id, name=None, category=None, physics=None, uid=None, animate=False, root="/evidence"):
    """The complete mesh make_objects generated for asset_id (from <root>/assets/<asset_id>/), at the pose estimated on
    its source frame: one registered object (instance_id uid, default asset_id) with its vertex colours. Move, turn
    or scale it like any object afterwards (its origin is the mesh's centre); physics as add_object's. animate: key
    the poses track_objects followed (frames before the first and after the last hold those poses)."""
    folder = Path(root) / "assets" / asset_id
    meta = json.loads((folder / "asset.json").read_text())
    data = np.load(folder / "mesh.npz")
    vertices, faces, colors = data["vertices"], data["faces"], data["colors"]
    mesh = bpy.data.meshes.new(asset_id)
    mesh.vertices.add(len(vertices))
    mesh.vertices.foreach_set("co", vertices.astype(np.float32).ravel())
    mesh.loops.add(3 * len(faces))
    mesh.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", np.arange(0, 3 * len(faces), 3, dtype=np.int32))
    mesh.polygons.foreach_set("loop_total", np.full(len(faces), 3, np.int32))
    mesh.polygons.foreach_set("use_smooth", np.ones(len(faces), bool))
    mesh.update()
    mesh.validate()
    srgb = colors[:, :3].astype(np.float32) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    attribute = mesh.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    attribute.data.foreach_set("color", np.concatenate([linear, np.ones((len(linear), 1), np.float32)], 1).ravel())
    mat = bpy.data.materials.new(asset_id + "_mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    shader, colour = nodes.get("Principled BSDF"), nodes.new("ShaderNodeVertexColor")
    colour.layer_name = "Col"
    mat.node_tree.links.new(colour.outputs["Color"], shader.inputs["Base Color"])
    shader.inputs["Roughness"].default_value = 0.6
    mat.diffuse_color = (*np.median(linear, 0).tolist(), 1.0)
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(asset_id, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = Matrix(meta["world_from_local"])
    obj.rotation_mode = "QUATERNION"
    name = name or meta.get("name") or asset_id
    register_object(obj, uid or asset_id, name, category or name, geometry_status="generated_mesh_from_frames",
                    source=f"make_objects {asset_id}: SAM 3D Objects on frame {meta['frame']}, tracks {meta['tracks']}")
    if animate:
        track = json.loads((folder / "track.json").read_text())
        frames = [int(f) for f in track["frames"]]
        loc, quat = [], []
        for f in frames:
            l, q, _ = Matrix(track["poses"][str(f)]).decompose()
            loc.append(l[:])
            quat.append(q[:])
        _keys(obj, [f + 1 for f in frames], loc, quat)
        obj["motion_type"] = "dynamic"
    if physics:
        set_physics(obj, **physics)
    return obj


def rest_on(obj, z=0.0, turn=True, max_turn_deg=35.0, max_points=None):
    """Seat obj on a horizontal support whose top is at height z. With turn, first lay it on the stable face of its
    convex hull closest to facing down (a face its centre of mass stands over), turning it the least about its centre:
    an object resting on a floor or a table rests on a face, and a pose estimated from one view is often tilted about
    the line of sight. No turn when that face is more than max_turn_deg away (the object leans on something or stands
    on a point). Then lower or raise it so its lowest point is at z. Returns the turn in degrees."""
    import bmesh
    co = np.empty(3 * len(obj.data.vertices), np.float32)
    obj.data.vertices.foreach_get("co", co)
    m = np.array(obj.matrix_world)
    world = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
    angle = 0.0
    if turn and len(world) >= 4:
        bm = bmesh.new()
        hull_points = world if not max_points or len(world) <= max_points else \
            world[np.linspace(0, len(world) - 1, max_points).astype(int)]
        for p in hull_points:
            bm.verts.new(p)
        hull = bmesh.ops.convex_hull(bm, input=bm.verts)
        faces = [f for f in hull["geom"] if isinstance(f, bmesh.types.BMFace)]
        tri = np.array([[v.co[:] for v in f.verts] for f in faces if len(f.verts) == 3])
        bm.free()
        if len(tri):
            centre = tri.reshape(-1, 3).mean(0)
            normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
            normal *= np.sign(((tri.mean(1) - centre) * normal).sum(1, keepdims=True) + 1e-12)   # outward
            area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
            # the centre of mass of the solid hull (tetrahedra from its vertex mean), uniform density
            vol = np.einsum("ij,ij->i", tri[:, 0] - centre, np.cross(tri[:, 1] - centre, tri[:, 2] - centre)) / 6
            com = ((tri.sum(1) + centre) / 4 * vol[:, None]).sum(0) / max(vol.sum(), 1e-12)
            best, best_cos = None, -2.0
            done = np.zeros(len(tri), bool)
            for i in np.argsort(-area):       # coplanar triangles make one face: group them, test the whole face
                if done[i]:
                    continue
                n = normal[i]
                same = (~done) & (normal @ n > np.cos(np.radians(2))) & \
                    (np.abs((tri.mean(1) - tri[i, 0]) @ n) < 1e-3 + 1e-3 * np.ptp(world, 0).max())
                done |= same
                pts = tri[same].reshape(-1, 3)
                a = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
                a /= np.linalg.norm(a)
                b = np.cross(n, a)
                uv = np.stack([pts @ a, pts @ b], 1)
                c = np.array([com @ a, com @ b])
                ring = _hull2d(uv)
                if len(ring) >= 3 and _inside(c, ring) and -n[2] > best_cos:
                    best, best_cos = n, -n[2]
            if best is not None:
                angle = float(np.degrees(np.arccos(np.clip(best_cos, -1, 1))))
                if 0.5 < angle <= max_turn_deg:
                    axis = np.cross(best, [0, 0, -1.0])
                    turn_q = Quaternion(Vector(axis / np.linalg.norm(axis)), np.radians(angle))
                    R = np.array(turn_q.to_matrix())
                    m2 = np.array(obj.matrix_world)
                    m2[:3, :3] = R @ m2[:3, :3]
                    m2[:3, 3] = R @ (m2[:3, 3] - com) + com
                    obj.matrix_world = Matrix(m2.tolist())
                    world = co.reshape(-1, 3) @ m2[:3, :3].T + m2[:3, 3]
                else:
                    angle = 0.0
    obj.location.z += z - float(world[:, 2].min())
    obj["rest"] = json.dumps({"z": z, "turned_deg": round(angle, 1)})
    return angle


def _hull2d(points):
    """Convex hull of 2D points (monotone chain), counter-clockwise."""
    pts = sorted(set(map(tuple, np.round(points, 9))))
    if len(pts) < 3:
        return pts
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _inside(point, ring, margin=0.05):
    """point inside the convex polygon ring (counter-clockwise), at least margin of its size from the edges."""
    ring = np.asarray(ring)
    size = np.ptp(ring, 0).max()
    for k in range(len(ring)):
        a, b = ring[k], ring[(k + 1) % len(ring)]
        edge = b - a
        if (edge[0] * (point[1] - a[1]) - edge[1] * (point[0] - a[0])) / max(np.linalg.norm(edge), 1e-12) < margin * size:
            return False
    return True
