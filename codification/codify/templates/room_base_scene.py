# The default scene of a room (fast room mode, 2026-11-01): every evidence object at its estimated pose (its prepared
# mesh where that agrees with its box), the floor and the walls the perception found, then arrange_room with automatic
# supports. It is a starting point: the positions, the floor and the walls are the perception's rough estimates.
import bpy
import numpy as np
import codify.assets as A

ev = A.load_evidence()
A.setup_scene(ev["video"], samples=16)
A.add_source_camera(ev["cameras"])
structure = ev.get("structure") or {}
floor = structure.get("floor") or {}
xr, yr = floor.get("x_range", [-5.0, 5.0]), floor.get("y_range", [-5.0, 5.0])
fz = floor.get("z", 0.0)

# The room's shell in the colours its depth points have: floor, walls, a ceiling above every camera, the lights.
points, rgb = A.load_points()


def median_colour(mask, default):
    return tuple(float(c) for c in np.median(rgb[mask], 0) / 255.0) if mask.sum() >= 50 else default


camera_top = max(c["position_world"][2] for c in ev["cameras"])
ceiling = min(max(camera_top + 0.4, floor.get("highest_surface_z_p99", fz + 2.6)), fz + 3.4)
floor_colour = median_colour(np.abs(points[:, 2] - fz) < 0.04, (0.68, 0.67, 0.64))
ceiling_colour = median_colour(points[:, 2] > ceiling - 0.15, (0.90, 0.89, 0.86))
A.add_floor((xr[0], xr[1]), (yr[0], yr[1]), z=fz, color=floor_colour)
wall_colours = []
for i, plane in enumerate(structure.get("vertical_planes", [])):
    near = np.abs(points[:, :2] @ np.asarray(plane["normal_xy"]) - plane["offset"]) < 0.05
    colour = median_colour(near & (points[:, 2] > fz + 0.3) & (points[:, 2] < ceiling - 0.3), (0.86, 0.85, 0.82))
    wall_colours.append(colour)
    A.add_wall(plane["p0"], plane["p1"], height=ceiling - fz, z0=fz, color=colour, name=f"Wall_{i}", collider=False)
span = (xr[1] - xr[0], yr[1] - yr[0])
A.add_environment(A.box_mesh("Ceiling", (span[0], span[1], 0.04), ((xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2, ceiling + 0.02),
                             color=ceiling_colour), collider=False)
world = bpy.context.scene.world
world.use_nodes = True
background = world.node_tree.nodes.get("Background")
background.inputs["Color"].default_value = (*(np.mean(wall_colours, 0) if wall_colours else ceiling_colour), 1.0)
background.inputs["Strength"].default_value = 0.5
for x in np.arange(xr[0] + 1.5, xr[1], 3.0):
    for y in np.arange(yr[0] + 1.5, yr[1], 3.0):
        lamp = bpy.data.lights.new("RoomLight", "AREA")
        lamp.energy, lamp.shape, lamp.size = 250.0, "DISK", 2.0
        light = bpy.data.objects.new("RoomLight", lamp)
        bpy.context.scene.collection.objects.link(light)
        light.location = (float(x), float(y), ceiling - 0.05)

# Built in, mounted or large: fixed colliders. On a wall or a ceiling: kept where they are. Furniture: not moved.
FIXED = {"wardrobe", "cabinet", "cabinet door", "cupboard", "countertop", "counter", "kitchen island", "range hood",
         "shelf", "bookshelf", "bookcase", "washing machine", "dryer", "dishwasher", "refrigerator", "fridge", "oven",
         "stove", "microwave", "sink", "door", "window", "curtain", "blinds", "painting", "picture", "picture frame",
         "wall artwork", "poster", "clock", "mirror", "backsplash", "headboard", "radiator", "heater", "tv",
         "television", "monitor", "light fixture", "light switch", "bed", "sofa", "couch", "table", "desk", "dresser",
         "nightstand", "toilet", "bathtub", "shower", "fireplace", "stairs"}
HANG = {"curtain", "blinds", "painting", "picture", "picture frame", "wall artwork", "poster", "clock", "mirror", "tv",
        "television", "light fixture", "light switch", "range hood", "shelf", "backsplash", "headboard", "window",
        "sign", "whiteboard", "calendar"}
# Built in, so left where they are. Everything else that stands is put on the floor by arrange_room.
KEEP = {"countertop", "counter", "cabinet", "cabinet door", "cupboard", "oven", "stove", "microwave", "sink", "stairs"}
STAND = {"kitchen island", "fireplace"}


def top_only(o):
    """A table or desk of which only the top was seen: lowering it would put its top on the floor; it needs legs."""
    return o["category"] in {"table", "desk"} and o["bounds"][5] - o["bounds"][4] < 0.30



# Proxy shapes (an object without an agreeing mesh) for categories the helper library names otherwise.
RECIPES = {"tv": "television", "monitor": "television", "screen": "television", "painting": "wall artwork",
           "picture": "wall artwork", "picture frame": "wall artwork", "poster": "wall artwork", "clock": "wall clock",
           "plant": "house plant", "plant pot": "house plant", "potted plant": "house plant", "flower": "vase",
           "flowers": "vase", "christmas tree": "house plant", "tree": "house plant", "sofa": "couch",
           "bed": "bed frame", "stool": "bar stool", "mug": "cup", "glass": "cup", "pan": "frying pan",
           "kettle": "tea kettle", "lamp": "desk lamp", "floor lamp": "desk lamp", "trash bin": "trash can",
           "bin": "trash can", "laptop": "laptop computer", "keyboard": "computer keyboard", "mouse": "computer mouse",
           "nightstand": "side table", "end table": "side table", "dresser": "cabinet", "wardrobe": "cabinet",
           "fridge": "refrigerator", "stove": "cooktop", "basket": "clothes basket", "cushion": "pillow",
           "books": "book", "magazine": "magazine", "speaker": "speaker", "router": "router", "toaster": "toaster"}

for uid, o in ev["objects"].items():
    pose, category = o["poses"][0], o["category"]
    color = (ev.get("appearance") or {}).get(uid, {}).get("median_srgb") or (0.55, 0.55, 0.53)
    role = "passive" if category in FIXED else "active"
    physics = {"role": role, "friction": 0.6, "restitution": 0.0, "shape": "CONVEX_HULL"}
    if role == "active":
        physics["mass_kg"] = 0.5
    A.add_object(uid, o.get("instance_name") or f"{category}_{uid}", category, o["bounds"], location=pose[1:4],
                 rotation_wxyz=pose[4:8], recipe=RECIPES.get(category) or o.get("proxy_recipe") or category,
                 color=color[:3],
                 poses=o["poses"] if o.get("motion_type", "static") != "static" else None, physics=physics)

A.arrange_room(hang=[u for u, o in ev["objects"].items() if o["category"] in HANG],
               keep=[u for u, o in ev["objects"].items() if o["category"] in KEEP or top_only(o)],
               stand=[u for u, o in ev["objects"].items() if o["category"] in STAND], auto_support=True)
A.layout_report()
