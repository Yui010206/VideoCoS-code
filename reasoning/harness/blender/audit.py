"""Validate declared resources in an already-loaded Blender scene.

This is an application-level resource audit, not an operating-system sandbox.
"""
from __future__ import annotations


class ValidationError(ValueError):
    """The loaded scene declares a resource outside the codified .blend."""


EVIDENCE_POLICY = "codified_scene_only_v1"


def audit_scene_resources(data, current_scene=None):
    """Reject declared external dependencies without opening their paths.

    Packed images/fonts/sounds are part of the codified scene. Linked libraries,
    movies, disk caches and import-capable nodes are unsupported in this mode.
    This checks the loaded Blender datablocks, not an OS-level file sandbox:
    Blender may already have attempted a dependency read while opening a file.
    """
    rejected = set()

    def packed(block):
        return bool(getattr(block, "packed_file", None) or getattr(block, "packed_files", ()))

    if getattr(data, "libraries", ()):
        rejected.add("linked Blender libraries")
    packed_images = 0
    for image in getattr(data, "images", ()):
        source = getattr(image, "source", "FILE")
        if source in ("GENERATED", "VIEWER"):
            continue
        if source not in ("FILE", "TILED") or not packed(image):
            rejected.add("unpacked or time-dependent images")
        elif source == "TILED" and len(getattr(image, "packed_files", ())) < len(getattr(image, "tiles", ())):
            rejected.add("partially packed tiled images")
        else:
            packed_images += 1
    for sound in getattr(data, "sounds", ()):
        if not packed(sound):
            rejected.add("unpacked sounds")
    for font in getattr(data, "fonts", ()):
        if getattr(font, "filepath", "") != "<builtin>" and not packed(font):
            rejected.add("unpacked fonts")
    for volume in getattr(data, "volumes", ()):
        if getattr(volume, "filepath", ""):
            rejected.add("external volume files")
    if getattr(data, "movieclips", ()):
        rejected.add("movie clips")
    if getattr(data, "cache_files", ()):
        rejected.add("external cache files")

    scenes = list(getattr(data, "scenes", ()))
    if current_scene is not None and all(s is not current_scene for s in scenes):
        scenes.append(current_scene)

    def disk_cache(cache):
        if cache is not None and getattr(cache, "use_disk_cache", False):
            rejected.add("simulation disk caches")

    for scene in scenes:
        editor = getattr(scene, "sequence_editor", None)
        if editor is not None and (getattr(editor, "strips", ()) or getattr(editor, "sequences", ())):
            rejected.add("video sequence strips")
        disk_cache(getattr(getattr(scene, "rigidbody_world", None), "point_cache", None))

    for obj in getattr(data, "objects", ()):
        disk_cache(getattr(obj, "point_cache", None))
        disk_cache(getattr(getattr(obj, "soft_body", None), "point_cache", None))
        for particle_system in getattr(obj, "particle_systems", ()):
            disk_cache(getattr(particle_system, "point_cache", None))
        for modifier in getattr(obj, "modifiers", ()):
            disk_cache(getattr(modifier, "point_cache", None))
            if getattr(modifier, "type", "") in ("MESH_CACHE", "MESH_SEQUENCE_CACHE"):
                rejected.add("external mesh cache modifiers")
            if getattr(modifier, "type", "") == "FLUID" and getattr(modifier, "domain_settings", None) is not None:
                rejected.add("fluid simulation disk caches")
            canvas = getattr(modifier, "canvas_settings", None)
            for surface in getattr(canvas, "canvas_surfaces", ()):
                disk_cache(getattr(surface, "point_cache", None))
            if getattr(modifier, "type", "") == "NODES":
                for bake in getattr(modifier, "bakes", ()):
                    target = getattr(bake, "bake_target", "")
                    if target == "INHERIT":
                        target = getattr(modifier, "bake_target", "")
                    if target == "DISK" or getattr(bake, "directory", ""):
                        rejected.add("geometry node disk bakes")

    trees = list(getattr(data, "node_groups", ()))
    for collection in ("materials", "worlds", "scenes"):
        for block in getattr(data, collection, ()):
            tree = getattr(block, "node_tree", None)
            if tree is not None:
                trees.append(tree)
    for tree in trees:
        for node in getattr(tree, "nodes", ()):
            node_type = getattr(node, "bl_idname", "")
            if node_type.startswith("GeometryNodeImport"):
                rejected.add("geometry import nodes")
            if node_type == "ShaderNodeScript" and getattr(node, "mode", "") == "EXTERNAL":
                rejected.add("external shader scripts")
    if rejected:
        # Do not copy arbitrary dependency paths or datablock text to reports.
        raise ValidationError("Scene-only mode rejects declared external resources: " + "; ".join(sorted(rejected)))
    return {"evidence_policy": EVIDENCE_POLICY, "declared_external_resources": False,
            "packed_image_count": packed_images,
            "scope": "Loaded datablock audit; not an operating-system file-access sandbox"}
