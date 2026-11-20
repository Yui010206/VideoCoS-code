"""Worker-side state passed to every query function.

``SceneContext`` holds the loaded Blender scene, its entity index, the current
evaluated frame and a per-query mesh cache. Runs inside Blender only.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

from ..protocol import integer
from .audit import audit_scene_resources


class SceneContext:
    """Entity index plus the evaluated frame and mesh cache of the running query."""

    def __init__(self):
        import bpy
        self.bpy = bpy
        self.scene = bpy.context.scene
        self.audit = audit_scene_resources(bpy.data, self.scene)
        self.scale = float(self.scene.unit_settings.scale_length)
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Invalid scene unit scale")
        self.duplicate_ids = []
        self._index(strict=True)

    def refresh(self):
        """Re-index after an edit: objects may have been added, removed or renamed, and the
        timeline may have changed. A duplicated object keeps its copied instance_id, so on
        refresh the first holder wins and the rest are reported instead of failing every query."""
        import bpy
        self.scene = bpy.context.scene
        self._index(strict=False)

    def _index(self, strict):
        self.first, self.last = self.scene.frame_start, self.scene.frame_end
        self.meshes = {"mesh:" + o.name: o for o in sorted(self.scene.objects, key=lambda x: x.name)
                       if o.type == "MESH"}
        if not self.meshes:
            raise ValueError("Scene contains no meshes")
        self.entities = {}
        self.semantic_objects = {}
        self.duplicate_ids = []
        for obj in sorted(self.scene.objects, key=lambda x: x.name):
            if "instance_id" not in obj:
                continue
            uid = str(obj["instance_id"])
            if not uid or len(uid) > 512 or uid in self.semantic_objects:
                if strict:
                    raise ValueError("Invalid or duplicate authored instance ID")
                self.duplicate_ids.append({"instance_id": uid, "object": obj.name})
                continue
            self.semantic_objects[uid] = obj

        direct_memberships = {uid: set() for uid in self.meshes}
        def walk(collection, parent=None):
            members = sorted({"mesh:" + o.name for o in collection.all_objects
                              if "mesh:" + o.name in self.meshes})
            if members:
                uid = "collection:" + collection.name
                self.entities[uid] = {"id": uid, "name": collection.name, "kind": "collection",
                                      "parent": parent, "members": members}
                # Object.users_collection scans every collection for every object.
                # Invert direct links once; ancestor membership remains excluded.
                for obj in collection.objects:
                    mid = "mesh:" + obj.name
                    if mid in direct_memberships:
                        direct_memberships[mid].add(uid)
                parent = uid
            for child in collection.children:
                walk(child, parent)
        for child in self.scene.collection.children:
            walk(child)
        for uid, obj in self.meshes.items():
            self.entities[uid] = {"id": uid, "name": obj.name, "kind": "mesh", "members": [uid],
                                  "collections": sorted(direct_memberships[uid])}
        self.depsgraph, self.frame, self.mesh_cache = None, None, {}

    def summary(self):
        return {"backend": "live_blender", "evidence_policy": "codified_scene_only_v1",
                "evidence_origin": "live_evaluated_blend", "units": "meters",
                "scale_length": self.scale, "unit_system": self.scene.unit_settings.system,
                "frame_range": [self.first, self.last], "default_frame": self.first,
                "fps": self.scene.render.fps / self.scene.render.fps_base,
                "mesh_count": len(self.meshes), "collection_count": len(self.entities) - len(self.meshes),
                "object_count": len(self.semantic_objects or self.meshes),
                "semantic_object_count": len(self.semantic_objects),
                "coordinate_system": "right-handed Blender world, +Z up",
                "available_details": ["basic", "transform", "geometry", "materials"],
                "camera_type": self.scene.camera.data.type if self.scene.camera else None,
                "resource_audit": self.audit,
                "limitations": [
                    "Collections are authored groups, not certified object instances; mesh objects may be parts.",
                    "Geometry and scale are reconstruction estimates, not ground truth.",
                    "Geometry uses the active view-layer/viewport dependency graph; render-only modifiers, subdivision levels and visibility settings can differ.",
                    "Each queried entity is capped at 500,000 evaluated vertices and 1,000,000 triangles; exceeding it returns an error, not approximate geometry.",
                    "Bounds are evaluated world AABBs; request metric=surface for triangle surface distance.",
                    "Surface queries can stop at a work or time cap and return bounds instead of an exact result.",
                    "Camera visibility uses sampled ray probes, not rendered pixels or exact first appearance.",
                    "Only mesh objects are indexed. Unrealized instances and non-mesh objects are not separately indexed.",
                    "Material defaults and texture presence do not reproduce rendered appearance.",
                    "No source video, annotations, answer files, text datablocks, or custom properties are exposed."]}

    def members(self, uid):
        if uid not in self.entities:
            raise ValueError("Unknown entity ID; use scene_inventory first")
        return self.entities[uid]["members"]

    def object(self, uid):
        """An authored instance ID or mesh ID; collections are not objects."""
        obj = self.semantic_objects.get(uid, self.meshes.get(uid))
        if obj is None:
            raise ValueError("Unknown editable object ID; use the scene object inventory")
        return obj

    def set_frame(self, frame):
        """Evaluate the scene at an integer frame; only frame-independent meshes stay cached."""
        frame = integer(frame, "frame", self.first, self.last)
        if frame != self.frame:
            self.scene.frame_set(frame)
            self.bpy.context.view_layer.update()
            self.depsgraph = self.bpy.context.evaluated_depsgraph_get()
            self.mesh_cache = {uid: mesh for uid, mesh in self.mesh_cache.items() if mesh["static"]}
            self.frame = frame
        return frame

    @contextmanager
    def query(self):
        """Run one query, then drop its caches and restore the scene's original frame."""
        frame, subframe = self.scene.frame_current, self.scene.frame_subframe
        try:
            yield self
        finally:
            self.depsgraph, self.frame, self.mesh_cache = None, None, {}
            if (self.scene.frame_current, self.scene.frame_subframe) != (frame, subframe):
                self.scene.frame_set(frame, subframe=subframe)
                self.bpy.context.view_layer.update()


def semantic_metadata(uid, obj):
    fields = ("category", "prototype_name", "motion_type", "geometry_status")
    return {"object_id": uid, "name": obj.name, "entity_id": "mesh:" + obj.name if obj.type == "MESH" else None,
            **{key: str(obj.get(key, "unknown"))[:512] for key in fields}}
