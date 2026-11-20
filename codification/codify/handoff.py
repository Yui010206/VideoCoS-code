"""Reasoning handoff for a codified .blend: metric gate -> prepare_blend -> live-query smoke tests -> isolation.

Uses the VideoCoS engine unchanged (BlendScene, VSIHarness, prepare_blend); the reasoning stage keeps opening
the .blend directly. A scene without metric scale is refused here, before prepare_blend. No model API calls.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from . import contract


def smoke_queries(BlendScene, VSIHarness, blend, blender):
    """The engine's query interface on the built scene: inventory, inspection, a measurement and the camera
    history; returns (results, failures)."""
    results, failures = {}, []
    with BlendScene(blend, blender) as scene:
        harness = VSIHarness(scene)
        summary = scene.summary()
        results["summary"] = {k: summary.get(k) for k in ("object_count", "semantic_object_count", "mesh_count",
                                                          "camera_type", "fps", "frame_range", "units")}
        results["resource_audit"] = summary.get("resource_audit")
        ids = sorted(scene.objects)
        inventory = harness.execute("scene_inventory", {"kind": "collection", "limit": 128})
        inspected = harness.execute("inspect_entities", {"ids": ids[:2], "detail": "basic"}) if ids else {}
        measured = harness.execute("measure_pair", {"a": ids[0], "b": ids[1], "metric": "center"}) if len(ids) > 1 else {}
        history = harness.execute("camera_history", {"entity_ids": ids[:32], "limit": 16, "detail": "summary"}) if ids else {}
        results.update(scene_inventory={"ok": inventory.get("ok"), "total_matches": inventory.get("total_matches")},
                       inspect_entities={"ok": inspected.get("ok")},
                       measure_pair={"ok": measured.get("ok"), "distance_m": measured.get("distance_m")},
                       camera_history={"ok": history.get("ok"), "camera_type": summary.get("camera_type")})
        failures += [f"{name} failed" for name in ("scene_inventory", "inspect_entities", "measure_pair")
                     if not results[name]["ok"]]
        if summary.get("camera_type") in ("PERSP", "ORTHO") and not history.get("ok"):
            failures.append("camera_history failed on a PERSP/ORTHO scene")
    return results, failures


def handoff(blend, blender, output_dir, videocos_repo, scale_status):
    """Metric gate, prepare_blend, smoke queries on the scene and on a copy moved to an empty directory (the
    .blend must not depend on anything beside itself). Writes handoff_report.json; returns the report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {"schema": "videocos.codification.handoff.v1", "blend": str(blend), "scale_status": scale_status,
              "model_api_calls": 0}
    if scale_status not in ("metric_calibrated", "metric_estimated"):
        result.update(status="refused", reasons=[f"scale {scale_status}: not metric; preview only"])
        contract.write_json(out / "handoff_report.json", result)
        return result
    sys.path.insert(0, str(Path(videocos_repo).resolve()))
    from videocos.engine.harness import VSIHarness
    from videocos.engine.scene import BlendScene
    from videocos.engine.source import prepare_blend
    result["prepared_source"] = str(prepare_blend(blend, out / "prepared", blender))
    result["smoke"], failures = smoke_queries(BlendScene, VSIHarness, blend, blender)
    with tempfile.TemporaryDirectory(prefix="codify_isolation_") as tmp:
        moved = Path(tmp) / "moved.blend"
        shutil.copyfile(blend, moved)
        result["isolation"], isolation_failures = smoke_queries(BlendScene, VSIHarness, moved, blender)
    result["failures"] = failures + [f"isolation: {f}" for f in isolation_failures]
    result["status"] = "ready" if not result["failures"] else "failed"
    contract.write_json(out / "handoff_report.json", result)
    return result
