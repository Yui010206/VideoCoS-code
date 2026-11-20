"""Given cameras: a dataset's own camera parameters as an input, instead of estimating them.

A given-cameras file (schema below) holds intrinsics and OpenCV world-to-camera extrinsics per
source frame, in the dataset's world frame (up axis and units declared, optionally the ground
height). fuse_objects then uses these cameras and that world frame; Depth Anything 3 still supplies
the depth, scaled to the given cameras (their path when they move, else the declared ground). Dataset adapters
(from_kubric_view, from_static_kubric) write the file once; nothing else knows the dataset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import contract

SCHEMA = "videocos.codification.given_cameras.v1"


def load(path, timeline, canonical_size):
    """Per-canonical-frame (K, w2c 3x4) on the canonical pixel grid, plus the declared world."""
    data = contract.read_json(path)
    if data.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported given-cameras file: {path}")
    by_source = {row["source_frame_index"]: row for row in data["frames"]}
    sx, sy = canonical_size[0] / data["width"], canonical_size[1] / data["height"]
    K, w2c = [], []
    for row in timeline:
        if row["source_frame_index"] not in by_source:
            raise ValueError(f"No given camera for source frame {row['source_frame_index']}")
        k = np.array(by_source[row["source_frame_index"]]["K"], float)
        k[0] *= sx
        k[1] *= sy
        k[0, 2] += 0.5 * sx - 0.5
        k[1, 2] += 0.5 * sy - 0.5
        K.append(k)
        w2c.append(np.array(by_source[row["source_frame_index"]]["w2c"], float))
    return np.array(K), np.array(w2c), data["world"]


def _kubric_file(output_path, width, height, focal_mm, sensor_mm, poses, source, source_sha256):
    """poses: per source frame (position xyz, quaternion wxyz of the Blender camera, which looks down -Z with +Y up)."""
    focal = focal_mm / sensor_mm * width                                    # Kubric fits the sensor to the width
    K = [[focal, 0.0, width / 2 - 0.5], [0.0, focal, height / 2 - 0.5], [0.0, 0.0, 1.0]]
    frames = []
    for index, (position, (w, x, y, z)) in enumerate(poses):
        R_blender = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                              [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                              [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        R_c2w = R_blender @ np.diag([1.0, -1.0, -1.0])                     # Blender camera axes -> OpenCV
        w2c = np.hstack([R_c2w.T, -R_c2w.T @ np.array(position)[:, None]])
        frames.append({"source_frame_index": index, "K": K, "w2c": w2c.tolist()})
    contract.write_json(output_path, {
        "schema": SCHEMA, "width": width, "height": height,
        "pixel_convention": "integer pixel centers (OpenCV)",
        "world": {"up": [0.0, 0.0, 1.0], "units": "Kubric scene units (meters)", "ground_z": 0.0,
                  "ground_note": "Kubric scenes put the floor at z = 0 by construction (simulator convention)"},
        "source": source, "source_sha256": source_sha256, "frames": frames})
    return output_path


def from_kubric_view(view_json, output_path):
    """A Kubric / Kubric-4D view JSON -> given-cameras file, from its camera block only (no scene state)."""
    view = contract.read_json(view_json)
    camera, (width, height) = view["camera"], view["scene"]["resolution"]
    return _kubric_file(output_path, width, height, camera["focal_length"], camera["sensor_width"],
                        list(zip(camera["positions"], camera["quaternions"])),
                        f"Kubric camera block of {Path(view_json).name}", contract.sha256(view_json))


def from_static_kubric(camera, frames, output_path, source):
    """A static Kubric camera as a benchmark states it (position, quaternion_wxyz, focal_length_mm,
    sensor_width_mm, resolution_wh) -> given-cameras file with the same camera on every frame."""
    width, height = camera["resolution_wh"]
    return _kubric_file(output_path, width, height, camera["focal_length_mm"], camera["sensor_width_mm"],
                        [(camera["position"], camera["quaternion_wxyz"])] * int(frames), source, None)
