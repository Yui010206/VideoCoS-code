"""Evidence conventions, camera math and file helpers shared by the harness and by code run inside Blender.

Depends on the standard library and numpy only (Blender's Python imports it too).

Conventions:
- World: right-handed Blender frame, +Z up, meters. Quaternions are wxyz, object-to-world.
- Camera rows: R_world_rgb maps canonical camera axes (+X right, +Y down, +Z forward)
  to world; position_world is the camera center. Blender camera = R_world_rgb @ diag(1,-1,-1).
- Pixels: OpenCV convention, integer pixel centers; boxes are half-open [x0, y0, x1, y1).
- Time: frame_index is zero-based; Blender frame = frame_index + 1; timestamp_ns comes
  from the rational canonical FPS, and source_pts_ns keeps the original stream time.
- Objects carry a proxy_recipe: the object_library category of their procedural stand-in.
- An evidence directory (cameras.json, objects.json, appearance.json, observations.jsonl, unresolved.json,
  world.json, structure.json, video.json, timeline.jsonl, manifest.json) is immutable once its manifest is
  written; verify_evidence checks the file hashes before any evaluation reads it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CAMERA_POSE_STATUSES = ("estimated", "interpolated", "held", "reference")
EVIDENCE_FIELDS = ("identity_source", "geometry_source", "pose_source", "rotation_status",
                   "dimension_status", "confidence")
FLIP_YZ = np.diag([1.0, -1.0, -1.0])


# ---------------------------------------------------------------- files and results

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_npz(path, names=None):
    """Arrays of an .npz archive (all, or those named), read-only. Each member is decompressed in one piece,
    several times faster than np.load on large compressed arrays (SAM 3 labels, depth maps)."""
    import zipfile
    with zipfile.ZipFile(path) as archive:
        members = [n[:-4] for n in archive.namelist() if n.endswith(".npy")]
        return {name: _npy(archive.read(name + ".npy")) for name in (names or members)}


def read_labels(path):
    """A prompt's SAM 3 label maps (labels_NNN.npz: uint16 frames x H x W), read-only."""
    return read_npz(path, ["labels"])["labels"]


def _npy(raw):
    import io
    header = io.BytesIO(raw)
    major, _ = np.lib.format.read_magic(header)
    read_header = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = read_header(header)
    return np.frombuffer(raw, dtype=dtype, offset=header.tell()).reshape(shape, order="F" if fortran else "C")


def write_json(path, value, indent=2):
    """Atomic write; NaN/inf are rejected rather than serialized."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=indent, allow_nan=False) + "\n")
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    os.replace(temp, path)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def tool_result(status, artifacts, summary, warnings=()):
    """Common return value of the six offline tools; large arrays stay in files."""
    return {"status": status, "artifacts": {k: str(v) for k, v in artifacts.items()},
            "summary": summary, "warnings": list(warnings)}


def frame_timestamp_ns(index, fps_num, fps_den):
    """Canonical elapsed time of frame_index, rounded to the nearest nanosecond."""
    return (2 * index * 1_000_000_000 * fps_den + fps_num) // (2 * fps_num)


# ---------------------------------------------------------------- rotations and cameras

def quat_wxyz_to_matrix(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat_wxyz(R):
    """Proper rotation matrix to a normalized wxyz quaternion with w >= 0."""
    R = np.asarray(R, float)
    trace = np.trace(R)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q) / np.linalg.norm(q)
    return (q if q[0] >= 0 else -q).tolist()


def camera_from_w2c(R_cw, t_cw):
    """World-to-camera [R | t] (OpenCV) to (R_world_rgb, camera center)."""
    R_cw, t_cw = np.asarray(R_cw, float), np.asarray(t_cw, float)
    return R_cw.T, -R_cw.T @ t_cw


def blender_camera_rotation(R_world_rgb):
    return np.asarray(R_world_rgb, float) @ FLIP_YZ


def project_pinhole(K, points_camera):
    p = np.asarray(points_camera, float)
    uv = (np.asarray(K, float) @ (p / p[:, 2:3]).T).T
    return uv[:, :2]


def pinhole_to_blender(K, width, height, sensor_width=36.0):
    """Blender PERSP parameters (sensor_fit HORIZONTAL) reproducing K exactly.

    Blender's image plane is continuous with pixel centers at u + 0.5, so the
    OpenCV principal point cx maps to cx + 0.5. Unequal fx/fy use pixel aspect.
    """
    K = np.asarray(K, float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if abs(K[0, 1]) > 1e-9 or fx <= 0 or fy <= 0:
        raise ValueError("Only skew-free pinhole intrinsics with positive focal lengths are supported")
    aspect_x, aspect_y = (1.0, fx / fy) if fx >= fy else (fy / fx, 1.0)
    ratio = aspect_y / aspect_x
    return {"type": "PERSP", "sensor_fit": "HORIZONTAL", "sensor_width": sensor_width,
            "lens": fx * sensor_width / width,
            "shift_x": (width / 2 - (cx + 0.5)) / width,
            "shift_y": ((cy + 0.5) - height / 2) * ratio / width,
            "pixel_aspect_x": aspect_x, "pixel_aspect_y": aspect_y,
            "resolution_x": int(width), "resolution_y": int(height)}


def blender_to_pinhole(lens, sensor_width, shift_x, shift_y, pixel_aspect_x, pixel_aspect_y, width, height):
    """Inverse of pinhole_to_blender for sensor_fit HORIZONTAL at 100% resolution."""
    ratio = pixel_aspect_y / pixel_aspect_x
    fx = lens * width / sensor_width
    return np.array([[fx, 0.0, width / 2 - shift_x * width - 0.5],
                     [0.0, fx / ratio, height / 2 + shift_y * width / ratio - 0.5],
                     [0.0, 0.0, 1.0]])


def fisheye624_project(params, points_camera):
    """Aria FisheyeRadTanThinPrism ("Fisheye624"): f, cx, cy, k0..k5, p0, p1, s0..s3."""
    params = np.asarray(params, float)
    f, cx, cy, k, p, s = params[0], params[1], params[2], params[3:9], params[9:11], params[11:15]
    xyz = np.asarray(points_camera, float)
    a, b = xyz[:, 0] / xyz[:, 2], xyz[:, 1] / xyz[:, 2]
    r = np.hypot(a, b)
    theta = np.arctan(r)
    theta2 = theta * theta
    radial = 1 + sum(k[i] * theta2 ** (i + 1) for i in range(6))
    scale = np.where(r < 1e-9, 1.0, theta / np.maximum(r, 1e-12)) * radial
    xr, yr = scale * a, scale * b
    rr = xr * xr + yr * yr
    temp = 2 * (xr * p[0] + yr * p[1])
    ud = xr * (1 + temp) + rr * p[0] + s[0] * rr + s[1] * rr * rr
    vd = yr * (1 + temp) + rr * p[1] + s[2] * rr + s[3] * rr * rr
    return np.stack([f * ud + cx, f * vd + cy], axis=1)


def fisheye624_scale(params, factor):
    """Parameters after resizing the image by `factor` (integer pixel centers)."""
    out = np.array(params, float)
    out[0] *= factor
    out[1:3] = (out[1:3] + 0.5) * factor - 0.5
    return out.tolist()


def fisheye624_rotate_cw90(params, height):
    """Parameters for the image rotated 90 degrees clockwise.

    A source pixel (u, v) moves to (height - 1 - v, u); the camera axes rotate the
    same way, so tangential and thin-prism coefficients rotate as 2D vectors.
    """
    f, cx, cy, *k = params[:9]
    p0, p1, s0, s1, s2, s3 = params[9:15]
    return [f, height - 1 - cy, cx, *k, -p1, p0, -s2, -s3, s0, s1]


# ---------------------------------------------------------------- bundle validation

def box_corners(item, pose=None):
    """World corners of an object's local bounds at a pose row (default: its first pose)."""
    b = np.array(item["bounds"]).reshape(3, 2)
    local = np.array([[b[0, i], b[1, j], b[2, k]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    pose = item["poses"][0] if pose is None else pose
    return local @ quat_wxyz_to_matrix(pose[4:8]).T + pose[1:4]


def project(row, points):
    """World points -> canonical pixels through an evidence camera row, or None if any point is behind it."""
    cam = (np.asarray(points) - row["position_world"]) @ np.asarray(row["R_world_rgb"])
    if np.any(cam[:, 2] < 0.05):
        return None
    projection = row["projection"]
    if projection["model"] == "fisheye624":
        K = projection["K"]
        return fisheye624_project([K[0][0], K[0][2], K[1][2], *projection["distortion_coeffs"]], cam)
    return project_pinhole(projection["K"], cam)


EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def comparison_frames(count, comparison=None):
    """The frames a run compares renders on: four evenly spaced frames (settings may name others). Perception
    publishes the frames and their object-id masks."""
    return [int(f) for f in sorted(comparison or {round(i * (count - 1) / 3) for i in range(4)})]


def write_manifest(directory, mode, sources, digest=None):
    """The manifest of an evidence directory: what it was made from and the hash of every file in it. An
    evidence version is never modified after this is written; verify_evidence checks that. digest identifies
    the sources (default: the hash of the file hashes)."""
    directory = Path(directory)
    files = {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob("*"))
             if p.is_file() and p.name != "manifest.json"}
    manifest = {"version": directory.name.removesuffix(".tmp"), "mode": mode,
                "digest": digest or hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
                "sources": sources, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "files": files}
    write_json(directory / "manifest.json", manifest)
    return manifest


def verify_evidence(directory):
    """The evidence directory's files still match its manifest (nothing was edited after it was published);
    returns the manifest."""
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    current = {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob("*"))
               if p.is_file() and p.name != "manifest.json"}
    if current != manifest["files"]:
        changed = sorted(set(current.items()) ^ set(manifest["files"].items()))
        raise RuntimeError(f"{directory} differs from its manifest: {changed[:5]}")
    return manifest
