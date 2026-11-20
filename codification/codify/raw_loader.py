"""Loader for a raw perception folder (codex_agent.raw_evidence copies this file there as raw.py; README.md beside it
describes the files). numpy and json only; overlay() also needs OpenCV.

    import sys; sys.path.insert(0, "<the folder>"); import raw
    raw.tracks()                       # {track id: {"name", "labels", "value", "frames", "detections"}}
    raw.mask("12", 30)                 # bool H x W: track 12 on frame 30 (frames are 0-based canonical indices)
    raw.depth(30)                      # float32 H x W metres along the optical axis, 0 = no reliable depth (None off keyframes)
    raw.points("12", 30)               # N x 3 world points (metres, +Z up) of the track's mask pixels with depth
    raw.track_points("12")             # the same over every frame with a detection and depth, concatenated
    raw.on_plane(30, u, v, z=0.0)      # where pixels' camera rays meet the horizontal plane at height z (no depth)
    raw.contact("12", 30, z=0.0)       # the bottom edge of track 12's mask put on that plane along the rays
    raw.overlay(30, "overlay_30.jpg")  # the frame with every track's mask outlined and labelled, to look at
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


@lru_cache(None)
def cameras():
    """{canonical frame index: camera row} (R_world_rgb maps camera +X right, +Y down, +Z forward into the world)."""
    return {row["frame"] - 1: row for row in json.loads((HERE / "cameras.json").read_text())}


@lru_cache(None)
def tracks():
    return {t["track"]: t for t in json.loads((HERE / "tracks.json").read_text())["tracks"]}


@lru_cache(8)
def _labels(path):
    return np.load(HERE / path)["labels"]


def mask(track, frame):
    t = tracks()[str(track)]
    return _labels(t["labels"])[frame] == t["value"]


@lru_cache(1)
def _depth():
    z = np.load(HERE / "depth.npz")
    return {int(f): i for i, f in enumerate(z["frames"])}, z["depth"]


def depth_frames():
    return sorted(_depth()[0])


def depth(frame):
    index, stack = _depth()
    return stack[index[frame]].astype(np.float32) if frame in index else None


def _erode(m, pixels):
    out = m.copy()
    for _ in range(pixels):
        inner = out.copy()
        inner[1:, :] &= out[:-1, :]
        inner[:-1, :] &= out[1:, :]
        inner[:, 1:] &= out[:, :-1]
        inner[:, :-1] &= out[:, 1:]
        out = inner
    return out


def unproject(frame, u, v, z):
    """World points of pixels (u right, v down; canonical pixels) at depths z on this frame's camera."""
    cam = cameras()[frame]
    K, R, C = (np.array(cam["projection"]["K"]), np.array(cam["R_world_rgb"]), np.array(cam["position_world"]))
    rays = np.stack([(np.asarray(u, float) - K[0, 2]) / K[0, 0], (np.asarray(v, float) - K[1, 2]) / K[1, 1],
                     np.ones(np.shape(u))], axis=-1)
    return (rays * np.asarray(z, float)[..., None]) @ R.T + C


def points(track, frame, erode=1):
    """World points (N x 3) of the track's mask pixels with depth on this frame; erode trims the mask's edge pixels,
    where depth mixes the object with what is behind it."""
    d = depth(frame)
    if d is None:
        return np.zeros((0, 3))
    m = mask(track, frame)
    if erode:
        m = _erode(m, erode)
    v, u = np.nonzero(m & (d > 0))
    return unproject(frame, u, v, d[v, u])


def track_points(track, frames=None, erode=1):
    """points() over the given frames (default: every frame with a detection and depth), concatenated."""
    have = set(_depth()[0])
    frames = frames if frames is not None else [f for f in tracks()[str(track)]["frames"] if f in have]
    found = [points(track, f, erode) for f in frames]
    return np.concatenate(found) if found else np.zeros((0, 3))


def on_plane(frame, u, v, z=0.0):
    """World points where the camera rays of pixels (u, v) meet the horizontal plane at height z; NaN for a ray that
    does not reach it in front of the camera."""
    cam = cameras()[frame]
    K, R, C = (np.array(cam["projection"]["K"]), np.array(cam["R_world_rgb"]), np.array(cam["position_world"]))
    d = np.stack([(np.asarray(u, float) - K[0, 2]) / K[0, 0], (np.asarray(v, float) - K[1, 2]) / K[1, 1],
                  np.ones(np.shape(u))], axis=-1) @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z - C[2]) / d[..., 2]
    t = np.where(t > 0, t, np.nan)
    return C + d * t[..., None]


def contact(track, frame, z=0.0):
    """The bottom edge of the track's mask on this frame (the lowest mask pixel of each image column) put on the plane
    z along the camera rays: for an object resting on that plane and seen to its bottom, where it touches it. N x 3."""
    m = mask(track, frame)
    cols = np.flatnonzero(m.any(axis=0))
    if not len(cols):
        return np.zeros((0, 3))
    rows = m.shape[0] - 1 - np.argmax(m[::-1, cols], axis=0)
    return on_plane(frame, cols, rows + 0.5, z)


def overlay(frame, path, track_ids=None):
    """Writes the frame with the masks of the given tracks (default: all detected on it) outlined and labelled."""
    import cv2
    image = cv2.imread(str(HERE / "frames" / f"{frame:06d}.jpg"))
    ids = track_ids or [t for t, row in tracks().items() if frame in row["frames"]]
    for k, t in enumerate(ids):
        m = mask(t, frame).astype(np.uint8)
        if not m.any():
            continue
        colour = tuple(int(c) for c in np.random.default_rng(int(t) if str(t).isdigit() else k).integers(60, 255, 3))
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, colour, 2)
        ys, xs = np.nonzero(m)
        cv2.putText(image, str(t), (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(image, str(t), (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    cv2.imwrite(str(path), image)
    return str(path)
