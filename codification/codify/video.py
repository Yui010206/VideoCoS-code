"""decode_video: canonical constant-frame-rate images, exact source PTS map, declared transforms.

The canonical timeline is CFR at a rational FPS. Each canonical frame keeps the source
frame index and original PTS it was taken from; VFR sources may repeat or drop source
frames, and that is recorded rather than hidden. Optional calibration-based
rectification is applied once, here, so every later tool works in the canonical pixel
grid. Display-rotation metadata is rejected rather than guessed.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from . import contract

CALIBRATION_SCHEMA = "videocos.codification.calibration.v1"


def cfr_mapping(source_pts_ns, fps_num, fps_den, start_index=0, end_index=None):
    """Map canonical frames to the nearest source frame (ties go to the earlier frame).

    source_pts_ns must be presentation-ordered. Canonical frame i is at
    pts[start_index] + timestamp_ns(i); frames are generated while that time stays
    within half a typical source period of the last source frame in the window.
    """
    pts = np.asarray(source_pts_ns[start_index:end_index], dtype=np.int64)
    if len(pts) == 1:       # a still image: one canonical frame at time 0
        row = {"frame_index": 0, "timestamp_ns": 0, "source_frame_index": start_index, "source_pts_ns": int(pts[0]),
               "source_offset_ns": 0, "repeated_source": False}
        return [row], {"canonical_frames": 1, "window_source_frames": 1, "repeated_source_frames": 0,
                       "unused_source_frames": 0, "max_abs_source_offset_ns": 0}
    if len(pts) < 2 or np.any(np.diff(pts) <= 0):
        raise ValueError("Source PTS must be strictly increasing presentation times")
    origin, rows = int(pts[0]), []
    tolerance = int(np.median(np.diff(pts))) // 2
    i = 0
    while True:
        t = contract.frame_timestamp_ns(i, fps_num, fps_den)
        target = origin + t
        if target > pts[-1] + tolerance:
            break
        k = int(np.searchsorted(pts, target))
        if k == len(pts) or (k > 0 and target - pts[k - 1] <= pts[k] - target):
            k -= 1
        rows.append({"frame_index": i, "timestamp_ns": t, "source_frame_index": start_index + k,
                     "source_pts_ns": int(pts[k]), "source_offset_ns": int(pts[k] - target),
                     "repeated_source": bool(rows) and rows[-1]["source_frame_index"] == start_index + k})
        i += 1
    used = {row["source_frame_index"] for row in rows}
    return rows, {"canonical_frames": len(rows), "window_source_frames": len(pts),
                  "repeated_source_frames": sum(r["repeated_source"] for r in rows),
                  "unused_source_frames": len(pts) - len(used),
                  "max_abs_source_offset_ns": max(abs(r["source_offset_ns"]) for r in rows)}


def probe(video_path):
    """Presentation-ordered PTS of every decoded frame plus stream metadata (PyAV)."""
    import av
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        pts, rotation = [], None
        for frame in container.decode(stream):
            pts.append(int(Fraction(frame.pts) * Fraction(frame.time_base) * 1_000_000_000))
            rotation = frame.rotation if rotation is None else rotation
        rate = stream.average_rate or stream.guessed_rate
        info = {"codec": stream.codec_context.name, "width": stream.width, "height": stream.height,
                "average_rate": [rate.numerator, rate.denominator], "time_base": str(stream.time_base),
                "decoded_frames": len(pts), "declared_frames": stream.frames, "display_rotation_deg": rotation or 0}
    order = np.argsort(pts, kind="stable")
    if np.any(order != np.arange(len(pts))):
        raise ValueError("Decoded frames are not in presentation order")
    return pts, info


def load_calibration(path):
    calibration = contract.read_json(path)
    if calibration.get("schema") != CALIBRATION_SCHEMA or calibration.get("model") != "fisheye624" \
            or len(calibration.get("params", [])) != 15:
        raise ValueError(f"Unsupported calibration declaration: {path}")
    return calibration


def rectification_maps(params, K, width, height):
    """Per-pixel source coordinates for a pinhole view K through a fisheye624 lens."""
    u, v = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    rays = np.linalg.inv(K) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    uv = contract.fisheye624_project(params, rays.T)
    return uv[:, 0].reshape(height, width).astype(np.float32), uv[:, 1].reshape(height, width).astype(np.float32)


def decode_video(video_path, output_dir, config):
    """Write canonical frames, timeline.jsonl and video_info.json into output_dir.

    config: start_frame / end_frame (source indices, end exclusive), stride or
    fps [num, den], optional rectify {calibration, size [w, h], hfov_deg}, optional max_side
    (downscale so the longer side fits; a declared K is scaled with it), jpeg_quality,
    cut_threshold (histogram correlation below which a cut is reported). A still image decodes
    as a one-frame video.
    """
    import av
    import cv2
    out = Path(output_dir)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=False)
    pts, info = probe(video_path)
    rate = Fraction(*info["average_rate"])
    fps = Fraction(*config["fps"]) if "fps" in config else rate / int(config.get("stride", 1))
    start, end = int(config.get("start_frame", 0)), config.get("end_frame")
    rows, mapping = cfr_mapping(pts, fps.numerator, fps.denominator, start, end)
    warnings, transforms = [], []
    if info["display_rotation_deg"]:
        raise ValueError(f"Display rotation metadata ({info['display_rotation_deg']} deg) is not supported in v1; "
                         "transcode the video upright first so pixels, masks and intrinsics share one grid")

    rectify, K, projection = config.get("rectify"), None, None
    if rectify:
        calibration = load_calibration(rectify["calibration"])
        width, height = rectify["size"]
        focal = (width / 2) / np.tan(np.radians(rectify["hfov_deg"]) / 2)
        K = np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]])
        map_x, map_y = rectification_maps(calibration["params"], K, width, height)
        inside = (map_x >= 0) & (map_x <= calibration["width"] - 1) & (map_y >= 0) & (map_y <= calibration["height"] - 1)
        if not inside.all():
            warnings.append(f"Rectified view samples outside the source image on {1 - inside.mean():.1%} of pixels")
        transforms.append({"type": "fisheye624_to_pinhole_rectification",
                           "calibration_sha256": contract.sha256(rectify["calibration"]),
                           "calibration_source": calibration.get("source"), "hfov_deg": rectify["hfov_deg"],
                           "K": K.tolist(), "size": [width, height]})
        projection = {"model": "pinhole", "K": K.tolist(), "width": width, "height": height,
                      "distortion_model": "none", "distortion_coeffs": []}

    width, height = (rectify["size"] if rectify else (info["width"], info["height"]))
    scale = min(1.0, int(config.get("max_side", max(width, height))) / max(width, height))
    if scale < 1:
        resized = (round(width * scale), round(height * scale))
        transforms.append({"type": "resize", "from_size": [width, height], "size": list(resized),
                           "scale": [resized[0] / width, resized[1] / height]})
        if projection:
            K = np.array(projection["K"])
            K[0] *= resized[0] / width
            K[1] *= resized[1] / height
            K[0, 2] += 0.5 * resized[0] / width - 0.5     # integer pixel centers stay at integer coordinates
            K[1, 2] += 0.5 * resized[1] / height - 0.5
            projection = {**projection, "K": K.tolist(), "width": resized[0], "height": resized[1]}

    wanted = {}
    for row in rows:
        wanted.setdefault(row["source_frame_index"], []).append(row["frame_index"])
    quality = int(config.get("jpeg_quality", 95))
    histograms, size, last = {}, None, max(wanted)
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if index > last:
                break
            if index not in wanted:
                continue
            image = frame.to_ndarray(format="rgb24")
            if rectify:
                image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            if scale < 1:
                image = cv2.resize(image, resized, interpolation=cv2.INTER_AREA)
            size = (image.shape[1], image.shape[0])
            for canonical in wanted[index]:
                cv2.imwrite(str(frames_dir / f"{canonical:06d}.jpg"), image[:, :, ::-1],
                            [cv2.IMWRITE_JPEG_QUALITY, quality])
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            histograms[index] = cv2.normalize(cv2.calcHist([gray], [0], None, [64], [0, 256]), None).ravel()
    written = sorted(frames_dir.glob("*.jpg"))
    if len(written) != len(rows):
        raise RuntimeError(f"Decoded {len(written)} canonical frames, expected {len(rows)}")
    threshold = float(config.get("cut_threshold", 0.5))
    cuts = [row["frame_index"] for previous, row in zip(rows, rows[1:])
            if cv2.compareHist(histograms[previous["source_frame_index"]].astype(np.float32),
                               histograms[row["source_frame_index"]].astype(np.float32),
                               cv2.HISTCMP_CORREL) < threshold]
    if cuts:
        raise RuntimeError(f"Possible camera cut before canonical frames {cuts}; split the clip at cuts "
                           "(raise cut_threshold only after checking the frames)")

    contract.write_jsonl(out / "timeline.jsonl", rows)
    video_info = {"source_video": str(Path(video_path).resolve()), "source_video_sha256": contract.sha256(video_path),
                  "source": info, "window_source_frames": [start, end],
                  "canonical": {"width": size[0], "height": size[1], "fps_num": fps.numerator,
                                "fps_den": fps.denominator, "frame_count": len(rows)},
                  "mapping": mapping, "transforms": transforms, "projection": projection,
                  "cut_candidates": cuts, "jpeg_quality": quality,
                  "frame_files": "frames/<frame_index:06d>.jpg (RGB, canonical pixel grid)"}
    contract.write_json(out / "video_info.json", video_info)
    summary = {"canonical_frames": len(rows), "fps": [fps.numerator, fps.denominator], "size": list(size),
               "source_window": [start, rows[-1]["source_frame_index"] + 1], "rectified": bool(rectify),
               **{k: mapping[k] for k in ("repeated_source_frames", "unused_source_frames")}}
    return contract.tool_result("success", {"frames_dir": frames_dir, "timeline": out / "timeline.jsonl",
                                            "video_info": out / "video_info.json"}, summary, warnings)
