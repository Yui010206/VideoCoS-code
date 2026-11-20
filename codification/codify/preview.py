"""Preview video of a finished run, for reports only (never shown to the model): the input beside the
reconstruction rendered from the source camera over the input's duration. Codification's output is the scene of
the input, so nothing after the last input frame is shown. A single image is shown still for two seconds, then
every movable part of the scene is moved through its joint range (what the scene lets one act on).
Output: final/preview.mp4 (H.264) and final/preview.json."""
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from . import contract

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(image, text):
    cv2.putText(image, text, (8, 22), FONT, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (8, 22), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def _fit(image, height):
    width = max(2, round(image.shape[1] * height / image.shape[0] / 2) * 2)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def preview_video(run, box, fps=10.0, size=512, max_frames=600, sweep_steps=12, samples=8, log=print):
    run = Path(run)
    final = run / "final"
    if not (final / "scene.blend").is_file():
        raise FileNotFoundError(f"{final / 'scene.blend'}: the run has no final scene")
    decode = run / "work" / "decode"
    count = len(contract.read_jsonl(decode / "timeline.jsonl"))
    video = contract.read_json(decode / "video_info.json")["canonical"]
    source_fps = video["fps_num"] / video["fps_den"]
    single = count == 1
    stride = max(1, math.ceil(source_fps / fps), math.ceil(count / max_frames))
    out_fps = fps if single else source_fps / stride
    inputs = list(range(0, count, stride))                     # canonical frames; the scene frame is f + 1
    if inputs[-1] != count - 1:
        inputs.append(count - 1)

    renders = final / "preview_renders"
    shutil.rmtree(renders, ignore_errors=True)
    args = ["render", "--frames", ",".join(str(f + 1) for f in inputs), "--size", str(size), "--samples",
            str(samples), "--threads", str(box.cpus), "--out", "/out/preview_renders"]
    if single:
        args += ["--sweep-joints", str(sweep_steps)]
    started = time.time()
    log(f"[preview] rendering {len(inputs)} frames" + (" and the joint sweep" if single else ""))
    result = box.blender(args, final, blend="scene.blend", timeout=6 * 3600)
    if result.get("status") != "ok":
        raise RuntimeError("preview render failed: " + str(result.get("stderr", result))[-2000:])
    render_seconds = time.time() - started
    info_path = renders / "render_info.json"
    sweep = contract.read_json(info_path).get("sweep", []) if info_path.is_file() else []

    composed = final / "preview_frames"
    shutil.rmtree(composed, ignore_errors=True)
    composed.mkdir()
    written = 0

    def emit(left, right):
        nonlocal written
        cv2.imwrite(str(composed / f"{written:06d}.png"), np.hstack([left, right]))
        written += 1

    source = None
    for f in inputs:
        render = cv2.imread(str(renders / f"render_{f + 1:06d}.png"))
        if render is None:
            continue
        source = _fit(cv2.imread(str(decode / f"frames/{f:06d}.jpg")), render.shape[0])
        for _ in range(round(2 * out_fps) if single else 1):
            emit(_label(source.copy(), "input"), _label(render.copy(), "reconstruction"))
    if source is None:
        raise RuntimeError("no input frame was rendered")
    k = 0
    for item in sweep:
        for _ in range(item["frames"]):
            image = cv2.imread(str(renders / f"sweep_{k:06d}.png"))
            k += 1
            if image is not None:
                emit(_label(source.copy(), "input"),
                     _label(image, f"{item['part']} ({item['type']} joint of {item['part_of']})"))

    output = final / "preview.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{out_fps:.4f}", "-i", str(composed / "%06d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], check=True)
    shutil.rmtree(composed)
    shutil.rmtree(renders)
    info = {"file": "final/preview.mp4", "fps": round(out_fps, 4), "stride": stride, "input_frames": len(inputs),
            "joint_sweep": sweep, "size": size, "samples": samples, "frames_written": written,
            "render_seconds": round(render_seconds, 1), "bytes": output.stat().st_size,
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (final / "preview.json").write_text(json.dumps(info, indent=2))
    return info
