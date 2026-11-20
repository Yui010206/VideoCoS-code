"""Render a submitted update to video, after the model loop has ended.

The settings come from here, never from the model: Cycles on the CPU, a fixed sample count, the
scene camera of the submitted scene (T5: the target camera the contract checked; T6/T7: the source
camera), and the frames the family is scored on (T7: only the predicted future).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .session import BlenderSession

SAMPLES = 32


def frames_for(task, scene_range):
    first, last = scene_range
    if task.family == "T7":
        return task.spec["observed_end"] + 1, task.spec["target_end"]
    return first, last


def render_submission(task, final_blend, out_dir, cpus=16, samples=SAMPLES, percentage=100, docker=True):
    """final.blend -> frames/ -> video.mp4 under out_dir; returns what was written."""
    out_dir = Path(out_dir)
    work = out_dir / "render_session"
    if work.exists():
        shutil.rmtree(work)
    with BlenderSession(final_blend, work, docker=docker, cpus=cpus, memory="16g") as session:
        first, last = frames_for(task, session.frame_range)
        directory = "/work/frames" if docker else str(work / "frames")
        result = session.request("render", {"directory": directory, "frame_start": first, "frame_end": last,
                                            "engine": "CYCLES", "samples": samples, "percentage": percentage})
    if result.get("ok") is False:
        raise RuntimeError(result["error"])
    frames = sorted((work / "frames").glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"no frames were written to {work / 'frames'}")
    video = out_dir / "video.mp4"
    listing = work / "frames" / "frames.txt"          # concat resolves entries relative to the list file
    listing.write_text("".join(f"file '{p.name}'\n" for p in frames))
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-r", str(result["fps"]), "-f", "concat",
                    "-safe", "0", "-i", str(listing), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "16", "-an", str(video)], check=True, cwd=work / "frames")
    return {"video": str(video), "frames": [first, last], "written": len(frames), "fps": result["fps"],
            "camera": result["camera"], "resolution": result["resolution"], "samples": samples}
