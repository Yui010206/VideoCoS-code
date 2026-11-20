#!/usr/bin/env python3
"""Fetch VSI-Bench room videos (the T1-T4 inputs) into bench/data/vsi_bench/videos/<dataset>/<scene>.mp4.

The videos come from ScanNet, ScanNet++ and ARKitScenes; VSI-Bench (huggingface.co/datasets/nyu-visionx/VSI-Bench)
ships them as three zips. This script is not a mirror: it reads the zips from VSI-Bench itself.

    python3 scripts/fetch_vsi_videos.py --demo                 # the two demo rooms (scene0164_02, scene0221_01), ~12 MB
    python3 scripts/fetch_vsi_videos.py scene0164_02 41069025   # any scene names of bench/meta/tasks/T1-T4.json
    python3 scripts/fetch_vsi_videos.py --all                   # every T1-T4 video: downloads the three zips (5.7 GB)

Single scenes are cut out of the remote zip with HTTP range requests, so only those bytes are downloaded.
Standard library only. Set HF_ENDPOINT to use a Hugging Face mirror.
"""
import argparse
import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = Path(os.environ.get("VIDEOCOS_BENCH", REPO / "bench"))
VIDEOS = BENCH / "data" / "vsi_bench" / "videos"
URL = os.environ.get("HF_ENDPOINT", "https://huggingface.co") + "/datasets/nyu-visionx/VSI-Bench/resolve/main/{}.zip"
DEMO = ["scene0164_02", "scene0221_01"]


class RemoteFile(io.RawIOBase):
    """A read-only, seekable view of a remote file through HTTP range requests."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.url, self.size = r.geturl(), int(r.headers["Content-Length"])   # follow the redirect once

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def readinto(self, buffer):
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(request, timeout=300) as r:
            data = r.read()
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def room_datasets():
    """scene name -> VSI dataset (arkitscenes | scannet | scannetpp), from the T1-T4 manifests."""
    rooms = {}
    for task in ("T1", "T2", "T3", "T4"):
        for sample in json.loads((BENCH / "meta" / "tasks" / f"{task}.json").read_text())["samples"]:
            rooms[sample["input"]["scene_name"]] = sample["input"]["dataset"]
    return rooms


def fetch_scenes(scenes, rooms):
    by_dataset = {}
    for scene in scenes:
        if scene not in rooms:
            sys.exit(f"{scene}: not a T1-T4 scene of {BENCH / 'meta' / 'tasks'}")
        by_dataset.setdefault(rooms[scene], []).append(scene)
    for dataset, names in sorted(by_dataset.items()):
        todo = [n for n in names if not (VIDEOS / dataset / f"{n}.mp4").is_file()]
        if not todo:
            continue
        print(f"{dataset}.zip: reading {len(todo)} member(s) over HTTP ranges", flush=True)
        archive = zipfile.ZipFile(io.BufferedReader(RemoteFile(URL.format(dataset)), buffer_size=1 << 22))
        for name in todo:
            out = VIDEOS / dataset / f"{name}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(f"{dataset}/{name}.mp4") as src, open(out.with_suffix(".part"), "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 22)
            out.with_suffix(".part").replace(out)
            print(f"  {out.relative_to(REPO) if out.is_relative_to(REPO) else out} ({out.stat().st_size / 1e6:.1f} MB)")


def fetch_all():
    for dataset in ("arkitscenes", "scannet", "scannetpp"):
        archive = VIDEOS / f"{dataset}.zip"
        if not archive.is_file():
            print(f"downloading {dataset}.zip", flush=True)
            with urllib.request.urlopen(URL.format(dataset), timeout=300) as r, open(archive.with_suffix(".part"), "wb") as f:
                shutil.copyfileobj(r, f, 1 << 22)
            archive.with_suffix(".part").replace(archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(VIDEOS)                                # members are <dataset>/<scene>.mp4
        print(f"{dataset}: {len(list((VIDEOS / dataset).glob('*.mp4')))} videos")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--demo", action="store_true", help="the two demo rooms")
    ap.add_argument("--all", action="store_true", help="all three zips (5.7 GB)")
    a = ap.parse_args()
    if a.all:
        fetch_all()
        return
    scenes = (DEMO if a.demo else []) + a.scenes
    if not scenes:
        ap.error("give scene names, --demo or --all")
    fetch_scenes(scenes, room_datasets())


if __name__ == "__main__":
    main()
