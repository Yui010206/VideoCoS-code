"""Tasks from an examples directory: one per bench record, split into what the model sees and what only
scoring reads.

    <examples>/<family>/<name>/scene.blend, scene.py, bench_records.json, ...

Each bench record has "input (give to the model)" and "eval_only (scoring only, never to the model)".
``Task.public`` is the first minus media paths (the model has no images); ``Task.eval_only`` is the
second and is read only by harness.score. T1-T4 records are completed from the frozen bench manifest
with their VSI id and question_type, which the reported VSI runs gave the model.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

INPUT_KEY = "input (give to the model)"
EVAL_KEY = "eval_only (scoring only, never to the model)"
MEDIA_KEYS = ("video", "start_image")
QUERY_FAMILIES = ("T1", "T2", "T3", "T4")
UPDATE_FAMILIES = ("T5", "T6", "T7")
# the frozen task manifests: <repo>/bench/meta/tasks, or $VIDEOCOS_BENCH/meta/tasks
BENCH_TASKS = Path(os.environ.get("VIDEOCOS_BENCH", Path(__file__).resolve().parents[2] / "bench")) / "meta" / "tasks"
FUTURE_SECONDS = 5.0                     # Physics-IQ scores the 5 s after the 3 s conditioning clip


@dataclass
class Task:
    id: str
    family: str
    example: str
    folder: Path
    public: dict
    eval_only: dict = field(repr=False)
    spec: dict = field(default_factory=dict)

    @property
    def kind(self):
        return "query" if self.family in QUERY_FAMILIES else "update"

    @property
    def scene_blend(self):
        return self.folder / "scene.blend"

    @property
    def scene_code(self):
        return (self.folder / "scene.py").read_text()


def _vsi_metadata():
    rows = {}
    for family in QUERY_FAMILIES:
        path = BENCH_TASKS / f"{family}.json"
        if path.is_file():
            for sample in json.loads(path.read_text())["samples"]:
                rows[sample["id"]] = sample
    return rows


def _spec(family, public, scene_frames):
    """What a submission must produce, from the task inputs and the scene (never from eval_only)."""
    if family == "T5":
        return {"target_camera": public["target_camera"], "frames": public["frames"]}
    if family == "T7":
        # The last observed frame is the scene's own frame_end; run.py fills it in once the scene is
        # open (scene_frames lets a caller who already knows it do it here).
        fps = float(public["fps"])
        last = scene_frames[1] if scene_frames else None
        return {"observed_end": last, "observed_seconds": public.get("seconds"), "fps": fps,
                "future_seconds": FUTURE_SECONDS,
                "target_end": None if last is None else last + round(FUTURE_SECONDS * fps)}
    return {}


def load_bench_tasks(scenes, families=None, ids=None):
    """Update tasks for the full bench: every sample of the frozen manifests (BENCH_TASKS/<family>.json),
    on its codified scene <scenes>/<family>/<scene>/codify/final/{scene.blend, scene.py} (read only; each
    task copies it). T5 has three target views per scene, so three tasks share one scene."""
    scenes, tasks, missing = Path(scenes), [], []
    for family in families or UPDATE_FAMILIES:
        if family not in UPDATE_FAMILIES:
            raise ValueError(f"{family}: the bench loader covers the update families {UPDATE_FAMILIES}")
        for sample in json.loads((BENCH_TASKS / f"{family}.json").read_text())["samples"]:
            if ids and sample["id"] not in ids:
                continue
            name = sample["scene"] if family == "T5" else sample["id"]
            folder = scenes / family / name / "codify" / "final"
            if not (folder / "scene.blend").is_file() or not (folder / "scene.py").is_file():
                missing.append(sample["id"])
                continue
            public = {k: v for k, v in sample["input"].items() if k not in MEDIA_KEYS}
            tasks.append(Task(id=sample["id"], family=family, example=name, folder=folder, public=public,
                              eval_only=sample["eval_only"], spec=_spec(family, public, None)))
    if missing:
        raise ValueError(f"{len(missing)} bench samples have no codified scene under {scenes}, e.g. {missing[:3]}")
    if not tasks:
        raise ValueError(f"no tasks under {scenes} for families={families} ids={ids}")
    return tasks


def load_tasks(examples, families=None, ids=None, scene_frames=None):
    """All tasks under ``examples``. ``scene_frames(folder)`` returns a scene's [first, last] frame,
    needed for T7's spec; it is supplied by the caller because reading it needs Blender."""
    examples = Path(examples)
    vsi = _vsi_metadata()
    tasks = []
    for records_path in sorted(examples.glob("*/*/bench_records.json")):
        folder = records_path.parent
        for record in json.loads(records_path.read_text()):
            family = record["task"]
            if families and family not in families:
                continue
            if ids and record["id"] not in ids:
                continue
            public = {k: v for k, v in record[INPUT_KEY].items() if k not in MEDIA_KEYS}
            if family in QUERY_FAMILIES:
                meta = vsi.get(record["id"])
                if meta is None:
                    raise ValueError(f"{record['id']} is not in the bench manifest; cannot recover its VSI id/type")
                public = {"id": meta["vsi_id"], "dataset": public.get("dataset"), "scene_name": public.get("scene_name"),
                          "question_type": meta["question_type"], "question": public["question"],
                          "options": public.get("options")}
            frames = scene_frames(folder) if (family == "T7" and scene_frames) else None
            tasks.append(Task(id=record["id"], family=family, example=folder.name, folder=folder, public=public,
                              eval_only=record[EVAL_KEY], spec=_spec(family, public, frames)))
    if not tasks:
        raise ValueError(f"no tasks under {examples} for families={families} ids={ids}")
    return tasks
