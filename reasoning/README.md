# Reasoning: read and write the executable scene

A language model solves the tasks by calling tools on a codified scene (`scene.blend` plus the `scene.py` that built
it). It never sees the video or any image.

- **Query tasks (T1–T4).** Five read-only tools (`scene_inventory`, `inspect_entities`, `measure_pair`,
  `viewpoint_relation`, `camera_history`), then `submit_answer`. Answers are scored with the VSI-Bench rule.
- **Update tasks (T5–T7).** The model edits the live scene with the queries plus `track_motion`, `run_python` (Python
  in the running Blender, with `bpy`, `mathutils` and the helper module `S` from `harness/blender/helpers.py`),
  `reset_scene` and `submit_update`. T5: make the target camera the scene camera. T6: animate about 15 s that reach the
  goal from the start frame. T7: set up and simulate the 5 s after the observed clip.

Each task gets its own Blender inside Docker: no network, a read-only root, and only the task's `work/` folder
writable. `submit_update` checks only the task contract below (objects count as unchanged within 1 mm and 0.1°) and
saves `work/final.blend`. A render pass then turns each accepted scene into `video.mp4` (Cycles on the CPU, 32 samples).

| Family | Contract |
|---|---|
| T5 | The scene camera equals the target camera (pose, focal length, sensor, resolution); the frame range is unchanged. |
| T6 | The start frame and the poses on it are unchanged, the timeline is at least 3 s long, the camera is unchanged. |
| T7 | The timeline ends 5 s after the last observed frame; the observed frames and the camera are unchanged; the rigid-body cache is baked. |

## Requirements

- Linux, Python 3.9 or newer (the standard library only; `pytest` for the tests) and `ffmpeg`.
- Blender 4.5 for Linux x64, unpacked. Set `VIDEOCOS_BLENDER_DIR` to its folder (default
  `~/apps/blender-4.5.14-linux-x64`).
- Docker usable without sudo, and the sandbox image, built once:

  ```bash
  docker build -t videocos-blender-runtime:1 - <<'DOCKERFILE'
  FROM ubuntu:24.04
  RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libice6 libgl1 libegl1 libglx-mesa0 \
      && rm -rf /var/lib/apt/lists/*
  DOCKERFILE
  ```
- A model, chosen with `--model`: `gemini-3.8-flash` (default, `GEMINI_API_KEY`), `gpt-5.6-luna` (`OPENAI_API_KEY`) or
  a local Qwen behind vLLM (`QWEN_LOCAL_KEY`, any value; endpoints in `run.py`). Keys are read from the environment
  only and redacted from the saved traces; prices are in `harness/llm.py`.

## Inputs

- **Task manifests:** `../bench/meta/tasks/T*.json` (set `VIDEOCOS_BENCH` for a bench elsewhere). The model sees a
  sample's `input` without media paths and never its `eval_only` part.
- **`--scenes DIR`** (T5–T7): codification outputs, `DIR/<family>/<id>/codify/final/{scene.blend,scene.py}`, where
  `<id>` is the sample id (for T5 the scene, shared by its three target views).
- **`--examples DIR`** (any family): `DIR/<group>/<scene>/{bench_records.json, scene.blend, scene.py}`. The demo's
  `../demo/examples` is such a folder (8 scenes, 45 tasks). For T1–T4 on your own codified rooms,
  `python3 make_room_examples.py --scenes ../codification/outputs --out runs/rooms` writes one folder per room with
  that room's questions.

## Run

From this folder:

```bash
python3 -m pytest -q tests/                                                   # 14 tests on ../demo/examples (6 use Docker)
python3 run.py --examples ../demo/examples --out runs/fake --fake            # a scripted model calls every tool; no API

# one task, rendered
GEMINI_API_KEY=... python3 run.py --examples ../demo/examples --out runs/one \
    --ids T7_0050_center_dominos-with-space --max-usd 1 --render

# T5–T7 on your codified scenes: the model phase, then the renders
GEMINI_API_KEY=... python3 run.py --scenes ../codification/outputs --out runs/full --families T7,T6,T5 --max-usd 110 --jobs 12
python3 run.py --scenes ../codification/outputs --out runs/full --families T7,T6,T5 --render-only --jobs 6 --render-cpus 16

# T1–T4 on your codified rooms
GEMINI_API_KEY=... python3 run.py --examples runs/rooms --out runs/t1t4 --max-usd 100 --jobs 16
```

Other options: `--families`, `--ids`, `--jobs`, `--render-cpus`, `--no-docker` (query tasks only); `python3 run.py -h`
lists them. With Gemini 3.8 Flash a question costs about $0.02–0.07, a T5 task $0.04 and a T6 or T7 task $0.3–0.6; the
paper's 476 T5–T7 tasks cost $97. At 16 CPUs a render takes about 1.5 min (T5), 20 min (T7) and 45 min (T6).

`--max-usd` is a hard cap for one invocation: before each request the meter reserves its worst-case cost and refuses
it if the total could pass the cap. Every API attempt is logged in `costs.jsonl`.

## Outputs

```
<out>/summary.json           statuses, spend, and the T1–T4 scores of this invocation
<out>/costs.jsonl            one line per API attempt
<out>/<family>/<id>/
    result.json              status, turns, tool calls, the answer (T1–T4) or the summary and final_blend (T5–T7)
    context.json             exactly what the model was given
    agent/api_trace.jsonl    every request, response and tool result
    work/edits.py            T5–T7: every snippet that ran, in order (replayable on the original scene)
    work/final.blend         T5–T7: the accepted scene
    video.mp4                T5–T7, after the render pass (render_session/frames/ keeps the PNG frames)
```

`status` is `completed` (accepted), `budget_exhausted` (nothing accepted within the budget; no video),
`missing_submission` or `incomplete` (the model never submitted), or `failed` (an exception, in `result.json`).

## Budgets

| | T1–T4 | T5 | T6, T7 |
|---|---|---|---|
| Model turns / tool calls | 6 / 18 | 4 / 6 | 24 / 40 |
| Output tokens per turn | 4,096 | 16,384 | 16,384 |
| Reasoning effort | provider default | medium | medium |

The last turn offers only the submission tool. `run_python` returns at most 8,000 printed characters per call and has
900 s; a crashed or hung Blender is restarted and the applied snippets replayed.

## Notes

- A task with a `result.json` counts as done. A task killed mid-run leaves a folder without one: delete it before
  running again.
- `result.json` stores absolute paths, so do not move a run folder before its render pass.
- The PNG frames of every render are kept and take most of a run's disk.
- After killing `run.py`, check `docker ps --filter name=vreason-` for a render still running.
- Known limitations: in T7 a body simulated during the observed clip can be read at a stale pose; `S.key` without a
  pose keys nothing; with its 4-turn limit a T5 task can run out of turns before setting the camera.
