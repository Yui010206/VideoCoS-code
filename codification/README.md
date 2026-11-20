# Codification: video → executable Blender scene

Turns a video, or a single image, into an editable Blender scene: `scene.blend` plus the `scene.py` that builds it.

| Step | Entry point | Needs | What it does |
|---|---|---|---|
| 1. Perception | `perception.py` | GPUs with ≥ 80 GB | Local models build the evidence: cameras and depth (Depth Anything 3), object names (Qwen3.8-27B) and object tracks (SAM 3). For rooms, also a checked object list and a SAM 3D mesh per object. |
| 2. Codification | `codification.py` | Docker, Blender 4.5, a logged-in Codex CLI | A Codex agent reads the evidence, writes `scene.py`, builds it in a sandboxed Blender, compares renders with the frames, revises and submits. |

## How it works

**Perception** runs cached stages, so a stopped run resumes: `decode` → `geometry` (DA3 cameras and metric depth;
DA3-Streaming for long room videos) → `camera_depth` (a gravity-aligned world and the floor) → `vocabulary` (Qwen names
the objects) → `tracks` (SAM 3 tracks every name) → `proposals`, `proposal_names`, `proposal_tracks` (class-agnostic SAM 3
masks that no track explains, named and tracked) → `fuse` (3D objects, published as an immutable evidence version).
Rooms add `objects` (Qwen checks every track; revisits are merged) and `meshes` (SAM 3D Objects completes every object
at least 0.25 m across).

**Codification** runs `codex exec` (default model `gpt-6.1-terra`, effort `xhigh`) with tools from an MCP server:
`inspect_video`, `execute_blender`, `execute_blender_file`, `render_scene`, `run_python` and `submit`; T5–T7 also get
`make_objects` and `track_objects` (SAM 3D on a GPU). Every build is a revision:

1. Blender runs `scene.py` in a Docker container (no network, evidence read-only, 8 CPUs, 8 GB, 15 min).
2. `check_scene` checks the structure, contacts and penetrations, and the silhouettes against the masks.
3. Cycles renders come back to the agent: source | render pairs and a top view.

`submit` accepts a revision that built, passed its check, whose renders the agent has seen, and that builds every
certain evidence object or says why not. The run ends as `success` (accepted), `partial` (no accepted submit: the most
complete eligible revision is kept) or `failed`; the result is copied to `codify/final/` with a `preview.mp4`
(input | render).

**Rooms (T1–T4)** differ only through their config (`configs/tasks/T1-T4.json`): the agent starts from `base_scene.py`
(every evidence object at its estimated pose, as its SAM 3D mesh where that agrees with its box, plus floor, walls,
ceiling and lights), sees two side views after every build, faces a stricter layout check (furniture must stand on the
floor), may build up to 16 revisions, and keeps the best failed build if none passes. T5–T7 get the raw perception
layers (frames, depth, masks, cameras) and make object meshes themselves with SAM 3D.

## Requirements

- Linux with NVIDIA GPUs of at least 80 GB for perception, one lane per GPU. Qwen3.8-27B needs about 72 GB free to
  start a server and about 55 GB when a step loads it in-process; a lane's own server sleeps in about 86 GB of host RAM.
- For codification: Docker usable without sudo, Blender 4.5 for Linux x64, and the Codex CLI on `PATH` after
  `codex login` (your own login; nothing is stored here). Every build is a container with 8 CPUs and 8 GB. T5–T7 runs
  also use a GPU for SAM 3D, about 20 GB each.
- `ffmpeg`, `ffprobe` and `nvidia-smi`.

Each model runs in its own Python environment; `configs/machine.json` names them. The versions we used:

| Environment | Runs | Packages |
|---|---|---|
| harness | the entry points, the MCP server, fusion | Python 3.12; `pip install numpy opencv-python-headless pillow scipy av tqdm` |
| vLLM | Qwen3.8-27B | vllm 0.28.0, torch 2.13.0, transformers 5.16.1 |
| DA3 | Depth Anything 3, DA3-Streaming | torch 2.7.0+cu128, xformers 0.0.30, [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) at `3d835ec1` |
| SAM 3 | tracks and proposals | torch 2.7.0+cu128, [sam3](https://github.com/facebookresearch/sam3) at `11dec293` |
| SAM 3D | room meshes, `make_objects` | torch 2.7.0+cu128, [sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects) at `f91db411`, pytorch3d 0.7.8, kaolin 0.18.0, spconv-cu121 2.3.8, gsplat 1.5.3, flash_attn 2.8.3, [MoGe](https://github.com/microsoft/MoGe) |

Weights (every model process runs offline, so download them first):

| Model | Key in `machine.json` | Source |
|---|---|---|
| Qwen3.8-27B | `vocabulary.weights` | `Qwen/Qwen3.8-27B` |
| Depth Anything 3 | `geometry.weights` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` |
| DA3-Streaming, SALAD, DINOv2 hub code | `geometry.streaming_paths` | `da3_streaming/` of the DA3 checkout; `dino_salad.ckpt` from [salad](https://github.com/serizba/salad) v1.0.0; a torch hub copy of `facebookresearch/dinov2` |
| SAM 3 | `tracks.weights` | `facebook/sam3` (`sam3.pt`) |
| SAM 3D Objects | `objects3d.config` | `facebook/sam-3d-objects` (`pipeline.yaml`) |
| MoGe ViT-L, DINOv2 ViT-L/14 | `env.HF_HOME`, `env.TORCH_HOME` | `Ruicheng/moge-vitl`; `facebookresearch/dinov2` |

## Configure

1. Replace every `/path/to/...` in `configs/machine.json`: the environments and weights above, `agent.blender_dir` (the
   folder with the `blender` binary), `env.HF_HOME` and `env.TORCH_HOME`. `bench_root` already points to this
   repository's `bench/`.
2. Build the sandbox image once (it needs network), from this folder: `$PY -c "from codify.sandbox import Sandbox;
   Sandbox('.').build_image()"`, where `$PY` is the harness Python.
3. Run `$PY perception.py check --gpus 0` until it prints `ready`.

## Quick start

```bash
cd codification
$PY perception.py run --gpus 0 --queue queues/demo.json          # the 8 demo inputs (bash ../scripts/download_demo.sh)
$PY codification.py run --queue queues/demo.json --gpus 0 --watch

# one input of your own: --task T1-T4 (a room walkthrough), T5, T6 (an image) or T7 (a short clip)
$PY perception.py run --gpus 0 --video clip.mp4 --task T7 --id my_clip
$PY codification.py run --video clip.mp4 --task T7 --id my_clip --gpus 0
```

Scenes land in `outputs/<id>/codify/final/` (`outputs/T5/<scene>/`, `outputs/T6/<id>/`, `outputs/T7/<id>/` for bench
samples). On one RTX PRO 6000 a 3 s T7 clip took 8 min of perception and 13 min of codification; a 33 s room took
98 min of perception and 15 min of codification. Starting a lane's Qwen server takes about 12 min.

## Inputs and queues

A queue is a JSON list of `{"id", "task", "video", "seconds"}` (a video file, for rooms) or `{"id", "task", "sample"}`
(a bench sample read from `bench/meta/tasks/`). `id` names the output folder.

| Queue | Inputs |
|---|---|
| `queues/demo.json` | the 8 demo inputs |
| `queues/examples.json` | one input per task family |
| `queues/vsi.json` | the 274 VSI-Bench rooms of T1–T4 |
| `queues/t5_t7.json` | 276 inputs: T5 100 scenes, T6 110, T7 66 |
| `queues/all.json` | both of the above |

## Running at scale

- **Perception lanes.** `$PY perception.py run --gpus 0,1,2,3 --queue queues/vsi.json` starts one lane per GPU. Lanes
  take inputs through lock files, so lanes on several machines can share one `--out`. A failed step is retried; a step
  that ran out of GPU memory waits for 60 GB and does not count as a try.
- **The 27B model.** By default each lane starts its own vLLM server and keeps it asleep between steps. To share one
  server per machine, start `$PY perception.py qwen-server --gpus 7 --port 8701` and give the lanes
  `--qwen-url http://127.0.0.1:8701`. `--no-qwen-server` loads the model in every step instead.
- **Groups.** `perception.py groups --num-groups N` splits a queue by estimated GPU time; run one group with
  `--num-groups N --group k` on both entry points.
- **Codification.** `$PY codification.py run --queue queues/t5_t7.json --gpus 6,7 --per-gpu 2 --parallel 4 --watch`.
  `--watch` picks inputs up as they are perceived; rooms need no GPU. `--minutes` sets the agent's time (default 20 for
  rooms, 35 for T5–T7), `--tries` the runs per input (default 2) and `--stagger S` the seconds between starts (Codex
  sessions started at the same moment can come up without their tools). The paper's rooms used
  `--minutes 30 --parallel 30 --order longest --stagger 30`; T5–T7 used the defaults.
- **Monitor.** `$PY progress.py --queue queues/vsi.json --watch 60`, `perception.py status`, `codification.py status`,
  and `perception.py unlock <id>` after a lane died. Logs: `outputs/perception.log`, `outputs/codification.log`.

## Outputs

```
outputs/<id>/
  perception/                  cached stages (work/) and the published evidence (evidence/vNNN/)
  objects/evidence_meshes/     rooms: the checked object list with SAM 3D meshes (the codification input)
  codify/final/                scene.blend, scene.py, inventory.json, final_check.json, preview.mp4
  codify/final_report.json     success | partial | failed, the selected revision and why
  codify/revisions/rNNN/       every build: scene.py, scene.blend, checks, renders
  codify/codex/                AGENTS.md (the agent's instructions), the session's events
```

## Time and GPU memory

Median wall minutes per step in the paper's run (two lanes per RTX PRO 6000 and a shared Qwen server):

| Step | Peak GPU memory | T1–T4 | T5 | T6 | T7 |
|---|---|---|---|---|---|
| geometry (DA3) | rooms 23 GB; others ≤ 17 GB | 11 | 0.6 | 0.4 | 0.6 |
| tracks (SAM 3) | rooms up to 80 GB; others ≤ 24 GB | 95 | 5.6 | 0.4 | 1.7 |
| proposal tracks (SAM 3) | ≤ 12 GB | 72 | 1.2 | 0.3 | 2.1 |
| objects + meshes (rooms) | about 55 + 40 GB | 26 | | | |
| codification (Codex) | T5–T7: SAM 3D, about 20 GB | 24 of 30 | 19 of 35 | 8 of 35 | 8 of 35 |

The other stages take minutes or less. A room took a lane a median of 4.6 h of perception (roughly 2 h plus 2.2 min
per second of video).

## Notes

- Run one `codification.py` per output folder: it takes no locks.
- After stopping `codification.py` by hand, also stop leftover `codex exec`, `codify.mcp_server` and `sam3d_worker.py`
  processes; they run in their own process groups.
- Rooms record absolute paths in `outputs/<id>/config_codify.json`: do not move `outputs/` between the two steps.
- Changing `machine.json` or the perception code recomputes the affected stages (their cache keys hash both).
- The sandbox mounts `codify/` read-only, so code the agent runs can read the harness source.
