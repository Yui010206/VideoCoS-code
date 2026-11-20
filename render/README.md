# Render: Blender plan → photoreal video (T6, T7)

The reasoning stage answers T6 and T7 with a Blender video, the plan. This stage renders the plan photoreal with the
open [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) Ref2VA model (the full model, 50 steps), served by
SGLang. One request per case:

- the real frame is the keyframe at frame 0 (`<Picture 1>`);
- the plan is the reference video (`<Video 1>`);
- the prompt is one fixed template in MiniMax's structured Ref2VA format, with the goal (T6) or the scene description
  (T7) filled in.

H3 keeps the look of the real frame, follows the plan's motion and timing, and adds what the simulation leaves out
(on T6, the hand).

| | T6: hand–object task | T7: future prediction |
|---|---|---|
| Keyframe | the start photo, upright | the last observed frame |
| Reference video | our plan, 15 s | our prediction of the next 5 s |
| Prompt template | `t6_h3/prompt_template_v4.txt` | `t6_h3/prompt_template_t7_ref.txt` |
| Output | 15 s at 24 fps with audio, short edge 480 | 5 s (120 frames at 24 fps), 864×480 |
| Generator | `t6_h3/generate.py` → `runs/ours_h3_v4` | `t6_h3/generate_t7_ref2va.py` → `runs/ours_h3full_t7` |
| H3-only baseline | `t6_h3/generate_h3only_t6.py` → `runs/h3only_t6` | `generate_t7_ref2va.py --no-reference` → `runs/h3only_t7` |

The H3-only baseline sends the same request without the plan. Ref2VA refuses a keyframe without a reference, so the
real frame is given again as the only (image) reference.

## Setup

- **GPUs.** One server runs on 1, 2 or 4 GPUs (3 cannot split H3's attention heads). We used two RTX PRO 6000 (96 GB)
  per server; a 1-GPU server also works with layer offload, more host RAM and about 1.7× the time. A server pins
  50–72 GB of host RAM.
- **Environment.** `bash setup_env.sh` creates the conda env `h3render` (`sglang[diffusion]==0.5.23`); every command
  below runs in it. Set `CONDA_SH` if `conda info --base` does not find your conda.
- **Weights** (Ref2VA is about 135 GB):

  ```bash
  hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" --include "Ref2VA/*" --local-dir /path/to/MiniMax-H3
  export H3_MODEL_PATH=/path/to/MiniMax-H3
  ```
- **Data read in place.** The bench (`VIDEOCOS_BENCH`, default `../bench`) and the reasoning run whose plans are
  rendered (`VIDEOCOS_REASONING_RUN`, default `../reasoning/runs/full_v1`).

## Prepare the inputs (CPU, minutes)

```bash
cd render && conda activate h3render
export VIDEOCOS_REASONING_RUN=../reasoning/runs/<your run>
python t6_h3/prepare.py          # inputs/T6/<id>/: start.png (at the plan's size), plan.mp4, case.json
python t6_h3/make_prompts.py     # prompts_v4/T6/<id>.txt
python t6_h3/make_t7_inputs.py   # inputs/T7/<id>/: last_observed.png, case.json
```

Cases without a plan are skipped. `--ids a,b` limits `prepare.py` and `make_prompts.py` to some cases.

## Render

All cases, on one or more servers:

```bash
bash scripts/run_t6_t7.sh 0,1:30021 2,3:30031              # GPUS:PORT per server: T7, then T6
bash scripts/run_t6_t7.sh --baseline 0,1:30021 2,3:30031   # then also the H3-only baselines
```

The script starts the servers one at a time, runs one generator chain per server (the chains share the case lists
through claim files), and stops the servers at the end. Running it again resumes. Logs are in `logs/`.

By hand, on one server:

```bash
bash serve_h3.sh 0,1 30021                  # wait until logs/serve_30021.log says "fired up"
S=http://127.0.0.1:30021
python t6_h3/generate_t7_ref2va.py --server $S --ids all                    # T7
python t6_h3/generate.py --server $S --ids all --name ours_h3_v4           # T6
python t6_h3/generate_t7_ref2va.py --server $S --ids all --no-reference    # T7 H3-only baseline
python t6_h3/generate_h3only_t6.py --server $S --ids all                    # T6 H3-only baseline
bash stop_h3.sh 30021
```

`--ids` takes `all`, `pilot` or a comma-separated list; `--dry-run` prints the first request without sending it.

## Outputs

```
runs/<name>/<T6|T7>/<id>/
  video.mp4       the render (T7: frames 1..120 of raw.mp4, so frame k shows k/24 s after the last observed frame)
  raw.mp4         T7: what the server returned (its frame 0 is the keyframe)
  request.json    the request as sent
  job.json        the server's job record and the timing
```

## Time

Median minutes per case on our GPUs:

| | 2-GPU server | 1-GPU server |
|---|---|---|
| T7 pair (5 s) | 10 | 16 |
| T6 case, 4:3 / 16:9 / 21:9 | 37 / 60 / 64–72 | 66 / 109 / – |
| H3-only T7 pair | 4.5 | – |
| H3-only T6 case, 4:3 / 16:9 / 21:9 | 7.3 / 11.3 / 16.3 | – |

A server took 1–7 min to start with the weights in the page cache and up to 20 min reading them from a network drive.

## Notes

- **Ports.** Besides `PORT`, a server binds `PORT+1` and a distributed port (30005, or the next free of 30042, 30079,
  30116, …). If `PORT+1` collides, the server still logs "fired up" but also "request broker exited" and answers every
  request with 503. Use ports off that sequence (30021, 30031, 30061, 30091 and 30013 work) and start servers one at
  a time.
- **Shared GPUs.** Between requests a server's GPUs look idle. If another job takes that memory, the next request runs
  out of memory and the server dies; keep the GPUs reserved for the whole run.
- **Interrupted runs.** A generator killed mid-case leaves that case without a video. Run the same command again: it
  takes over claims of dead generators. A single case: `--ids <id>`.
- **Aspect ratio.** The generators send the H3 aspect ratio nearest to the input (21:9, 16:9, 4:3, 1:1, 3:4 or 9:16).
- **Disk.** SGLang also keeps every result in `outputs/` and the uploads in `inputs/uploads/`; both can be deleted after
  a run.
