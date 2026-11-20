# VideoCoS bench: the seven tasks

`meta/tasks/T1.json … T7.json` are the frozen task manifests (2,838 samples). Every stage of this repository reads
them from here; set `VIDEOCOS_BENCH` to use a bench folder elsewhere. The media they name live under `data/`, which
is not part of the repository: download the demo subset (`scripts/download_demo.sh`) or the full sources (below).

| Task | Name | Samples | Source | Input to the system | Output | Subgroups |
|---|---|---|---|---|---|---|
| T1 | Entity Querying | 251 | VSI-Bench-Debiased | room walkthrough video + question | answer (number or option letter) | object_count 251 |
| T2 | Geometric Querying | 1,297 | VSI-Bench-Debiased | same | same | object_size 353, room_size 200, abs_distance 434, rel_distance 310 |
| T3 | Viewpoint-conditioned Querying | 496 | VSI-Bench-Debiased | same | same | rel_direction 382, route_plan 114 |
| T4 | Temporal Querying | 318 | VSI-Bench-Debiased | same | same | appearance_order 318 |
| T5 | Novel-view Synthesis | 300 | Kubric-4D (GCD test split) | 60-frame source video + source and target camera | the same 60 frames from the target camera, 576×384 | small, medium, large view change: 100 each |
| T6 | Goal-conditioned Planning | 110 | Ego2Act benchmark | one egocentric start photo + goal + three constraints | a ~15 s video that carries out the manipulation | long 42, medium 54, short 14 |
| T7 | Future-state Prediction | 66 | Physics-IQ Verified | 3 s conditioning clip at 30 fps (one camera per scenario) + scene description | the next 5.000 s at 30 fps from the same camera | in_scope 38, out_of_scope 28 |

Each manifest is `{"task", "name", "source", "output", "scoring", "n", "subgroups", "samples": [...]}` (T5–T7 add a
`protocol`). Each sample is `{"id", "subgroup", ..., "input": {...}, "eval_only": {...}}`:

- `input` is everything a system may see. Media paths in it (`video`, `start_image`) are relative to this folder.
- `eval_only` is for scoring only and is never given to a system: the answer (T1–T4), the target view's RGB,
  segmentation, depth and camera (T5), the human recordings and step counts (T6), the take-1 and take-2 futures and
  their motion masks (T7).
- T1–T4 samples also carry `vsi_id` and `question_type`; their ids are VSI-Bench ids (`T3_vsi_3675`). T5 samples
  carry `scene`, `source_view`, `target_view` and the angles; T7 samples carry the Physics-IQ scenario, camera and
  category.

## Download

All media go under `data/` next to the manifests, at the paths the manifests name. The repository does not include
them; there are two ways to get them.

### Demo subset (80 MB)

From the repository root:

```bash
bash scripts/download_demo.sh
```

It downloads [Shoubin/VideoCoS-demo](https://huggingface.co/datasets/Shoubin/VideoCoS-demo) from Hugging Face: the
inputs of two samples per task family (T5, T6, T7), the codified scenes of eight inputs and the paper's outputs for
them. It then fetches the two demo room videos (T1–T4) from VSI-Bench.

### Full benchmark (about 20 GB on disk)

Run these from this folder (`bench/`). `hf` comes with `pip install -U huggingface_hub`.

**T1–T4: VSI-Bench room videos (274 videos, 5.7 GB)**

```bash
python3 ../scripts/fetch_vsi_videos.py --all          # or single rooms: python3 ../scripts/fetch_vsi_videos.py <scene> ...
```

The script downloads the three video zips of [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench)
and unpacks them into `data/vsi_bench/videos/`.

**T5: Kubric-4D test split (100 scenes, 6.6 GB kept)**

```bash
mkdir -p data/t5_kubric4d
curl -L https://s3.us-east-1.amazonaws.com/tri-ml-public.s3.amazonaws.com/urp/datasets/tcow/gcd_kubric4d_test.tar.gz \
  | tar xzf - -C data/t5_kubric4d --exclude='*/frames_p0_v*'
```

This is the official test split of [Kubric-4D](https://gcd.cs.columbia.edu/#datasets) (Generative Camera Dolly),
scenes `scn02900` to `scn02999`. The archive is 237 GB. The command streams it and keeps only what T5 reads: every
view's RGB, segmentation and depth video and its camera (`<scene>_p0_v<view>_{rgb,segm,depth}.mp4`, `.json`). It skips
the per-frame folders. To keep the archive, download it first and run the same `tar` command on it.

**T6: Ego2Act (110 cases, 2.1 GB)**

```bash
hf download ego2act/ego2act-bench --repo-type dataset --local-dir data/t6_ego2act/bench
```

**T7: Physics-IQ Verified (66 scenarios, 5.7 GB of 25 GB)**

```bash
hf download Anates-Labs-Research/Physics-IQ-Verified --repo-type dataset --local-dir data/t7_physics_iq_verified \
  --include "split-videos/conditioning/30FPS/*" --include "split-videos/testing/30FPS/*" --include "video-masks/real/30FPS/*"
```

Give `--include` once per pattern.

### Layout

```
data/vsi_bench/videos/{arkitscenes,scannet,scannetpp}/<scene>.mp4                          T1–T4
data/t5_kubric4d/<scene>/<scene>_p0_v<view>{_rgb,_segm,_depth}.mp4, <scene>_p0_v<view>.json    T5
data/t6_ego2act/bench/cases/<case>/start.jpg, human/                                         T6
data/t7_physics_iq_verified/split-videos/{conditioning,testing}/30FPS/, video-masks/real/30FPS/   T7
```

## Sources and licenses

| Tasks | Source | License |
|---|---|---|
| T1–T4 | [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) (VSI-Bench-Debiased questions) | Apache-2.0; the videos come from ScanNet, ScanNet++ and ARKitScenes, whose own terms apply |
| T5 | [Kubric-4D](https://gcd.cs.columbia.edu/#datasets) (Van Hoorick et al., Generative Camera Dolly) | CC BY 4.0 |
| T6 | [Ego2Act](https://huggingface.co/datasets/ego2act/ego2act-bench) | CC BY 4.0 |
| T7 | [Physics-IQ Verified](https://huggingface.co/datasets/Anates-Labs-Research/Physics-IQ-Verified), derived from [Physics-IQ](https://github.com/google-deepmind/physics-IQ-benchmark) | CC BY 4.0 |
