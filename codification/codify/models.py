"""Local model adapters: estimate_geometry (Depth Anything 3) and segment_track (SAM 3).

Long videos use DA3-Streaming (overlapping chunks, Sim(3) alignment, loop closure; geometry_stage picks it from
the config) and SAM 3 in overlapping windows joined by mask IoU (a short video is one window).

The host-side functions start this file as a worker inside each model's own Python
environment (config: python, gpu, local weights). Workers run with HF_HUB_OFFLINE=1 and
local checkpoint paths, so execution never fetches weights or calls hosted inference.
Large arrays stay in the stage directory; the returned record lists their paths.
"""
from __future__ import annotations

import argparse
import json
import re
import os
import shutil
import subprocess
import tempfile
import sys
import time
from pathlib import Path

import numpy as np

# Scene-independent indoor vocabulary, fixed before looking at any evaluation labels or
# questions. Value = object_library recipe category used for the procedural proxy.
VOCABULARY = {
    "chair": "chair", "armchair": "armchair", "sofa": "couch", "stool": "bar stool",
    "table": "table", "coffee table": "coffee table", "desk": "table", "bed": "bed frame",
    "cabinet": "cabinet", "bookshelf": "bookcase", "wall shelf": "shelf", "tv stand": "tv stand",
    "television": "television", "computer monitor": "television", "laptop": "laptop computer",
    "lamp": "desk lamp", "potted plant": "house plant", "vase": "vase", "picture frame": "picture",
    "wall clock": "wall clock", "book": "book", "cup": "cup", "bowl": "bowl", "plate": "plate",
    "bottle": "bottle", "jar": "jar", "box": "box", "basket": "clothes basket",
    "trash can": "trash can", "pillow": "pillow", "curtain": "curtain", "rug": "rug",
    "refrigerator": "refrigerator", "microwave": "consumer electronic device", "kettle": "tea kettle",
    "cutting board": "cutting board", "speaker": "speaker", "remote control": "remote control",
    "candle": "candle", "toy": "toy",
}


# Open vocabulary: one fixed instruction for every input, written before looking at any dataset's
# labels or questions. A local VLM answers it on evenly spaced frames; the names become SAM 3 prompts.
VOCABULARY_PROMPT = (
    "List the distinct physical objects you can see in this image. Write one object per line as a short "
    "lowercase English noun or noun phrase naming its kind (for example: mug, laptop, office chair, wooden "
    "block, rubber duck). Include furniture, appliances, devices, tools, toys, food, small items, and people "
    "or hands if visible. Do not list the floor, ground, walls, ceiling or sky, or what they are made of or "
    "covered with (tiles, bricks, paving stones, planks, grass, carpet), and do not list parts of an object. "
    "No colors, numbers, bullets or explanations.")
NOT_OBJECTS = {"floor", "wall", "walls", "ceiling", "sky", "ground", "background", "shadow", "reflection"}


def names_from_text(text):
    """VLM answer -> object names (one per line, or several on a line separated by commas or semicolons; bullets,
    numbering and duplicates removed)."""
    names = []
    for line in text.splitlines():
        for part in re.split(r"[,;]", line):
            name = part.strip().lstrip("-*•0123456789.) ").strip(" .,;:'\"").lower()
            if name and len(name) <= 40 and name not in NOT_OBJECTS and name not in names:
                names.append(name)
    return names


def _git_revision(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _hardware(torch, started, device="cuda"):
    if device == "cpu":
        import platform
        info = {"device": "cpu", "cpu": platform.processor() or platform.machine(), "threads": torch.get_num_threads(),
                "peak_gpu_memory_gb": None}
    else:
        info = {"device": "cuda", "gpu": torch.cuda.get_device_name(0),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}
    return {**info, "torch": torch.__version__, "wall_seconds": round(time.time() - started, 2)}


# ---------------------------------------------------------------- workers (model envs)

def run_da3(frames_dir, out_dir, cfg):
    """Multi-view depth + cameras on keyframes; depth is camera-Z (verified in DA3's unprojection)."""
    import torch
    from depth_anything_3 import api
    from depth_anything_3.api import DepthAnything3
    started = time.time()
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    stride = int(cfg.get("keyframe_stride", 1))
    keyframes = list(range(0, len(frames), stride))
    if keyframes[-1] != len(frames) - 1:
        keyframes.append(len(frames) - 1)
    device = cfg.get("device", "cuda")
    if device == "cpu":
        # DA3 picks its autocast dtype from this CUDA query; use bf16 on CPU too (native on AVX512-BF16).
        torch.set_num_threads(int(cfg.get("cpu_threads", 32)))
        torch.cuda.is_bf16_supported = lambda *args, **kwargs: True
    model = DepthAnything3.from_pretrained(cfg["weights"]).to(device).eval()
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    prediction = model.inference([str(frames[i]) for i in keyframes], process_res=int(cfg.get("process_res", 504)),
                                 process_res_method=cfg.get("process_res_method", "upper_bound_resize"),
                                 ref_view_strategy=cfg.get("ref_view_strategy", "middle"),
                                 use_ray_pose=bool(cfg.get("use_ray_pose", False)))
    np.savez_compressed(Path(out_dir) / "geometry.npz", keyframes=np.array(keyframes),
                        depth=prediction.depth.astype(np.float32), conf=prediction.conf.astype(np.float32),
                        w2c=prediction.extrinsics.astype(np.float64), K=prediction.intrinsics.astype(np.float64),
                        processed_hw=np.array(prediction.depth.shape[1:]))
    repo = Path(api.__file__).resolve().parents[2]
    return {"model": cfg["model_id"], "weights": cfg["weights"], "weights_revision": cfg["revision"],
            "code_revision": _git_revision(repo), "precision": f"bf16 autocast on {device}", "seed": cfg.get("seed", 0),
            "is_metric": int(getattr(prediction, "is_metric", 0) or 0),
            "metric_scale_factor": float(getattr(prediction, "scale_factor", float("nan"))),
            "depth_type": "camera_z", "extrinsics": "OpenCV world-to-camera",
            "intrinsics_grid": "processed image, integer pixel centers", "keyframe_count": len(keyframes),
            "parameters": {k: cfg.get(k) for k in ("process_res", "process_res_method", "ref_view_strategy",
                                                    "use_ray_pose", "keyframe_stride")},
            **_hardware(torch, started, device)}


def run_da3_global(frames_dir, out_dir, cfg):
    """run_da3 for a walkthrough: keyframes every keyframe_stride frames, sparser when that would exceed max_views,
    in one joint inference, so every camera and depth map shares one frame (chunks aligned afterwards drift apart
    over thousands of frames, and the same object seen again lands elsewhere)."""
    frames = len(list(Path(frames_dir).glob("*.jpg")))
    stride = max(int(cfg.get("keyframe_stride", 1)), -(-frames // int(cfg["max_views"])))
    info = run_da3(frames_dir, out_dir, {**cfg, "keyframe_stride": stride})
    info["parameters"]["max_views"] = int(cfg["max_views"])
    return info


def run_da3_streaming(frames_dir, out_dir, cfg):
    """run_da3 for long videos: DA3-Streaming (DA3 repo, after VGGT-Long), same geometry.npz.

    Overlapping chunks are aligned by Sim(3) on their shared frames and refined with SALAD loop
    closures; DA3-Streaming saves each frame's depth in the aligned scale and its confidence minus 1.
    """
    import torch
    from depth_anything_3 import api
    started = time.time()
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))
    streaming = cfg["streaming"]
    sys.path.insert(0, streaming["code_dir"])
    import da3_streaming
    from loop_utils.config_utils import load_config
    # SALAD's DINOv2 backbone is built through torch.hub; the SALAD checkpoint holds all of its
    # weights, so build it from the cached hub code without downloading anything. Its xformers
    # attention dispatches to Hopper-only kernels on this GPU, so it uses PyTorch attention.
    hub_load = torch.hub.load
    def local_hub(repo, model, *args, **kwargs):
        if repo != "facebookresearch/dinov2":
            return hub_load(repo, model, *args, **kwargs)
        os.environ["XFORMERS_DISABLED"] = "1"
        return hub_load(streaming["dinov2_hub_dir"], model, source="local", pretrained=False)
    torch.hub.load = local_hub
    config = load_config(str(Path(streaming["code_dir"]) / "configs" / "base_config.yaml"))
    config["Weights"].update(DA3=str(Path(cfg["weights"]) / "model.safetensors"),
                             DA3_CONFIG=str(Path(cfg["weights"]) / "config.json"), SALAD=streaming["salad_weights"])
    config["Model"].update(chunk_size=int(streaming["chunk_size"]), overlap=int(streaming["overlap"]),
                           loop_enable=bool(streaming.get("loop_enable", True)), align_lib="torch",
                           ref_view_strategy=cfg.get("ref_view_strategy", "middle"),
                           save_depth_conf_result=True, save_debug_info=False, delete_temp_files=True)
    config["Loop"]["SIM3_Optimizer"]["lang_version"] = "python"
    work = Path(out_dir) / "streaming"
    streamer = da3_streaming.DA3_Streaming(str(frames_dir), str(work), config)
    chunks, inference = [], streamer.model.inference
    def recorded(images, **kwargs):
        prediction = inference(images, process_res=int(cfg.get("process_res", 504)),
                               process_res_method=cfg.get("process_res_method", "upper_bound_resize"), **kwargs)
        chunks.append({"views": len(images), "is_metric": int(getattr(prediction, "is_metric", 0) or 0),
                       "metric_scale_factor": float(getattr(prediction, "scale_factor", float("nan")))})
        return prediction
    streamer.model.inference = recorded
    torch.cuda.reset_peak_memory_stats()
    streamer.run()
    streamer.close()
    count = len(streamer.img_list)
    c2w = np.loadtxt(work / "camera_poses.txt").reshape(count, 4, 4)
    per_frame = []
    for i in range(count):
        with np.load(work / "results_output" / f"frame_{i}.npz") as saved:   # closed before the directory is removed
            per_frame.append({k: saved[k] for k in ("depth", "conf", "intrinsics")})
    depth = np.stack([p["depth"] for p in per_frame]).astype(np.float32)
    np.savez_compressed(Path(out_dir) / "geometry.npz", keyframes=np.arange(count), depth=depth,
                        conf=np.stack([p["conf"] for p in per_frame]).astype(np.float32) + 1.0,
                        w2c=np.linalg.inv(c2w)[:, :3, :], K=np.stack([p["intrinsics"] for p in per_frame]).astype(np.float64),
                        processed_hw=np.array(depth.shape[1:]))
    shutil.rmtree(work / "results_output")
    shutil.rmtree(work / "pcd")
    sequence = chunks[:len(streamer.chunk_indices)]
    repo = Path(api.__file__).resolve().parents[2]
    return {"model": cfg["model_id"], "weights": cfg["weights"], "weights_revision": cfg["revision"],
            "code_revision": _git_revision(repo), "precision": "bf16 autocast on cuda", "seed": cfg.get("seed", 0),
            "is_metric": int(all(c["is_metric"] for c in sequence)),
            "metric_scale_factor": float(np.median([c["metric_scale_factor"] for c in sequence])),
            "depth_type": "camera_z", "extrinsics": "OpenCV world-to-camera",
            "intrinsics_grid": "processed image, integer pixel centers", "keyframe_count": count,
            "parameters": {**{k: cfg.get(k) for k in ("process_res", "process_res_method", "ref_view_strategy")},
                           "streaming": streaming},
            "streaming": {"method": "DA3-Streaming (chunked Sim(3) alignment, SALAD loop closure)",
                          "chunks": [list(c) for c in streamer.chunk_indices], "chunk_inference": sequence,
                          "loop_closures": [list(map(int, pair)) for pair in streamer.loop_list],
                          "chunk_sim3_scales": [float(np.ravel(s)[0]) for s, _, _ in streamer.sim3_list],
                          "effective_config": config},
            **_hardware(torch, started)}


def _vllm_engine(weights, max_images=1, max_model_len=16384, kv_cache_gb=6):
    """The local vision-language model in vLLM: bf16, eager mode (the answers are short, so no CUDA graph capture)
    and a fixed KV cache, so the engine stays near the weights' 55 GB instead of taking 90% of the card."""
    # Greedy decoding needs no FlashInfer sampling kernels (they are compiled on first use, with ninja); the
    # environment's own tools come first on PATH for anything else vLLM starts.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    from vllm import LLM
    return LLM(model=str(weights), dtype="bfloat16", max_model_len=max_model_len, enforce_eager=True, seed=0,
               limit_mm_per_prompt={"image": max_images}, kv_cache_memory_bytes=int(kv_cache_gb * 2 ** 30))


def _vllm_answers(llm, processor, requests, max_tokens, repetition_penalty=1.0):
    """Greedy answers to [(PIL images, prompt text)], in order, from one batched vLLM call; the prompts are built
    with the model's chat template exactly as for transformers (thinking off)."""
    from vllm import SamplingParams
    prompts = []
    for images, text in requests:
        messages = [{"role": "user", "content": [*({"type": "image"} for _ in images), {"type": "text", "text": text}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        prompts.append({"prompt": prompt, "multi_modal_data": {"image": images if len(images) > 1 else images[0]}})
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, repetition_penalty=repetition_penalty)
    return [o.outputs[0].text for o in llm.generate(prompts, params, use_tqdm=False)]


def _qwen_server():
    """The qwen module when a Qwen server answers: the lane's (CODIFY_QWEN_URL set), or else the shared one of this
    machine (perception.py qwen-server, port 8701; a codification run re-running a perception stage finds it this
    way); None when there is none."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import qwen
    if not os.environ.get(qwen.URL_ENV):
        url = f"http://127.0.0.1:{qwen.SHARED_PORT}"
        if not qwen.healthy(url):
            return None
        os.environ[qwen.URL_ENV], os.environ[qwen.SHARED_ENV] = url, "1"
    return qwen


def _served_info(started, timing):
    return {**timing, "device": "cuda", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "peak_gpu_memory_gb": None, "wall_seconds": round(time.time() - started, 2)}


def _vllm_info(started, loaded, answered):
    """Timing and memory of a vLLM worker: the engine runs in its own process, so the memory is what the card has in
    use while it is up (the lane is alone on its card)."""
    import torch
    import vllm
    free, total = torch.cuda.mem_get_info()
    used = round((total - free) / 1e9, 3)
    return {"engine": "vllm", "vllm": vllm.__version__, "device": "cuda", "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "peak_gpu_memory_gb": used,
            "load_seconds": round(loaded - started, 2), "generate_seconds": round(answered - loaded, 2),
            "wall_seconds": round(time.time() - started, 2)}


def run_vlm_vocabulary(frames_dir, out_dir, cfg):
    """Object names seen in evenly spaced frames, from a local vision-language model and VOCABULARY_PROMPT (with
    the text that comes with the input, when there is one: the objects it mentions are listed when they are seen).
    All frames go to vLLM in one batch."""
    from PIL import Image
    from transformers import AutoProcessor
    started = time.time()
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    k = min(len(frames), int(cfg.get("frames", 32)))
    picks = sorted({round(i * (len(frames) - 1) / max(k - 1, 1)) for i in range(k)})
    prompt = VOCABULARY_PROMPT + (f' The video comes with this text: "{cfg["context"]}" Include the objects it '
                                  "mentions when you can see them." if cfg.get("context") else "")
    served = _qwen_server()
    if served:                      # the lane's sleeping server: no model load here
        texts, timing = served.answers([([str(frames[f])], prompt) for f in picks], int(cfg.get("max_new_tokens", 300)),
                                       float(cfg.get("repetition_penalty", 1.1)))
        loaded = answered = time.time()
    else:
        processor = AutoProcessor.from_pretrained(cfg["weights"])
        llm = _vllm_engine(cfg["weights"])
        loaded = time.time()
        texts = _vllm_answers(llm, processor, [([Image.open(frames[f]).convert("RGB")], prompt) for f in picks],
                              int(cfg.get("max_new_tokens", 300)), float(cfg.get("repetition_penalty", 1.1)))
        answered = time.time()
    per_frame = {}
    for f, text in zip(picks, texts):
        # The answer lists salient objects first; long tails tend to be enumerations, not observations.
        per_frame[f] = names_from_text(text)[: int(cfg.get("names_per_frame", 20))]
        print(f"frame {f}: {per_frame[f]}", flush=True)
    counts = {}
    for f in picks:
        for name in per_frame[f]:
            counts[name] = counts.get(name, 0) + 1
    # A name seen in one frame of many is often a misreading; with few frames every name counts.
    min_count = 2 if len(picks) >= 8 else 1
    names = sorted((n for n, c in counts.items() if c >= min_count), key=lambda n: (-counts[n], n))
    names = names[: int(cfg.get("max_names", 60))]
    with (Path(out_dir) / "vocabulary.json").open("w") as stream:
        json.dump({"prompt": VOCABULARY_PROMPT, "frames": picks, "per_frame": per_frame, "counts": counts,
                   "min_count": min_count, "names": names}, stream, indent=1)
    return {"model": cfg["model_id"], "weights": cfg["weights"], "weights_revision": cfg["revision"],
            "precision": "bf16", "decoding": "greedy", "seed": cfg.get("seed", 0), "frames": len(picks),
            "names": len(names), **(_served_info(started, timing) if served else _vllm_info(started, loaded, answered))}


def _sam3_propagate(predictor, session, prompt, count):
    """One text prompt on frame 0, propagated forward with per-frame detection.

    Returns uint16 labels (count, H, W) holding obj_id + 1, and {obj_id: detections}.
    """
    predictor.handle_request({"type": "reset_session", "session_id": session})
    predictor.handle_request({"type": "add_prompt", "session_id": session, "frame_index": 0, "text": prompt})
    labels, records = None, {}
    for response in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session,
                                                     "propagation_direction": "forward"}):
        frame, out = response["frame_index"], response["outputs"]
        masks = out["out_binary_masks"]
        if labels is None:
            labels = np.zeros((count, *masks.shape[1:]), np.uint16)
        for obj_id, prob, mask in zip(out["out_obj_ids"].tolist(), out["out_probs"].tolist(), masks):
            ys, xs = np.nonzero(mask)
            if not len(xs):  # an ID kept by the tracker with an empty mask is not a detection
                continue
            labels[frame][mask] = obj_id + 1
            records.setdefault(obj_id, []).append(
                {"frame_index": frame, "score": float(prob), "mask_area_px": int(len(xs)),
                 "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]})
    return labels, records


def stitch_windows(windows, overlap, labels, records, link_iou=0.5):
    """Join the per-window SAM 3 tracks of one prompt into whole-video tracks.

    windows: [(start, end)] canonical frame ranges, consecutive ones sharing `overlap` frames;
    labels[k]: uint16 (end - start, H, W) holding obj_id + 1; records[k]: {obj_id: detections with
    window-local frame_index}. A track of window k + 1 continues the track of window k that its
    masks overlap best on the shared frames (IoU >= link_iou, one to one). Each frame keeps the
    masks of one window, split at the middle of the overlap. Returns uint16 labels (frames, H, W)
    and [(label_value, detections, [[window, obj_id], ...])] ordered by first detection.
    """
    parent = {}
    for k in range(len(windows) - 1):
        a = labels[k][windows[k + 1][0] - windows[k][0]:]
        b = labels[k + 1][:len(a)]
        pairs = []
        for va in np.unique(a[a > 0]):
            inside = a == va
            for vb in np.unique(b[inside & (b > 0)]):
                other = b == vb
                pairs.append((float((inside & other).sum() / (inside | other).sum()), int(va), int(vb)))
        linked_a, linked_b = set(), set()
        for iou, va, vb in sorted(pairs, reverse=True):
            if iou >= link_iou and va not in linked_a and vb not in linked_b:
                linked_a.add(va)
                linked_b.add(vb)
                parent[(k + 1, vb - 1)] = (k, va - 1)
    def root(key):
        while key in parent:
            key = parent[key]
        return key
    owned = [(start if k == 0 else start + overlap // 2,
              windows[k + 1][0] + overlap // 2 if k + 1 < len(windows) else end) for k, (start, end) in enumerate(windows)]
    rows, members = {}, {}
    for k, ((start, _), (lo, hi)) in enumerate(zip(windows, owned)):
        for obj_id, detections in records[k].items():
            key = root((k, obj_id))
            members.setdefault(key, []).append([k, obj_id])
            rows.setdefault(key, []).extend({**d, "frame_index": start + d["frame_index"]} for d in detections
                                            if lo <= start + d["frame_index"] < hi)
    joined, value = [], {}
    for key in sorted((key for key in rows if rows[key]), key=lambda key: min(d["frame_index"] for d in rows[key])):
        value[key] = len(joined) + 1
        joined.append((value[key], sorted(rows[key], key=lambda d: d["frame_index"]), members[key]))
    full = np.zeros((windows[-1][1], *labels[0].shape[1:]), np.uint16)
    for k, ((start, _), (lo, hi)) in enumerate(zip(windows, owned)):
        lut = np.zeros(int(labels[k].max()) + 1, np.uint16)
        for obj_id in records[k]:
            lut[obj_id + 1] = value.get(root((k, obj_id)), 0)
        full[lo:hi] = lut[labels[k][lo - start:hi - start]]
    return full, joined


def run_sam3(frames_dir, out_dir, cfg):
    """Text-prompted detection + tracking on every canonical frame, one prompt per pass: one SAM 3 session per
    window of cfg["window"] frames (the whole video when it is not set or longer), consecutive windows sharing
    window_overlap frames and joined by stitch_windows."""
    import torch
    import sam3
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    started = time.time()
    torch.manual_seed(cfg.get("seed", 0))
    out = Path(out_dir)
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    count, size, overlap = len(frames), int(cfg.get("window") or len(frames)), int(cfg.get("window_overlap", 10))
    link_iou = float(cfg.get("link_iou", 0.5))
    if size >= count:
        windows, overlap = [(0, count)], 0
    else:
        windows = [(start, min(start + size, count)) for start in range(0, max(count - overlap, 1), size - overlap)]
    prompts = list(cfg.get("prompts", VOCABULARY))
    predictor = Sam3VideoPredictor(checkpoint_path=cfg["weights"])
    torch.cuda.reset_peak_memory_stats()
    records = [[] for _ in prompts]
    for k, (start, end) in enumerate(windows):
        window_dir = out / f"window_{k:03d}"
        window_dir.mkdir()
        for i in range(start, end):
            (window_dir / f"{i - start:06d}.jpg").symlink_to(frames[i])
        session = predictor.handle_request({"type": "start_session", "resource_path": str(window_dir)})["session_id"]
        for j, prompt in enumerate(prompts):
            labels, found = _sam3_propagate(predictor, session, prompt, end - start)
            np.savez_compressed(window_dir / f"labels_{j:03d}.npz", labels=labels)
            records[j].append(found)
        predictor.handle_request({"type": "close_session", "session_id": session})
        print(f"window {k + 1}/{len(windows)}: frames {start}-{end - 1}", flush=True)
    tracks = []
    for j, prompt in enumerate(prompts):
        window_labels = [np.load(out / f"window_{k:03d}" / f"labels_{j:03d}.npz")["labels"] for k in range(len(windows))]
        labels, joined = stitch_windows(windows, overlap, window_labels, records[j], link_iou)
        np.savez_compressed(out / f"labels_{j:03d}.npz", labels=labels)
        for value, detections, members in joined:
            tracks.append({"track_id": str(len(tracks) + 1), "prompt": prompt, "prompt_index": j, "label_value": value,
                           "detections": detections, "window_tracks": members})
        print(f"prompt {j + 1}/{len(prompts)} {prompt!r}: {len(joined)} tracks", flush=True)
    for k in range(len(windows)):
        shutil.rmtree(out / f"window_{k:03d}")
    with (out / "tracks.json").open("w") as stream:
        json.dump({"prompts": prompts, "evaluated_frames": list(range(count)), "frame_count": count,
                   "windows": [list(w) for w in windows], "window_overlap": overlap, "link_iou": link_iou,
                   "label_files": "labels_<prompt_index:03d>.npz: uint16 (frames, H, W); value = label_value",
                   "tracks": tracks}, stream)
    return {"model": cfg["model_id"], "weights": cfg["weights"], "weights_revision": cfg["revision"],
            "code_revision": _git_revision(Path(sam3.__file__).resolve().parents[1]),
            "precision": "SAM 3 default (bf16 autocast inside the predictor)", "seed": cfg.get("seed", 0),
            "prompt_count": len(prompts), "track_count": len(tracks),
            "sampling": f"every canonical frame in windows of {size} frames overlapping by {overlap}, one session "
                        "each: a text prompt on the window's first frame, forward propagation with per-frame "
                        f"detection; window tracks joined by mask IoU >= {link_iou} on the shared frames",
            **_hardware(torch, started)}


# ---------------------------------------------------------------- host side (tool functions)

PROPOSAL_NAME_PROMPT = (
    "A region of a video frame is outlined in yellow (its surroundings slightly darkened). If the outline holds a "
    "physical object that could be picked up, moved or used (a box, a shoe, a towel, a bottle, a toy, a tool, a "
    "piece of furniture ...), even partly hidden, name its kind with a short lowercase English noun phrase; give "
    "your best guess even if it is small or blurry. Answer 'none' if the outline holds the ground or floor or what "
    "covers it (grass, leaves, gravel, pavement, carpet, floorboards), a wall, the sky, distant scenery (trees, "
    "buildings, hills), a shadow or a reflection.")


def _greedy_nms(boxes, scores, threshold):
    """Indices kept by greedy non-maximum suppression on [x0, y0, x1, y1] boxes."""
    order = np.argsort(-scores)
    kept = []
    for i in order:
        if all(_box_iou(boxes[i], boxes[j]) < threshold for j in kept):
            kept.append(i)
    return kept


def _box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def run_sam3_proposals(frames_dir, out_dir, cfg):
    """Class-agnostic object proposals on a few frames: SAM 3's interactive (SAM 1-style) predictor prompted with a
    point grid, three masks per point, kept by predicted IoU, area and non-maximum suppression. No names involved."""
    import torch
    import sam3
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    started = time.time()
    torch.manual_seed(cfg.get("seed", 0))
    model = build_sam3_image_model(checkpoint_path=cfg["weights"], load_from_HF=False, enable_inst_interactivity=True)
    processor = Sam3Processor(model)
    predictor = model.inst_interactive_predictor
    torch.cuda.reset_peak_memory_stats()
    pps = int(cfg.get("points_per_side", 16))
    per_frame = {}
    for f in cfg["frames"]:
        pil = Image.open(Path(frames_dir) / f"{f:06d}.jpg").convert("RGB")
        image = np.array(pil)
        H, W = image.shape[:2]
        # The interactive predictor has no backbone of its own: the image model's features, prepared the way
        # Sam3Image.predict_inst does, then batched point prompts through the predictor's internals.
        with torch.inference_mode():
            state = processor.set_image(pil)
            _, vision_feats, _, _ = predictor.model._prepare_backbone_features(state["backbone_out"]["sam2_backbone_out"])
            vision_feats[-1] = vision_feats[-1] + predictor.model.no_mem_embed
            feats = [feat.permute(1, 2, 0).view(1, -1, *size)
                     for feat, size in zip(vision_feats[::-1], predictor._bb_feat_sizes[::-1])][::-1]
            predictor._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
            predictor._is_image_set, predictor._orig_hw = True, [(H, W)]
        xs = (np.arange(pps) + 0.5) / pps * W
        ys = (np.arange(pps) + 0.5) / pps * H
        points = np.array([[x, y] for y in ys for x in xs], np.float32)
        masks, scores = [], []
        for start in range(0, len(points), 64):
            chunk = points[start:start + 64]
            _, coords, labels, _ = predictor._prep_prompts(chunk[:, None, :], np.ones((len(chunk), 1), np.int32),
                                                           None, None, True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                out_masks, ious, _ = predictor._predict(coords, labels, None, None, multimask_output=True)
            masks.append((out_masks > 0).reshape(-1, H, W).cpu().numpy())
            scores.append(ious.reshape(-1).float().cpu().numpy())
        masks, scores = np.concatenate(masks), np.concatenate(scores)
        areas = masks.reshape(len(masks), -1).sum(1) / (H * W)
        keep = (scores >= float(cfg.get("pred_iou", 0.75))) & (areas >= float(cfg.get("min_area_frac", 0.0005))) \
            & (areas <= float(cfg.get("max_area_frac", 0.4)))
        masks, scores = masks[keep], scores[keep]
        boxes = []
        for m in masks:
            ys_, xs_ = np.nonzero(m)
            boxes.append([int(xs_.min()), int(ys_.min()), int(xs_.max()) + 1, int(ys_.max()) + 1])
        kept = _greedy_nms(np.array(boxes, float).reshape(-1, 4), scores, float(cfg.get("nms_iou", 0.7)))[: int(cfg.get("max_per_frame", 64))]
        np.savez_compressed(Path(out_dir) / f"proposals_{f:06d}.npz", masks=masks[kept].astype(bool))
        per_frame[str(f)] = [{"box": boxes[i], "score": float(scores[i]), "area_px": int(masks[i].sum())} for i in kept]
        print(f"frame {f}: {len(kept)} proposals", flush=True)
    with (Path(out_dir) / "proposals_raw.json").open("w") as stream:
        json.dump({"frames": cfg["frames"], "per_frame": per_frame, "points_per_side": pps}, stream)
    return {"model": cfg["model_id"], "weights": cfg["weights"], "weights_revision": cfg["revision"],
            "code_revision": _git_revision(Path(sam3.__file__).resolve().parents[1]), "frames": len(cfg["frames"]),
            "proposals": sum(len(v) for v in per_frame.values()), "seed": cfg.get("seed", 0), **_hardware(torch, started)}


def run_vlm_crop_names(frames_dir, out_dir, cfg):
    """One name per proposal crop from the local vision-language model (PROPOSAL_NAME_PROMPT, with the text that
    comes with the input when there is one, whose names it uses); 'none' for crops that show no whole object. All
    crops go to vLLM in one batch."""
    from PIL import Image
    from transformers import AutoProcessor
    started = time.time()
    prompt = PROPOSAL_NAME_PROMPT + (f' The video comes with this text: "{cfg["context"]}" If the outline shows '
                                     "an object it mentions, use that name." if cfg.get("context") else "")
    served = _qwen_server()
    if served:                      # the lane's sleeping server: no model load here
        texts, timing = served.answers([([c], prompt) for c in cfg["crops"]], 16)
        loaded = answered = time.time()
    else:
        processor = AutoProcessor.from_pretrained(cfg["weights"])
        llm = _vllm_engine(cfg["weights"])
        loaded = time.time()
        texts = _vllm_answers(llm, processor, [([Image.open(c).convert("RGB")], prompt) for c in cfg["crops"]], 16)
        answered = time.time()
    names = {Path(c).name: (names_from_text(text) or ["none"])[0] for c, text in zip(cfg["crops"], texts)}
    with (Path(out_dir) / "crop_names.json").open("w") as stream:
        json.dump({"prompt": PROPOSAL_NAME_PROMPT, "names": names}, stream, indent=1)
    return {"model": cfg["model_id"], "weights": cfg["weights"], "weights_revision": cfg["revision"],
            "precision": "bf16", "decoding": "greedy", "crops": len(cfg["crops"]),
            **(_served_info(started, timing) if served else _vllm_info(started, loaded, answered))}


def run_sam3_track_masks(frames_dir, out_dir, cfg):
    """Track proposals from their own masks with SAM 3's tracker (SAM 2-style memory tracking; the model's vision
    backbone is loaded from the same checkpoint): each proposal is a mask prompt on its frame, groups of objects
    are propagated forward and backward over the video (or a window around the prompt frame), and the masks come
    out as label maps like the text tracks'. cfg["proposals"]: [{frame_index, mask_file, mask_index, name}]."""
    import torch
    import sam3
    from sam3.model_builder import build_tracker
    started = time.time()
    torch.manual_seed(cfg.get("seed", 0))
    tracker = build_tracker(apply_temporal_disambiguation=False, with_backbone=True)
    ckpt = torch.load(cfg["weights"], map_location="cpu", weights_only=True)
    ckpt = ckpt.get("model", ckpt)
    state = {k[len("tracker."):]: v for k, v in ckpt.items() if k.startswith("tracker.")}
    state.update({k.replace("detector.backbone.", "backbone."): v for k, v in ckpt.items()
                  if k.startswith("detector.backbone.vision_backbone")})
    missing, _ = tracker.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"tracker weights missing: {missing[:5]}")
    tracker = tracker.cuda().eval()
    del ckpt, state
    torch.cuda.reset_peak_memory_stats()
    count = len(list(Path(frames_dir).glob("*.jpg")))
    proposals = json.loads(Path(cfg["proposals_file"]).read_text()) if cfg.get("proposals_file") else cfg["proposals"]
    group_size, window = int(cfg.get("group_size", 16)), int(cfg.get("track_window", 0)) or count
    # A group shares one seed frame, so every object has its conditioning frame before tracking starts.
    groups = []
    for frame in sorted({p["frame_index"] for p in proposals}):
        same = [p for p in proposals if p["frame_index"] == frame]
        groups += [same[i:i + group_size] for i in range(0, len(same), group_size)]
    tracks, label_files = [], []
    frames = sorted(Path(frames_dir).glob("*.jpg"), key=lambda p: int(p.stem))
    for g, group in enumerate(groups):
        seed = group[0]["frame_index"]
        # The tracker reads every frame of the folder it is given: only the window's frames go in (links in a
        # temporary folder), so a group costs its window, not the whole video.
        lo, hi = max(0, seed - window + 1), min(count, seed + window)
        clip = tempfile.TemporaryDirectory(dir=out_dir) if (lo, hi) != (0, count) else None
        if clip:
            for i, f in enumerate(range(lo, hi)):
                (Path(clip.name) / f"{i:06d}.jpg").symlink_to(frames[f].resolve())
        inference = tracker.init_state(video_path=clip.name if clip else str(frames_dir),
                                       offload_video_to_cpu=hi - lo > 300)
        for slot, item in enumerate(group):
            mask = np.load(Path(cfg["proposals_dir"]) / item["mask_file"])["masks"][item["mask_index"]]
            tracker.add_new_mask(inference, seed - lo, slot + 1, torch.from_numpy(mask).bool())
        labels, records = np.zeros((count, *mask.shape), np.uint16), {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            passes = [(seed - lo, min(window, count - seed), False), (seed - lo, min(window, seed + 1), True)]
            for start_frame, length, reverse in passes:
                for frame, obj_ids, _, video_res, scores in tracker.propagate_in_video(
                        inference, start_frame, length, reverse, propagate_preflight=True, tqdm_disable=True):
                    frame = int(frame) + lo
                    masks = (video_res > 0).cpu().numpy()
                    present = scores.flatten().float().cpu().numpy() if scores is not None else np.ones(len(obj_ids))
                    for i, oid in enumerate(obj_ids):
                        m = masks[i].reshape(labels.shape[1:])
                        ys, xs = np.nonzero(m)
                        # The tracker's presence score: a frame where it says the object is gone is no detection.
                        if not len(xs) or present[i] <= 0 or (labels[frame] == oid).any():
                            continue
                        labels[frame][m & (labels[frame] == 0)] = oid
                        records.setdefault(oid, {})[frame] = {
                            "frame_index": int(frame), "score": float(scores[i]) if scores is not None else 1.0,
                            "mask_area_px": int(len(xs)),
                            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]}
        np.savez_compressed(Path(out_dir) / f"labels_{g:03d}.npz", labels=labels)
        label_files.append(f"labels_{g:03d}.npz")
        for slot, item in enumerate(group):
            rows = records.get(slot + 1, {})
            if rows:
                tracks.append({"track_id": str(len(tracks) + 1), "prompt": item["name"], "prompt_index": g,
                               "label_value": slot + 1, "seed_frame": item["frame_index"],
                               "detections": [rows[f] for f in sorted(rows)]})
        print(f"group {g + 1}: {len(group)} proposals -> {sum(1 for t in tracks if t['prompt_index'] == g)} tracks", flush=True)
        del inference
        if clip:
            clip.cleanup()
        torch.cuda.empty_cache()
    with (Path(out_dir) / "tracks.json").open("w") as stream:
        json.dump({"prompts": [f"proposals {g}" for g in range(len(label_files))], "evaluated_frames": list(range(count)),
                   "frame_count": count, "label_files": "labels_<prompt_index:03d>.npz: uint16 (frames, H, W); value = label_value",
                   "tracks": tracks}, stream)
    return {"model": cfg["model_id"], "weights": cfg["weights"], "weights_revision": cfg["revision"],
            "code_revision": _git_revision(Path(sam3.__file__).resolve().parents[1]), "proposals": len(proposals),
            "track_count": len(tracks), "track_window": window, **_hardware(torch, started)}


def track_proposals(frames_dir, proposals_dir, names_path, tracks_dir, output_dir, config):
    """The named proposals (not 'none') tracked from their masks, then those repeating a text track or another
    proposal track (the same region, or one inside the other under the same name; shares of pixels >= dedupe_iou
    over shared frames) dropped; the rest become a tracks directory to join to the others."""
    from .contract import read_json, read_labels, tool_result, write_json
    proposals, answers = read_json(Path(proposals_dir) / "proposals.json"), read_json(names_path)["names"]
    items = []
    for crop in proposals["crops"]:
        name = answers.get(crop["file"], "none").strip().lower()
        if name == "none" or name in NOT_OBJECTS or len(name) > 40:
            continue
        f, idx = crop["frame_index"], int(crop["file"].split("_")[1].split(".")[0])
        items.append({"frame_index": f, "mask_file": f"proposals_{f:06d}.npz", "mask_index": idx, "name": name})
    out = Path(output_dir)
    if not items:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "tracks.json", {"prompts": [], "evaluated_frames": [], "frame_count": 0, "tracks": []})
        return tool_result("success", {"tracks": out / "tracks.json"}, {"proposals": 0, "tracks": 0, "kept": 0})
    listing = out.parent / f"{out.name}_proposals.json"     # by file: thousands of proposals overflow a command line
    write_json(listing, items)
    info = _run_worker("sam3_track_masks", frames_dir, output_dir,
                       {**config, "proposals_file": str(listing), "proposals_dir": str(proposals_dir)})
    tracks = read_json(out / "tracks.json")
    # Masks of every track on a sample of its frames, for the overlap tests; each label file (a whole video's
    # frames) is read once for all its tracks.
    def sample_masks(track_list, directory):
        masks = {}
        for prompt_index in sorted({t["prompt_index"] for t in track_list}):
            labels = read_labels(Path(directory) / f"labels_{prompt_index:03d}.npz")
            for t in track_list:
                if t["prompt_index"] == prompt_index:
                    frames = [d["frame_index"] for d in t["detections"]]
                    masks[t["track_id"]] = {f: labels[f] == t["label_value"] for f in frames[:: max(1, len(frames) // 8)]}
            del labels
        return masks
    existing = read_json(Path(tracks_dir) / "tracks.json")["tracks"]
    existing_masks = sample_masks(existing, tracks_dir)
    candidate_masks = sample_masks(tracks["tracks"], out)

    def covered(a, b):
        """Share of a's pixels inside b on the frames both have."""
        shared = [f for f in a if f in b]
        if not shared:
            return 0.0
        return float(np.mean([(a[f] & b[f]).sum() / max(a[f].sum(), 1) for f in shared]))
    threshold = float(config.get("dedupe_iou", 0.5))
    # The best supported track of each object first (the longest, then the earliest seed); a larger mask is not
    # a better one, since a proposal can span an object and its neighbour.
    candidates = sorted(tracks["tracks"], key=lambda t: (-len(t["detections"]), t["seed_frame"]))
    # One object twice: the same region, or one inside the other under the same name. A differently named region
    # inside a larger one is kept (a part, or a thing inside, held by or resting on another, which a merged mask
    # can swallow); fusion decides whether the two are one object.
    repeats = lambda t, m, other, om: ((covered(m, om) >= threshold and covered(om, m) >= threshold) or
                                       (t["prompt"] == other["prompt"] and max(covered(m, om), covered(om, m)) >= threshold))
    by_id = {t["track_id"]: t for t in existing}
    kept, kept_masks, dropped = [], [], {"covered_by_tracks": 0, "duplicate_proposal": 0}
    for t in candidates:
        m = candidate_masks[t["track_id"]]
        if any(repeats(t, m, by_id[tid], em) for tid, em in existing_masks.items()):
            dropped["covered_by_tracks"] += 1
            continue
        if any(repeats(t, m, kt, km) for kt, km in zip(kept, kept_masks)):
            dropped["duplicate_proposal"] += 1
            continue
        kept.append(t)
        kept_masks.append(m)
    for new_id, t in enumerate(kept, 1):
        t["track_id"] = str(new_id)
    write_json(out / "tracks.json", {**tracks, "tracks": kept, "dropped": dropped})
    return tool_result("success", {"tracks": out / "tracks.json"},
                       {"proposals": len(items), "tracks": info["track_count"], "kept": len(kept), **dropped,
                        "names": sorted({t["prompt"] for t in kept}), "wall_seconds": info["wall_seconds"],
                        "peak_gpu_memory_gb": info["peak_gpu_memory_gb"]})


PROPOSAL_TRACK_CODE = ("models.py:run_sam3_track_masks", "models.py:track_proposals", "models.py:_run_worker")


def propose_objects(frames_dir, timeline_path, tracks_dir, output_dir, config):
    """Class-agnostic proposals on evenly spaced frames, then the ones no SAM 3 track already explains (no track
    mask covering the same region, each at least unexplained_overlap inside the other) cut out as crops for naming."""
    import cv2
    from .contract import read_json, read_labels, read_jsonl, tool_result
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    k = min(len(frames), int(config.get("frames", 8)))
    picks = sorted({round(i * (len(frames) - 1) / max(k - 1, 1)) for i in range(k)})
    worker_cfg = {**config, "frames": picks}
    info = _run_worker("sam3_proposals", frames_dir, output_dir, worker_cfg)
    out = Path(output_dir)
    raw = read_json(out / "proposals_raw.json")
    tracks = read_json(Path(tracks_dir) / "tracks.json")
    # A track explains a proposal when both cover the same region (each mostly inside the other). A region inside
    # a larger tracked mask (a part, a thing inside a clear container or lying on a tracked object) is named, and
    # masks covering more than a quarter of the frame (grass, a floor, a table top) explain nothing.
    tracked = {f: [] for f in picks}
    for prompt_index in sorted({t["prompt_index"] for t in tracks["tracks"]}):
        labels = read_labels(Path(tracks_dir) / f"labels_{prompt_index:03d}.npz")
        for f in picks:
            for value in np.unique(labels[f]):
                if value and (labels[f] == value).mean() <= float(config.get("background_area_frac", 0.25)):
                    tracked[f].append(labels[f] == value)
        del labels
    share = float(config.get("unexplained_overlap", 0.6))
    (out / "crops").mkdir(exist_ok=True)
    crops, explained = [], 0
    for f in picks:
        masks = np.load(out / f"proposals_{f:06d}.npz")["masks"]
        image = cv2.imread(str(frames[f]))
        H, W = image.shape[:2]
        for i, item in enumerate(raw["per_frame"][str(f)]):
            m = masks[i]
            if m.any() and any((m & t).sum() >= share * max(m.sum(), t.sum()) for t in tracked[f]):
                explained += 1
                continue
            x0, y0, x1, y1 = item["box"]
            pad = int(0.15 * max(x1 - x0, y1 - y0)) + 4
            ya, yb, xa, xb = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
            crop = image[ya:yb, xa:xb].copy()
            local = m[ya:yb, xa:xb]
            crop[~local] = (crop[~local] * 0.7).astype(np.uint8)           # surroundings a little darker
            scale = max(1.0, 320 / max(crop.shape[:2]))                      # the VLM gets at least 320 px
            crop = cv2.resize(crop, (round(crop.shape[1] * scale), round(crop.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
            contours, _ = cv2.findContours(cv2.resize(local.astype(np.uint8), (crop.shape[1], crop.shape[0]),
                                                      interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(crop, contours, -1, (0, 255, 255), 2)
            name = f"{f:06d}_{i:02d}.jpg"
            cv2.imwrite(str(out / "crops" / name), crop)
            crops.append({"file": name, "frame_index": f, "box": item["box"], "score": item["score"], "area_px": item["area_px"]})
    with (out / "proposals.json").open("w") as stream:
        json.dump({"frames": picks, "crops": crops, "explained_by_tracks": explained}, stream, indent=1)
    return tool_result("success", {"proposals": out / "proposals.json", "crops": out / "crops"},
                       {"frames": len(picks), "proposals": info["proposals"], "explained_by_tracks": explained,
                        "unexplained_crops": len(crops), "wall_seconds": info["wall_seconds"],
                        "peak_gpu_memory_gb": info["peak_gpu_memory_gb"]})


CROP_CHUNK = 600


def name_proposals(proposals_dir, output_dir, config):
    """One name per unexplained proposal crop from the local VLM ('none' for ground, walls and the like); every named
    crop is then tracked from its mask (track_proposals)."""
    from .contract import read_json, tool_result, write_json
    proposals = read_json(Path(proposals_dir) / "proposals.json")
    crops = [str(Path(proposals_dir) / "crops" / c["file"]) for c in proposals["crops"]]
    out = Path(output_dir)
    if not crops:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "crop_names.json", {"prompt": PROPOSAL_NAME_PROMPT, "names": {}})
        return tool_result("success", {"names": out / "crop_names.json"}, {"crops": 0, "named": 0})
    if len(crops) <= CROP_CHUNK:
        info = _run_worker("vlm_crop_names", "/", output_dir, {**config, "crops": crops})
        names = read_json(out / "crop_names.json")["names"]
    else:       # the worker takes its settings on the command line, too long for thousands of crop paths at once
        out.mkdir(parents=True, exist_ok=False)
        names, seconds = {}, 0.0
        for k in range(0, len(crops), CROP_CHUNK):
            part = out / f"part_{k // CROP_CHUNK:02d}"
            info = _run_worker("vlm_crop_names", "/", part, {**config, "crops": crops[k:k + CROP_CHUNK]})
            names.update(read_json(part / "crop_names.json")["names"])
            seconds += info["wall_seconds"]
        info = {**info, "wall_seconds": seconds}
        write_json(out / "crop_names.json", {"prompt": PROPOSAL_NAME_PROMPT, "names": names})
        write_json(out / "model_info.json", info)
    named = sorted({n.strip().lower() for n in names.values() if n.strip().lower() not in ("none", *NOT_OBJECTS)})
    return tool_result("success", {"names": out / "crop_names.json"},
                       {"crops": len(crops), "named": sum(n.strip().lower() not in ("none", *NOT_OBJECTS) for n in names.values()),
                        "names": named, "wall_seconds": info["wall_seconds"]})


PROPOSAL_CODE = ("models.py:run_sam3_proposals", "models.py:_greedy_nms", "models.py:_box_iou", "models.py:propose_objects",
                 "models.py:_run_worker")
PROPOSAL_NAME_CODE = ("models.py:run_vlm_crop_names", "models.py:name_proposals", "models.py:names_from_text",
                      "models.py:_run_worker")


def _run_worker(kind, frames_dir, output_dir, config):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TQDM_DISABLE="1",
               CUDA_VISIBLE_DEVICES="" if config.get("device") == "cpu" else str(config.get("gpu", 0)))
    command = [config["python"], str(Path(__file__).resolve()), kind, str(Path(frames_dir).resolve()),
               str(out.resolve()), json.dumps(config)]
    with (out / "worker.log").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(out))
    if result.returncode:
        raise RuntimeError(f"{kind} worker failed with exit {result.returncode}; inspect {out / 'worker.log'}")
    return json.loads((out / "model_info.json").read_text())


def estimate_geometry(frames_dir, timeline_path, output_dir, config):
    """Per-keyframe cameras and metric depth from Depth Anything 3 (no ground truth)."""
    from .contract import read_jsonl, tool_result
    info = _run_worker("da3", frames_dir, output_dir, config)
    if len(read_jsonl(timeline_path)) != len(list(Path(frames_dir).glob("*.jpg"))):
        raise RuntimeError("Timeline and canonical frames disagree")
    warnings = [] if info["is_metric"] else ["Model output is not metric; scale must be reported as relative"]
    return tool_result("success", {"geometry": Path(output_dir) / "geometry.npz",
                                   "model_info": Path(output_dir) / "model_info.json"},
                       {k: info[k] for k in ("model", "keyframe_count", "is_metric", "wall_seconds", "peak_gpu_memory_gb")},
                       warnings)


def estimate_geometry_global(frames_dir, timeline_path, output_dir, config):
    """estimate_geometry for walkthroughs (run_da3_global); same artifacts and summary."""
    from .contract import read_jsonl, tool_result
    info = _run_worker("da3_global", frames_dir, output_dir, config)
    if len(read_jsonl(timeline_path)) != len(list(Path(frames_dir).glob("*.jpg"))):
        raise RuntimeError("Timeline and canonical frames disagree")
    warnings = [] if info["is_metric"] else ["Model output is not metric; scale must be reported as relative"]
    return tool_result("success", {"geometry": Path(output_dir) / "geometry.npz",
                                   "model_info": Path(output_dir) / "model_info.json"},
                       {k: info[k] for k in ("model", "keyframe_count", "is_metric", "wall_seconds", "peak_gpu_memory_gb")},
                       warnings)


def estimate_geometry_streaming(frames_dir, timeline_path, output_dir, config):
    """estimate_geometry for long videos (DA3-Streaming); same artifacts and summary."""
    from .contract import read_jsonl, tool_result
    info = _run_worker("da3_streaming", frames_dir, output_dir, config)
    if len(read_jsonl(timeline_path)) != len(list(Path(frames_dir).glob("*.jpg"))):
        raise RuntimeError("Timeline and canonical frames disagree")
    warnings = [] if info["is_metric"] else ["Model output is not metric; scale must be reported as relative"]
    return tool_result("success", {"geometry": Path(output_dir) / "geometry.npz",
                                   "model_info": Path(output_dir) / "model_info.json"},
                       {k: info[k] for k in ("model", "keyframe_count", "is_metric", "wall_seconds", "peak_gpu_memory_gb")}
                       | {"chunks": len(info["streaming"]["chunks"]), "loop_closures": len(info["streaming"]["loop_closures"])},
                       warnings)


def discover_vocabulary(frames_dir, timeline_path, output_dir, config):
    """Object names for segment_track from a local VLM over evenly spaced frames (no labels, no questions)."""
    from .contract import read_json, tool_result
    info = _run_worker("vlm_vocabulary", frames_dir, output_dir, config)
    names = read_json(Path(output_dir) / "vocabulary.json")["names"]
    return tool_result("success", {"vocabulary": Path(output_dir) / "vocabulary.json",
                                   "model_info": Path(output_dir) / "model_info.json"},
                       {"model": info["model"], "frames": info["frames"], "names": len(names),
                        "wall_seconds": info["wall_seconds"], "peak_gpu_memory_gb": info["peak_gpu_memory_gb"]},
                       [] if names else ["The VLM named no objects; segment_track gets no prompts"])


VOCABULARY_CODE = ("models.py:run_vlm_vocabulary", "models.py:names_from_text", "models.py:discover_vocabulary",
                   "models.py:_run_worker")


def geometry_stage(config):
    """(tool function, code hashed into the stage key) for a geometry config: DA3, DA3 on a walkthrough's keyframes
    (max_views), or DA3-Streaming."""
    if "max_views" in config:
        return estimate_geometry_global, ("models.py:run_da3_global", "models.py:run_da3",
                                          "models.py:estimate_geometry_global", "models.py:_run_worker")
    if "streaming" in config:
        return estimate_geometry_streaming, ("models.py:run_da3_streaming", "models.py:estimate_geometry_streaming",
                                             "models.py:_run_worker")
    return estimate_geometry, ("models.py:run_da3", "models.py:estimate_geometry", "models.py:_run_worker")


def segment_track(frames_dir, timeline_path, output_dir, config):
    """Instance masks, stable per-run track IDs and category hypotheses from SAM 3 (in windows on long videos)."""
    from .contract import read_jsonl, tool_result
    info = _run_worker("sam3", frames_dir, output_dir, config)
    if len(read_jsonl(timeline_path)) != len(list(Path(frames_dir).glob("*.jpg"))):
        raise RuntimeError("Timeline and canonical frames disagree")
    return tool_result("success", {"tracks": Path(output_dir) / "tracks.json",
                                   "model_info": Path(output_dir) / "model_info.json"},
                       {k: info[k] for k in ("model", "prompt_count", "track_count", "wall_seconds", "peak_gpu_memory_gb")})


TRACKS_CODE = ("models.py:run_sam3", "models.py:_sam3_propagate", "models.py:stitch_windows", "models.py:segment_track",
               "models.py:_run_worker")


def _mask_iou_lean(pred_masks, gt_masks):
    """sam3.perflib.masks_ops.mask_iou without its (N, M, H*W) temporary, which needs tens of GB when a prompt finds
    hundreds of objects in a window (scene0304_00, scene0686_00: 38.6 and 55.6 GiB more on a 95 GiB card, an
    out-of-memory retried forever). The intersections are a product of 0/1 matrices in float32 (autocast off): every
    count is an integer below 2**24, so exact, and the IoUs are the original's to the bit."""
    import torch
    assert pred_masks.dtype == gt_masks.dtype == torch.bool
    N, H, W = pred_masks.shape
    M, _, _ = gt_masks.shape
    if H * W >= 2 ** 24:        # counts no longer exact in float32: the original, a few rows at a time
        pred_flat, gt_flat = pred_masks.view(N, 1, H * W), gt_masks.view(1, M, H * W)
        rows = max(1, 2 ** 28 // max(1, M * H * W))
        intersection = torch.cat([(pred_flat[i:i + rows] & gt_flat).sum(dim=2) for i in range(0, N, rows)]).float()
        union = torch.cat([(pred_flat[i:i + rows] | gt_flat).sum(dim=2) for i in range(0, N, rows)]).float()
        return intersection / union.clamp(min=1)
    with torch.autocast(device_type=pred_masks.device.type, enabled=False):
        a, b = pred_masks.view(N, H * W).float(), gt_masks.view(M, H * W).float()
        intersection = a @ b.T
        union = a.sum(dim=1)[:, None] + b.sum(dim=1)[None, :] - intersection
        return intersection / union.clamp(min=1)


def _patch_sam3_mask_iou():
    """Every module that imported SAM 3's mask_iou by name gets the lean one (the SAM 3 checkout is not changed)."""
    import importlib
    for name in ("sam3.perflib.masks_ops", "sam3.perflib.nms", "sam3.perflib.associate_det_trk",
                 "sam3.model.sam3_video_base"):
        module = importlib.import_module(name)
        if hasattr(module, "mask_iou"):
            module.mask_iou = _mask_iou_lean


def main():
    parser = argparse.ArgumentParser(description="Model worker; started by estimate_geometry/segment_track")
    parser.add_argument("kind", choices=["da3", "da3_global", "da3_streaming", "sam3", "vlm_vocabulary",
                                         "sam3_proposals", "vlm_crop_names", "sam3_track_masks"])
    parser.add_argument("frames_dir")
    parser.add_argument("output_dir")
    parser.add_argument("config")
    args = parser.parse_args()
    run = {"da3": run_da3, "da3_global": run_da3_global, "da3_streaming": run_da3_streaming, "sam3": run_sam3,
           "vlm_vocabulary": run_vlm_vocabulary, "sam3_proposals": run_sam3_proposals,
           "vlm_crop_names": run_vlm_crop_names, "sam3_track_masks": run_sam3_track_masks}[args.kind]
    config = json.loads(args.config)
    if args.kind in ("sam3", "sam3_proposals", "sam3_track_masks"):
        _patch_sam3_mask_iou()
    info = run(args.frames_dir, args.output_dir, config)
    (Path(args.output_dir) / "model_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    sys.exit(main())
