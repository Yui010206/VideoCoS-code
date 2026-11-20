"""SAM 3D Objects worker: an object's mask on one frame -> a complete mesh (vertex colours) and its pose in that frame's
camera. Runs in the SAM 3D environment (not this package's): objects3d.py starts it once per run and sends one JSON
request per line on stdin; it answers one JSON line on stdout (logs go to stderr).

Request: {"image": jpg, "mask": npy (bool H x W), "depth": npy (float32 H x W, metres along the optical axis, 0 = none)
or null, "K": 3 x 3 (pixels), "out": dir, "faces": 20000, "seed": 42, "stage1_shortcut": false (the sparse structure in
a few shortcut steps), "stage2_steps": null (the pipeline's default)}.
Writes out/mesh.npz (vertices in the object's local frame, +Z up as Blender has them; faces; RGBA vertex colours) and
answers {"ok", "camera_from_local" (4 x 4, the local frame -> the camera's +X right, +Y down, +Z forward), "faces",
"layout_iou" [before, after the render-and-compare refinement], "seconds"}.
With depth, the pose is conditioned on the frame's measured points (metric, the run's own scale) and refined against
the mask with the frame's own intrinsics; without, SAM 3D estimates the points itself.
usage: <sam3d env python> sam3d_worker.py --repo <sam-3d-objects checkout> --config <checkpoints>/pipeline.yaml
"""
import argparse
import json
import os
import sys
import time
import traceback

PROTOCOL = os.fdopen(os.dup(1), "w", buffering=1)       # the real stdout, for answers only
os.dup2(2, 1)                                           # the model's prints go to stderr
sys.stdout = sys.stderr

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
parser.add_argument("--config", required=True)
ARGS = parser.parse_args()
os.environ.setdefault("CONDA_PREFIX", os.path.dirname(os.path.dirname(sys.executable)))
os.environ["LIDRA_SKIP_INIT"] = "true"
sys.path[:0] = [os.path.join(ARGS.repo, "notebook"), ARGS.repo]

import numpy as np                                       # noqa: E402
import open3d as o3d                                     # noqa: E402
import torch                                             # noqa: E402
from PIL import Image                                    # noqa: E402
from pytorch3d.transforms import quaternion_to_matrix    # noqa: E402

from inference import Inference                          # noqa: E402
from sam3d_objects.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera  # noqa: E402

Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float32)    # row vectors: (x, y, z) -> (x, -z, y)
RC = camera_to_pytorch3d_camera(device="cpu").rotation[0].numpy()          # p_pytorch3d = p_camera @ RC


def decimate(vertices, faces, colors, target):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices.astype(np.float64)),
                                     o3d.utility.Vector3iVector(faces.astype(np.int32)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:, :3].astype(np.float64) / 255.0)
    mesh = mesh.remove_duplicated_vertices().remove_degenerate_triangles().remove_unreferenced_vertices()
    if len(mesh.triangles) > target:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target))
    rgb = (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8)
    return (np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles, np.int32),
            np.concatenate([rgb, np.full((len(rgb), 1), 255, np.uint8)], 1))


def run(model, request):
    started = time.time()
    image = np.array(Image.open(request["image"]).convert("RGB"))
    mask = np.load(request["mask"]).astype(bool)
    H, W = mask.shape
    K = np.asarray(request["K"], float)
    pointmap, intrinsics = None, None
    if request.get("depth"):
        depth = np.load(request["depth"]).astype(np.float32)
        v, u = np.mgrid[0:H, 0:W]
        points = np.stack([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth], -1)
        points[depth <= 0] = np.nan
        pointmap = torch.from_numpy((points.reshape(-1, 3) @ RC).reshape(H, W, 3).astype(np.float32))
        intrinsics = torch.tensor([[K[0, 0] / W, 0, K[0, 2] / W], [0, K[1, 1] / H, K[1, 2] / H], [0, 0, 1]],
                                  dtype=torch.float32)
    pipeline = model._pipeline
    compute = pipeline.compute_pointmap

    def with_our_intrinsics(img, pm=None):
        # SAM 3D infers intrinsics from the point map for its render-and-compare step; from a point map with holes
        # that inference fails (negative focal lengths), so the frame's own intrinsics go in instead.
        out = compute(img, pm)
        if intrinsics is not None:
            out["intrinsics"] = intrinsics.to(out["intrinsics"].device)
        return out

    pipeline.compute_pointmap = with_our_intrinsics
    try:
        output = pipeline.run(model.merge_mask_to_rgba(image, mask), None, int(request.get("seed", 42)),
                              stage1_only=False, with_mesh_postprocess=False, with_texture_baking=False,
                              with_layout_postprocess=True, use_vertex_color=True, pointmap=pointmap,
                              use_stage1_distillation=bool(request.get("stage1_shortcut")),
                              stage2_inference_steps=request.get("stage2_steps"))
    finally:
        pipeline.compute_pointmap = compute
    glb = output["glb"]
    local = np.asarray(glb.vertices, np.float32) @ Y_UP_TO_Z_UP
    colors = np.asarray(glb.visual.vertex_colors, np.uint8)
    R = quaternion_to_matrix(output["rotation"].squeeze().float().cpu()).numpy()
    flip = np.ones(3, np.float32)
    if R[2, 1] < 0:          # the shape came out upside down in its own frame (SceneRig's rule): turn it 180 degrees
        flip[[0, 2] if R[1, 2] > 0 else [1, 2]] = -1
    scale = output["scale"][0].float().cpu().numpy().reshape(-1)
    t = output["translation"][0].float().cpu().numpy().reshape(3)
    # pytorch3d applies scale, rotate (row vectors: p @ R), translate; the camera frame is RC away from its own.
    linear = RC @ R.T @ np.diag(scale * np.ones(3) * flip)
    camera_from_local = np.eye(4)
    camera_from_local[:3, :3], camera_from_local[:3, 3] = linear, RC @ t
    vertices, faces, colors = decimate(local, np.asarray(glb.faces), colors, request.get("faces", 20000))
    os.makedirs(request["out"], exist_ok=True)
    np.savez_compressed(os.path.join(request["out"], "mesh.npz"), vertices=vertices, faces=faces, colors=colors)
    iou = [output.get(k) for k in ("iou_before_optim", "iou")]
    return {"ok": True, "camera_from_local": camera_from_local.tolist(), "faces": int(len(faces)),
            "layout_iou": [None if x is None else round(float(x), 3) for x in iou],
            "seconds": round(time.time() - started, 1)}


def main():
    started = time.time()
    model = Inference(ARGS.config, compile=False)
    PROTOCOL.write(json.dumps({"ready": True, "load_seconds": round(time.time() - started, 1)}) + "\n")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            answer = run(model, json.loads(line))
        except Exception as exc:
            traceback.print_exc()
            answer = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:600]}
        PROTOCOL.write(json.dumps(answer) + "\n")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
