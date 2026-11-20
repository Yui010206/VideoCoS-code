#!/usr/bin/env python3
"""VideoCoS codification harness: input -> perception evidence -> an MLLM writes the Blender scene -> .blend.

  perceive   local perception (DA3 cameras and depth, a local VLM's object names, SAM 3 tracks and proposals, fused
             3D objects) published as an immutable evidence version; stages are cached by content, so it resumes
  codify     perceive, then the model loop: the model reads the evidence and frames, writes scene.py, builds it in a
             sandboxed Blender, follows the checks' repairs and names a final revision (--script: a scripted
             stand-in that makes no API calls)
  preview    a report video of a finished run: input | render over the input's duration

The input is a diagnostic-bench sample (--sample <id>: its media, its task family's config configs/tasks/*.json and
what the sample adds) or --video with --config; a config includes configs/machine.json. This is the internal
tool of the two entry points, perception.py and codification.py (README.md).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from codify import bench, contract, perception  # noqa: E402
from codify.perception import now  # noqa: E402

PATH_KEYS = ("calibration", "given_cameras", "given", "include", "bench_root")
TASK_CONFIGS = Path(__file__).resolve().parent / "configs" / "tasks"
MACHINE_CONFIG = TASK_CONFIGS.parent / "machine.json"


def _merge(base, over):
    out = dict(base)
    for key, value in over.items():
        out[key] = _merge(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return out


def _read(path):
    """One config file with its includes merged underneath it; relative paths resolve against the file."""
    path = Path(path).resolve()
    config = json.loads(path.read_text())

    def resolve(value, key=""):
        if isinstance(value, dict):
            return {k: resolve(v, k) for k, v in value.items()}
        if isinstance(value, str) and key in PATH_KEYS and not Path(value).is_absolute():
            return str((path.parent / value).resolve())
        return value
    config = resolve(config)
    include = config.pop("include", None)
    return _merge(_read(include), config) if include else config


def load_config(path):
    """The example's config over the machine config it includes. An example that chunks a long video
    (geometry.streaming) gets the machine's DA3-Streaming paths (geometry.streaming_paths) merged in."""
    config = _read(path)
    geometry = config.get("geometry", {})
    paths = geometry.pop("streaming_paths", None)
    if isinstance(geometry.get("streaming"), dict) and paths:
        geometry["streaming"] = {**paths, **geometry["streaming"]}
    return config, Path(path).resolve()


def resolve_input(args):
    """(config, config path, media path) of a run: --sample (a bench sample id: its media, its task family's config
    configs/tasks/<family>.json and the settings the sample adds) or --video with --config."""
    if getattr(args, "sample", None):
        from codify import bench
        family = bench.FAMILIES.get(args.sample.split("_")[0])
        if family is None:
            raise SystemExit(f"{args.sample}: not a bench sample id (T1_... to T7_...)")
        config, config_path = load_config(TASK_CONFIGS / f"{family}.json")
        media, settings = bench.inputs(config, args.sample, args.output)
        return _merge(config, settings), config_path, Path(media).resolve()
    if not (args.video and args.config):
        raise SystemExit("give --sample, or --video and --config")
    config, config_path = load_config(args.config)
    return config, config_path, Path(args.video).resolve()


def _problems(config, needs_model):
    problems = []
    for key in ("geometry", "tracks", "vocabulary"):
        for field in ("python", "weights"):
            if key in config and not Path(config[key][field]).exists():
                problems.append(f"{key}.{field} not found: {config[key][field]}")
    for field in ("code_dir", "salad_weights", "dinov2_hub_dir"):
        path = config.get("geometry", {}).get("streaming", {}).get(field)
        if path and not Path(path).exists():
            problems.append(f"geometry.streaming.{field} not found: {path}")
    calibration = config.get("decode", {}).get("rectify", {}).get("calibration")
    if calibration and not Path(calibration).is_file():
        problems.append(f"calibration not found: {calibration}")
    given = config.get("evidence", {}).get("given")
    if given and not (Path(given) / "manifest.json").is_file():
        problems.append(f"evidence.given is not an evidence directory: {given}")
    if needs_model:
        settings = config.get("agent", {})
        key_env = settings.get("model", {}).get("api_key_env", "OPENAI_API_KEY")
        if not settings.get("model", {}).get("model"):
            problems.append("agent.model.model is not set")
        if not os.environ.get(key_env):
            problems.append(f"environment variable {key_env} (API key) is not set")
        budget = settings.get("budget", {})
        if budget.get("max_usd") is None or not budget.get("usd_per_million_tokens"):
            problems.append("agent.budget.max_usd and usd_per_million_tokens are required before paid calls")
    return problems


def _start(args, needs_model, extra_plan=None):
    """Shared start of perceive and codify: the plan (run.json), the output directory and cached stages reused
    from another run. Returns None after --prepare-only."""
    config, config_path, video_path = resolve_input(args)
    if getattr(args, "gpu", None) is not None:          # which card, never what is computed (not part of stage keys)
        for section in ("geometry", "vocabulary", "tracks", "objects3d"):
            config.setdefault(section, {})["gpu"] = args.gpu
    out = Path(args.output).resolve()
    video_sha = contract.sha256(video_path)
    if (out / "run.json").is_file():
        previous = contract.read_json(out / "run.json")
        if previous.get("source_video_sha256") != video_sha or previous.get("finished_utc"):
            raise SystemExit(f"{out} holds an unrelated or finished run; choose an empty output directory")
    elif out.exists() and any(p.name not in ("given_cameras.json", bench.UPRIGHT) for p in out.iterdir()):
        raise SystemExit(f"{out} is not empty and holds no run; choose an empty output directory")
    problems = _problems(config, needs_model)
    plan = {"video": str(video_path), "source_video_sha256": video_sha, "config": str(config_path),
            "sample": getattr(args, "sample", None),
            "mode": contract.read_json(Path(config["evidence"]["given"]) / "manifest.json")["mode"]
            if config.get("evidence", {}).get("given") else perception.mode_of(config),
            **(extra_plan or {}), "problems": problems}
    if args.prepare_only:
        print(json.dumps(plan, indent=2))
        return None
    if problems:
        raise SystemExit("; ".join(problems))
    out.mkdir(parents=True, exist_ok=True)
    contract.write_json(out / "run.json", {**plan, "started_utc": now()})
    if args.reuse:
        # Stage keys decide what is reused: a copied stage whose inputs, settings or code changed runs again.
        source = Path(args.reuse).resolve() / "work"
        names = ["decode", "vocabulary", "geometry", "tracks", "camera_depth", "proposals", "proposal_names",
                 "proposal_tracks"] + sorted(p.name[:-len(".stage.json")] for p in source.glob("tracks_joined_*.stage.json")) \
            + sorted(p.name[:-len(".stage.json")] for p in source.glob("fuse_*.stage.json"))
        for name in names:
            if (source / f"{name}.stage.json").is_file() and not (out / "work" / name).exists():
                (out / "work").mkdir(parents=True, exist_ok=True)
                shutil.copytree(source / name, out / "work" / name, copy_function=os.link)   # cached stages: hard links
                shutil.copyfile(source / f"{name}.stage.json", out / "work" / f"{name}.stage.json")
    log_path = out / "run.log"

    def log(message):
        line = f"{now()} {message}"
        print(line, flush=True)
        with log_path.open("a") as stream:
            stream.write(line + "\n")
    return config, config_path, video_path, out, plan, log


def perceive(args):
    started = _start(args, needs_model=False)
    if started is None:
        return 0
    config, config_path, video_path, out, plan, log = started
    evidence = perception.Perception(out, video_path, config, log).perceive()
    manifest = contract.read_json(evidence / "manifest.json")
    objects = contract.read_json(evidence / "objects.json") if (evidence / "objects.json").is_file() else {}
    contract.write_json(out / "run.json", {**plan, "started_utc": contract.read_json(out / "run.json")["started_utc"],
                                           "perceived_utc": now(), "evidence": manifest["version"],
                                           "evidence_digest": manifest["digest"], "objects": len(objects),
                                           "dynamic_objects": sum(o["motion_type"] == "dynamic" for o in objects.values())})
    print(json.dumps({"evidence": str(evidence), "objects": len(objects)}, indent=2))
    return 0


def codify(args):
    """Local perception first, then the model with four tools, sandboxed Blender and revisions."""
    import importlib.util
    from codify import agent, sandbox, tools
    scripted = args.script is not None
    codex = args.agent == "codex"
    config, config_path, _ = resolve_input(args)
    settings = config.get("agent", {})
    if args.max_usd is not None:            # a lower cap for this run than the config's
        settings.setdefault("budget", {})["max_usd"] = min(args.max_usd, settings.get("budget", {}).get("max_usd", args.max_usd))
    box = sandbox.Sandbox(settings["blender_dir"], cpus=settings.get("sandbox_cpus", 8),
                          memory=settings.get("sandbox_memory", "8g"), timeout=settings.get("sandbox_timeout", 900),
                          room=bool(settings.get("fast_room")))
    try:
        box.check()
        isolation = None
    except sandbox.SandboxUnavailable as exc:
        isolation = str(exc)
    plan = {"model": "scripted (" + str(args.script) + ")" if scripted else
            f"codex ({args.codex_model or 'its configured model'}, effort {args.codex_effort or 'configured'})" if codex
            else settings.get("model", {}).get("model"),
            "agent": "codex" if codex else "api", "perception_given_to_agent": not (codex and args.blind),
            "perception_form": "none" if codex and args.blind else "raw layers" if codex and args.raw else "fused evidence",
            "review": {"model": args.judge_model, "effort": args.judge_effort, "rounds": args.judge_rounds}
            if codex and args.judge else None,
            "objects": "complete meshes (SAM 3D Objects, make_objects)" if codex and args.sam3d else "authored",
            "budget": {"minutes": args.codex_minutes} if codex
            else settings.get("budget", {}), "sandbox_image": box.image, "task": config.get("task")}
    if isolation:
        plan["problems_sandbox"] = isolation
    started = _start(args, needs_model=not (scripted or codex), extra_plan=plan)
    if started is None:
        return 1 if isolation else 0
    if isolation:
        raise SystemExit(isolation)
    config, config_path, video_path, out, plan, log = started
    workspace = tools.Workspace(out, perception.Perception(out, video_path, config, log), box, settings)
    workspace.prepare_evidence()      # all local perception first; the model starts with the full evidence
    if codex:
        from codify import codex_agent
        outcome = codex_agent.run_codex(workspace, settings, video_path, config, task=config.get("task"), log=log,
                                        minutes=args.codex_minutes, model=args.codex_model, effort=args.codex_effort,
                                        blind=args.blind, raw=args.raw,
                                        judge={"model": args.judge_model, "effort": args.judge_effort,
                                               "rounds": args.judge_rounds} if args.judge else None,
                                        # the machine's SAM 3D setup under the run's (a config written before
                                        # the section existed carries only --gpu there)
                                        objects3d={**load_config(MACHINE_CONFIG)[0].get("objects3d", {}),
                                                   **config.get("objects3d", {}), **({"workers": args.sam3d_workers}
                                                   if args.sam3d_workers else {})} if args.sam3d else None)
    else:
        if scripted:
            spec = importlib.util.spec_from_file_location("codify_script", args.script)
            script = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(script)
            client = script.make_model(workspace)
        else:
            client = agent.api_model(settings)
        budget = agent.Budget(settings.get("budget", {}), paid=not scripted)
        outcome = agent.run_agent(workspace, client, settings, budget, log, task=config.get("task"))
    report = agent.finalize(workspace, outcome, settings, config.get("handoff"), log)
    stages = {p.name.replace(".stage.json", ""): contract.read_json(p) for p in sorted((out / "work").glob("*.stage.json"))}
    models = {name: contract.read_json(out / "work" / name / "model_info.json")
              for name in ("geometry", "vocabulary", *[s for s in stages if s.startswith("tracks")])
              if (out / "work" / name / "model_info.json").is_file()}
    blender_seconds = sum(r["exec"]["seconds"] for r in workspace.revisions.values())
    contract.write_json(out / "run.json", {
        **plan, "started_utc": contract.read_json(out / "run.json")["started_utc"], "finished_utc": now(),
        "status": report["status"], "selected_revision": report.get("selected_revision"),
        "evidence": workspace.evidence.name,
        "inputs": {"video": {"path": str(video_path), "sha256": plan["source_video_sha256"]},
                   "config": {"path": str(config_path), "sha256": contract.sha256(config_path)}},
        "perception_stages": {k: {f: v for f, v in r.items() if f != "result"} for k, r in stages.items()},
        "models": models, "comparison_frames": workspace.comparison,
        "revisions": {rid: {"executed": r["executed"], "eligible": workspace.eligible(rid)[0],
                            "evidence_version": r["evidence_version"], "code_sha256": r["code_sha256"],
                            "blend_sha256": r.get("blend_sha256")} for rid, r in workspace.revisions.items()},
        "costs": {**report["budget"], "local_gpu_seconds": {k: m.get("wall_seconds") for k, m in models.items()},
                  "sandbox_exec_seconds": round(blender_seconds, 1)},
        "code_sha256": {p.name: contract.sha256(p) for p in sorted((CODE / "codify").glob("*.py"))}})
    print(json.dumps({k: report.get(k) for k in ("status", "selected_revision", "final_check_pass", "reasons")}, indent=2))
    return 0 if report["status"] == "success" else 2


def preview(args):
    """Report-only video of a finished run: the input beside the rendered reconstruction over the input's duration."""
    from codify import preview as preview_video, sandbox
    run = Path(args.run).resolve()
    settings = load_config(MACHINE_CONFIG)[0]["agent"]          # the sandbox is the machine's, whatever the run was
    box = sandbox.Sandbox(settings["blender_dir"], cpus=settings.get("sandbox_cpus", 8),
                          memory=settings.get("sandbox_memory", "8g"), timeout=settings.get("sandbox_timeout", 900),
                          room=bool(settings.get("fast_room")))
    info = preview_video.preview_video(run, box, fps=args.fps, size=args.size, max_frames=args.max_frames)
    print(json.dumps(info, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func, help_text in (("perceive", perceive, "local perception -> evidence version (resumable)"),
                                  ("codify", codify, "perceive, then the model writes, builds and revises the scene")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--sample", help="a diagnostic-bench sample id (T1_... to T7_...): its input and task config")
        p.add_argument("--video", help="a video file or a single image (with --config, instead of --sample)")
        p.add_argument("--config", help="the example's config (includes the machine config)")
        p.add_argument("--output", required=True, help="run directory (a perceive run can be continued by codify)")
        p.add_argument("--reuse", help="another run whose cached perception stages (work/) are copied first")
        p.add_argument("--gpu", type=int, help="CUDA device for the perception workers (default: the machine config's)")
        p.add_argument("--prepare-only", action="store_true", help="print the plan and check the setup; no inference")
        if name == "codify":
            p.add_argument("--script", help="scripted stand-in model (tests/scripted_baseline.py); no API calls")
            p.add_argument("--max-usd", type=float, help="cap this run's model spend below the config's")
            p.add_argument("--agent", choices=["api", "codex"], default="api",
                           help="who writes the scene: the API model loop (default) or the Codex CLI on the user's "
                                "subscription, with the same tools over MCP (codify/codex_agent.py)")
            p.add_argument("--codex-minutes", type=float, default=35, help="Codex's time budget (minutes)")
            p.add_argument("--codex-model", help="Codex model (default: ~/.codex/config.toml's)")
            p.add_argument("--codex-effort", help="Codex reasoning effort (default: ~/.codex/config.toml's)")
            p.add_argument("--blind", action="store_true",
                           help="with --agent codex: no perception for the agent (frames, timing and given cameras only)")
            p.add_argument("--raw", action="store_true",
                           help="with --agent codex: the perception's raw layers for the agent (cameras, depth, SAM 3 "
                                "masks with track ids) instead of its fused objects")
            p.add_argument("--judge", action="store_true",
                           help="with --agent codex: every build carries the video check, and submit goes through an "
                                "independent Codex reviewer (codify/judge.py)")
            p.add_argument("--judge-model", default="gpt-6.1-terra", help="the reviewer's Codex model")
            p.add_argument("--judge-effort", default="xhigh", help="the reviewer's reasoning effort")
            p.add_argument("--judge-rounds", type=int, default=3, help="reviews per run (the last always accepts)")
            p.add_argument("--sam3d", action="store_true",
                           help="with --agent codex: the make_objects tool (SAM 3D Objects: complete meshes of the "
                                "objects from the frames; the machine config's objects3d section)")
            p.add_argument("--sam3d-workers", type=int,
                           help="SAM 3D workers that make objects at once (about 20 GB of GPU memory each)")
        p.set_defaults(func=func)
    p = sub.add_parser("preview", help="report-only video of a finished run: input | reconstruction over the input "
                                       "(a single image: then its movable parts through their joints)")
    p.add_argument("--run", required=True)
    p.add_argument("--fps", type=float, default=10.0, help="at most this many rendered frames per second of input")
    p.add_argument("--size", type=int, default=512, help="render size (longest side)")
    p.add_argument("--max-frames", type=int, default=600, help="cap on rendered input frames (long videos stride)")
    p.set_defaults(func=preview)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
