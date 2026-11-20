#!/usr/bin/env python3
"""Run the reasoning harness over an examples directory.

    GEMINI_API_KEY=... python run.py --examples EXAMPLES --out runs/NAME --max-usd 10
    python run.py --examples EXAMPLES --out runs/NAME --families T7 --fake      # no model, no cost

One task per bench record. A query task (T1-T4) is one VSI question answered through the five query
tools; an update task (T5-T7) edits the live scene and submits it, then (with --render) the
submission is rendered. Every task writes its own folder; a finished task is skipped on rerun.
Scoring (T1-T4) runs after all tasks, from the saved answers.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
import traceback
from pathlib import Path

from harness.agent import UPDATE_WORDING, atomic_json, run_agent
from harness.llm import BudgetExceeded, ChatClient, MeteredClient
from harness.prompts import SYSTEM, SYSTEM_UPDATE
from harness.render import render_submission
from harness.score import score_query_tasks
from harness.session import BlenderSession
from harness.tasks import load_bench_tasks, load_tasks
from harness.tools import QueryTools, UpdateTools

MODELS = {
    "gemini-3.8-flash": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                         "api_style": "chat", "key_env": "GEMINI_API_KEY"},
    # As in the reported Luna VSI runs (VideoCoS outputs/lunafull-rest): the Responses API at OpenAI.
    "gpt-5.6-luna": {"base_url": "https://api.openai.com/v1", "api_style": "responses", "key_env": "OPENAI_API_KEY"},
    # 2026-11-07: Qwen3.6-27B served by vLLM on this machine (served name qwen3.6-27b; thinking on, the model's own sampling
    # defaults, as in the earlier Qwen3.6-27B VSI runs). Any key works (QWEN_LOCAL_KEY=EMPTY); local, so no cost.
    # A long thinking turn (4,096 tokens at ~15-26 tokens/s on these cards) outlasts the 120 s request timeout: 900 s here.
    "qwen3.6-27b": {"base_url": "http://127.0.0.1:8711/v1", "api_style": "chat", "key_env": "QWEN_LOCAL_KEY", "local": True,
                    "timeout": 900},
    # 2026-11-08: Qwen3.8-27B (pretrained_models/Qwen3.8-27B), served the same way as Qwen3.6-27B on port 8712.
    "qwen3.8-27b": {"base_url": "http://127.0.0.1:8712/v1", "api_style": "chat", "key_env": "QWEN_LOCAL_KEY", "local": True,
                    "timeout": 900},
    # The same Qwen3.8-27B served once more on GPU 0 (port 8713) for the BVB Astra scenes.
    "qwen3.8-27b-gpu0": {"base_url": "http://127.0.0.1:8713/v1", "api_style": "chat", "key_env": "QWEN_LOCAL_KEY", "local": True,
                         "timeout": 900},
}
# The reported VSI runs: 6 turns, 18 tool calls, 4,096 output tokens (8,192 on the forced submission).
QUERY_LIMITS = {"max_turns": 6, "max_tool_calls": 18, "max_output_tokens": 4096, "max_final_output_tokens": 8192,
                "timeout": 120, "unanswered_fallback": True}   # 2026-11-07: harness/agent.py fallback_turn
# Update tasks write code: more turns, longer outputs, thinking capped, truncated turns recovered.
UPDATE_LIMITS = {"max_turns": 24, "max_tool_calls": 40, "max_output_tokens": 16384, "max_final_output_tokens": 8192,
                 "timeout": 300, "reasoning_effort": "medium", "max_length_recoveries": 2,
                 "recovery_output_tokens": 8192}
# T5 only turns the instruction (the target camera) into code: a few turns, the last one reserved for submitting.
T5_LIMITS = {**UPDATE_LIMITS, "max_turns": 4, "max_tool_calls": 6}
UPDATE_QUESTION = ("Complete the update task described in the context (brief and task inputs). Read scene_code "
                   "and scene_objects first. Edit the live scene with run_python in small steps, check each step "
                   "with the query tools and track_motion, and finish with submit_update.")


def run_task(task, out, api, meter, args):
    folder = out / task.family / task.id
    if (folder / "result.json").is_file():
        return json.loads((folder / "result.json").read_text())
    folder.mkdir(parents=True, exist_ok=True)
    started, row = time.time(), {"id": task.id, "family": task.family, "kind": task.kind, "example": task.example}
    try:
        with BlenderSession(task.scene_blend, folder / "work", docker=not args.no_docker) as session:
            if task.family == "T7":
                first, last = session.frame_range
                fps = float(task.public["fps"])
                task.spec.update(observed_end=last, target_end=last + round(task.spec["future_seconds"] * fps))
            tools = QueryTools(session, task) if task.kind == "query" else UpdateTools(session, task)
            context = tools.context()
            atomic_json(folder / "context.json", context)
            limits = QUERY_LIMITS if task.kind == "query" else T5_LIMITS if task.family == "T5" else UPDATE_LIMITS
            question = task.public["question"] if task.kind == "query" else UPDATE_QUESTION
            result = run_agent(question, context, tools.tools(), tools.execute, {**api, **limits},
                               folder / "agent", client=meter.for_task(task.id, task.kind), final_submission_turn=True,
                               system=SYSTEM if task.kind == "query" else SYSTEM_UPDATE,
                               wording=None if task.kind == "query" else UPDATE_WORDING)
        if result.get("error_kind") == "BudgetExceeded":
            # The loop turns every exception into status "failed"; a cap stop must stay resumable (no result.json).
            raise BudgetExceeded(result.get("error", "run cap reached"))
        submission = result.get("submission") or {}
        row.update(status=result["status"], turns=result.get("model_turns"), tool_calls=result.get("tool_calls"))
        if task.kind == "query":
            row.update(answer=result.get("answer", "") if result["status"] == "completed" else "",
                       raw_answer=result.get("raw_answer", ""), question_type=task.public["question_type"])
        else:
            row.update(summary=submission.get("summary"), limitations=submission.get("limitations"),
                       blocks=len(tools.calls), final_blend=submission.get("final_blend"))
            if submission and args.render:
                row["render"] = render_submission(task, submission["final_blend"], folder, cpus=args.render_cpus,
                                                  docker=not args.no_docker)
        if result.get("error"):
            row["error"] = result["error"]
    except BudgetExceeded as exc:
        row.update(status="budget_exceeded", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - one task must not end the run
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-2000:])
    row["spec"] = task.spec
    row["seconds"] = round(time.time() - started, 1)
    row["spent_usd_so_far"] = round(meter.spent, 4)
    if row.get("status") != "budget_exceeded":
        atomic_json(folder / "result.json", row)
    return row


def render_all(tasks, out, args):
    """Render the submitted updates of an earlier run, in parallel; results go into each result.json."""
    todo = []
    for task in tasks:
        path = out / task.family / task.id / "result.json"
        if task.kind == "update" and path.is_file():
            row = json.loads(path.read_text())
            if row.get("final_blend") and not row.get("render"):
                todo.append((task, row, path))
    print(f"render: {len(todo)} submissions, {args.jobs} at a time x {args.render_cpus} cpus", flush=True)

    def one(item):
        task, row, path = item
        task.spec.update(row.get("spec") or {})          # T7: the frames filled in when the task ran
        started = time.time()
        try:
            row["render"] = render_submission(task, row["final_blend"], path.parent, cpus=args.render_cpus,
                                              docker=not args.no_docker)
        except Exception as exc:  # noqa: BLE001
            row["render_error"] = f"{type(exc).__name__}: {exc}"
        row["render_seconds"] = round(time.time() - started, 1)
        atomic_json(path, row)
        return row
    with cf.ThreadPoolExecutor(args.jobs) as pool:
        for row in pool.map(one, todo):
            print(f"{row['family']} {row['id']} {row.get('render', {}).get('written')} frames "
                  f"{row.get('render_error', '')} {row['render_seconds']}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--examples", help="<examples>/<family>/<name>/{scene.blend, scene.py, bench_records.json}")
    source.add_argument("--scenes", help="codify outputs <scenes>/<family>/<id>/codify/final/ (read only), "
                                         "joined with the frozen bench manifests; update families only")
    parser.add_argument("--out", required=True)
    parser.add_argument("--families", default="", help="comma-separated, e.g. T1,T2,T3,T4 or T7")
    parser.add_argument("--ids", default="", help="comma-separated bench record ids")
    parser.add_argument("--model", default="gemini-3.8-flash", choices=sorted(MODELS))
    parser.add_argument("--max-usd", type=float, default=0.0, help="hard cap for this run (required unless --fake)")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--render", action="store_true", help="render submitted updates to video")
    parser.add_argument("--render-only", action="store_true",
                        help="no model: render every submitted update in --out that has no video yet")
    parser.add_argument("--render-cpus", type=int, default=16)
    parser.add_argument("--no-docker", action="store_true", help="run Blender without the sandbox (query tasks only)")
    parser.add_argument("--fake", action="store_true", help="scripted model for plumbing tests; no API, no cost")
    args = parser.parse_args()

    families = [f for f in args.families.split(",") if f] or None
    ids = [i for i in args.ids.split(",") if i] or None
    tasks = load_bench_tasks(args.scenes, families, ids) if args.scenes else load_tasks(args.examples, families, ids)
    if args.no_docker and any(t.kind == "update" for t in tasks):
        raise SystemExit("update tasks run model code and need the Docker sandbox")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        return render_all(tasks, out, args)

    if args.fake:
        from harness.fake import FakeClient
        api, local = {"model": "fake", "api_key": "", "base_url": "http://fake"}, True
        clients = {"query": FakeClient(QUERY_LIMITS), "update": FakeClient(UPDATE_LIMITS)}
    else:
        spec = MODELS[args.model]
        key = os.environ.get(spec["key_env"])
        if not key:
            raise SystemExit(f"{spec['key_env']} is not set")
        if args.max_usd <= 0:
            raise SystemExit("--max-usd is required for a paid run")
        api = {"model": args.model, "base_url": spec["base_url"], "api_style": spec["api_style"], "api_key": key}
        slow = {"timeout": spec["timeout"]} if spec.get("timeout") else {}     # a model's own request timeout (local Qwen)
        clients = {"query": ChatClient({**api, **QUERY_LIMITS, **slow}), "update": ChatClient({**api, **UPDATE_LIMITS, **slow})}
        local = bool(spec.get("local"))
    meter = MeteredClient(clients, args.max_usd if not args.fake else 1e9, out / "costs.jsonl", local=local)
    config = {"examples": args.examples and str(Path(args.examples).resolve()),
              "scenes": args.scenes and str(Path(args.scenes).resolve()), "model": api["model"], "families": families,
              "ids": ids, "query_limits": QUERY_LIMITS, "update_limits": UPDATE_LIMITS, "t5_limits": T5_LIMITS,
              "cap_usd": args.max_usd,
              "render": args.render, "tasks": len(tasks)}
    atomic_json(out / "config.json", config)
    print(f"{len(tasks)} tasks -> {out} | model {api['model']} | cap ${args.max_usd} | jobs {args.jobs}", flush=True)

    rows, started = [], time.time()
    with cf.ThreadPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(run_task, task, out, api, meter, args): task for task in tasks}
        for future in cf.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"[{len(rows)}/{len(tasks)}] {row['family']} {row['id']} {row.get('status')} "
                  f"turns={row.get('turns')} calls={row.get('tool_calls')} | ${meter.spent:.3f} | {row.get('seconds')}s",
                  flush=True)

    by_id = {t.id: t for t in tasks}
    query_rows = [{"id": r["id"], "family": r["family"], "question_type": r.get("question_type"),
                   "answer": r.get("answer", ""), "ground_truth": by_id[r["id"]].eval_only["ground_truth"]}
                  for r in rows if r["kind"] == "query" and r.get("status") != "budget_exceeded"]
    summary = {"tasks": len(rows), "status": {}, "spent_usd": round(meter.spent, 4), "requests": meter.requests,
               "minutes": round((time.time() - started) / 60, 1),
               "query_scores": score_query_tasks(query_rows) if query_rows else None,
               "updates": [{k: r.get(k) for k in ("id", "family", "status", "blocks", "summary", "render")}
                           for r in rows if r["kind"] == "update"]}
    for r in rows:
        summary["status"][r.get("status")] = summary["status"].get(r.get("status"), 0) + 1
    atomic_json(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("tasks", "status", "spent_usd", "requests", "minutes")}, indent=1))
    if summary["query_scores"]:
        q = summary["query_scores"]
        print("T1-T4:", q["mean_percent"], q["by_family_percent"])


if __name__ == "__main__":
    main()
