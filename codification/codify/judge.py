"""The independent reviewer of a submitted revision: a separate, read-only Codex session (gpt-6.1-terra by default) that
sees only the revision's video consistency report and its evidence cards, never the builder's reasoning, and returns
what to change: {"verdict": "pass" | "revise", "fixes": [...], "unreliable": [...]}.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from . import contract

PROMPT = """You review a 3D scene that another agent built in Blender from {kind}. You did not build it and you do not
see how it was built: judge only from the evidence below.

Evidence. The scene was seen from the input's own camera on {n} frame(s) spread over the whole input and compared,
object by object, with what a perception model observed there (SAM 3 masks and measured depth):
- score {score}: the share of the observed object pixels that the matching scene object covers (1 = every observed
  silhouette matched).
- objects, worst first: iou = the object's silhouette against its observed mask (1 = identical); depth_error = built
  surface minus measured depth, relative (+ = too far from the camera, - = too near; measured depth is approximate).
- seen_not_built: observed masks that no scene object explains (a missing or misplaced object, or a mask that is not
  an object).
- built_not_seen: scene objects where the input shows nothing.
The attached images are cards for these objects, in this order: {cards}. Each is a crop of a source frame with the
observed mask outlined in green and the built object's silhouette in magenta.

A mask can be wrong: it can cover a patch of floor or wall, a shadow, a reflection, part of an object or several
objects. When a card or the numbers show that, list it under unreliable instead of asking for a change.

Task. Decide what the builder must change. For each problem that is the scene's: the object id, what is wrong as the
evidence shows it (too small, shifted, too far, wrong shape or orientation, missing, should not be there), and the
change with its direction and rough amount. Order by how much each changes what a viewer understands of the scene
(positions, sizes, counts, orientations). At most {max_fixes} fixes. Verdict "pass" when no remaining problem would
change positions, sizes, counts or orientations a viewer notices; otherwise "revise".

Answer with only this JSON:
{{"verdict": "pass" or "revise", "fixes": [{{"object": "...", "problem": "...", "change": "..."}}],
  "unreliable": [{{"item": "...", "reason": "..."}}]}}

Report:
{report}
"""


def _report_text(report, rows=15):
    return json.dumps({"score": report["score"], "frames": report["frames"],
                       "objects": [{k: o[k] for k in ("id", "category", "iou", "depth_error", "frames_visible")}
                                   for o in report["objects"][:rows]],
                       "seen_not_built": report["missing"][:10], "built_not_seen": report["unseen"][:10]}, indent=1)


def _verdict(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reviewer's answer")
    answer = json.loads(text[start:end + 1])
    if answer.get("verdict") not in ("pass", "revise"):
        raise ValueError(f"verdict {answer.get('verdict')!r}")
    return {"verdict": answer["verdict"], "fixes": list(answer.get("fixes") or []),
            "unreliable": list(answer.get("unreliable") or [])}


def review(workspace, rid, report, settings, log=print):
    """Run the reviewer on revision rid's report and cards; returns the verdict with its folder, time and tokens.
    settings: {"model", "effort", "minutes", "max_fixes"}."""
    from .codex_agent import usage_of
    root = workspace.run / "judge"
    root.mkdir(exist_ok=True)
    folder = root / f"{rid}_{len(list(root.glob(f'{rid}_*'))) + 1}"
    folder.mkdir()
    kind = "a video" if len(workspace.timeline) > 1 else "an image"
    cards = report.get("cards", [])
    prompt = PROMPT.format(kind=kind, n=len(report["frames"]), score=report["score"],
                           cards=", ".join(c["id"] for c in cards) or "none",
                           max_fixes=settings.get("max_fixes", 8), report=_report_text(report))
    (folder / "prompt.txt").write_text(prompt)
    command = ["codex", "exec", "--json", "--skip-git-repo-check", "-C", str(folder), "-s", "read-only",
               "-m", settings.get("model", "gpt-6.1-terra"),
               "-c", f'model_reasoning_effort="{settings.get("effort", "xhigh")}"',
               "-o", str(folder / "last_message.txt")]
    for card in cards:
        command += ["-i", card["path"]]
    command += ["-"]
    started = time.time()
    with open(folder / "events.jsonl", "w") as events, open(folder / "stderr.log", "w") as stderr:
        process = subprocess.Popen(command, cwd=folder, stdin=subprocess.PIPE, stdout=events, stderr=stderr, text=True,
                                   start_new_session=True)
        process.stdin.write(prompt)
        process.stdin.close()
        try:
            process.wait(timeout=60 * settings.get("minutes", 10))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    usage, _ = usage_of(folder / "events.jsonl")
    out = {"revision": rid, "folder": str(folder), "minutes": round((time.time() - started) / 60, 2), "tokens": usage,
           "model": settings.get("model", "gpt-6.1-terra"), "score": report["score"]}
    try:
        out.update(_verdict((folder / "last_message.txt").read_text()))
    except (OSError, ValueError) as exc:
        out.update(verdict="error", error=str(exc)[:300], fixes=[], unreliable=[])
    contract.write_json(folder / "verdict.json", out)
    log(f"judge {rid}: {out['verdict']} (score {report['score']}, {len(out['fixes'])} fixes, {out['minutes']} min)")
    return out
