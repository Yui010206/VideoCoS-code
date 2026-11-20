"""Instance check by a local vision-language model (run in the vllm env), for codify.instances.check_instances.

Each item is one 3D instance seen in a few frames, outlined in red. The model answers in JSON: what it is (a short
name), whether it is a whole object, a part of one, part of the room's structure, or nothing, and what it is a part of.

    python vlm_check_worker.py <job.json> <answers.json>
job.json: {"weights": <model folder>, "items": [{"id": ..., "images": [paths], "hints": [names]}]}
Runs the model in vLLM (models._vllm_engine): every item in one batch, greedy, thinking off.
"""
import json
import os
import re
import sys
import time

from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import _vllm_answers, _vllm_engine  # noqa: E402

PROMPT = (
    "These {n} images are frames from one video of a room. The same thing is outlined in red in each of them. "
    "{hints}\n"
    "Answer with one JSON object and nothing else, with these keys:\n"
    '"name": a short lowercase English noun phrase for what is outlined (e.g. "armchair", "throw pillow", '
    '"trash can");\n'
    '"kind": one of "object", "part", "structure", "none":\n'
    '  "object" = a whole, separate physical thing a person would count on its own, fixed ones included (a chair, a '
    "pillow lying on a bed or sofa, a lamp, a towel, a backpack, a book, a door, a window, a curtain, a mirror, a "
    "bathtub, a toilet, a sink, a cabinet, a radiator);\n"
    '  "part" = a piece of a larger object that is not counted on its own (a sofa\'s seat or back cushion, a '
    "lampshade, a bed skirt, a headboard, a drawer, a door knob, a faucet on a sink);\n"
    '  "structure" = a bare surface of the room itself (a wall, the floor, the ceiling, a baseboard, a tiled wall or '
    "floor area, wall-to-wall carpet); doors, windows and fixtures are objects, not structure;\n"
    '  "none" = no single physical thing: a shadow, a reflection, light, an empty area, or an outline that holds '
    'several different things at once (e.g. a stretch of wall with a headboard, a lamp and a curtain in it);\n'
    '"part_of": if kind is "part", the whole object it belongs to (e.g. "sofa"), else null.')


def parse(text):
    match = re.search(r"\{.*\}", text, re.S)
    try:
        answer = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        answer = {}
    kind = str(answer.get("kind", "")).strip().lower()
    return {"name": str(answer.get("name", "")).strip().lower()[:60] or None,
            "kind": kind if kind in ("object", "part", "structure", "none") else None,
            "part_of": (str(answer["part_of"]).strip().lower()[:60] if answer.get("part_of") else None), "raw": text[:400]}


def main():
    job = json.load(open(sys.argv[1]))
    started = time.time()
    processor = AutoProcessor.from_pretrained(job["weights"])
    requests = []
    for item in job["items"]:
        hints = ", ".join(dict.fromkeys(item.get("hints") or []))
        hints = f"Earlier detections called it: {hints}." if hints else ""
        requests.append(([Image.open(p).convert("RGB") for p in item["images"]],
                         PROMPT.format(n=len(item["images"]), hints=hints)))
    if os.environ.get("CODIFY_QWEN_URL"):      # the lane's sleeping server: no model load here
        import qwen
        texts, timing = qwen.answers([(item["images"], text) for item, (_, text) in zip(job["items"], requests)], 80)
        loaded = time.time()
    else:
        llm = _vllm_engine(job["weights"], max_images=max([len(r[0]) for r in requests] or [1]))
        loaded = time.time()
        texts = _vllm_answers(llm, processor, requests, 80)
    answers = {str(item["id"]): parse(text) for item, text in zip(job["items"], texts)}
    json.dump({"answers": answers, "seconds": round(time.time() - started, 1), "load_seconds": round(loaded - started, 1),
               "items": len(job["items"]), "weights": job["weights"], "engine": "vllm"}, open(sys.argv[2], "w"), indent=1)

if __name__ == "__main__":
    main()
