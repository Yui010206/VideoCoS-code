#!/usr/bin/env python
"""One prompt for every case: prompt_template_v4.txt (the strict prompt of the final runs), written once in the
structured format of the official H3 full-reference guide (guides/), with only the case's goal filled in. The real
frame (<Picture 1>) and the plan video (<Video 1>) carry everything else that is case specific; the T6 constraints are
the same for all 110 cases and are written into the template (one hand, only visible objects, physically plausible).

    python t6_h3/make_prompts.py [--ids all|pilot|a,b]      # writes prompts_v4/T6/<id>.txt
"""
import argparse
import json
from pathlib import Path

from common import INPUTS, PROMPTS, chosen

TEMPLATE = Path(__file__).resolve().parent / "prompt_template_v4.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="all")
    ap.add_argument("--template", default=str(TEMPLATE), help="template file (default: the strict v4 prompt)")
    ap.add_argument("--out", default=str(PROMPTS), help="output folder (default: prompts_v4/T6)")
    a = ap.parse_args()
    template = Path(a.template).read_text()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = chosen(a.ids)
    for case_id in ids:
        goal = json.loads((INPUTS / case_id / "case.json").read_text())["goal"].strip()
        goal = goal if goal.endswith((".", "!", "?")) else goal + "."
        (out / f"{case_id}.txt").write_text(template.replace("{goal}", goal))
    print(f"wrote {len(ids)} prompts to {out}")


if __name__ == "__main__":
    main()
