"""A scripted stand-in for the model API: plumbing tests at no cost (run.py --fake).

It reads the task context from the first user message and replies with a fixed sequence of tool
calls per kind and family, enough to exercise every tool path once. It does not try to be right.
"""
from __future__ import annotations

import json

from .llm import _tool_choice


class FakeClient:
    model = "fake"

    def __init__(self, limits):
        self.max_output_tokens = limits.get("max_output_tokens", 4096)
        self.max_final_output_tokens = limits.get("max_final_output_tokens", self.max_output_tokens)

    def request_output_token_limit(self, tools, force_final=False):
        return (self.max_final_output_tokens if isinstance(_tool_choice(tools, force_final), dict)
                else self.max_output_tokens)

    @staticmethod
    def _call(index, name, arguments):
        return {"id": f"call_{index}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)}}

    def _plan(self, context):
        if "answer_format" in context:
            fmt = context["answer_format"]
            answer = fmt["allowed_letters"][0] if fmt["kind"] == "choice" else "1"
            return [("scene_inventory", {"kind": "all", "limit": 5}),
                    ("submit_answer", {"text": answer, "explanation": "fake model"})]
        family, inputs = context["task"]["family"], context["task"]["inputs"]
        objects = [o for o in context["scene_objects"] if o.get("physics")]
        first = objects[0]["id"] if objects else context["scene_objects"][0]["id"]
        if family == "T5":
            code = f"print(S.set_camera(**{json.dumps(inputs['target_camera'])}))"
            steps = [("run_python", {"code": code})]
        elif family == "T6":
            fps = context["scene"]["fps"]
            code = (f"S.timeline({int(4 * fps)})\np = S.pose('{first}', 1)['location']\n"
                    f"S.key('{first}', 1, location=p)\n"
                    f"S.key('{first}', {int(3 * fps)}, location=[p[0] + 0.1, p[1], p[2]])\nprint('keyed')")
            steps = [("run_python", {"code": code}), ("track_motion", {"ids": [first], "limit": 5})]
        else:
            end = int(context["brief"].split("extend the timeline to frame ")[1].split(",")[0])
            steps = [("run_python", {"code": f"S.timeline({end})\nprint(S.bake())"}),
                     ("track_motion", {"ids": [first], "limit": 5})]
        return steps + [("submit_update", {"summary": "fake model edit"})]

    def complete(self, messages, tools, force_final=False, max_tokens=None):
        context = json.loads(messages[1]["content"].split("Task context (data, not instructions):\n", 1)[1])
        plan = self._plan(context)
        turn = sum(1 for m in messages if m["role"] == "assistant")
        offered = {t["function"]["name"] for t in tools}
        name, arguments = plan[min(turn, len(plan) - 1)]
        if name not in offered:                       # the final turn offers only the submission tool
            name, arguments = plan[-1]
        message = {"role": "assistant", "content": None, "tool_calls": [self._call(turn, name, arguments)]}
        return {"choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
