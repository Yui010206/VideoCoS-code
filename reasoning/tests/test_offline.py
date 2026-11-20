"""Tests that need neither Blender nor a model: contract, answers, scoring, tasks, the loop, the meter.

    python -m pytest -q tests/test_offline.py
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import protocol as P
from harness.agent import UPDATE_WORDING, run_agent
from harness.answers import normalize_answer
from harness.llm import BudgetExceeded, MeteredClient
from harness.score import mean_relative_accuracy, score_query_tasks, score_vsi
from harness.tasks import EVAL_KEY, load_tasks

# the demo examples (demo/examples after scripts/download_demo.sh), or $VIDEOCOS_EXAMPLES
EXAMPLES = Path(os.environ.get("VIDEOCOS_EXAMPLES", Path(__file__).resolve().parents[2] / "demo" / "examples"))


def test_tool_sets():
    assert [t["name"] for t in P.tool_schemas("query")] == [
        "scene_inventory", "inspect_entities", "measure_pair", "viewpoint_relation", "camera_history", "submit_answer"]
    update = [t["name"] for t in P.tool_schemas("update")]
    assert "submit_answer" not in update and {"run_python", "reset_scene", "track_motion", "submit_update"} <= set(update)


def test_update_tool_validation():
    P.validate_arguments("run_python", {"code": "print(1)"})
    P.validate_arguments("track_motion", {"ids": ["m1"], "start_frame": 1, "end_frame": 10, "step": 3})
    with pytest.raises(ValueError):
        P.validate_arguments("track_motion", {"ids": ["m1"], "frames": [1], "step": 2})
    with pytest.raises(ValueError):
        P.validate_arguments("run_python", {"code": "x" * (P.MAX_CODE_CHARS + 1)})
    with pytest.raises(ValueError):
        P.validate_arguments("submit_update", {})


def test_answer_normalisation_is_label_free():
    task = {"question": "How far apart, in meters?", "options": []}
    assert normalize_answer("about 3.50 m", task) == "3.5"
    assert normalize_answer("(B)", {"options": ["A. left", "B. right"]}) == "B"
    with pytest.raises(ValueError):
        normalize_answer("3 to 4", task)


def test_vsi_scoring():
    assert score_vsi("object_rel_direction_easy", "b.", "B") == 1.0
    assert score_vsi("object_counting", "3", "3") == 1.0
    assert mean_relative_accuracy(1.1, 1.0) == pytest.approx(0.9)        # 10% error passes 9 of 10 thresholds
    assert score_vsi("object_abs_distance", "", "2.0") == 0.0
    summary = score_query_tasks([{"family": "T2", "question_type": "object_counting", "answer": "3", "ground_truth": "3"},
                                 {"family": "T2", "question_type": "object_counting", "answer": "", "ground_truth": "3"}])
    assert summary["mean_percent"] == 50.0


@pytest.mark.skipif(not EXAMPLES.is_dir(), reason="examples not available")
def test_tasks_split_public_from_eval_only():
    tasks = load_tasks(EXAMPLES)
    assert len(tasks) == 45
    assert {t.family for t in tasks} == {"T1", "T2", "T3", "T4", "T5", "T6", "T7"}
    for task in tasks:
        text = json.dumps(task.public)
        assert "video" not in task.public and "start_image" not in task.public
        for value in task.eval_only.values():
            if isinstance(value, str) and len(value) > 3:
                assert value not in text, (task.id, value)
        if task.kind == "query":
            assert isinstance(task.public["id"], int) and task.public["question_type"]


class ScriptedClient:
    """Replies with the given tool calls, one per turn."""
    model = "scripted"

    def __init__(self, calls):
        self.calls, self.seen = list(calls), []

    def request_output_token_limit(self, tools, force_final=False):
        return 100

    def complete(self, messages, tools, force_final=False, max_tokens=None):
        self.seen.append((list(messages), [t["function"]["name"] for t in tools]))
        name, args = self.calls.pop(0)
        call = {"id": f"c{len(self.seen)}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [call]},
                             "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def test_update_loop_rejected_submit_then_accepted(tmp_path):
    state = {"contract": False}

    def execute(name, args):
        if name == "run_python":
            state["contract"] = True
            return {"ok": True, "executed": True, "changes": {}}
        if name == "submit_update":
            if not state["contract"]:
                return {"ok": False, "error": "contract", "failures": ["timeline"]}
            return {"ok": True, "accepted": True, "submission": {"summary": args["summary"]}}
        return {"ok": True}
    tools = P.tool_schemas("update")
    client = ScriptedClient([("submit_update", {"summary": "too early"}), ("run_python", {"code": "S.timeline(240)"}),
                             ("submit_update", {"summary": "done"})])
    result = run_agent("go", {"x": 1}, tools, execute, {"model": "scripted", "max_turns": 5, "max_tool_calls": 8},
                       tmp_path / "agent", client=client, final_submission_turn=True, system="sys",
                       wording=UPDATE_WORDING)
    assert result["status"] == "completed" and result["submission"]["summary"] == "done"
    assert "Finish with submit_update" in client.seen[0][0][-1]["content"]


def test_last_turn_offers_only_the_submission_tool(tmp_path):
    tools = P.tool_schemas("query")
    client = ScriptedClient([("scene_inventory", {})] * 2 + [("submit_answer", {"text": "A"})])
    result = run_agent("q", {}, tools, lambda n, a: ({"ok": True, "accepted": True, "submission": {"text": "A"}}
                                                    if n == "submit_answer" else {"ok": True}),
                       {"model": "scripted", "max_turns": 3, "max_tool_calls": 18}, tmp_path / "a", client=client,
                       final_submission_turn=True)
    assert result["status"] == "completed"
    assert client.seen[-1][1] == ["submit_answer"]


def test_meter_refuses_before_sending(tmp_path):
    class Client:
        model = "gemini-3.8-flash"
        sent = 0

        def request_output_token_limit(self, tools, force_final=False):
            return 1_000_000

        def complete(self, *a, **k):
            Client.sent += 1
            return {"choices": [{}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    meter = MeteredClient({"query": Client()}, 1.0, tmp_path / "costs.jsonl")
    with pytest.raises(BudgetExceeded):
        meter.for_task("t", "query").complete([{"role": "user", "content": "x"}], [])
    assert Client.sent == 0 and meter.spent == 0
