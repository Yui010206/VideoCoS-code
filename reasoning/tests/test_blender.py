"""Live tests in the Docker sandbox: the task contract passes when it should and fails when it should.

    python -m pytest -q tests/test_blender.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.session import BlenderSession
from harness.tasks import load_tasks
from harness.tools import UpdateTools

# the demo examples (demo/examples after scripts/download_demo.sh), or $VIDEOCOS_EXAMPLES
EXAMPLES = Path(os.environ.get("VIDEOCOS_EXAMPLES", Path(__file__).resolve().parents[2] / "demo" / "examples"))
pytestmark = pytest.mark.skipif(not shutil.which("docker") or not EXAMPLES.is_dir()
                                or subprocess.run(["docker", "image", "inspect", "videocos-blender-runtime:1"],
                                                  capture_output=True).returncode != 0,
                                reason="needs Docker, the sandbox image and the examples")


def _task(family, example):
    return next(t for t in load_tasks(EXAMPLES, [family]) if t.example == example)


def _run(tools, code):
    result = tools.execute("run_python", {"code": code})
    assert result.get("executed"), result
    return result


def test_t5_contract(tmp_path):
    task = _task("T5", "scn02900")
    with BlenderSession(task.scene_blend, tmp_path) as session:
        tools = UpdateTools(session, task)
        assert tools.execute("submit_update", {"summary": "nothing"})["ok"] is False
        result = _run(tools, f"S.set_camera(**{task.public['target_camera']!r})")
        assert result["changes"]["scene"]["camera"]
        accepted = tools.execute("submit_update", {"summary": "camera"})
        assert accepted["ok"] is True and (tmp_path / "final.blend").is_file()


def test_t6_contract_and_frame_one(tmp_path):
    task = _task("T6", "T6_blue_pouch_eyedrops")
    with BlenderSession(task.scene_blend, tmp_path) as session:
        tools = UpdateTools(session, task)
        first = tools.context()["scene_objects"][0]["id"]
        failures = tools.execute("submit_update", {"summary": "x"})["failures"]
        assert any("shorter than 3 s" in f for f in failures)
        _run(tools, f"S.timeline(100)\np = S.pose('{first}', 1)['location']\nS.key('{first}', 1, location=p)\n"
                    f"S.key('{first}', 80, location=[p[0] + 0.2, p[1], p[2]])")
        motion = tools.execute("track_motion", {"ids": [first], "frames": [1, 80]})
        assert motion["ok"] and motion["tracks"][0]["displacement_m"] == pytest.approx(0.2, abs=1e-3)
        assert tools.execute("submit_update", {"summary": "moved"})["ok"] is True
        _run(tools, f"q = mathutils.Quaternion(S.pose('{first}', 1)['quaternion_wxyz'])\n"
                    f"S.key('{first}', 1, quaternion_wxyz=q @ mathutils.Quaternion((0, 0, 1), math.radians(30)))")
        failures = tools.execute("submit_update", {"summary": "frame 1 turned"})["failures"]
        assert any("turned 30.0 deg at frame 1" in f for f in failures)
        _run(tools, f"p = S.pose('{first}', 1)['location']\nS.key('{first}', 1, location=[p[0], p[1], p[2] + 0.05])")
        failures = tools.execute("submit_update", {"summary": "frame 1 moved"})["failures"]
        assert any("frame 1 must stay" in f for f in failures)


def test_t7_contract_and_reset(tmp_path):
    task = _task("T7", "T7_0050_center_dominos-with-space")
    with BlenderSession(task.scene_blend, tmp_path) as session:
        last = session.frame_range[1]
        task.spec.update(observed_end=last, target_end=last + 150)
        tools = UpdateTools(session, task)
        assert tools.execute("submit_update", {"summary": "x"})["ok"] is False
        _run(tools, "S.timeline(240)\nprint(S.bake())")
        assert tools.execute("submit_update", {"summary": "baked"})["ok"] is True
        first = tools.context()["scene_objects"][0]["id"]
        _run(tools, f"o = S.obj('{first}')\no.location.z += 0.5\no.keyframe_insert('location', frame=10)")
        failures = tools.execute("submit_update", {"summary": "moved"})["failures"]
        assert any("observed frames" in f for f in failures)
        assert tools.execute("reset_scene", {})["ok"] is True
        assert session.frame_range == [1, last]


def test_frame_move_is_not_an_edit(tmp_path):
    task = _task("T7", "T7_0050_center_dominos-with-space")
    with BlenderSession(task.scene_blend, tmp_path) as session:
        tools = UpdateTools(session, task)
        only_frame = _run(tools, "bpy.context.scene.frame_set(90)")["changes"]
        assert set(only_frame) == {"frame_now", "frame_change"} and only_frame["frame_now"] == 90
        assert "from 1 to 90" in only_frame["frame_change"]
        stays = _run(tools, "print(bpy.context.scene.frame_current)")
        assert stays["printed"].strip() == "90" and set(stays["changes"]) == {"frame_now"}
        edit = _run(tools, "bpy.context.scene.frame_set(1)\nS.obj('m10').location.x += 0.1")["changes"]
        assert [row["object"] for row in edit["moved"]] == ["m10 (Power cable)"]
        assert edit["moved"][0]["shift_m"] == pytest.approx(0.1, abs=1e-3) and edit["frame_now"] == 1


def test_changes_are_what_renders(tmp_path):
    task = _task("T7", "T7_0050_center_dominos-with-space")
    with BlenderSession(task.scene_blend, tmp_path) as session:
        tools = UpdateTools(session, task)
        direct = _run(tools, "S.obj('m10').location.x += 0.3")["changes"]      # unanimated, no keyframe
        assert [row["object"] for row in direct["moved"]] == ["m10 (Power cable)"]
        assert direct["moved"][0]["shift_m"] == pytest.approx(0.3, abs=1e-3)
        unkeyed = _run(tools, "S.obj('m9').location.x += 0.3")["changes"]      # animated, no keyframe: not rendered
        assert "moved" not in unkeyed
        keyed = _run(tools, "o = S.obj('m9')\no.location.x += 0.05\no.keyframe_insert('location', frame=40)")
        assert "moved" not in keyed["changes"]                                  # frame 1 itself is unchanged
        at40 = _run(tools, "print(round(S.pose('m9', 40)['location'][0] - S.pose('m9', 1)['location'][0], 4))")
        assert float(at40["printed"]) == pytest.approx(0.05, abs=1e-3)       # the keyed value is at frame 40


def test_crash_is_restored(tmp_path):
    task = _task("T7", "T7_0050_center_dominos-with-space")
    with BlenderSession(task.scene_blend, tmp_path) as session:
        tools = UpdateTools(session, task)
        _run(tools, "S.timeline(200)")
        result = tools.execute("run_python", {"code": "import os\nos._exit(3)"})
        assert result["ok"] is False and "restored" in result["error"]
        assert session.frame_range == [1, 200]            # the earlier edit was replayed
