"""The modeling step done by an agentic coding CLI, OpenAI Codex (`codex exec`, on the user's subscription), in place
of the API loop of agent.py: the same opening material (frames, annotated frames, full evidence, task text), the same
instructions and helper library, the same tools over MCP (codify.mcp_server), the same check and the same final rule.
What changes is the agent: Codex's own loop, shell and file editing, its model and reasoning effort, and a time budget
instead of a request and money cap.

Codex works in <run>/codex/ (AGENTS.md, the opening images, scene files it writes); the MCP server keeps the run's
state in the run directory, which is read back when Codex exits or its time is up. A submitted, accepted revision is
final ("success"); otherwise the most complete eligible revision is kept ("partial", agent.fallback).

blind: no perception for the agent, to measure what perception adds. Codex gets only the input's frames and timing
and, where the input gives them (T5), the cameras; no task text. scene.py sees an evidence
folder with nothing else (evidence/blind), the tools lose perception (no masks, boxes, depth, object crops or
find_objects) and the final rule asks no evidence object. The run's full perception is still published, for the
evaluation only.

judge (with full evidence or raw layers): every build also carries the video check (consistency.py: the scene from
the input's own camera against the perception's masks and depth) and submit goes through an independent reviewer
(judge.py: a separate read-only Codex session that sees only that check and its cards) at most `rounds` times; the
final scene is the reviewed revision with the best check score.

raw: the perception's own layers instead of its fused objects, for Codex to use itself: the cameras (given or
estimated), the gravity-aligned world, metric depth on the depth model's keyframes and SAM 3's instance masks with
their track ids and names, plus a small loader (raw_loader.py, copied as raw.py) in evidence/raw. No boxes, poses,
merges or motion classes; the tools and the final rule are the blind ones. The task's text is given as with the full
evidence.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import agent, contract
from .perception import now

CODE = Path(__file__).resolve().parents[1]

FINISH_API = ("Finish with only a JSON object {\"selected_revision\": \"rNNN\", \"summary\": \"...\", \"left_out\": "
              "[{\"id\": \"...\", \"reason\":\n\"...\"}], \"limitations\": [\"...\"]}. The selected revision holds "
              "every certain object; each uncertain one is built or\nlisted in left_out with the reason the frames show "
              "(a duplicate of another id, not a physical object, ...).")
FINISH_CODEX = ("Finish by calling the submit tool with selected_revision, summary, left_out ([{\"id\": \"...\", "
                "\"reason\": \"...\"}]) and limitations. The selected revision holds every certain object; each "
                "uncertain one is built or listed in left_out with the reason the frames show (a duplicate of another "
                "id, not a physical object, ...). If submit answers 'Not accepted', fix what it names and submit again.")


BLIND = """You write the Blender 4.5 Python (scene.py -> scene.blend) of a semantic, editable scene that reproduces one
input, a video or an image: its objects and their motion, the environment and the camera. Other systems then query the
.blend, render it from new viewpoints, simulate it forward or edit it.

Input. Only the input itself: its frames (the opening images, more with inspect_video) and its timing{given}. No
perception is provided: there is no object list and no boxes, masks, depth or ground plane. You
estimate everything from the frames: which physical objects there are, their shapes, sizes, colors, positions and
motion, the support surface and {camera_task}. The input's files are at /evidence inside the sandbox (video.json,
timeline.jsonl{camera_file}, frames/ with the input frames at the comparison frames listed in frames.json), and read-only
at {host} here.

Tools (MCP server `codify`). inspect_video: frames by index, or a region of a frame zoomed at full resolution.
execute_blender (scene code as an argument) or execute_blender_file (a scene.py in this directory, so you edit a file
and build it again): builds a complete scene.py as a new revision and returns errors, printed output, the inventory and
its check (structure, contacts and penetrations, objects that move after release, the declared plan, one simulated
second after the last input frame) with source|render pairs at the comparison frames {comparison} and a top view; the
images come back with the result and are saved at the paths it lists. render_scene: other frames or views of a built
revision. submit: the final revision. Build only with these tools; do not run Blender yourself.

Work: look at the frames; list every physical object they show; estimate the camera and each object's size, position
and orientation (the support surface's perspective and the known sizes of everyday objects give the metric scale);
build a complete scene early; compare every source|render pair, fix what differs (positions, sizes, shapes, colors,
camera), build again.
Finish by calling submit with selected_revision, summary and limitations; if it answers 'Not accepted', fix what it
names and submit again.

A finished scene.py:
1. ev = A.load_evidence(); A.setup_scene(ev["video"]): every input frame on the timeline. {camera_contract}
2. One registered mesh per physical object (A.add_object, or your own mesh with A.register_object): instance_id "m1",
   "m2", ..., a category, bounds, location and rotation; an object the frames show moving is keyframed along that
   motion (A.add_object(..., poses=...) with poses over the input's timestamps, motion_type "dynamic"), every other one
   is static. The mesh has the shape the frames show (a laptop is a thin base with a lid, a bottle has a neck, a bowl
   is hollow): a plain box is right only for a box-like thing; colors from the frames; with a camera that matches a
   frame, A.project_texture(obj, ev["frames"][f], camera) gives a static object that frame's look on its visible faces.
   An object with movable pieces is declared in A.declare_plan (hinge, slider, separate or fixed, as the helper
   library says). End scene.py with A.separate_objects().
3. The environment the frames show, as environment meshes (A.add_floor, A.add_wall, A.box_mesh + A.add_environment):
   the supporting surface where resting objects stand, continued past the image edges, then the backdrop.
4. Lights and a world background that reproduce the frames' exposure and shadows.
5. Physics, so the scene can be simulated forward from the last input frame: every object gets a rigid body (role
   "active" for anything free to fall, roll, slide, tip or be pushed, and for anything held until the last frame;
   "passive" for what stays put), mass_kg, friction, restitution and collision shape from what it is made of and its
   size; environment meshes are fixed colliders; a resting object sits on its support without intersecting it.
   Say what each object does when things act on it, from what it is made of and what the frames and the text
   show (set_physics behavior=..., properties={{...}}): "rigid", "brittle" (breaks or crushes under load: ceramic,
   glass, styrofoam, an egg; break_load_n), "magnetic" (a magnet: pull_n, reach_m, attracts), "soft" (clay, dough, a
   pillow, a sponge, a balloon), "cloth" (paper, a napkin, fabric), "granular" (sand), "liquid", "heat_source" (a
   flame); the helper library says what each needs. Magnets pull in the simulation; soft, cloth, granular and liquid
   objects are absorbing colliders; every behavior and its properties are read by whoever simulates the scene next.

Rules. +Z is up, units are meters. Never edit /evidence. scene.py runs in an empty Blender without network and may use
only /evidence, bpy, numpy and codify.assets (import codify.assets as A). At most {max_revisions} built scenes (failed
builds do not count). You have about {minutes} minutes in all: build a complete first scene early (within about ten
minutes), then refine it, and submit before the time is up (when it runs out, the most complete eligible revision is
kept; a run with no built scene has no output).

Helper library (codify.assets; its evidence-based helpers have no evidence objects to work on here):
{api}
"""


RAW = """You write the Blender 4.5 Python (scene.py -> scene.blend) of a semantic, editable scene that reproduces one
input, a video or an image: its objects and their motion, the environment and the camera. Other systems then query the
.blend, render it from new viewpoints, simulate it forward or edit it.

Input. The input's frames (the opening images, more with inspect_video) and timing, the camera on every frame
({camera_source}), and the raw layers of a local perception, before any fusion: metric depth on {depth_frames} frame(s)
and SAM 3 instance masks with track ids (each track is one name's segment followed over the frames) and a name for each
track. There is no object list: no boxes, poses, merged identities, motion classes or shapes. The tracks are as found:
one physical object can have several tracks (under different names), a track can be a part of an object, several
objects, a background region or a mistake, and a name can be wrong. You decide what the physical objects are and
estimate each one's shape, size, position, orientation and motion yourself, from the frames, the masks and the depth.
The files are read-only at {host} (README.md there says what each holds) and at /evidence inside the sandbox. raw.py
there loads them: raw.mask(track, frame), raw.depth(frame), raw.points(track, frame) (the mask's pixels with depth as
world points), raw.track_points(track), raw.on_plane(frame, u, v, z) (where pixels' camera rays meet the horizontal
plane at height z), raw.contact(track, frame, z) (the bottom edge of a mask put on that plane along the rays), and
raw.overlay(frame, path) (the masks outlined and labelled). inspect_video(frame_ids, tracks=[...]) shows tracks'
masks on frames. Run Python with {python} (numpy, scipy, OpenCV) or in a build; write your scripts and images in this
directory.{task}

Tools (MCP server `codify`). inspect_video: frames by index, or a region of a frame zoomed at full resolution.
execute_blender (scene code as an argument) or execute_blender_file (a scene.py in this directory, so you edit a file
and build it again): builds a complete scene.py as a new revision and returns errors, printed output, the inventory and
its check (structure, contacts and penetrations, objects that move after release, the declared plan, one simulated
second after the last input frame) with source|render pairs at the comparison frames {comparison} and a top view; the
images come back with the result and are saved at the paths it lists. render_scene: other frames or views of a built
revision. submit: the final revision. Build only with these tools; do not run Blender yourself.

Work: look at the frames and the tracks' masks; decide which tracks are which physical objects; fit each object's shape
and pose to its points over the frames where it is seen and check the fit against the frames (orientation from the
points' principal axes and the frames, not only an upright box: an object can lie on its side or lean). Depth sees only
the surfaces that face the camera: a mask's points are an object's front and top, its centre lies behind them and its
hidden side is not in them; depth is noisy at mask edges, on thin or dark parts and far from the camera. For an object
resting on a surface whose height you know (the floor at z = 0, a table top from its points), raw.contact puts the
bottom edge of its mask on that surface without depth: where it stands; the points and the frames give its height,
shape and orientation. Every physical object the frames show is in the scene, also one no track covers; build a
complete scene early; compare every source|render pair, fix what differs (positions, sizes, shapes, colors, camera),
build again.
Finish by calling submit with selected_revision, summary and limitations; if it answers 'Not accepted', fix what it
names and submit again.

A finished scene.py:
1. ev = A.load_evidence(); A.setup_scene(ev["video"]); A.add_source_camera(ev["cameras"]): every input frame on the
   timeline and the camera of the input on every frame.
2. One registered mesh per physical object (A.add_object, or your own mesh with A.register_object): instance_id "m1",
   "m2", ..., a category, bounds, location and rotation; an object the frames show moving is keyframed along that
   motion (A.add_object(..., poses=...) with poses over the input's timestamps, motion_type "dynamic"), every other one
   is static. The mesh has the shape the frames show (a laptop is a thin base with a lid, a bottle has a neck, a bowl
   is hollow): a plain box is right only for a box-like thing; colors from the frames. A.project_texture(obj,
   ev["frames"][f], camera, mask=raw.mask(track, f)) gives a static object that frame's look on its visible faces from
   its own pixels only (its tracks' masks, OR-ed when several); never project without the object's mask: the frame
   would paint what lies in front of or behind it onto it. An object with movable pieces is declared in
   A.declare_plan (hinge, slider, separate or fixed, as the helper library says). End scene.py with
   A.separate_objects().
3. The environment the frames show, as environment meshes (A.add_floor, A.add_wall, A.box_mesh + A.add_environment):
   the supporting surface where resting objects stand (the world's floor is at z = {floor}), continued past the image
   edges, then the backdrop, in the colors the frames show. Do not project a frame onto a surface objects rest on:
   the objects would stay painted on it once they move.
4. Lights and a world background that reproduce the frames' exposure and shadows.
5. Physics, so the scene can be simulated forward from the last input frame: every object gets a rigid body (role
   "active" for anything free to fall, roll, slide, tip or be pushed, and for anything held until the last frame;
   "passive" for what stays put), mass_kg, friction, restitution and collision shape from what it is made of and its
   size; environment meshes are fixed colliders; a resting object sits on its support without intersecting it.
   Say what each object does when things act on it, from what it is made of and what the frames and the text
   show (set_physics behavior=..., properties={{...}}): "rigid", "brittle" (breaks or crushes under load: ceramic,
   glass, styrofoam, an egg; break_load_n), "magnetic" (a magnet: pull_n, reach_m, attracts), "soft" (clay, dough, a
   pillow, a sponge, a balloon), "cloth" (paper, a napkin, fabric), "granular" (sand), "liquid", "heat_source" (a
   flame); the helper library says what each needs. Magnets pull in the simulation; soft, cloth, granular and liquid
   objects are absorbing colliders; every behavior and its properties are read by whoever simulates the scene next.

Rules. +Z is up, units are meters. Never edit /evidence. scene.py runs in an empty Blender without network and may use
only /evidence, bpy, numpy and codify.assets (import codify.assets as A); work out numbers with your own scripts here
and write them into scene.py. At most {max_revisions} built scenes (failed builds do not count). You have about
{minutes} minutes in all: build a complete first scene early (within about ten minutes), then refine it, and submit
before the time is up (when it runs out, the most complete eligible revision is kept; a run with no built scene has no
output).

Helper library (codify.assets; its evidence-based helpers have no evidence objects to work on here):
{api}
"""


VIDEO_CHECK = """
# Checking against the video

Every build also returns video_check: the scene seen from the input's own camera on frames spread over the whole
input and compared, object by object, with the perception's masks and depth. score is the share of the observed object
pixels that the right scene object covers; worst lists the objects that miss most (iou of the silhouette against its
mask; depth_error = built surface minus measured depth, + = too far, - = too near); seen_not_built are observed masks
nothing explains; built_not_seen are objects where the input shows nothing. Use it to find what to fix; a mask can be
wrong (a patch of floor, a shadow, part of an object). When you submit, an independent reviewer that sees only this
check on more frames, with crops of the worst objects, decides. If submit answers 'Not accepted' with a list of
changes, make them, build again and submit again (it reviews at most {rounds} times). The final scene is the reviewed
revision with the best score.
"""


RUN_PYTHON = """
# Running code

Shell commands may fail in this session (the coding CLI's own sandbox cannot start on this machine). To compute from
the evidence, use the run_python tool: it runs a Python script (numpy only; no PIL, cv2 or scipy) in the isolated sandbox with the evidence folder
read-only at /evidence{raw} and prints back what it prints. Scene code goes to the build tools as before.
"""

OBJECTS3D = """
# Objects as complete meshes (make_objects)

Build each physical object from the input itself, not from primitives: make_objects turns an object's masks on one
frame into its complete shape (its hidden sides completed by SAM 3D Objects) with vertex colours, posed with that
frame's camera and measured depth. The model loads when the run starts; the first call may wait for it.
1. Inventory: {inventory}look at the masks on frames where everything is well seen ({look}) and decide which {refs} make up each
   physical object. One can be a part of an object (a lid, a cap, a label, a handle, a latch) and several can be one
   object: give all of an object's {refs} together, never a part alone. Leave out what is not an object (patches of
   floor or wall, shadows, reflections) and the surfaces things stand on (floor, ground, table top): build those as
   planes or boxes.
   In a room with many objects, make meshes first for the furniture, appliances and fixtures, then for the other
   objects that are well seen; small or barely seen ones can stay boxes.{same}
2. Call make_objects with every object (up to 16 per call; more calls for more){ids}. Read each card: the magenta outline
   should cover the green masks, and the side and top views should look like the object. When one is wrong (merged
   with a neighbour, cut off, flat, oddly shaped), call again with its id and a frame where it is seen whole, or a
   corrected set of {refs}.
3. In scene.py place each with obj = A.add_asset("<id>", name, category, physics=...): the mesh is already at its
   estimated pose and size, so change only what the evidence contradicts. A.rest_on(obj, z) lays a resting object on
   the stable face nearest to facing down (a pose seen from one view is often tilted about the line of sight) and
   seats it on a support whose top is at z. The pose is the one on the source frame (with a camera that does not move,
   the latest frame where the object is well seen). Primitives (A.add_object, boxes) are for the structure and for
   objects make_objects could not make.
4. An object that moves: track_objects follows its mesh through the frames (fitted to each frame's depth inside its
   masks), and A.add_asset(id, animate=True) keys those poses. Read what it returns: how far each object moved and
   turned, and the frames where the fit failed (fast, hidden or blurred there: fix those keys from the masks).
"""


def blind_evidence(workspace):
    """evidence/blind: what the input itself carries (timing, the input frames at the comparison frames, and the
    cameras when the input gives them), with no perception. Returns (directory, cameras given)."""
    full, folder = workspace.evidence, workspace.run / "evidence" / "blind"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("video.json", "timeline.jsonl"):
        shutil.copyfile(full / name, folder / name)
    given = contract.read_json(full / "manifest.json")["mode"] == "rgb_plus_cameras"
    if given:
        shutil.copyfile(full / "cameras.json", folder / "cameras.json")
    contract.write_json(folder / "objects.json", {})
    (folder / "frames").mkdir(exist_ok=True)
    for f in workspace.comparison:
        shutil.copyfile(workspace.decode_dir / f"frames/{f:06d}.jpg", folder / f"frames/{f:06d}.jpg")
    contract.write_json(folder / "frames.json", {"comparison": workspace.comparison,
                                                 "frames": "frames/<frame>.jpg: the input frames (no masks)"})
    return folder, given


def blind_instructions(workspace, settings, minutes, folder, given, frames):
    """AGENTS.md without perception and without the task's text: the frames, the timing and given cameras only."""
    video, count = workspace.video, len(workspace.timeline)
    canonical = video.get("canonical", {})
    duration = workspace.timeline[-1]["timestamp_ns"] / 1e9 if count > 1 else 0.0
    system = BLIND.format(
        given=" and the camera on every frame, which the input gives" if given else "",
        camera_task="the layout in the given camera's world (the ground at z = 0)" if given else "the camera (none is given)",
        camera_file=", cameras.json" if given else "", host=folder, comparison=workspace.comparison,
        camera_contract=("A.add_source_camera(ev[\"cameras\"]): the given camera on every frame." if given else
                         "One perspective camera at the input's resolution that reproduces the frames' view, as the scene "
                         "camera: its focal length and pose estimated from the frames, fixed when the camera does not move."),
        max_revisions=settings["max_revisions"], minutes=minutes, api=agent.assets_reference(room=bool(settings.get("fast_room"))))
    opening = (f"Input: {canonical.get('width')}x{canonical.get('height')} pixels, {count} frame(s)"
               + (f" over {duration:.2f} s" if count > 1 else " (a single image)") + ".")
    note = (f"\n\nThe opening images attached to the first message are the source frames {frames}, in that order.\n")
    return "# Task\n\n" + system + "\n# Opening\n\n" + opening + note


ARRANGE = """
Arrange the room like a set dresser. The evidence puts each object where its masks and depth put it, which can be
well off: objects float over the counter they stand on or sink into it, lean, or cross their neighbours, and some
stand on surfaces the evidence lacks (a window sill, a shelf board, a counter top). End scene.py with
A.arrange_room() in place of A.separate_objects(): it turns standing objects upright, seats each on the surface under
it, keeps what hangs and separates what crosses, and prints its report (ARRANGEMENT, UNSUPPORTED). Then go through
the objects it lists as unsupported, with the frames (inspect_video on their ids and frames): build what they stand on
(A.add_support([ids]) before arrange_room, or the furniture itself where the evidence lacks it), pass the ids that
hang on a wall or a ceiling in hang= and those that stand on the floor in stand=, and leave out in submit what is
misplaced, a duplicate of a built object (same_as) or not an object. The check fails while more than {floating} of
the objects float above what is under them or more than {deep} of them cross another by more than {deep_cm} cm; its
failure names them.
"""


FAST_ROOM = """
Time. You have {minutes} minutes in all, so work in few, large steps. base_scene.py in this directory is a complete
scene.py for this room: every evidence object at its estimated pose (its prepared mesh where it agrees with its box),
the floor and the walls the perception found, then A.arrange_room(auto_support=True). Build it first
(execute_blender_file with path base_scene.py), then render the source camera on a few frames and compare them with the
input. Change only what the frames show is wrong, the most visible first: a large object missing, in the wrong
place, duplicated or of the wrong color or shape (a rug, a tree, a sofa, a bed: they fill the frames), a missing wall
or surface, objects arrange_room left unsupported (hang=, stand=, A.add_support, or leave them out), and build once
more. Submit by minute {submit_by}: when the time runs out the last eligible revision is
kept, so a submitted scene with small faults beats a better one that is not finished. make_objects, find_objects,
track_objects and inventory take minutes each; use them only for an object the scene cannot do without.
"""


FAST_ROOM_FULL = """
How to use the perception. The evidence is extra information from local models, not the truth. What it mainly gives you
is the objects: their names, crops, masks and prepared meshes. Their positions, heights and sizes, the floor and the walls
are rough starting values from depth seen by one moving camera; objects can be repeated, missing, misnamed or not objects
at all. The scene is yours: what is in the room and where it stands is for you to decide from the video. Change the
evidence's values wherever the video says otherwise, however far, and do not take a value as right because it came from
the evidence.

Time. You have {minutes} minutes in all; use them. First watch the whole video (inspect_video on at least 12 frames spread
over all of it) and write down the room as a whole: which rooms it has, how big they are, what stands along each wall,
what is next to what. Then build base_scene.py in this directory (execute_blender_file with path base_scene.py), which is
only a starting point: every evidence object at its estimated pose (its prepared mesh where it agrees with its box), the
floor and the walls the perception found, then A.arrange_room(auto_support=True) and A.layout_report().

Then improve the scene in rounds until minute {min_submit}. In every round look at the whole scene, not only from the
source camera: an object can look right from there and still be wrong, because an error along the line of sight does not
show. Every build returns, besides the source|render pairs, a top view without the ceiling and two side views without walls
or ceiling (side_x, side_y, with a red line at floor level), and prints LAYOUT; render_scene gives other frames and views.
Check these, in this order, and fix what is wrong:
1. On the floor. Furniture and appliances stand on the floor: tables, chairs, sofas, beds, shelves, wardrobes, bins, toilets.
   In the side views nothing hangs in the air except what really hangs on a wall or a ceiling. A table or counter that is
   only a top gets its legs or body down to the floor. A Support_ slab is a placeholder for something the evidence lacks:
   replace it with the real thing (the table, the counter, the shelf on its wall) or move what stands on it.
2. The room. The floor covers the rooms the video shows and no more, at the right height (the camera is a person's height
   above it). Every side the video shows has its wall, in the right place, with its doors and openings; a wall may hide
   the camera's view in a frame only if it does so in the video.
3. The layout. Compare the top view with the room you wrote down: which wall each thing is on, what is next to what, how
   far apart. Move, turn and resize what is off; a large object first.
4. The objects. One object per physical thing: remove duplicates (one kind, close together, or one thing seen from two
   places; LAYOUT lists overlaps) and what is not an object (left_out in submit, with the reason); add what the video shows
   and the scene lacks; correct wrong kinds (perception's names can be wrong: inspect_video with object_ids shows a crop);
   give each a plain name for what it is.
5. Then the source|render pairs at frames over the whole video (render_scene with frame_ids): shapes, colors, what is
   still missing.
Fix, build again, look again. The build's repairs carry measurements: fix them too, but they do not replace your own look.
submit is refused before minute {min_submit} while the selected revision still has repairs; submit by minute {submit_by}
(when the time runs out, the last eligible revision is kept). make_objects, find_objects, track_objects and inventory take
minutes each; use them for an object the scene cannot do without.
"""


RAW_README = """# Raw perception of this input

Local models' outputs before any fusion into objects: no boxes, poses, merges or motion classes. Frame indices are the
0-based canonical frames (frames/000000.jpg is frame 0); cameras.json's "frame" is that index + 1 (the Blender frame).
World: metres, +Z up, the floor at z = 0 ({up_note}).

- video.json, timeline.jsonl: the input's size and frame rate, and the timestamp of every frame.
- frames/<frame>.jpg: every input frame ({W} x {H}).
- cameras.json: one row per frame: position_world (the camera centre), R_world_rgb (the camera axes +X right, +Y down,
  +Z forward, in the world), projection.K (pinhole, in these frames' pixels), pose_status. The cameras are {cameras}.
- world.json: how the world frame was set (up, floor, scale: {scale}).
- depth.npz: frames (the {depth_frames} frame(s) with depth) and depth (float16, frames x {H} x {W}): metres along the
  camera's optical axis, 0 where the depth model is not reliable (low confidence, depth edges); from {depth_model}.
- masks/labels_NNN.npz: labels (uint16, {frame_count} frames x {H} x {W}), one file per searched name; a track's mask on
  frame f is labels[f] == its value.
- tracks.json: names_searched (the names SAM 3 was prompted with: a vision-language model's list of what the frames
  show, and names it gave to unprompted proposals) and tracks: track (its id), name, labels (its file), value, frames
  (where it is detected) and detections (score, mask area, bbox x0 y0 x1 y1 per frame). {tracks} tracks, as found: one
  object can have several tracks, a track can be a part, several objects, background or a mistake.
- raw.py: a loader (its docstring lists the functions): masks, depth, a mask's world points, rays meeting a horizontal
  plane (raw.on_plane, raw.contact: where a resting object's mask bottom touches the floor, without depth), overlays.
- objects.json: empty (there is no object list).

A pixel (u, v) with depth z on frame f is at R_world_rgb @ (z * [(u - cx) / fx, (v - cy) / fy, 1]) + position_world,
with that frame's camera. Depth only covers the surfaces facing the camera (an object's centre is behind its points)
and is noisiest at mask edges, on thin or dark parts and far away.
"""


def raw_evidence(workspace):
    """evidence/raw: the published evidence's cameras and world, every input frame, metric depth on the depth model's
    keyframes (canonical resolution, 0 where unreliable: the reconstruction the fusion starts from) and the joined SAM 3
    tracks' label maps with a track table, the loader as raw.py and README.md; no objects. Returns (directory, number
    of frames with depth)."""
    import cv2
    import numpy as np
    from . import fusion
    perception, full = workspace.perception, workspace.evidence
    folder = workspace.run / "evidence" / "raw"
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "masks").mkdir(parents=True)
    for name in ("video.json", "timeline.jsonl", "cameras.json", "world.json"):
        shutil.copyfile(full / name, folder / name)
    contract.write_json(folder / "objects.json", {})
    shutil.copytree(workspace.decode_dir / "frames", folder / "frames")
    contract.write_json(folder / "frames.json", {"comparison": workspace.comparison,
                                                 "frames": "frames/<frame>.jpg: every input frame (no masks)"})
    cfg = {**fusion.DEFAULTS, **perception.config.get("fuse", {})}
    rec = fusion.reconstruction(perception.work / "geometry", workspace.decode_dir, cfg,
                                np.random.default_rng(cfg["seed"]))
    depth = np.stack([cv2.resize(np.where(rec.valid[i], rec.depth[i], 0).astype(np.float32), (rec.W, rec.H),
                                 interpolation=cv2.INTER_NEAREST) for i in range(len(rec.keyframes))])
    np.savez_compressed(folder / "depth.npz", frames=np.array(rec.keyframes), depth=depth.astype(np.float16))
    tracks = contract.read_json(perception.parts["tracks"] / "tracks.json")
    rows = []
    for t in tracks["tracks"]:
        name = f"labels_{t['prompt_index']:03d}.npz"
        if not (folder / "masks" / name).is_file():
            shutil.copyfile(perception.parts["tracks"] / name, folder / "masks" / name)
        rows.append({"track": t["track_id"], "name": t["prompt"], "labels": f"masks/{name}", "value": t["label_value"],
                     "frames": [d["frame_index"] for d in t["detections"]],
                     "detections": [{k: d.get(k) for k in ("frame_index", "score", "mask_area_px", "bbox_xyxy")}
                                    for d in t["detections"]]})
    contract.write_json(folder / "tracks.json", {"names_searched": tracks["prompts"], "tracks": rows})
    shutil.copyfile(Path(__file__).with_name("raw_loader.py"), folder / "raw.py")
    given = rec.world["intrinsics_source"] == "given cameras"
    (folder / "README.md").write_text(RAW_README.format(
        up_note="the given cameras' world, its ground at z = 0" if given else
        f"up {rec.world['up_status']} from the depth, the floor plane through the origin",
        W=rec.W, H=rec.H, cameras="given with the input" if given else "estimated by the depth model (held when the "
        "camera does not move)", scale=f"{rec.world['scale_status']}, {rec.world['scale_source']}",
        depth_frames=len(rec.keyframes), depth_model=rec.model_info.get("model", "the depth model"),
        frame_count=tracks.get("frame_count", len(workspace.timeline)), tracks=len(rows)))
    return folder, len(rec.keyframes)


def raw_instructions(workspace, settings, minutes, folder, depth_frames, task, frames):
    """AGENTS.md with the raw perception layers instead of the fused evidence, and the task's text."""
    video, count = workspace.video, len(workspace.timeline)
    canonical = video.get("canonical", {})
    duration = workspace.timeline[-1]["timestamp_ns"] / 1e9 if count > 1 else 0.0
    given = contract.read_json(workspace.evidence / "manifest.json")["mode"] == "rgb_plus_cameras"
    system = RAW.format(
        camera_source="given with the input" if given else "estimated by the perception", depth_frames=depth_frames,
        host=folder, python=sys.executable, task=f"\n\n{task['text']}" if task and task.get("text") else "",
        comparison=workspace.comparison, floor="0", max_revisions=settings["max_revisions"], minutes=minutes,
        api=agent.assets_reference(room=bool(settings.get("fast_room"))))
    opening = (f"Input: {canonical.get('width')}x{canonical.get('height')} pixels, {count} frame(s)"
               + (f" over {duration:.2f} s" if count > 1 else " (a single image)") + ".")
    note = (f"\n\nThe opening images attached to the first message are the source frames {frames}, in that order.\n")
    return "# Task\n\n" + system + "\n# Opening\n\n" + opening + note


def instructions(workspace, settings, minutes, opening_text, frames, annotated):
    """AGENTS.md: the API loop's system prompt with Codex's tools, finish and time budget, then the opening text."""
    source = "given" if workspace.perception.config.get("evidence", {}).get("given") else "local"
    system = agent.SYSTEM.format(evidence=agent.EVIDENCE[source], evidence_rule=agent.EVIDENCE_RULE[source],
                                 max_revisions=settings["max_revisions"], max_requests="no fixed number of",
                                 max_images=settings.get("max_images_per_call", 6), comparison=workspace.comparison,
                                 api=agent.assets_reference(room=bool(settings.get("fast_room"))))
    budget_api = ("At most {} built scenes (failed builds do not count) and no fixed number of model turns;\nthe run also "
                  "has a cost budget and stops when it is spent, so build a complete first scene within your first few\n"
                  "turns and refine it afterwards: a run that stops before any scene is built has no output."
                  ).format(settings["max_revisions"])
    budget_codex = (f"At most {settings['max_revisions']} built scenes (failed builds do not count). You have about "
                    f"{minutes} minutes in all: build a complete first scene early (within about ten minutes), then "
                    "refine it with the repairs and the source|render pairs, and submit before the time is up (when "
                    "it runs out, the most complete eligible revision is kept; a run with no built scene has no "
                    "output).")
    for old, new in ((FINISH_API, FINISH_CODEX), (budget_api, budget_codex)):
        if old not in system:
            raise RuntimeError(f"the API prompt changed; update codex_agent ({old[:40]}...)")
        system = system.replace(old, new)
    evidence = workspace.evidence
    tools_note = f"""
# Working here (Codex)

The codify tools are MCP tools of the server `codify`: inspect_video, find_objects, find_objects_result (wait for a
running search), execute_blender (scene code as an argument), execute_blender_file (the same build from a file in this
directory: write scene.py here, edit it, build it again), render_scene and submit. Their images come back with the
result and are also saved at the paths it lists (open them with your image viewer when you need a closer look).
Build only with these tools: they run Blender in the sandbox against the evidence and keep the revisions, checks and
renders that decide what can be final. Do not run Blender yourself.

Inside the sandbox scene.py reads the evidence at /evidence; the same files are here, read-only:
{evidence} (objects.json, cameras.json, world.json, structure.json, frames/, masks/ ...). Read them with the shell
when you need numbers the summary below does not show; never edit them.

The opening images are attached to the first message: {len(frames)} source frames (frames {frames}) and then
{len(annotated)} annotated frames (frames {annotated}), in that order.

"""
    return "# Task\n\n" + system + tools_note + "# Opening evidence\n\n" + opening_text + "\n"


def usage_of(events_path):
    """Token usage summed over Codex's turn.completed events, and the number of agent turns."""
    total, turns = {}, 0
    if not Path(events_path).is_file():
        return total, turns
    for line in Path(events_path).read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed":
            turns += 1
            for key, value in (event.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    total[key] = total.get(key, 0) + value
    return total, turns


def run_codex(workspace, settings, video_path, config, task=None, log=print, minutes=35, model=None, effort=None,
              blind=False, raw=False, judge=None, objects3d=None):
    """Codex writes and revises the scene through the codify tools; returns the API loop's outcome dict.
    judge: None, or the reviewer's settings ({"model", "effort", "rounds", "frames", "cards"}): every build then
    carries the video check and submit goes through the independent review (not in blind runs: no observations).
    objects3d: None, or the SAM 3D worker's settings (the machine config's objects3d: python, repo, config, gpu, faces):
    the agent then has make_objects (objects3d.py) and builds objects as complete meshes (not in blind runs)."""
    settings = {**settings, "max_revisions": settings.get("codex_max_revisions", 12),
                "max_probe_builds": settings.get("codex_max_probe_builds", 3)}
    # Use the time (2026-11-02, the default for room runs): submit opens at 3/4 of the budget while the selected
    # revision has repairs; CODIFY_MIN_SUBMIT_MINUTES overrides it (0: the old quick submit).
    opens = os.environ.get("CODIFY_MIN_SUBMIT_MINUTES")
    opens = float(opens) if opens not in (None, "") else (0.75 * minutes if settings.get("fast_room") else 0.0)
    if opens:
        settings["min_submit_minutes"] = min(opens, max(1.0, minutes - 3))
    if judge and not blind:
        settings.update(video_check={"frames": 6}, judge={"model": "gpt-6.1-terra", "effort": "xhigh", "rounds": 3,
                                                          "frames": 12, "cards": 6, "minutes": 10, **judge})
    if objects3d and not blind:
        settings["objects3d"] = dict(objects3d)
    workspace.settings = settings
    meshes = None
    if objects3d and not blind and not raw:        # a room: its sizeable objects completed before the agent starts
        prepared = workspace.evidence / "assets" / "default.json"
        if prepared.is_file():                     # completed offline with the perception (prepare_room.py)
            meshes = contract.read_json(prepared)
            log(f"meshes prepared offline: {len(meshes['ids'])} of {meshes['tried']} agree with their boxes")
        else:
            from . import objects3d as generate
            meshes = generate.prepare_meshes(workspace, settings["objects3d"], log)
    run = workspace.run
    work = run / "codex"
    work.mkdir(exist_ok=True)
    if blind:
        folder, given = blind_evidence(workspace)
        kind, frames, first, _, _, _ = agent.opening(workspace, settings, None)
        images = [workspace.images[i]["path"] for i in first]
        (work / "AGENTS.md").write_text(blind_instructions(workspace, settings, minutes, folder, given, frames))
    elif raw:
        folder, depth_frames = raw_evidence(workspace)
        kind, frames, first, _, _, _ = agent.opening(workspace, settings, None)
        images = [workspace.images[i]["path"] for i in first]
        (work / "AGENTS.md").write_text(raw_instructions(workspace, settings, minutes, folder, depth_frames, task, frames))
    else:
        kind, frames, first, annotated_frames, annotated, opening_text = agent.opening(workspace, settings, task)
        (work / "AGENTS.md").write_text(instructions(workspace, settings, minutes, opening_text, frames, annotated_frames))
        images = [workspace.images[i]["path"] for i in first + annotated]
    with open(work / "AGENTS.md", "a") as stream:
        stream.write(RUN_PYTHON.format(raw=" (the raw layers: sys.path.insert(0, '/evidence'); import raw)" if raw else ""))
    if settings.get("judge"):
        with open(work / "AGENTS.md", "a") as stream:
            stream.write(VIDEO_CHECK.format(rounds=settings["judge"]["rounds"]))
    if settings.get("objects3d"):
        with open(work / "AGENTS.md", "a") as stream:
            stream.write(OBJECTS3D.format(
                look="inspect_video with tracks=['all']" if raw else "inspect_video with layers=['masks']",
                refs="tracks" if raw else "evidence objects",
                inventory="" if raw else ((
                    "in a room, start with the inventory tool on the categories that have several evidence objects\n"
                    "   (most first; each call completes up to 64 objects with SAM 3D, minutes): from its cards, map and "
                    "overlaps decide\n   per category which evidence objects are one physical object (keep one id, "
                    "the others same_as it), which\n   are something else (give them their category) and which are not "
                    "objects (uncertain ones to left_out). Its\n   meshes are made: A.add_asset(id) builds each kept "
                    "one. Submit waits until the four most numerous categories\n   of sizeable things have been through it. "
                    if settings.get("inventory_gate", True) else
                    "in a room, the inventory tool lays out an evidence category's objects by their completed meshes\n"
                    "   (minutes): use it only when a category's count looks wrong in the frames. ")
                    + (f"Before you started, every evidence object at least {meshes['min_size_m']} m across was "
                       f"completed\n   ({len(meshes['ids'])} of {meshes['tried']} agree with their fused boxes; "
                       "/evidence/assets/default.json lists them):\n   A.add_object(uid=<that id>, ...) builds such an "
                       "object as its mesh at its estimated pose instead of a proxy\n   (mesh=False keeps the proxy), so "
                       "build the room's objects with add_object under their evidence ids. " if meshes else "")
                    + "Then "),
                ids="" if raw else " (the evidence objects' ids go in tracks), each under the id of its evidence "
                                   "object: A.add_asset(id) registers it under that id, which the coverage at submit "
                                   "counts",
                same="" if raw else (
                    "\n   Evidence objects can repeat one physical object (tracked again when the camera came back, or "
                    "split\n   into parts): build each physical object once, under one of its ids, and list the others "
                    "in submit's\n   left_out with same_as: that id, so the scene's counts are the room's; one that is a "
                    "patch of wall,\n   floor, ceiling or window: same_as: \"environment\" (the room's structure "
                    "stands for it).")))
    if settings.get("fast_room") and not blind and not raw:     # first in AGENTS.md: it decides the first build
        shutil.copyfile(CODE / "codify" / "templates" / "room_base_scene.py", work / "base_scene.py")
        text = (work / "AGENTS.md").read_text()
        if settings.get("min_submit_minutes"):
            head = FAST_ROOM_FULL.format(minutes=f"{minutes:g}", submit_by=f"{max(5.0, minutes - 2):g}",
                                         min_submit=f"{settings['min_submit_minutes']:g}")
        else:
            head = FAST_ROOM.format(minutes=f"{minutes:g}", submit_by=f"{max(5.0, minutes - 2):g}")
        (work / "AGENTS.md").write_text(head.lstrip() + "\n" + text)
    if settings.get("arrange") and not blind and not raw:
        layout = settings["arrange"] if isinstance(settings["arrange"], dict) else {}
        with open(work / "AGENTS.md", "a") as stream:
            stream.write(ARRANGE.format(floating=f"{100 * layout.get('max_floating_share', 0.1):.0f}%",
                                        deep=f"{100 * layout.get('max_deep_share', 0.15):.0f}%",
                                        deep_cm=f"{100 * layout.get('deep_m', 0.08):.0f}"))
    session = work / "session.json"
    contract.write_json(session, json.loads(json.dumps({"run": str(run), "video": str(video_path), "config": config,
                                                        "settings": settings, "blind": blind,
                                                        "evidence_mode": "blind" if blind else "raw" if raw else None},
                                                       default=str)))
    workspace._save_revisions()
    env = {k: os.environ[k] for k in ("PATH", "HOME", "USER", "LANG", "TMPDIR") if k in os.environ}
    env["PYTHONPATH"] = str(CODE)
    server_env = "{" + ", ".join(f'{k} = {json.dumps(v)}' for k, v in env.items()) + "}"
    command = ["codex", "exec", "--json", "--skip-git-repo-check", "-C", str(work), "-s", "workspace-write",
               *(["-m", model] if model else []), *(["-c", f'model_reasoning_effort="{effort}"'] if effort else []),
               "-c", f"mcp_servers.codify.command={json.dumps(sys.executable)}",
               "-c", f"mcp_servers.codify.args={json.dumps(['-m', 'codify.mcp_server', '--session', str(session)])}",
               "-c", f"mcp_servers.codify.env={server_env}",
               "-c", "mcp_servers.codify.startup_timeout_sec=600",
               "-c", "mcp_servers.codify.tool_timeout_sec=1800",
               "-c", 'mcp_servers.codify.default_tools_approval_mode="approve"',
               "-o", str(work / "last_message.txt")]
    for path in images:
        command += ["-i", path]
    command += ["-"]
    prompt = ("Read AGENTS.md in this directory first: it holds the task, the instructions, the helper library and the "
              "full evidence. The attached images are the opening frames it lists. Build the scene with the codify "
              "tools, refine it, and finish with submit.")
    if settings.get("fast_room") and not blind and not raw:
        prompt += (f" This is a timed room run ({minutes:g} minutes): your first build is base_scene.py as it is "
                   "(execute_blender_file with path base_scene.py), as the top of AGENTS.md says.")
    contract.write_json(work / "command.json", {"argv": command, "started_utc": now(), "minutes": minutes})
    log(f"codex: {' '.join(command[:12])} ... ({len(images)} images, {minutes} min)")
    started = time.time()
    reasons = []
    with open(work / "events.jsonl", "w") as events, open(work / "codex_stderr.log", "w") as stderr:
        process = subprocess.Popen(command, cwd=work, stdin=subprocess.PIPE, stdout=events, stderr=stderr, text=True,
                                   start_new_session=True)
        process.stdin.write(prompt)
        process.stdin.close()
        try:
            process.wait(timeout=60 * minutes)
        except subprocess.TimeoutExpired:
            reasons.append(f"time budget of {minutes} minutes spent")
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    wall = time.time() - started
    if process.returncode not in (0, None) and not reasons:
        reasons.append(f"codex exited with code {process.returncode}")
    # The run's state as the MCP server left it.
    workspace.revisions = contract.read_json(run / "revisions.json") if (run / "revisions.json").is_file() else {}
    state = contract.read_json(work / "state.json") if (work / "state.json").is_file() else {}
    if state.get("evidence"):
        workspace.evidence = run / "evidence" / state["evidence"]
    submission = contract.read_json(work / "submission.json") if (work / "submission.json").is_file() else None
    usage, turns = usage_of(work / "events.jsonl")
    budget = {"agent": "codex", "perception_given": not blind,
              "perception_form": "none" if blind else "raw layers" if raw else "fused evidence", "model": model or "codex default",
              "reasoning_effort": effort or "codex default",
              "wall_seconds": round(wall, 1), "agent_turns": turns, "tokens": usage, "api_usd": 0.0,
              "billing": "Codex subscription (no API spend)"}
    shutil.copyfile(work / "tool_trace.jsonl", run / "tool_trace.jsonl") if (work / "tool_trace.jsonl").is_file() else None
    reviews = contract.read_json(work / "reviews.json") if (work / "reviews.json").is_file() else []
    if reviews:
        budget["review"] = {"rounds": len(reviews), "verdicts": [r["verdict"] for r in reviews],
                            "scores": [r.get("score") for r in reviews],
                            "minutes": round(sum(r.get("minutes", 0) for r in reviews), 2),
                            "tokens": {k: sum((r.get("tokens") or {}).get(k, 0) for r in reviews)
                                       for k in ("input_tokens", "cached_input_tokens", "output_tokens")}}
    if submission and submission.get("accepted") and submission.get("selected_revision") in workspace.revisions:
        chosen = submission["selected_revision"]
        # The final scene is the reviewed revision with the best video-check score (the last can be worse).
        scored = {r["revision"]: r["score"] for r in reviews if r.get("score") is not None
                  and r["revision"] in workspace.revisions and workspace.eligible(r["revision"])[0]}
        if scored and max(scored.values()) > scored.get(chosen, -1.0) + 0.02:
            best = max(scored, key=scored.get)
            reasons.append(f"{best} kept over the submitted {chosen}: video-check score {scored[best]} vs "
                           f"{scored.get(chosen)}")
            chosen = best
        return {"status": "success", "selected_revision": chosen, "final": submission,
                "reasons": reasons, "budget": budget}
    status, selected = agent.fallback(workspace, reasons)
    return {"status": status, "selected_revision": selected, "final": submission, "reasons": reasons, "budget": budget}
