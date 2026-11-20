"""The one tool contract: schemas, argument validation, limits and frame sampling.

Shared by the host and the Blender worker; it reads no files and needs no Blender.
Query tasks (T1-T4) get QUERY_TOOLS + submit_answer, exactly the set the reported VSI runs used.
Update tasks (T5-T7) get the same five queries + track_motion, run_python, reset_scene, submit_update.
"""
from __future__ import annotations

import copy
import json

PREFIX = "VIDEOCOS_RPC "
MAX_MESSAGE_BYTES = 128 * 1024
MAX_RESPONSE_BYTES = 262144
MAX_VERTICES = 500_000
MAX_TRIANGLES = 1_000_000
MAX_REFINEMENT_FRAMES = 128
HEAVY_QUERY_SECONDS = 180.0  # worker budget for camera_history/surface; host allows 270 s


class QueryCapacityError(ValueError):
    """A bounded query cannot inspect the requested geometry within its caps."""


STR = {'type': 'string', 'maxLength': 512}
INT = {'type': 'integer'}
IDS = {'type': 'array', 'items': STR, 'minItems': 1, 'maxItems': 6, 'uniqueItems': True}


def _object(properties, required=()):
    return {'type': 'object', 'properties': properties, 'required': list(required), 'additionalProperties': False}


def _tool(name, description, properties, required=()):
    return {'name': name, 'description': description, 'parameters': _object(properties, required)}


TOOLS = (
    _tool('scene_inventory', 'Find authored entities by name. Default kind is collection. Search meshes for floor/wall parts. Collections may group objects; meshes may be parts: do not count both as separate objects.', {
        'query': STR, 'kind': {'type': 'string', 'enum': ['all', 'collection', 'mesh']},
        'offset': {'type': 'integer', 'minimum': 0},
        'limit': {'type': 'integer', 'minimum': 1, 'maximum': 128}}),
    _tool('inspect_entities', 'Read evaluated entities at an integer frame (default: first frame). basic returns world-AABB centers, sizes and membership; transform adds poses; materials gives authored appearance; geometry pages one entity. Dimensions are world-axis bounds, not oriented physical lengths.', {
        'ids': IDS, 'frame': INT,
        'detail': {'type': 'string', 'enum': ['basic', 'transform', 'materials', 'geometry']},
        'geometry_offset': {'type': 'integer', 'minimum': 0},
        'geometry_limit': {'type': 'integer', 'minimum': 1, 'maximum': 128}}, ['ids']),
    _tool('measure_pair', 'Measure two entities. center: center distance; bounds: world AABB separation (lower-bound proxy); surface: evaluated triangle distance (bounded work). Use one call per pair when comparing candidates.', {
        'a': STR, 'b': STR, 'metric': {'type': 'string', 'enum': ['center', 'bounds', 'surface']},
        'frame': INT}, ['a', 'b', 'metric']),
    _tool('viewpoint_relation', 'Stand at observer and face facing. Compute subject position: positive right_m is right, positive forward_m is front; negative is left/back. Uses horizontal world-bounds centers and +Z up.', {
        'observer': STR, 'facing': STR, 'subject': STR, 'frame': INT}, ['observer', 'facing', 'subject']),
    _tool('camera_history', 'Read the source camera over time. Supply entity_id OR entity_ids (union visibility of up to 32 instances). For category appearance, select all relevant whole-object instances, not a single part. Default: 64 coarse samples. Supply frames OR start/end/step, up to 128 samples. Automatically scan the first detected visibility bracket, up to 128 extra frames; use first_visible_hit when nonnull, otherwise first_sampled_hit and refinement status. Earlier brief appearances can still be missed; hidden objects report visibility=excluded; truncated=true means the time budget stopped sampling early. summary is compact; poses/full add camera details.', {
        'entity_id': STR,
        'entity_ids': {'type': 'array', 'items': STR, 'minItems': 1, 'maxItems': 32, 'uniqueItems': True},
        'frames': {'type': 'array', 'items': INT, 'minItems': 1, 'maxItems': 128, 'uniqueItems': True},
        'start_frame': INT, 'end_frame': INT, 'step': {'type': 'integer', 'minimum': 1},
        'limit': {'type': 'integer', 'minimum': 1, 'maximum': 128},
        'detail': {'type': 'string', 'enum': ['summary', 'poses', 'full']}}),
    _tool('submit_answer', 'Finish with only the requested option letter or numeric value in text. An explanation is optional. Submission is not graded here.', {
        'text': {'type': 'string', 'maxLength': 200},
        'explanation': {'type': 'string', 'maxLength': 6000}}, ['text']),
)
MAX_CODE_CHARS = 60_000
UPDATE_TOOLS = (
    _tool('track_motion', 'Follow up to 6 entities over time in the CURRENT (possibly edited) scene: world-AABB centre at each sampled frame, plus path length, peak and final speed. Supply frames OR start/end/step (up to 128 samples; default 32 evenly spaced over the timeline). Use it to check that an edit or a simulation does what you intended.', {
        'ids': IDS, 'frames': {'type': 'array', 'items': INT, 'minItems': 1, 'maxItems': 128, 'uniqueItems': True},
        'start_frame': INT, 'end_frame': INT, 'step': {'type': 'integer', 'minimum': 1},
        'limit': {'type': 'integer', 'minimum': 1, 'maximum': 128}}, ['ids']),
    _tool('run_python', 'Execute Python in the live Blender session that holds this scene. bpy, mathutils, math and the helper module S are imported; variables persist between calls, and so do your edits. Returns printed output (truncated), the error if one was raised, and what changed in the scene. Edit the scene here; check it with the query tools and track_motion.', {
        'code': {'type': 'string', 'maxLength': MAX_CODE_CHARS}}, ['code']),
    _tool('reset_scene', 'Discard every edit and reload the original scene. Variables are cleared.', {}),
    _tool('submit_update', 'Finish with the current edited scene. The task contract is checked first (for example the frame range or the camera the task requires); if it fails, nothing is submitted and the reasons come back. Rendering and scoring happen afterwards and are never shown to you.', {
        'summary': {'type': 'string', 'maxLength': 3000},
        'limitations': {'type': 'array', 'items': {'type': 'string', 'maxLength': 400}, 'maxItems': 8}}, ['summary']),
)
TOOL_NAMES = frozenset(tool['name'] for tool in TOOLS)
UPDATE_TOOL_NAMES = frozenset(tool['name'] for tool in UPDATE_TOOLS)
QUERY_ONLY = tuple(tool for tool in TOOLS if tool['name'] != 'submit_answer')
# Worker-only methods used by the host scene; never offered to the model.
SCHEMAS = {tool['name']: tool['parameters'] for tool in TOOLS + UPDATE_TOOLS} | {
    'summary': _object({}),
    '_scene_metadata': _object({'offset': {'type': 'integer', 'minimum': 0},
                                'limit': {'type': 'integer', 'minimum': 1, 'maximum': 128}}),
    '_scene_states': _object({'ids': {**IDS, 'maxItems': 20}, 'frame': INT}, ['ids', 'frame']),
}


def tool_schemas(kind="query"):
    """Fresh copies of the model-visible tools: the six VSI tools for a query task, or the five
    queries plus the update tools for an update task (submit_answer is not offered there)."""
    if kind == "query":
        return copy.deepcopy(list(TOOLS))
    if kind == "update":
        return copy.deepcopy(list(QUERY_ONLY) + list(UPDATE_TOOLS))
    raise ValueError("kind is query or update")


def validate(value, spec, path='arguments'):
    """Check a value against the JSON-schema subset used by SCHEMAS."""
    kind = spec.get('type')
    types = {'object': dict, 'array': list, 'string': str, 'integer': int}
    if kind in types and type(value) is not types[kind]:
        raise ValueError(f'{path} must have type {kind}')
    if 'enum' in spec and value not in spec['enum']:
        raise ValueError(f'{path} has an unsupported value')
    if kind == 'object':
        props, required = spec['properties'], set(spec.get('required', ()))
        missing, extra = sorted(required - set(value)), sorted(set(value) - set(props))
        if missing or extra:
            details = ([f'missing required fields: {", ".join(missing)}'] if missing else []) + (
                [f'unsupported fields: {", ".join(extra)}'] if extra else [])
            raise ValueError(f'{path}: {"; ".join(details)}')
        for key, child in value.items():
            validate(child, props[key], path + '.' + key)
    elif kind == 'array':
        if not spec.get('minItems', 0) <= len(value) <= spec.get('maxItems', 128):
            raise ValueError(f'{path} has an invalid item count')
        if spec.get('uniqueItems') and len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
            raise ValueError(f'{path} requires distinct items')
        for child in value:
            validate(child, spec['items'], path + '[]')
    elif kind == 'integer':
        if not spec.get('minimum', value) <= value <= spec.get('maximum', value):
            raise ValueError(f'{path} is outside the allowed range')
    elif kind == 'string' and len(value) > spec.get('maxLength', 1024):
        raise ValueError(f'{path} is too long')


def validate_arguments(method, arguments):
    """Schema plus cross-field checks; the same rules apply on the host and in Blender."""
    if method not in SCHEMAS:
        raise ValueError('Unknown method')
    validate(arguments, SCHEMAS[method])
    if method == 'inspect_entities':
        detail = arguments.get('detail', 'basic')
        if detail != 'geometry' and {'geometry_offset', 'geometry_limit'} & arguments.keys():
            raise ValueError('geometry_offset and geometry_limit require detail=geometry')
        if detail == 'geometry' and len(arguments['ids']) != 1:
            raise ValueError('Geometry detail accepts exactly one entity ID per call; basic/materials/transform accept up to 6')
    if method == 'track_motion':
        if 'frames' in arguments and {'start_frame', 'end_frame', 'step'} & arguments.keys():
            raise ValueError('Supply frames OR start_frame/end_frame/step, not both')
    if method == 'camera_history':
        if 'entity_id' in arguments and 'entity_ids' in arguments:
            raise ValueError('Supply entity_id OR entity_ids, not both')
        if 'frames' in arguments and {'start_frame', 'end_frame', 'step'} & arguments.keys():
            raise ValueError('Supply frames OR start_frame/end_frame/step, not both')
        if len(arguments.get('frames', ())) > arguments.get('limit', 128):
            raise ValueError('Explicit frames exceed limit; increase limit or request fewer frames')
        if {'start_frame', 'end_frame'} <= arguments.keys() and arguments['end_frame'] < arguments['start_frame']:
            raise ValueError('end_frame must be at or after start_frame')
    return arguments


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def string(value, name, limit=512):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{name} must be a string of at most {limit} characters")
    return value


def sample_frames(arguments, first, last):
    """Bounded exact-frame requests; never silently drop explicitly requested frames."""
    # Uniform sampling defaults to 64 observations. An explicit step is an
    # exact-frame request and may use the full 128-observation capacity.
    limit = arguments.get("limit", 128 if "step" in arguments else 64)
    if "frames" in arguments:
        frames = sorted({integer(f, "frame", first, last) for f in arguments["frames"]})
        if "limit" in arguments and len(frames) > limit:
            raise ValueError("frames exceeds limit; increase limit or request fewer frames")
        return frames
    start = integer(arguments.get("start_frame", first), "start_frame", first, last)
    end = integer(arguments.get("end_frame", last), "end_frame", first, last)
    if end < start:
        raise ValueError("end_frame must be >= start_frame")
    if "step" in arguments:
        step = arguments["step"]
        count = (end - start) // step + 1
        if count > limit:
            raise ValueError("frame range exceeds limit; increase step or limit")
        return list(range(start, end + 1, step))
    if limit == 1:
        return [start]
    return sorted({round(start + (end - start) * i / (limit - 1)) for i in range(limit)})


def compact_numbers(value):
    """Round display summaries only; exact geometry/calculations stay unrounded."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: compact_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [compact_numbers(item) for item in value]
    return value
