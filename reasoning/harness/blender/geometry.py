"""Bounded, dependency-free distance between evaluated triangle surfaces.

The caller supplies *world-space vertices in meters*, after applying Blender's
evaluated object transform and scene unit scale.  These functions do not read
files, sample geometry, infer solid occupancy, or access Blender.  In particular,
one closed shell contained inside another has a positive surface distance.

``exact`` means the complete search was resolved using triangle geometry (to
floating-point precision), not that a solid-intersection test was performed.
If the work or time budget is exhausted, ``distance`` is a measured upper bound
and ``lower_bound`` explicitly records how far the answer could still improve.
Inputs are validated and indexed with numpy; BVH nodes split lazily on first visit.
"""

from __future__ import annotations

import heapq
import math
from numbers import Integral
import time

import numpy as np


def _sub(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def _add_scaled(a, b, scale):
    return tuple(a[i] + b[i] * scale for i in range(3))


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _distance2(a, b):
    return _dot(_sub(a, b), _sub(a, b))


def _clamp(value):
    return max(0.0, min(1.0, value))


def _closest_on_segment(point, a, b):
    direction = _sub(b, a)
    length2 = _dot(direction, direction)
    if length2 == 0:
        return a
    return _add_scaled(a, direction, _clamp(_dot(_sub(point, a), direction) / length2))


def _inside_triangle(point, triangle, normal):
    # Used only for points on the triangle's plane.  Keeping the half-plane
    # comparison strict avoids treating nearby, disjoint surfaces as touching.
    a, b, c = triangle
    return all(_dot(_cross(_sub(v, u), _sub(point, u)), normal) >= 0.0
               for u, v in ((a, b), (b, c), (c, a)))


def _closest_on_triangle(point, triangle):
    a, b, c = triangle
    candidates = [_closest_on_segment(point, u, v)
                  for u, v in ((a, b), (b, c), (c, a))]
    normal = _cross(_sub(b, a), _sub(c, a))
    normal2 = _dot(normal, normal)
    if normal2 > 0.0:
        projection = _add_scaled(point, normal, -_dot(_sub(point, a), normal) / normal2)
        if _inside_triangle(projection, triangle, normal):
            candidates.append(projection)
    return min(candidates, key=lambda candidate: _distance2(point, candidate))


def _segment_pair(a0, a1, b0, b1):
    """Closest points on two closed segments, including degenerate segments."""
    u, v, r = _sub(a1, a0), _sub(b1, b0), _sub(a0, b0)
    aa, bb = _dot(u, u), _dot(v, v)
    if aa == 0.0:
        return a0, _closest_on_segment(a0, b0, b1)
    if bb == 0.0:
        return _closest_on_segment(b0, a0, a1), b0
    uv, ur, vr = _dot(u, v), _dot(u, r), _dot(v, r)
    # The equivalent aa * bb - uv ** 2 expression loses all significant
    # digits for long, nearly parallel edges.  Cross products retain their
    # small angular separation and permit interior/interior nearest points.
    normal = _cross(u, v)
    denominator = _dot(normal, normal)
    s = _clamp(_dot(_cross(v, r), normal) / denominator) if denominator > 0.0 else 0.0
    t = (uv * s + vr) / bb
    if t < 0.0:
        t, s = 0.0, _clamp(-ur / aa)
    elif t > 1.0:
        t, s = 1.0, _clamp((uv - ur) / aa)
    return _add_scaled(a0, u, s), _add_scaled(b0, v, t)


def _segment_triangle_hit(start, end, triangle):
    """Detect face-interior crossings; coplanar cases are handled by edges."""
    a, b, c = triangle
    normal = _cross(_sub(b, a), _sub(c, a))
    d0, d1 = _dot(_sub(start, a), normal), _dot(_sub(end, a), normal)
    if (d0 > 0.0 and d1 > 0.0) or (d0 < 0.0 and d1 < 0.0):
        return None
    denominator = d0 - d1
    if denominator == 0.0:
        return None
    t = d0 / denominator
    if not 0.0 <= t <= 1.0:
        return None
    point = _add_scaled(start, _sub(end, start), t)
    return point if _inside_triangle(point, triangle, normal) else None


def _triangle_pair(a, b):
    """Return squared distance and witnesses for two closed triangles."""
    edges_a = ((a[0], a[1]), (a[1], a[2]), (a[2], a[0]))
    edges_b = ((b[0], b[1]), (b[1], b[2]), (b[2], b[0]))
    for start, end in edges_a:
        hit = _segment_triangle_hit(start, end, b)
        if hit is not None:
            return 0.0, hit, hit
    for start, end in edges_b:
        hit = _segment_triangle_hit(start, end, a)
        if hit is not None:
            return 0.0, hit, hit
    candidates = [(point, _closest_on_triangle(point, b)) for point in a]
    candidates.extend((_closest_on_triangle(point, a), point) for point in b)
    candidates.extend(_segment_pair(*ea, *eb) for ea in edges_a for eb in edges_b)
    point_a, point_b = min(candidates, key=lambda pair: _distance2(*pair))
    return _distance2(point_a, point_b), point_a, point_b


def _bounds_distance2(a, b):
    return sum(max(a[0][i] - b[1][i], b[0][i] - a[1][i], 0.0) ** 2 for i in range(3))


def _corners(vertices, triangles, name):
    """Validated (M, 3, 3) array of triangle corner coordinates."""
    try:
        points, indices = np.asarray(vertices, dtype=float), np.asarray(triangles)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: vertices must be finite XYZ coordinates and triangles vertex indices") from exc
    points = points.reshape(0, 3) if points.size == 0 else points
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"{name}: vertices must be finite XYZ coordinates")
    if indices.size == 0:
        raise ValueError(f"{name}: mesh has no triangle faces; point clouds are unsupported")
    if (indices.ndim != 2 or indices.shape[1] != 3 or indices.dtype.kind not in "iu"
            or indices.min() < 0 or indices.max() >= len(points)):
        raise ValueError(f"{name}: triangles must contain three valid vertex indices")
    return points[indices]


class _Node:
    """Triangle subset of one mesh; leaves hold at most four triangles."""
    __slots__ = ("corners", "low", "high", "indices", "bounds", "_children", "_items")

    def __init__(self, corners, low, high, indices):
        self.corners, self.low, self.high, self.indices = corners, low, high, indices
        self.bounds = (tuple(low[indices].min(0).tolist()), tuple(high[indices].max(0).tolist()))
        self._children = self._items = None

    @property
    def leaf(self):
        return len(self.indices) <= 4

    def children(self):
        if self._children is None:
            axis = max(range(3), key=lambda i: self.bounds[1][i] - self.bounds[0][i])
            key = self.low[self.indices, axis] + self.high[self.indices, axis]
            ordered = self.indices[np.argsort(key, kind="stable")]
            middle = len(ordered) // 2
            self._children = tuple(_Node(self.corners, self.low, self.high, part)
                                   for part in (ordered[:middle], ordered[middle:]))
        return self._children

    def items(self):
        """(index, bounds, triangle) for a leaf, as plain Python values."""
        if self._items is None:
            self._items = [(int(i), (tuple(self.low[i].tolist()), tuple(self.high[i].tolist())),
                            self.corners[i].tolist()) for i in self.indices]
        return self._items


def _root(corners):
    return _Node(corners, corners.min(1), corners.max(1), np.arange(len(corners)))


def mesh_surface_distance(vertices_a, triangles_a, vertices_b, triangles_b, max_pairs=2_000_000, deadline=None):
    """Find the unsigned surface distance, or return explicit bounded progress.

    ``max_pairs`` bounds examined triangle pairs, including cheap AABB checks.
    BVH node visits are independently bounded by ``max(64, 8 * max_pairs)``.
    ``deadline`` is an optional ``time.monotonic()`` value that stops the search.
    Distances and witnesses use the input coordinates' units; callers must supply meters.
    """
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, Integral) or max_pairs < 1:
        raise ValueError("max_pairs must be a positive integer")
    corners_a = _corners(vertices_a, triangles_a, "mesh_a")
    corners_b = _corners(vertices_b, triangles_b, "mesh_b")
    best, closest_a, closest_b = _triangle_pair(corners_a[0].tolist(), corners_b[0].tolist())
    if not math.isfinite(best):
        raise ValueError("Geometry exceeds supported floating-point range")
    pairs_checked = pairs_evaluated = 1
    node_visits = nodes_pruned = 0
    node_limit = max(64, int(max_pairs) * 8)
    queue = []
    sequence = 0

    def push(node_a, node_b):
        nonlocal sequence, nodes_pruned
        lower = _bounds_distance2(node_a.bounds, node_b.bounds)
        if lower >= best:
            nodes_pruned += 1
            return
        sequence += 1
        heapq.heappush(queue, (lower, sequence, node_a, node_b))

    def result(exact, lower, reason):
        upper = math.sqrt(max(0.0, best))
        return {
            "distance": upper,
            "closest_a": list(closest_a),
            "closest_b": list(closest_b),
            "exact": exact,
            "status": "exact" if exact else "bounded",
            "stop_reason": reason,
            "lower_bound": upper if exact else math.sqrt(max(0.0, min(lower, best))),
            "upper_bound": upper,
            "units": "meters",
            "method": "triangle_surface_distance",
            "semantics": "Unsigned surface separation; containment does not imply zero distance.",
            "triangle_counts": [len(corners_a), len(corners_b)],
            "triangle_pairs_checked": pairs_checked,
            "triangle_pairs_evaluated": pairs_evaluated,
            "bvh_node_pairs_visited": node_visits,
            "bvh_node_pairs_pruned": nodes_pruned,
            "max_pairs": int(max_pairs),
        }

    if best == 0.0:
        return result(True, 0.0, "surfaces_intersect")
    push(_root(corners_a), _root(corners_b))
    while queue:
        if node_visits >= node_limit:
            return result(False, queue[0][0], "bvh_work_limit_reached")
        if deadline is not None and time.monotonic() > deadline:
            return result(False, queue[0][0], "time_limit_reached")
        lower, _, node_a, node_b = heapq.heappop(queue)
        node_visits += 1
        if lower >= best:
            nodes_pruned += 1
            continue
        if not (node_a.leaf and node_b.leaf):
            if not node_a.leaf and (node_b.leaf or len(node_a.indices) >= len(node_b.indices)):
                for child in node_a.children():
                    push(child, node_b)
            else:
                for child in node_b.children():
                    push(node_a, child)
            continue
        for i, box_a, triangle_a in node_a.items():
            for j, box_b, triangle_b in node_b.items():
                if i == 0 and j == 0:
                    continue  # Already evaluated for the initial upper bound.
                if pairs_checked >= max_pairs:
                    remaining_lower = min(lower, queue[0][0]) if queue else lower
                    return result(False, remaining_lower, "triangle_pair_limit_reached")
                pairs_checked += 1
                if _bounds_distance2(box_a, box_b) >= best:
                    continue
                pairs_evaluated += 1
                distance, point_a, point_b = _triangle_pair(triangle_a, triangle_b)
                if not math.isfinite(distance):
                    raise ValueError("Geometry exceeds supported floating-point range")
                if distance < best:
                    best, closest_a, closest_b = distance, point_a, point_b
                if best == 0.0:
                    return result(True, 0.0, "surfaces_intersect")
    return result(True, best, "search_complete")
