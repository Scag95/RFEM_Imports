import math
from dataclasses import dataclass

Point3 = tuple[float, float, float]
Segment3 = tuple[float, float, float, float, float, float]

@dataclass
class EntityGeometry:
    handle: str
    layer: str
    segments: list[Segment3]

def point_to_segment_distance(point: Point3, segment: Segment3) -> float:
    px, py, pz = point
    x1, y1, z1, x2, y2, z2 = segment
    vx, vy, vz = x2 - x1, y2 - y1, z2 - z1
    wx, wy, wz = px - x1, py - y1, pz - z1

    segment_len_sq = vx * vx + vy * vy + vz * vz
    if segment_len_sq <= 1e-12:
        dx, dy, dz = px - x1, py - y1, pz - z1
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    t = (wx * vx + wy * vy + wz * vz) / segment_len_sq
    t = max(0.0, min(1.0, t))
    cx, cy, cz = x1 + t * vx, y1 + t * vy, z1 + t * vz
    dx, dy, dz = px - cx, py - cy, pz - cz
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def segments_are_connected(segment_a: Segment3, segment_b: Segment3, tol: float) -> bool:
    a_start: Point3 = (segment_a[0], segment_a[1], segment_a[2])
    a_end: Point3 = (segment_a[3], segment_a[4], segment_a[5])
    b_start: Point3 = (segment_b[0], segment_b[1], segment_b[2])
    b_end: Point3 = (segment_b[3], segment_b[4], segment_b[5])

    return (
        point_to_segment_distance(a_start, segment_b) <= tol
        or point_to_segment_distance(a_end, segment_b) <= tol
        or point_to_segment_distance(b_start, segment_a) <= tol
        or point_to_segment_distance(b_end, segment_a) <= tol
    )
