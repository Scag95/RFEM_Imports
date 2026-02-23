from collections import defaultdict

from src.core.geometry import EntityGeometry, Segment3

def get_entity_segments(entity) -> list[Segment3]:
    entity_type = entity.dxftype()
    entity_segments: list[Segment3] = []

    if entity_type == "LINE":
        start = entity.dxf.start
        end = entity.dxf.end
        entity_segments.append((start.x, start.y, start.z, end.x, end.y, end.z))
        return entity_segments

    if entity_type == "LWPOLYLINE":
        elevation = float(getattr(entity.dxf, "elevation", 0.0))
        points = [(p[0], p[1], elevation) for p in entity.get_points("xy")]
        if len(points) < 2:
            return entity_segments

        for (x1, y1, z1), (x2, y2, z2) in zip(points, points[1:]):
            entity_segments.append((x1, y1, z1, x2, y2, z2))

        if entity.closed:
            x1, y1, z1 = points[-1]
            x2, y2, z2 = points[0]
            entity_segments.append((x1, y1, z1, x2, y2, z2))
        return entity_segments

    if entity_type == "POLYLINE":
        points = [
            (v.dxf.location.x, v.dxf.location.y, v.dxf.location.z) for v in entity.vertices
        ]
        if len(points) < 2:
            return entity_segments

        for (x1, y1, z1), (x2, y2, z2) in zip(points, points[1:]):
            entity_segments.append((x1, y1, z1, x2, y2, z2))

        if entity.is_closed:
            x1, y1, z1 = points[-1]
            x2, y2, z2 = points[0]
            entity_segments.append((x1, y1, z1, x2, y2, z2))
        return entity_segments

    return entity_segments

def collect_entity_geometries(msp) -> tuple[list[EntityGeometry], dict[str, int]]:
    entity_geometries: list[EntityGeometry] = []
    layer_counts: dict[str, int] = defaultdict(int)

    for index, entity in enumerate(msp):
        layer = entity.dxf.layer
        entity_segments = get_entity_segments(entity)
        if not entity_segments:
            continue

        handle = getattr(entity.dxf, "handle", None)
        handle_id = str(handle) if handle else f"NO_HANDLE_{index}"
        entity_geometries.append(
            EntityGeometry(
                handle=handle_id,
                layer=layer,
                segments=entity_segments,
            )
        )
        layer_counts[layer] += len(entity_segments)

    return entity_geometries, layer_counts

def flatten_segments_for_viewer(
    entities: list[EntityGeometry]
) -> list[tuple[float, float, float, float, float, float, str]]:
    flat_segments: list[tuple[float, float, float, float, float, float, str]] = []
    for entity in entities:
        for x1, y1, z1, x2, y2, z2 in entity.segments:
            flat_segments.append((x1, y1, z1, x2, y2, z2, entity.layer))
    return flat_segments
