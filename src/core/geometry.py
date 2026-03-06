from __future__ import annotations
import math
from dataclasses import dataclass


from collections import defaultdict
from math import isclose
from typing import Iterable


from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple

from shapely.geometry import LineString
from shapely.ops import unary_union, polygonize
from shapely.geometry.polygon import orient

Point3 = tuple[float, float, float]
Segment3 = tuple[float, float, float, float, float, float]
Segment3D = tuple[float, float, float, float, float, float]
Segment2D = tuple[float, float, float, float]

@dataclass
class EntityGeometry:
    handle: str
    layer: str
    segments: list[Segment3]
    segments: list[tuple[float, float, float, float, float, float]]



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

def corregir_direccion_barras(
    geometries: list[EntityGeometry],
    tolerance_z: float = 0.1,
    tolerance_xy: float = 1e-3,
) -> tuple[list[EntityGeometry], int]:
    """
    Corrige la dirección de las barras horizontales reconstruyendo los polígonos
    reales en planta XY.

    Regla:
    - Solo procesa entidades cuya capa NO contiene "MONTANTE".
    - Considera horizontal si |z2 - z1| <= tolerance_z.
    - Agrupa por cotas Z parecidas.
    - Polygoniza las líneas en XY.
    - Usa la orientación del contorno del polígono:
        * exterior: antihorario
        * huecos: horario
    - Si un segmento no logra asociarse a ningún contorno polygonizado,
      se deja igual.

    Devuelve:
        (geometries_corregidas, cantidad_segmentos_corregidos)
    """
    corrected_geometries: list[EntityGeometry] = []
    corrected_count = 0

    for entity in geometries:
        layer_name = (entity.layer or "").upper()
        if "MONTANTE" in layer_name:
            corrected_geometries.append(entity)
            continue

        horizontales: list[tuple[int, Segment3D, float]] = []
        for idx, seg in enumerate(entity.segments):
            x1, y1, z1, x2, y2, z2 = seg
            if abs(z2 - z1) <= tolerance_z:
                z_ref = 0.5 * (z1 + z2)
                horizontales.append((idx, seg, z_ref))

        if not horizontales:
            corrected_geometries.append(entity)
            continue

        grupos_z = agrupar_por_cota_z(horizontales, tolerance_z)

        replacements: Dict[int, Segment3D] = {}

        for grupo in grupos_z:
            segmentos_grupo = [(idx, seg) for idx, seg, _ in grupo]

            # Mapa: borde no dirigido -> borde dirigido correcto según polygonize
            directed_edges = construir_mapa_bordes_orientados(segmentos_grupo, tolerance_xy)

            for idx, seg in segmentos_grupo:
                x1, y1, z1, x2, y2, z2 = seg
                undirected_key = edge_key_undirected((x1, y1), (x2, y2), tolerance_xy)

                if undirected_key not in directed_edges:
                    # No se pudo asociar a un polígono real: se deja igual
                    replacements[idx] = seg
                    continue

                (ax, ay), (bx, by) = directed_edges[undirected_key]

                p_start = snap_point((x1, y1), tolerance_xy)
                p_end = snap_point((x2, y2), tolerance_xy)

                a_snap = snap_point((ax, ay), tolerance_xy)
                b_snap = snap_point((bx, by), tolerance_xy)

                if p_start == a_snap and p_end == b_snap:
                    replacements[idx] = seg
                elif p_start == b_snap and p_end == a_snap:
                    replacements[idx] = (x2, y2, z2, x1, y1, z1)
                    corrected_count += 1
                else:
                    # Si por tolerancias no coincide perfecto, dejar como está
                    replacements[idx] = seg

        new_segments = list(entity.segments)
        for idx, seg_corr in replacements.items():
            new_segments[idx] = seg_corr

        corrected_geometries.append(
            EntityGeometry(
                handle=entity.handle,
                layer=entity.layer,
                segments=new_segments,
            )
        )

    return corrected_geometries, corrected_count


# ---------------------------------------------------------------------
# Agrupación por Z
# ---------------------------------------------------------------------

def agrupar_por_cota_z(
    horizontales_info: list[tuple[int, Segment3D, float]],
    tolerance_z: float,
) -> list[list[tuple[int, Segment3D, float]]]:
    if not horizontales_info:
        return []

    horizontales_ordenados = sorted(horizontales_info, key=lambda item: item[2])

    grupos = []
    actual = [horizontales_ordenados[0]]
    z_ref_actual = horizontales_ordenados[0][2]

    for item in horizontales_ordenados[1:]:
        _, _, z_ref = item
        if abs(z_ref - z_ref_actual) <= tolerance_z:
            actual.append(item)
        else:
            grupos.append(actual)
            actual = [item]
            z_ref_actual = z_ref

    grupos.append(actual)
    return grupos


# ---------------------------------------------------------------------
# Polygonize + orientación
# ---------------------------------------------------------------------

def construir_mapa_bordes_orientados(
    segmentos_grupo: list[tuple[int, Segment3D]],
    tolerance_xy: float,
) -> dict[
    tuple[tuple[int, int], tuple[int, int]],
    tuple[tuple[float, float], tuple[float, float]]
]:
    """
    Devuelve un mapa:
        borde_no_dirigido -> borde_dirigido_correcto

    El borde dirigido correcto se extrae de:
    - exteriores CCW
    - interiores CW
    """
    lines = []

    for _, seg in segmentos_grupo:
        x1, y1, _z1, x2, y2, _z2 = seg
        p1 = unsnap_point((x1, y1), tolerance_xy)
        p2 = unsnap_point((x2, y2), tolerance_xy)

        if p1 != p2:
            lines.append(LineString([p1, p2]))

    if not lines:
        return {}

    merged = unary_union(lines)
    polygons = list(polygonize(merged))

    directed_edges = {}

    for poly in polygons:
        # exterior en sentido antihorario
        poly_ccw = orient(poly, sign=1.0)

        # anillo exterior
        exterior_coords = list(poly_ccw.exterior.coords)
        agregar_edges_de_ring(exterior_coords, directed_edges, tolerance_xy)

        # huecos interiores: los dejamos en el sentido que trae shapely
        # normalmente CW cuando el exterior está CCW
        for ring in poly_ccw.interiors:
            interior_coords = list(ring.coords)
            agregar_edges_de_ring(interior_coords, directed_edges, tolerance_xy)

    return directed_edges


def agregar_edges_de_ring(
    coords: list[tuple[float, float]],
    directed_edges: dict,
    tolerance_xy: float,
) -> None:
    for i in range(len(coords) - 1):
        a = coords[i]
        b = coords[i + 1]

        if snap_point(a, tolerance_xy) == snap_point(b, tolerance_xy):
            continue

        key = edge_key_undirected(a, b, tolerance_xy)

        # Guardar una única dirección "correcta" por borde
        if key not in directed_edges:
            directed_edges[key] = (a, b)


# ---------------------------------------------------------------------
# Utilidades de tolerancia
# ---------------------------------------------------------------------

def snap_point(pt: tuple[float, float], tol: float) -> tuple[int, int]:
    return (round(pt[0] / tol), round(pt[1] / tol))


def unsnap_point(pt: tuple[float, float], tol: float) -> tuple[float, float]:
    return (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)


def edge_key_undirected(
    p1: tuple[float, float],
    p2: tuple[float, float],
    tol: float,
) -> tuple[tuple[int, int], tuple[int, int]]:
    a = snap_point(p1, tol)
    b = snap_point(p2, tol)
    return (a, b) if a <= b else (b, a)