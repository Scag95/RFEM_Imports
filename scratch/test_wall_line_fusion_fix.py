import sys
import os
import math
import ezdxf

from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor, point_to_segment_distance_xy
from src.core.geometry import EntityGeometry

def test_wall_fusion(dxf_path):
    print("=" * 60)
    print(f"TESTING CONTINUOUS WALL FUSION: {os.path.basename(dxf_path)}")
    print("=" * 60)

    doc = ezdxf.readfile(dxf_path)
    geometries, _ = collect_entity_geometries(doc.modelspace())

    processor = StructuralProcessor(geometries)
    processor.execute_preprocesar_cadwork()
    processor.execute_merge_durmientes_testeros(z_drop=250.0)
    processor.execute_merge_durmientes_testeros_contrapeados(z_drop=250.0)

    tol_cota_nivel = 1125.0
    processor.entity_geometries = [e for e in processor.entity_geometries if "COTA" not in e.layer.upper()]

    durmientes = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "DURMIENTE")]
    testeros = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "TESTERO")]
    montantes = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "MONTANTE")]

    cota_entities = []

    # 1. Level lines for ALL DURMIENTE plates
    for d in durmientes:
        if not d.segments:
            continue
        x1, y1, z1, x2, y2, z2 = d.segments[0]
        z_base = min(z1, z2)
        z_cota = z_base + tol_cota_nivel
        suffix = "(No Pasante)" if processor._layer_is_no_pasante(d.layer) else "(Pasante)"
        layer_name = f"Linea Cota de Nivel {suffix}"
        cota_entities.append(EntityGeometry(
            handle=f"COTA_GEN_{d.handle}",
            layer=layer_name,
            segments=[(x1, y1, z_cota, x2, y2, z_cota)]
        ))

    # 2. Level lines for PASANTE TESTEROs that have montantes above them
    pasante_testeros = [t for t in testeros if not processor._layer_is_no_pasante(t.layer)]
    pasante_montantes = [m for m in montantes if not processor._layer_is_no_pasante(m.layer)]

    for t in pasante_testeros:
        if not t.segments:
            continue
        tx1, ty1, tz1, tx2, ty2, tz2 = t.segments[0]
        tz_base = min(tz1, tz2)

        has_montantes_above = False
        for m in pasante_montantes:
            if not m.segments:
                continue
            mx1, my1, mz1, mx2, my2, mz2 = m.segments[0]
            mz_min = min(mz1, mz2)
            mz_max = max(mz1, mz2)
            if (tz_base - 100.0 <= mz_min <= tz_base + 350.0) and (mz_max >= tz_base + 1000.0):
                mx_mid = (mx1 + mx2) / 2.0
                my_mid = (my1 + my2) / 2.0
                dist_xy = point_to_segment_distance_xy(mx_mid, my_mid, tx1, ty1, tx2, ty2)
                if dist_xy <= 200.0:
                    has_montantes_above = True
                    break

        if has_montantes_above:
            z_cota = tz_base + tol_cota_nivel
            layer_name = "Linea Cota de Nivel (Pasante)"
            cota_entities.append(EntityGeometry(
                handle=f"COTA_GEN_{t.handle}",
                layer=layer_name,
                segments=[(tx1, ty1, z_cota, tx2, ty2, z_cota)]
            ))

    processor.entity_geometries.extend(cota_entities)

    # REFINED FUSE_LEVEL_LINES: Merge collinear segments FIRST, then clamp ONLY wall endpoints
    tol_snap = 300.0
    cota_entities_to_fuse = [e for e in processor.entity_geometries if "COTA" in e.layer.upper()]
    other_entities = [e for e in processor.entity_geometries if e not in cota_entities_to_fuse]

    all_zs = [e.segments[0][2] for e in cota_entities_to_fuse if e.segments]
    z_clusters = processor._cluster_z_levels(all_zs, tol_z=150.0)

    def get_clustered_z(z: float) -> float:
        for cz in z_clusters:
            if abs(z - cz) <= 150.0:
                return cz
        return round(z / 50.0) * 50.0

    floor_layer_groups = {}
    for e in cota_entities_to_fuse:
        if not e.segments:
            continue
        seg = e.segments[0]
        z_floor = get_clustered_z(seg[2])
        key = (z_floor, e.layer)
        if key not in floor_layer_groups:
            floor_layer_groups[key] = []
        floor_layer_groups[key].append(e)

    fused_cotas = []

    for (z_floor, layer_name), group in floor_layer_groups.items():
        x_lines = []
        y_lines = []

        for e in group:
            seg = e.segments[0]
            x1, y1, z1, x2, y2, z2 = seg
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dy <= 10.0:
                y_val = (y1 + y2) / 2.0
                min_x, max_x = min(x1, x2), max(x1, x2)
                x_lines.append((y_val, min_x, max_x, z1))
            elif dx <= 10.0:
                x_val = (x1 + x2) / 2.0
                min_y, max_y = min(y1, y2), max(y1, y2)
                y_lines.append((x_val, min_y, max_y, z1))

        # 1. Merge X-parallel lines
        y_grouped = {}
        for y_val, min_x, max_x, z in x_lines:
            y_key = round(y_val / 50.0) * 50.0
            if y_key not in y_grouped:
                y_grouped[y_key] = []
            y_grouped[y_key].append((y_val, min_x, max_x, z))

        floor_merged_segments = []

        for y_key, intervals in y_grouped.items():
            avg_y = sum(item[0] for item in intervals) / len(intervals)
            intervals.sort(key=lambda x: x[1])
            merged = []
            for y_val, cur_min, cur_max, z in intervals:
                if not merged:
                    merged.append([cur_min, cur_max, z])
                else:
                    prev = merged[-1]
                    # SNAP TOLERANCE: Merge adjacent or slightly separated segments on the same wall line!
                    if cur_min <= prev[1] + tol_snap:
                        prev[1] = max(prev[1], cur_max)
                    else:
                        merged.append([cur_min, cur_max, z])
            for min_x, max_x, z in merged:
                floor_merged_segments.append([min_x, avg_y, z, max_x, avg_y, z])

        # 2. Merge Y-parallel lines
        x_grouped = {}
        for x_val, min_y, max_y, z in y_lines:
            x_key = round(x_val / 50.0) * 50.0
            if x_key not in x_grouped:
                x_grouped[x_key] = []
            x_grouped[x_key].append((x_val, min_y, max_y, z))

        for x_key, intervals in x_grouped.items():
            avg_x = sum(item[0] for item in intervals) / len(intervals)
            intervals.sort(key=lambda x: x[1])
            merged = []
            for x_val, cur_min, cur_max, z in intervals:
                if not merged:
                    merged.append([cur_min, cur_max, z])
                else:
                    prev = merged[-1]
                    if cur_min <= prev[1] + tol_snap:
                        prev[1] = max(prev[1], cur_max)
                    else:
                        merged.append([cur_min, cur_max, z])
            for min_y, max_y, z in merged:
                floor_merged_segments.append([avg_x, min_y, z, avg_x, max_y, z])

        # 3. NOW clamp ONLY the endpoints of each fully fused continuous wall line to montantes
        montantes = [m for m in other_entities if "MONTANTE" in m.layer.upper()]
        floor_montantes = [
            m for m in montantes
            if m.segments and (min(m.segments[0][2], m.segments[0][5]) - 250.0 <= z_floor <= max(m.segments[0][2], m.segments[0][5]) + 250.0)
        ]

        for k in range(len(floor_merged_segments)):
            seg = floor_merged_segments[k]
            sx1, sy1, sz1, sx2, sy2, sz2 = seg
            sdx = abs(sx2 - sx1)
            sdy = abs(sy2 - sy1)
            s_is_x = sdx >= sdy

            wall_m_projs = []
            for m in floor_montantes:
                mseg = m.segments[0]
                mx = (mseg[0] + mseg[3]) / 2.0
                my = (mseg[1] + mseg[4]) / 2.0
                if s_is_x:
                    if abs(my - (sy1 + sy2) / 2.0) <= 150.0:
                        if min(sx1, sx2) - 300.0 <= mx <= max(sx1, sx2) + 300.0:
                            wall_m_projs.append(mx)
                else:
                    if abs(mx - (sx1 + sx2) / 2.0) <= 150.0:
                        if min(sy1, sy2) - 300.0 <= my <= max(sy1, sy2) + 300.0:
                            wall_m_projs.append(my)

            if wall_m_projs:
                min_m = min(wall_m_projs)
                max_m = max(wall_m_projs)
                if s_is_x:
                    nx1 = max(min_m, min(max_m, sx1))
                    nx2 = max(min_m, min(max_m, sx2))
                    floor_merged_segments[k] = [nx1, sy1, sz1, nx2, sy2, sz2]
                else:
                    ny1 = max(min_m, min(max_m, sy1))
                    ny2 = max(min_m, min(max_m, sy2))
                    floor_merged_segments[k] = [sx1, ny1, sz1, sx2, ny2, sz2]

        for idx, seg in enumerate(floor_merged_segments):
            if math.hypot(seg[3] - seg[0], seg[4] - seg[1]) <= 1.0:
                continue
            fused_cotas.append(
                EntityGeometry(
                    handle=f"COTA_FUSED_{z_floor:.0f}_{idx}",
                    layer=layer_name,
                    segments=[(seg[0], seg[1], seg[2], seg[3], seg[4], seg[5])],
                )
            )

    print(f"Total fused level lines: {len(fused_cotas)}")
    for c in fused_cotas[:10]:
        length = math.hypot(c.segments[0][3]-c.segments[0][0], c.segments[0][4]-c.segments[0][1])
        print(f"  Fused Line [{c.handle}] Layer: {c.layer:<35} Z: {c.segments[0][2]:.1f} Len: {length:.1f}")

test_wall_fusion(r"c:\PythonProjects\RFEM_Imports\Panel No Pasante_unido.dxf")
test_wall_fusion(r"c:\PythonProjects\RFEM_Imports\LT082-LGT-EST-FCH-M3-G01-001-Estructura_1A_editado.dxf")
