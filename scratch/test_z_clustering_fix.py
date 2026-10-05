import sys
import os
import math
import ezdxf

from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor, point_to_segment_distance_xy
from src.core.geometry import EntityGeometry

def test_z_clustering(dxf_path):
    print("=" * 60)
    print(f"TESTING Z CLUSTERING FIX ON: {os.path.basename(dxf_path)}")
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

    # Improved fuse_level_lines with Z clustering by 100mm tolerance
    cota_entities_to_fuse = [e for e in processor.entity_geometries if "COTA" in e.layer.upper()]
    other_entities = [e for e in processor.entity_geometries if e not in cota_entities_to_fuse]

    # Cluster Z values
    all_zs = [e.segments[0][2] for e in cota_entities_to_fuse if e.segments]
    z_clusters = processor._cluster_z_levels(all_zs, tol_z=100.0) if hasattr(processor, '_cluster_z_levels') else []

    def get_clustered_z(z):
        for cz in z_clusters:
            if abs(z - cz) <= 100.0:
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

    print(f"Total floor Z-layer groups after clustering: {len(floor_layer_groups)}")
    for (z_val, layer_name), group in sorted(floor_layer_groups.items()):
        print(f"  Floor Z: {z_val:<8.1f} Layer: {layer_name:<35} Count: {len(group)}")

test_z_clustering(r"c:\PythonProjects\RFEM_Imports\Panel No Pasante_unido.dxf")
test_z_clustering(r"c:\PythonProjects\RFEM_Imports\LT082-LGT-EST-FCH-M3-G01-001-Estructura_1A_editado.dxf")
