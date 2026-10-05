import sys
import os
import math
import ezdxf

from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor, point_to_segment_distance_xy
from src.core.geometry import EntityGeometry

dxf_path = r"c:\PythonProjects\RFEM_Imports\LT082-LGT-EST-FCH-M3-G01-001-Estructura_1A_editado.dxf"
doc = ezdxf.readfile(dxf_path)
geometries, _ = collect_entity_geometries(doc.modelspace())

processor = StructuralProcessor(geometries)
processor.execute_preprocesar_cadwork()
processor.execute_merge_durmientes_testeros(z_drop=250.0)
processor.execute_merge_durmientes_testeros_contrapeados(z_drop=250.0)

# Proposed new generate_level_lines implementation
tol_cota_nivel = 1125.0
processor.entity_geometries = [e for e in processor.entity_geometries if "COTA" not in e.layer.upper()]

durmientes = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "DURMIENTE")]
testeros = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "TESTERO")]
montantes = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "MONTANTE")]

cota_entities = []

# Generate level lines for ALL durmientes on Linea Cota de Nivel (Pasante)
# (and if layer was No Pasante, we also generate Pasante so all floors get main level lines)
for d in durmientes:
    if not d.segments:
        continue
    x1, y1, z1, x2, y2, z2 = d.segments[0]
    z_base = min(z1, z2)
    z_cota = z_base + tol_cota_nivel
    layer_name = "Linea Cota de Nivel (Pasante)"
    cota_entities.append(EntityGeometry(
        handle=f"COTA_GEN_{d.handle}",
        layer=layer_name,
        segments=[(x1, y1, z_cota, x2, y2, z_cota)]
    ))
    if processor._layer_is_no_pasante(d.layer):
        # Optionally also add No Pasante for layer symmetry if user filters by No Pasante
        cota_entities.append(EntityGeometry(
            handle=f"COTA_GEN_NP_{d.handle}",
            layer="Linea Cota de Nivel (No Pasante)",
            segments=[(x1, y1, z_cota, x2, y2, z_cota)]
        ))

# Generate level lines for TESTEROs that serve as base for upper floor walls
for t in testeros:
    if not t.segments:
        continue
    tx1, ty1, tz1, tx2, ty2, tz2 = t.segments[0]
    tz_base = min(tz1, tz2)

    has_montantes_above = False
    for m in montantes:
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
        cota_entities.append(EntityGeometry(
            handle=f"COTA_GEN_{t.handle}",
            layer="Linea Cota de Nivel (Pasante)",
            segments=[(tx1, ty1, z_cota, tx2, ty2, z_cota)]
        ))
        if processor._layer_is_no_pasante(t.layer):
            cota_entities.append(EntityGeometry(
                handle=f"COTA_GEN_NP_{t.handle}",
                layer="Linea Cota de Nivel (No Pasante)",
                segments=[(tx1, ty1, z_cota, tx2, ty2, z_cota)]
            ))

processor.entity_geometries.extend(cota_entities)
success, msg = processor.fuse_level_lines()

cotas = [e for e in processor.entity_geometries if "COTA" in e.layer.upper()]
print(f"Total fused level lines: {len(cotas)}")

by_z_layer = {}
for c in cotas:
    z = round(c.segments[0][2])
    key = (z, c.layer)
    by_z_layer[key] = by_z_layer.get(key, 0) + 1

print("\n--- LEVEL LINES BY FLOOR Z AND LAYER ---")
for (z_val, layer_name), count in sorted(by_z_layer.items()):
    print(f"Floor Z: {z_val:<8} Layer: {layer_name:<40} Count: {count}")
