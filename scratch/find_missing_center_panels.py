import sys
import os
import math
import ezdxf

from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor, point_to_segment_distance_xy

dxf_path = r"c:\PythonProjects\RFEM_Imports\LT082-LGT-EST-FCH-M3-G01-001-Estructura_1A_editado.dxf"
doc = ezdxf.readfile(dxf_path)
geometries, _ = collect_entity_geometries(doc.modelspace())

processor = StructuralProcessor(geometries)

# Step 1: Execute preprocesar_cadwork and inspect panels formed
processor.execute_preprocesar_cadwork()
processor.execute_merge_durmientes_testeros(z_drop=250.0)
processor.execute_merge_durmientes_testeros_contrapeados(z_drop=250.0)
processor.generate_level_lines()

# Collect all montantes
montantes = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "MONTANTE")]
cotas = [e for e in processor.entity_geometries if "COTA" in e.layer.upper()]

print(f"Total montantes: {len(montantes)}")
print(f"Total cotas: {len(cotas)}")

# For every montante M, check if there is a level line intersecting M (i.e. crossing M's X/Y position at M's floor level + 1125mm)
uncovered_montantes = []
for m in montantes:
    if not m.segments:
        continue
    mx1, my1, mz1, mx2, my2, mz2 = m.segments[0]
    mz_min = min(mz1, mz2)
    mz_max = max(mz1, mz2)

    # A full-height montante should have a level line at mz_min + 1125.0
    if mz_max - mz_min < 1500.0:  # Skip short montantes (cripples / dinteles / alfeizares)
        continue

    expected_cota_z = mz_min + 1125.0
    mx_mid, my_mid = (mx1 + mx2) / 2.0, (my1 + my2) / 2.0

    # Search for level line near this montante in 2D and Z
    has_cota = False
    for c in cotas:
        cx1, cy1, cz1, cx2, cy2, cz2 = c.segments[0]
        if abs(cz1 - expected_cota_z) <= 250.0:
            dist = point_to_segment_distance_xy(mx_mid, my_mid, cx1, cy1, cx2, cy2)
            if dist <= 150.0:
                has_cota = True
                break
    if not has_cota:
        uncovered_montantes.append((m, mz_min, mz_max, expected_cota_z))

print(f"\nUncovered Full-Height Montantes Count: {len(uncovered_montantes)}")
by_z_layer_uncovered = {}
for m, mz_min, mz_max, expected_z in uncovered_montantes:
    key = (round(expected_z), m.layer)
    by_z_layer_uncovered[key] = by_z_layer_uncovered.get(key, 0) + 1

for (z_val, layer_name), count in sorted(by_z_layer_uncovered.items()):
    print(f"  Expected Cota Z: {z_val:<8} Montante Layer: {layer_name:<40} Count: {count}")

if uncovered_montantes:
    print("\nSample Uncovered Montantes:")
    for m, mz_min, mz_max, expected_z in uncovered_montantes[:10]:
        x1, y1 = m.segments[0][0], m.segments[0][1]
        print(f"   Handle [{m.handle}] Layer: {m.layer:<35} Z: {mz_min:.1f}..{mz_max:.1f} Pos: ({x1:.1f}, {y1:.1f})")
