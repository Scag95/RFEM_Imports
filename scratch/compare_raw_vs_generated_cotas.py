import sys
import os
import math
import ezdxf

from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor, point_to_segment_distance_xy

dxf_path = r"c:\PythonProjects\RFEM_Imports\LT082-LGT-EST-FCH-M3-G01-001-Estructura_1A_editado.dxf"
doc = ezdxf.readfile(dxf_path)
geometries, _ = collect_entity_geometries(doc.modelspace())

raw_cotas = [g for g in geometries if "COTA" in g.layer.upper()]
print(f"Total raw cotas in DXF: {len(raw_cotas)}")

raw_cotas_by_z = {}
for c in raw_cotas:
    z = round(c.segments[0][2])
    key = (z, c.layer)
    raw_cotas_by_z[key] = raw_cotas_by_z.get(key, 0) + 1

print("\n--- RAW DXF COTAS BY Z AND LAYER ---")
for (z, layer), count in sorted(raw_cotas_by_z.items()):
    print(f"Z: {z:<6} Layer: {layer:<45} Count: {count}")

processor = StructuralProcessor(geometries)
processor.execute_preprocesar_cadwork()
processor.execute_merge_durmientes_testeros(z_drop=250.0)
processor.execute_merge_durmientes_testeros_contrapeados(z_drop=250.0)
processor.generate_level_lines()

gen_cotas = [e for e in processor.entity_geometries if "COTA" in e.layer.upper()]
print(f"\nTotal generated cotas: {len(gen_cotas)}")

gen_cotas_by_z = {}
for c in gen_cotas:
    z = round(c.segments[0][2])
    key = (z, c.layer)
    gen_cotas_by_z[key] = gen_cotas_by_z.get(key, 0) + 1

print("\n--- GENERATED COTAS BY Z AND LAYER ---")
for (z, layer), count in sorted(gen_cotas_by_z.items()):
    print(f"Z: {z:<6} Layer: {layer:<45} Count: {count}")

# Check raw cotas that do not match any generated cota in Z and 2D location
unmatched_raw = []
for rc in raw_cotas:
    rx1, ry1, rz1, rx2, ry2, rz2 = rc.segments[0]
    rmx, rmy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0
    r_len = math.hypot(rx2 - rx1, ry2 - ry1)
    if r_len <= 10.0:
        continue

    matched = False
    for gc in gen_cotas:
        gx1, gy1, gz1, gx2, gy2, gz2 = gc.segments[0]
        if abs(gz1 - rz1) <= 150.0:
            dist = point_to_segment_distance_xy(rmx, rmy, gx1, gy1, gx2, gy2)
            if dist <= 200.0:
                matched = True
                break
    if not matched:
        unmatched_raw.append((rc, rz1, r_len, (rx1, ry1), (rx2, ry2)))

print(f"\nUnmatched Raw Cotas Count: {len(unmatched_raw)}")
for rc, z, r_len, p1, p2 in unmatched_raw[:20]:
    print(f"  Raw Cota [{rc.handle}] Layer: {rc.layer:<42} Z: {z:.1f} P1: ({p1[0]:.1f}, {p1[1]:.1f}) P2: ({p2[0]:.1f}, {p2[1]:.1f}) Len: {r_len:.1f}")
