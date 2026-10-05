import sys
import os
import math
import ezdxf

from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor

dxf_path = r"c:\PythonProjects\RFEM_Imports\LT082-LGT-EST-FCH-M3-G01-001-Estructura_1A_editado.dxf"
doc = ezdxf.readfile(dxf_path)
geometries, _ = collect_entity_geometries(doc.modelspace())

processor = StructuralProcessor(geometries)
processor.execute_preprocesar_cadwork()
processor.execute_merge_durmientes_testeros(z_drop=250.0)

# Inspect all plates and montantes
plates = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "DURMIENTE") or processor._is_layer_match(e.layer, "TESTERO")]
montantes = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "MONTANTE")]

print(f"Total plates remaining: {len(plates)}")
print(f"Total montantes: {len(montantes)}")

# Check floor Z levels of plates
durmientes = [e for e in plates if processor._is_layer_match(e.layer, "DURMIENTE")]
testeros = [e for e in plates if processor._is_layer_match(e.layer, "TESTERO")]

print("\n--- DURMIENTES Z LEVELS ---")
dur_z_set = set(round(min(e.segments[0][2], e.segments[0][5])) for e in durmientes)
print(sorted(list(dur_z_set)))

print("\n--- TESTEROS Z LEVELS ---")
tes_z_set = set(round(min(e.segments[0][2], e.segments[0][5])) for e in testeros)
print(sorted(list(tes_z_set)))

# Proposed rule for level lines:
# 1. All DURMIENTES get a level line at min(z1, z2) + 1125.0
# 2. Any TESTERO that has montantes extending ABOVE it (e.g. montantes with z_min >= tz - 100 and z_max > tz + 1000) also gets a level line at min(z1, z2) + 1125.0!

cota_candidates = []

for d in durmientes:
    z_base = min(d.segments[0][2], d.segments[0][5])
    cota_candidates.append((d, z_base + 1125.0, "DURMIENTE"))

for t in testeros:
    tx1, ty1, tz1, tx2, ty2, tz2 = t.segments[0]
    tz_base = min(tz1, tz2)
    # Check if there are montantes extending above this testero
    has_montantes_above = False
    for m in montantes:
        mx1, my1, mz1, mx2, my2, mz2 = m.segments[0]
        mz_min = min(mz1, mz2)
        mz_max = max(mz1, mz2)
        if (tz_base - 100.0 <= mz_min <= tz_base + 300.0) and (mz_max >= tz_base + 1000.0):
            # Check XY proximity
            dist_xy = processor._point_to_segment_distance_xy((mx1+mx2)/2.0, (my1+my2)/2.0, tx1, ty1, tx2, ty2) if hasattr(processor, '_point_to_segment_distance_xy') else max(abs((mx1+mx2)/2.0 - (tx1+tx2)/2.0), abs((my1+my2)/2.0 - (ty1+ty2)/2.0))
            if dist_xy <= 300.0:
                has_montantes_above = True
                break
    if has_montantes_above:
        cota_candidates.append((t, tz_base + 1125.0, "TESTERO_BASE"))

print(f"\nTotal level line candidate plates: {len(cota_candidates)}")

level_zs = sorted(list(set(round(c[1]) for c in cota_candidates)))
print(f"Generated Level Line Z Heights: {level_zs}")
