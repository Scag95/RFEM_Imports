import sys
import os
import ezdxf

from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor, point_to_segment_distance_xy

dxf_path = r"c:\PythonProjects\RFEM_Imports\LT082-LGT-EST-FCH-M3-G01-001-Estructura_1A_editado.dxf"
doc = ezdxf.readfile(dxf_path)
geometries, _ = collect_entity_geometries(doc.modelspace())

processor = StructuralProcessor(geometries)
processor.execute_preprocesar_cadwork()
processor.execute_merge_durmientes_testeros(z_drop=250.0)
processor.execute_merge_durmientes_testeros_contrapeados(z_drop=250.0)

testeros = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "TESTERO") and not processor._layer_is_no_pasante(e.layer)]
montantes = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "MONTANTE") and not processor._layer_is_no_pasante(e.layer)]

print(f"Pasante testeros count: {len(testeros)}")
print(f"Pasante montantes count: {len(montantes)}")

for t in testeros[:10]:
    t_z1, t_z2 = t.segments[0][2], t.segments[0][5]
    tz_base = min(t_z1, t_z2)
    tx1, ty1, tx2, ty2 = t.segments[0][0], t.segments[0][1], t.segments[0][3], t.segments[0][4]
    
    # Find all montantes near this testero in XY
    near_m = []
    for m in montantes:
        mx1, my1, mz1, mx2, my2, mz2 = m.segments[0]
        mz_min = min(mz1, mz2)
        mz_max = max(mz1, mz2)
        dist_xy = point_to_segment_distance_xy((mx1+mx2)/2.0, (my1+my2)/2.0, tx1, ty1, tx2, ty2)
        if dist_xy <= 300.0:
            near_m.append((m, mz_min, mz_max))
            
    print(f"\nTestero handle={t.handle} layer={t.layer} Z={tz_base:.1f}: found {len(near_m)} near montantes in XY:")
    for m, mz_min, mz_max in near_m[:5]:
        z_delta = mz_min - tz_base
        print(f"  - Montante handle={m.handle} layer={m.layer} mz_min={mz_min:.1f} (delta={z_delta:.1f}) mz_max={mz_max:.1f}")
