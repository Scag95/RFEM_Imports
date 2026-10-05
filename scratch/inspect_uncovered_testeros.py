import sys
import os
import math
import ezdxf

from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor, point_to_segment_distance_xy

dxf_path = r"c:\PythonProjects\RFEM_Imports\Panel No Pasante_unido.dxf"
doc = ezdxf.readfile(dxf_path)
geometries, _ = collect_entity_geometries(doc.modelspace())

processor = StructuralProcessor(geometries)
processor.execute_preprocesar_cadwork()
processor.execute_merge_durmientes_testeros(z_drop=250.0)
processor.execute_merge_durmientes_testeros_contrapeados(z_drop=250.0)

testeros = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "TESTERO") and not processor._layer_is_no_pasante(e.layer)]
montantes = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "MONTANTE")]

# Find Testero [103D] or any testero at Z ~ 38055.0
target_testeros = [t for t in testeros if abs(min(t.segments[0][2], t.segments[0][5]) - 38055.0) <= 10.0]
print(f"Testeros at Z ~ 38055.0 count: {len(target_testeros)}")

for t in target_testeros[:5]:
    tx1, ty1, tz1, tx2, ty2, tz2 = t.segments[0]
    tz_base = min(tz1, tz2)
    t_len = math.hypot(tx2-tx1, ty2-ty1)

    m_near = []
    for m in montantes:
        mx1, my1, mz1, mx2, my2, mz2 = m.segments[0]
        mz_min = min(mz1, mz2)
        mz_max = max(mz1, mz2)
        dist_xy = point_to_segment_distance_xy((mx1+mx2)/2.0, (my1+my2)/2.0, tx1, ty1, tx2, ty2)
        if dist_xy <= 300.0:
            m_near.append((m, mz_min, mz_max))

    print(f"\nTestero [{t.handle}] layer={t.layer} Z={tz_base:.1f} len={t_len:.1f} P1=({tx1:.1f}, {ty1:.1f}) P2=({tx2:.1f}, {ty2:.1f}):")
    print(f"  Total near montantes in XY: {len(m_near)}")
    for m, mz_min, mz_max in m_near[:10]:
        print(f"    Montante [{m.handle}] layer={m.layer} Z_min={mz_min:.1f} Z_max={mz_max:.1f} (delta_min={mz_min-tz_base:.1f})")
