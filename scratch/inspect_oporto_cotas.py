import sys
import os
import math
import ezdxf

from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor, point_to_segment_distance_xy
from src.core.geometry import EntityGeometry

def inspect_dxf_cotas(dxf_path):
    print("=" * 60)
    print(f"INSPECTING LEVEL LINES FOR: {os.path.basename(dxf_path)}")
    print("=" * 60)

    doc = ezdxf.readfile(dxf_path)
    geometries, _ = collect_entity_geometries(doc.modelspace())

    processor = StructuralProcessor(geometries)
    processor.execute_preprocesar_cadwork()
    processor.execute_merge_durmientes_testeros(z_drop=250.0)
    processor.execute_merge_durmientes_testeros_contrapeados(z_drop=250.0)

    processor.generate_level_lines()

    cotas = [e for e in processor.entity_geometries if "COTA" in e.layer.upper()]
    print(f"Total fused level lines: {len(cotas)}")

    # Check total length of level lines per floor Z
    by_z = {}
    for c in cotas:
        z = round(c.segments[0][2])
        p1 = (c.segments[0][0], c.segments[0][1])
        p2 = (c.segments[0][3], c.segments[0][4])
        length = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        key = (z, c.layer)
        if key not in by_z:
            by_z[key] = []
        by_z[key].append((c.handle, length, p1, p2))

    for (z, layer), lines in sorted(by_z.items()):
        total_len = sum(l[1] for l in lines)
        print(f"Z: {z:<6} Layer: {layer:<35} Lines count: {len(lines)} Total length: {total_len:.1f} mm")
        for handle, length, p1, p2 in lines:
            print(f"   Line [{handle}] len={length:.1f} P1=({p1[0]:.1f}, {p1[1]:.1f}) P2=({p2[0]:.1f}, {p2[1]:.1f})")

inspect_dxf_cotas(r"c:\PythonProjects\RFEM_Imports\Oporto Prueba.dxf")
inspect_dxf_cotas(r"c:\PythonProjects\RFEM_Imports\Panel No Pasante_unido.dxf")
