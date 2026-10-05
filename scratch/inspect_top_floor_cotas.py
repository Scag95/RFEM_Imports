import sys
import os
import ezdxf

from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor

dxf_path = r"c:\PythonProjects\RFEM_Imports\LT082-LGT-EST-FCH-M3-G01-001-Estructura_1A_editado.dxf"
doc = ezdxf.readfile(dxf_path)
geometries, _ = collect_entity_geometries(doc.modelspace())

processor = StructuralProcessor(geometries)
processor.execute_preprocesar_cadwork()
processor.execute_merge_durmientes_testeros(z_drop=250.0)
processor.execute_merge_durmientes_testeros_contrapeados(z_drop=250.0)

# Check plates on upper floor (Z >= 18000)
upper_plates = [e for e in processor.entity_geometries if (processor._is_layer_match(e.layer, "DURMIENTE") or processor._is_layer_match(e.layer, "TESTERO")) and min(e.segments[0][2], e.segments[0][5]) >= 17000.0]

print("=== UPPER FLOOR PLATES (Z >= 17000) ===")
for p in upper_plates:
    z1, z2 = p.segments[0][2], p.segments[0][5]
    print(f"Layer: {p.layer:<35} Handle: {p.handle:<10} Z: {min(z1,z2):.1f} P1: ({p.segments[0][0]:.1f}, {p.segments[0][1]:.1f}) P2: ({p.segments[0][3]:.1f}, {p.segments[0][4]:.1f})")

# Test generate_level_lines with NO PASANTE restricted to DURMIENTE ONLY
processor.generate_level_lines()

cotas_top = [e for e in processor.entity_geometries if "COTA" in e.layer.upper() and min(e.segments[0][2], e.segments[0][5]) >= 18000.0]
print("\n=== TOP FLOOR LEVEL LINES (Z >= 18000) ===")
for c in cotas_top:
    z = c.segments[0][2]
    p1 = (c.segments[0][0], c.segments[0][1])
    p2 = (c.segments[0][3], c.segments[0][4])
    print(f"Layer: {c.layer:<35} Handle: {c.handle:<15} Z: {z:.1f} P1: ({p1[0]:.1f}, {p1[1]:.1f}) P2: ({p2[0]:.1f}, {p2[1]:.1f})")
