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

print("=== ALL PLATES AFTER ALL MERGES ===")
plates = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "DURMIENTE") or processor._is_layer_match(e.layer, "TESTERO")]

# Group plates by Z and layer
by_z_layer = {}
for p in plates:
    z_min = min(p.segments[0][2], p.segments[0][5])
    z_max = max(p.segments[0][2], p.segments[0][5])
    key = (round(z_min, 1), p.layer)
    by_z_layer[key] = by_z_layer.get(key, 0) + 1

for (z_val, layer_name), count in sorted(by_z_layer.items(), key=lambda x: (x[0][0], x[0][1])):
    print(f"Z: {z_val:<8.1f} Layer: {layer_name:<40} Count: {count}")

print("\n=== MONTANTES HIGHEST Z RANGE ===")
montantes = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "MONTANTE")]
m_z_top = max(max(m.segments[0][2], m.segments[0][5]) for m in montantes)
m_z_bottom_for_top = max(min(m.segments[0][2], m.segments[0][5]) for m in montantes)

print(f"Highest montante top Z: {m_z_top:.1f}")
print(f"Highest montante bottom Z: {m_z_bottom_for_top:.1f}")

# Check montantes on highest floor
top_montantes = [m for m in montantes if min(m.segments[0][2], m.segments[0][5]) >= m_z_bottom_for_top - 500.0]
print(f"Top floor montantes count: {len(top_montantes)}")
for m in top_montantes[:10]:
    z1, z2 = m.segments[0][2], m.segments[0][5]
    print(f"Top montante handle: {m.handle:<10} layer: {m.layer:<35} Z: {min(z1,z2):.1f} .. {max(z1,z2):.1f}")
