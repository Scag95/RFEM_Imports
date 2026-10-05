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

montantes = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "MONTANTE")]
pasante_montantes = [m for m in montantes if not processor._layer_is_no_pasante(m.layer)]
no_pasante_montantes = [m for m in montantes if processor._layer_is_no_pasante(m.layer)]

print(f"Total montantes: {len(montantes)}")
print(f"Pasante montantes count: {len(pasante_montantes)}")
print(f"No Pasante montantes count: {len(no_pasante_montantes)}")

pasante_m_by_z = {}
for m in pasante_montantes:
    z_min = min(m.segments[0][2], m.segments[0][5])
    z_key = round(z_min)
    pasante_m_by_z[z_key] = pasante_m_by_z.get(z_key, 0) + 1

print("\n--- PASANTE MONTANTES BY Z_MIN ---")
for z_val, count in sorted(pasante_m_by_z.items()):
    print(f"Z_min: {z_val:<8} Count: {count}")

no_pasante_m_by_z = {}
for m in no_pasante_montantes:
    z_min = min(m.segments[0][2], m.segments[0][5])
    z_key = round(z_min)
    no_pasante_m_by_z[z_key] = no_pasante_m_by_z.get(z_key, 0) + 1

print("\n--- NO PASANTE MONTANTES BY Z_MIN ---")
for z_val, count in sorted(no_pasante_m_by_z.items()):
    print(f"Z_min: {z_val:<8} Count: {count}")
