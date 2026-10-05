import sys
import os
import math
import ezdxf

from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor, point_to_segment_distance_xy

dxf_path = r"c:\PythonProjects\RFEM_Imports\LT082-LGT-EST-FCH-M3-G01-001-Estructura_1A_editado.dxf"
doc = ezdxf.readfile(dxf_path)
geometries, _ = collect_entity_geometries(doc.modelspace())

print(f"Total raw DXF entities loaded: {len(geometries)}")

# Group raw entities by layer keyword
layer_summary = {}
for g in geometries:
    layer_summary[g.layer] = layer_summary.get(g.layer, 0) + 1

print("\n--- RAW DXF LAYER SUMMARY ---")
for layer, count in sorted(layer_summary.items()):
    print(f"Layer: {layer:<45} Count: {count}")

processor = StructuralProcessor(geometries)

# Inspect preprocesar_cadwork internal state
plates = []
montantes = []
dinteles = []
alfeizares = []
capialzados = []

for entity in processor.entity_geometries:
    if not entity.segments:
        continue
    x1, y1, z1, x2, y2, z2 = entity.segments[0]
    data = {
        'entity': entity,
        'handle': entity.handle,
        'start': [x1, y1, z1],
        'end': [x2, y2, z2],
        'orig_layer': entity.layer
    }
    layer_upper = entity.layer.upper()
    if "TESTERO" in layer_upper or "DURMIENTE" in layer_upper:
        plates.append(data)
    elif "MONTANTE" in layer_upper:
        montantes.append(data)

print(f"\nInitial plates found: {len(plates)}")
print(f"Initial montantes found: {len(montantes)}")

# Check floor Z range of plates
z_levels = sorted(list(set(round(min(p['start'][2], p['end'][2])) for p in plates)))
print(f"\nPlate Z Levels (floor bases/ceilings): {z_levels}")

# Execute full pipeline step by step
processor.execute_preprocesar_cadwork()
print(f"\nAfter preprocesar_cadwork: {len(processor.entity_geometries)} geometries")

processor.execute_merge_durmientes_testeros(z_drop=250.0)
print(f"After execute_merge_durmientes_testeros: {len(processor.entity_geometries)} geometries")

processor.execute_merge_durmientes_testeros_contrapeados(z_drop=250.0)
print(f"After execute_merge_durmientes_testeros_contrapeados: {len(processor.entity_geometries)} geometries")

processor.generate_level_lines()
print(f"After generate_level_lines: {len(processor.entity_geometries)} geometries")

cotas = [e for e in processor.entity_geometries if "COTA" in e.layer.upper()]
print(f"\nFinal COTA entities count: {len(cotas)}")
cotas_by_layer = {}
for c in cotas:
    cotas_by_layer[c.layer] = cotas_by_layer.get(c.layer, 0) + 1

for layer, count in sorted(cotas_by_layer.items()):
    print(f"  Layer: {layer:<40} Count: {count}")
