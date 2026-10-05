import sys
import os
import ezdxf

from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor

dxf_path = r"c:\PythonProjects\RFEM_Imports\Panel No Pasante_unido.dxf"
doc = ezdxf.readfile(dxf_path)
geometries, _ = collect_entity_geometries(doc.modelspace())

processor = StructuralProcessor(geometries)
processor.execute_preprocesar_cadwork()

plates = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "DURMIENTE") or processor._is_layer_match(e.layer, "TESTERO")]
montantes = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "MONTANTE")]

# Check if any element exists in Y = 9276..12699 or Y = 44763..48107
elements_in_gap1 = []
elements_in_gap2 = []

for e in plates + montantes:
    y1, y2 = e.segments[0][1], e.segments[0][4]
    my = (y1 + y2) / 2.0
    if 9350.0 <= my <= 12600.0:
        elements_in_gap1.append(e)
    elif 44850.0 <= my <= 48000.0:
        elements_in_gap2.append(e)

print(f"Elements in Y gap [9350 .. 12600]: {len(elements_in_gap1)}")
for e in elements_in_gap1[:5]:
    print(f"  Gap 1 element [{e.handle}] Layer: {e.layer}")

print(f"Elements in Y gap [44850 .. 48000]: {len(elements_in_gap2)}")
for e in elements_in_gap2[:5]:
    print(f"  Gap 2 element [{e.handle}] Layer: {e.layer}")
