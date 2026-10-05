import sys
import os
import glob
import ezdxf

from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor

dxfs = glob.glob(r"c:\PythonProjects\RFEM_Imports\*.dxf")

for dxf_path in dxfs:
    print("\n" + "=" * 60)
    print(f"TESTING DXF: {os.path.basename(dxf_path)}")
    print("=" * 60)
    try:
        doc = ezdxf.readfile(dxf_path)
        geometries, _ = collect_entity_geometries(doc.modelspace())
        processor = StructuralProcessor(geometries)
        processor.execute_preprocesar_cadwork()

        dur_before = len([e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "DURMIENTE")])

        processor.execute_merge_durmientes_testeros(z_drop=250.0)
        dur_after_m1 = len([e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "DURMIENTE")])

        processor.execute_merge_durmientes_testeros_contrapeados(z_drop=250.0)
        dur_after_m2 = len([e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "DURMIENTE")])

        print(f"Durmientes: Initial={dur_before} -> After Merge1={dur_after_m1} -> After Merge2={dur_after_m2}")
        print(f"Durmientes removed in Merge1: {dur_before - dur_after_m1}")
        print(f"Durmientes removed in Merge2: {dur_after_m1 - dur_after_m2}")
    except Exception as e:
        print(f"Error processing {dxf_path}: {e}")
