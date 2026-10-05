import sys
import os
sys.path.insert(0, os.path.abspath("."))
import math
import ezdxf

from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor

def run_pipeline(dxf_path):
    print("=" * 60)
    print(f"PIPELINE VERIFICATION: {os.path.basename(dxf_path)}")
    print("=" * 60)

    doc = ezdxf.readfile(dxf_path)
    geometries, _ = collect_entity_geometries(doc.modelspace())

    processor = StructuralProcessor(geometries)

    # 1. Preprocesar Cadwork
    ok, msg1 = processor.execute_preprocesar_cadwork()
    print("[1] Preprocesar Cadwork:")
    print(msg1)

    # Check level lines after preprocesar (should be 0)
    cotas_after_prep = [e for e in processor.entity_geometries if "COTA" in e.layer.upper()]
    print(f"--> Level lines immediately after preprocesar: {len(cotas_after_prep)} (Expected: 0)")

    # 2. Merge durmientes testeros
    ok, msg2 = processor.execute_merge_durmientes_testeros(z_drop=250.0)
    print("\n[2] Merge Durmientes Testeros:")
    print(msg2.splitlines()[0] if msg2 else "")

    # 3. Merge contrapeados
    ok, msg3 = processor.execute_merge_durmientes_testeros_contrapeados(z_drop=250.0)
    print("\n[3] Merge Contrapeados:")
    print(msg3.splitlines()[0] if msg3 else "")

    # 4. Generate Level Lines
    ok, msg4 = processor.generate_level_lines()
    print("\n[4] Generar Líneas de Sobrecarga:")
    print(msg4)

    final_cotas = [e for e in processor.entity_geometries if "COTA" in e.layer.upper()]
    pasante_cotas = [e for e in final_cotas if "PASANTE" in e.layer.upper() and "NO PASANTE" not in e.layer.upper()]
    no_pasante_cotas = [e for e in final_cotas if "NO PASANTE" in e.layer.upper()]

    print(f"\nFinal level lines summary:")
    print(f"  - Total level lines: {len(final_cotas)}")
    print(f"  - Pasante level lines: {len(pasante_cotas)}")
    print(f"  - No Pasante level lines: {len(no_pasante_cotas)}")

    # Check for zero-length segments
    zero_len = [e for e in final_cotas if math.hypot(e.segments[0][3]-e.segments[0][0], e.segments[0][4]-e.segments[0][1]) <= 1.0]
    print(f"  - Zero-length level lines: {len(zero_len)} (Expected: 0)")
    assert len(cotas_after_prep) == 0, "Level lines were generated too early!"
    assert len(zero_len) == 0, "Zero-length level lines exist!"
    print("ALL ASSERTIONS PASSED CLEANLY!\n")

run_pipeline(r"c:\PythonProjects\RFEM_Imports\LT082-LGT-EST-FCH-M3-G01-001-Estructura_1A_editado.dxf")
run_pipeline(r"c:\PythonProjects\RFEM_Imports\Oporto Prueba.dxf")
