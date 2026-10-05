import sys
import os
import math
import ezdxf

sys.path.insert(0, os.path.abspath("."))
from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor, point_to_segment_distance_xy

def test_local_xy_testero_rule(dxf_path):
    print("=" * 60)
    print(f"TESTING LOCAL XY TESTERO RULE: {os.path.basename(dxf_path)}")
    print("=" * 60)

    doc = ezdxf.readfile(dxf_path)
    geometries, _ = collect_entity_geometries(doc.modelspace())
    processor = StructuralProcessor(geometries)

    processor.execute_preprocesar_cadwork()
    processor.execute_merge_durmientes_testeros(z_drop=250.0)
    processor.execute_merge_durmientes_testeros_contrapeados(z_drop=250.0)

    durmientes = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "DURMIENTE")]
    testeros = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "TESTERO")]
    montantes = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "MONTANTE")]

    durmiente_info = []
    for d in durmientes:
        if d.segments:
            dx1, dy1, dz1, dx2, dy2, dz2 = d.segments[0]
            dz_min = min(dz1, dz2)
            durmiente_info.append((dx1, dy1, dx2, dy2, dz_min))

    testero_generated = 0
    testero_skipped_no_pasante_layer = 0
    testero_skipped_local_durmiente = 0

    for t in testeros:
        if not t.segments:
            continue
        if processor._layer_is_no_pasante(t.layer):
            testero_skipped_no_pasante_layer += 1
            continue
            
        tx1, ty1, tz1, tx2, ty2, tz2 = t.segments[0]
        tz_base = min(tz1, tz2)
        tx_mid = (tx1 + tx2) / 2.0
        ty_mid = (ty1 + ty2) / 2.0

        # Check if there is a local durmiente near (tx_mid, ty_mid) at tz_base
        has_local_durmiente = False
        for dx1, dy1, dx2, dy2, dz in durmiente_info:
            if abs(dz - tz_base) <= 350.0:
                dist_xy = point_to_segment_distance_xy(tx_mid, ty_mid, dx1, dy1, dx2, dy2)
                if dist_xy <= 200.0:
                    has_local_durmiente = True
                    break

        if has_local_durmiente:
            testero_skipped_local_durmiente += 1
            continue

        has_montantes_above = False
        for m in montantes:
            if not m.segments:
                continue
            mx1, my1, mz1, mx2, my2, mz2 = m.segments[0]
            mz_min = min(mz1, mz2)
            mz_max = max(mz1, mz2)
            if (tz_base - 100.0 <= mz_min <= tz_base + 350.0) and (mz_max >= tz_base + 1000.0):
                mx_mid = (mx1 + mx2) / 2.0
                my_mid = (my1 + my2) / 2.0
                dist_xy = point_to_segment_distance_xy(mx_mid, my_mid, tx1, ty1, tx2, ty2)
                if dist_xy <= 200.0:
                    has_montantes_above = True
                    break

        if has_montantes_above:
            testero_generated += 1

    print(f"Testeros generated: {testero_generated}")
    print(f"Testeros skipped (No Pasante layer): {testero_skipped_no_pasante_layer}")
    print(f"Testeros skipped (local Durmiente at same location): {testero_skipped_local_durmiente}")

if __name__ == "__main__":
    test_local_xy_testero_rule(r"c:\PythonProjects\RFEM_Imports\LT082-LGT-EST-FCH-M3-G01-001-Estructura_1A_editado.dxf")
    test_local_xy_testero_rule(r"c:\PythonProjects\RFEM_Imports\Oporto Prueba.dxf")
