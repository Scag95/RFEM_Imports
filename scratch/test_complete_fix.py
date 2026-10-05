import sys
import os
import math
import ezdxf

from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor
from src.core.geometry import EntityGeometry

def test_complete_fix(dxf_path):
    print("=" * 60)
    print(f"TESTING COMPLETE LEVEL LINE FIX: {os.path.basename(dxf_path)}")
    print("=" * 60)

    doc = ezdxf.readfile(dxf_path)
    geometries, _ = collect_entity_geometries(doc.modelspace())

    processor = StructuralProcessor(geometries)

    # 1. Preprocesar Cadwork
    processor.execute_preprocesar_cadwork()

    # 2. Merge Durmientes Testeros
    processor.execute_merge_durmientes_testeros(z_drop=250.0)

    # 3. Merge Contrapeados
    processor.execute_merge_durmientes_testeros_contrapeados(z_drop=250.0)

    # 4. Generate Level Lines with updated logic
    tol_cota_nivel = 1125.0
    processor.entity_geometries = [e for e in processor.entity_geometries if "COTA" not in e.layer.upper()]

    durmientes = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "DURMIENTE")]
    testeros = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "TESTERO")]
    montantes = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "MONTANTE")]

    cota_entities = []

    for d in durmientes:
        if not d.segments:
            continue
        x1, y1, z1, x2, y2, z2 = d.segments[0]
        z_base = min(z1, z2)
        z_cota = z_base + tol_cota_nivel
        suffix = "(No Pasante)" if processor._layer_is_no_pasante(d.layer) else "(Pasante)"
        layer_name = f"Linea Cota de Nivel {suffix}"
        cota_entities.append(EntityGeometry(
            handle=f"COTA_GEN_{d.handle}",
            layer=layer_name,
            segments=[(x1, y1, z_cota, x2, y2, z_cota)]
        ))

    for t in testeros:
        if not t.segments:
            continue
        tx1, ty1, tz1, tx2, ty2, tz2 = t.segments[0]
        tz_base = min(tz1, tz2)
        # Check if there are montantes extending above this testero
        has_montantes_above = False
        for m in montantes:
            if not m.segments:
                continue
            mx1, my1, mz1, mx2, my2, mz2 = m.segments[0]
            mz_min = min(mz1, mz2)
            mz_max = max(mz1, mz2)
            if (tz_base - 100.0 <= mz_min <= tz_base + 350.0) and (mz_max >= tz_base + 1000.0):
                dist_xy = max(abs((mx1+mx2)/2.0 - (tx1+tx2)/2.0), abs((my1+my2)/2.0 - (ty1+ty2)/2.0))
                if dist_xy <= 300.0:
                    has_montantes_above = True
                    break
        if has_montantes_above:
            z_cota = tz_base + tol_cota_nivel
            suffix = "(No Pasante)" if processor._layer_is_no_pasante(t.layer) else "(Pasante)"
            layer_name = f"Linea Cota de Nivel {suffix}"
            cota_entities.append(EntityGeometry(
                handle=f"COTA_GEN_{t.handle}",
                layer=layer_name,
                segments=[(tx1, ty1, z_cota, tx2, ty2, z_cota)]
            ))

    processor.entity_geometries.extend(cota_entities)
    success, msg = processor.fuse_level_lines()

    final_cotas = [e for e in processor.entity_geometries if "COTA" in e.layer.upper()]
    floor_zs = sorted(list(set(round(c.segments[0][2]) for c in final_cotas)))

    print(f"Total level lines generated & fused: {len(final_cotas)}")
    print(f"Floor Z levels with level lines: {floor_zs}")
    print(f"Pasante level lines: {len([c for c in final_cotas if 'PASANTE' in c.layer.upper() and 'NO PASANTE' not in c.layer.upper()])}")
    print(f"No Pasante level lines: {len([c for c in final_cotas if 'NO PASANTE' in c.layer.upper()])}")
    print(f"Zero length level lines: {len([c for c in final_cotas if math.hypot(c.segments[0][3]-c.segments[0][0], c.segments[0][4]-c.segments[0][1]) <= 1.0])}\n")

test_complete_fix(r"c:\PythonProjects\RFEM_Imports\LT082-LGT-EST-FCH-M3-G01-001-Estructura_1A_editado.dxf")
test_complete_fix(r"c:\PythonProjects\RFEM_Imports\Oporto Prueba.dxf")
