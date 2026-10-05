import sys
import os
import math
import ezdxf

from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor
from src.core.geometry import EntityGeometry

def test_file(dxf_path):
    print(f"\n==========================================")
    print(f"Testing level line generation on: {os.path.basename(dxf_path)}")
    print(f"==========================================")
    doc = ezdxf.readfile(dxf_path)
    geometries, _ = collect_entity_geometries(doc.modelspace())

    processor = StructuralProcessor(geometries)
    processor.execute_preprocesar_cadwork()
    processor.execute_merge_durmientes_testeros(z_drop=250.0)

    # Call generate_level_lines logic
    tol_cota_nivel = 1125.0

    # 1. Remove existing cota entities
    processor.entity_geometries = [e for e in processor.entity_geometries if "COTA" not in e.layer.upper()]

    durmientes = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "DURMIENTE")]
    testeros = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "TESTERO")]

    cota_entities = []

    for d in durmientes:
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
        tx1, ty1, tz1, tx2, ty2, tz2 = t.segments[0]
        tz_base = min(tz1, tz2)
        has_durmiente_below = False
        for d in durmientes:
            dx1, dy1, dz1, dx2, dy2, dz2 = d.segments[0]
            dz_base = min(dz1, dz2)
            if dz_base < tz_base - 100.0:
                dist_xy = max(abs((tx1+tx2)/2.0 - (dx1+dx2)/2.0), abs((ty1+ty2)/2.0 - (dy1+dy2)/2.0))
                if dist_xy < 500.0:
                    has_durmiente_below = True
                    break
        if not has_durmiente_below:
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
    print(msg)

    final_cotas = [e for e in processor.entity_geometries if "COTA" in e.layer.upper()]
    non_zero_cotas = [c for c in final_cotas if math.hypot(c.segments[0][3]-c.segments[0][0], c.segments[0][4]-c.segments[0][1]) > 1.0]
    print(f"Total non-zero level lines: {len(non_zero_cotas)}")

test_file(r"c:\PythonProjects\RFEM_Imports\LT082-LGT-EST-FCH-M3-G01-001-Estructura_1A_editado.dxf")
test_file(r"c:\PythonProjects\RFEM_Imports\Oporto Prueba.dxf")
