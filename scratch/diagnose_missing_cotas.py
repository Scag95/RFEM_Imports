import sys
import os
import glob
import math
import ezdxf

from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor, point_to_segment_distance_xy

dxfs = glob.glob(r"c:\PythonProjects\RFEM_Imports\*.dxf")

for dxf_path in dxfs:
    print("\n" + "=" * 60)
    print(f"DIAGNOSING: {os.path.basename(dxf_path)}")
    print("=" * 60)
    try:
        doc = ezdxf.readfile(dxf_path)
        geometries, _ = collect_entity_geometries(doc.modelspace())

        processor = StructuralProcessor(geometries)
        processor.execute_preprocesar_cadwork()
        processor.execute_merge_durmientes_testeros(z_drop=250.0)
        processor.execute_merge_durmientes_testeros_contrapeados(z_drop=250.0)
        processor.generate_level_lines()

        cotas = [e for e in processor.entity_geometries if "COTA" in e.layer.upper()]
        durmientes = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "DURMIENTE")]
        testeros = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "TESTERO")]
        montantes = [e for e in processor.entity_geometries if processor._is_layer_match(e.layer, "MONTANTE")]

        real_durmientes = [d for d in durmientes if math.hypot(d.segments[0][3]-d.segments[0][0], d.segments[0][4]-d.segments[0][1]) > 10.0]

        uncovered_dur = []
        for d in real_durmientes:
            dx1, dy1, dz1, dx2, dy2, dz2 = d.segments[0]
            dz_base = min(dz1, dz2)
            expected_z = dz_base + 1125.0
            dmx, dmy = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0

            covered = False
            for c in cotas:
                cx1, cy1, cz1, cx2, cy2, cz2 = c.segments[0]
                if abs(cz1 - expected_z) <= 250.0:
                    dist = point_to_segment_distance_xy(dmx, dmy, cx1, cy1, cx2, cy2)
                    if dist <= 200.0:
                        covered = True
                        break
            if not covered:
                uncovered_dur.append((d, dz_base))

        # Also check panels / montantes that had their durmiente merged into testero
        pasante_testeros = [t for t in testeros if not processor._layer_is_no_pasante(t.layer)]
        pasante_montantes = [m for m in montantes if not processor._layer_is_no_pasante(m.layer)]

        uncovered_tes = []
        for t in pasante_testeros:
            tx1, ty1, tz1, tx2, ty2, tz2 = t.segments[0]
            t_len = math.hypot(tx2 - tx1, ty2 - ty1)
            if t_len <= 10.0:
                continue
            tz_base = min(tz1, tz2)

            has_montantes_above = False
            for m in pasante_montantes:
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
                expected_z = tz_base + 1125.0
                tmx, tmy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0
                covered = False
                for c in cotas:
                    cx1, cy1, cz1, cx2, cy2, cz2 = c.segments[0]
                    if abs(cz1 - expected_z) <= 250.0:
                        dist = point_to_segment_distance_xy(tmx, tmy, cx1, cy1, cx2, cy2)
                        if dist <= 200.0:
                            covered = True
                            break
                if not covered:
                    uncovered_tes.append((t, tz_base))

        print(f"Total REAL Durmientes: {len(real_durmientes)} | Uncovered: {len(uncovered_dur)}")
        print(f"Total Pasante Testeros: {len(pasante_testeros)} | Uncovered (with montantes above): {len(uncovered_tes)}")
        print(f"Total Fused Level Lines: {len(cotas)}")

        if uncovered_dur:
            print("  Uncovered Durmientes:")
            for d, z in uncovered_dur[:5]:
                print(f"    [{d.handle}] Layer: {d.layer} Z: {z:.1f}")
        if uncovered_tes:
            print("  Uncovered Testeros:")
            for t, z in uncovered_tes[:5]:
                print(f"    [{t.handle}] Layer: {t.layer} Z: {z:.1f}")

    except Exception as e:
        print(f"Error: {e}")
