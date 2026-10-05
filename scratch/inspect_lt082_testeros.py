import sys
import os
import ezdxf

sys.path.insert(0, os.path.abspath("."))
from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor

def inspect_lt082_testeros():
    dxf_path = r"c:\PythonProjects\RFEM_Imports\LT082-LGT-EST-FCH-M3-G01-001-Estructura_1A_editado.dxf"
    doc = ezdxf.readfile(dxf_path)
    geometries, _ = collect_entity_geometries(doc.modelspace())
    sp = StructuralProcessor(geometries)

    sp.execute_preprocesar_cadwork()
    sp.execute_merge_durmientes_testeros(z_drop=250.0)

    testeros = [e for e in sp.entity_geometries if "TESTERO" in e.layer.upper()]
    durmientes = [e for e in sp.entity_geometries if "DURMIENTE" in e.layer.upper()]

    print("--- DURMIENTES ---")
    d_layers = {}
    for d in durmientes:
        d_layers[d.layer] = d_layers.get(d.layer, 0) + 1
    print(d_layers)

    print("--- TESTEROS ---")
    t_layers = {}
    for t in testeros:
        t_layers[t.layer] = t_layers.get(t.layer, 0) + 1
    print(t_layers)

    # Check how many testeros have not self._layer_is_no_pasante(t.layer)
    pasante_testeros = [t for t in testeros if not sp._layer_is_no_pasante(t.layer)]
    print(f"Total testeros treated as pasante (not _layer_is_no_pasante): {len(pasante_testeros)}")

    for t in pasante_testeros:
        print(f"Pasante testero handle={t.handle}, layer={t.layer}, z={min(t.segments[0][2], t.segments[0][5])}")

if __name__ == "__main__":
    inspect_lt082_testeros()
