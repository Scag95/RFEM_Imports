import sys
import os

sys.path.insert(0, os.path.abspath("."))
import ezdxf
from src.core.dxf_utils import collect_entity_geometries
from src.domain.structural_processor import StructuralProcessor

def test_no_pasante_duplication():
    dxf_path = r"c:\PythonProjects\RFEM_Imports\LT082-LGT-EST-FCH-M3-G01-001-Estructura_1A_editado.dxf"
    doc = ezdxf.readfile(dxf_path)
    geometries, _ = collect_entity_geometries(doc.modelspace())
    sp = StructuralProcessor(geometries)
    
    # 1. Run fusion
    ok, msg = sp.fuse_durmientes_y_testeros()
    print("Fusion result:", ok, msg)
    
    # 2. Inspect testeros and durmientes
    testeros = [e for e in sp.entity_geometries if "TESTERO" in e.layer.upper()]
    durmientes = [e for e in sp.entity_geometries if "DURMIENTE" in e.layer.upper()]
    
    print(f"Total testeros: {len(testeros)}")
    print(f"Total durmientes: {len(durmientes)}")
    
    # Check layer breakdown
    t_layers = {}
    for t in testeros:
        t_layers[t.layer] = t_layers.get(t.layer, 0) + 1
    print("Testero layers:", t_layers)
    
    d_layers = {}
    for d in durmientes:
        d_layers[d.layer] = d_layers.get(d.layer, 0) + 1
    print("Durmiente layers:", d_layers)

    # Now run generate_level_lines
    ok_cota, msg_cota = sp.generate_level_lines()
    print("Generate level lines result:", ok_cota, msg_cota)
    
    cotas = [e for e in sp.entity_geometries if "COTA" in e.layer.upper()]
    print(f"Generated level lines count: {len(cotas)}")
    
    # Check Z heights of cotas
    z_heights = set()
    for c in cotas:
        if c.segments:
            z_heights.add(round(c.segments[0][2], 1))
    print("Cota Z heights:", sorted(list(z_heights)))

if __name__ == "__main__":
    test_no_pasante_duplication()
