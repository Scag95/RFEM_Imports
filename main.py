import ezdxf as dxf

doc = dxf.readfile("VA031-LGT-ARQ-FCH-M3-G01-001-Ejes_Entramado.dxf")
msp = doc.modelspace()

for e in msp:
    print(e.dxftype(), e.dxf.layer)
