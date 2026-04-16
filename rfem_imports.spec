# PyInstaller build spec for the RFEM DXF Manager desktop app.

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


project_root = Path.cwd()

datas = collect_data_files("ezdxf")
binaries = collect_dynamic_libs("PyQt6")
hiddenimports = [
    "PyQt6.Qt3DCore",
    "PyQt6.Qt3DExtras",
    "PyQt6.Qt3DRender",
]

excludes = [
    # Stdlib no utilizado en runtime
    "tkinter",
    "pydoc",
    "doctest",
    "distutils",
    "setuptools",
    "pip",
    "email",
    "lib2to3",
    # Submódulos de NumPy innecesarios en runtime
    "numpy.f2py",
    "numpy.testing",
    "numpy.tests",
    "numpy.distutils",
    # Paquetes científicos/visualización no presentes en este proyecto
    "matplotlib",
    "scipy",
    "pandas",
    "IPython",
    "PIL",
]

# UPX desactivado: en builds de directorio con SSD la descompresion en
# startup es mas lenta que leer archivos sin comprimir del disco.
upx_exclude_list = []


a = Analysis(
    ["main.py"],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=2,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RFEM_Imports",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=upx_exclude_list,
    name="RFEM_Imports",
)
