import os
import sys

# Workaround for Qt3D RHI pipeline failures on some Windows/D3D drivers.
os.environ.setdefault("QT3D_RENDERER", "opengl")
os.environ.setdefault("QT_OPENGL", "desktop")

from PyQt6.QtWidgets import QApplication

# Importamos la clase MainWindow desde la nueva estructura modular
from src.ui.main_window import MainWindow

def main() -> int:
    # 1. Crear la aplicación Qt 
    app = QApplication(sys.argv)

    # 2. Instanciar y mostrar la ventana principal
    window = MainWindow()
    window.show()

    # 3. Entrar en el bucle de eventos
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
