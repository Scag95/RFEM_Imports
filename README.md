# RFEM DXF Manager - 3D Viewer

Una herramienta de escritorio desarrollada con PyQt6 y Qt3D para procesar, visualizar y estructurar geometrías de diseño (líneas, polilíneas) a partir de archivos `.dxf`, enfocada en las necesidades de análisis estructural para su potencial importación en RFEM.

## Estructura del Proyecto

La aplicación ha sido refactorizada para separar responsabilidades y facilitar su mantenimiento y evolución:

```text
RFEM_Imports/
├── main.py                     # Punto de entrada de la aplicación
├── requirements.txt            # Dependencias del proyecto (ezdxf, PyQt6, etc.)
├── src/
│   ├── core/
│   │   ├── geometry.py         # Definición de tipos core (Point3, Segment3, EntityGeometry) y funciones matemáticas (distancias e intersecciones).
│   │   └── dxf_utils.py        # Conversión de geometrías nativas de ezdxf a entidades manejables por la aplicación.
│   ├── domain/
│   │   └── structural_processor.py # Lógica de negocio (algoritmos para reconocer, alinear, agrupar y unir montantes, durmientes y esquinas estructurales).
│   └── ui/
│       ├── main_window.py      # Controlador principal de la vista, menús y estado de la aplicación.
│       ├── viewer_3d.py        # Integración con OpenGL mediante Qt3DWindow para renderizar las geometrías. 
│       └── dialogs.py          # Definición de las ventanas modales de alerta, borrado de capas e información.
```

## Características Principales

- **Visualizador 3D interactivo**: Carga cualquier cantidad de segmentos separados por capas, con la habilidad de rotar el punto de vista, hacer paneo y ajustar el zoom.
- **Detección estructural**: Posibilidad de escanear automáticamente las entidades para emparejar durmientes y testeros con montantes (tanto en cruces como en esquinas). 
- **Auto-conexión inteligente**: Las uniones entre perfiles se resuelven de forma automática respetando tolerancias horizontales y verticales.
- **Exportación de DXF**: Guarda un nuevo archivo `dxf` tras limpiar capas en la interfaz y alinear las conexiones con las tolerancias precisadas por el usuario.

## Requisitos y Uso

1. Requiere Python 3.10+.
2. Instalar el fichero de requerimientos: `pip install -r requirements.txt` (usualmente `ezdxf` y `PyQt6`).
3. Ejecutar la aplicación:
   ```bash
   python main.py
   ```
4. Navegar a **Archivo > Cargar DXF** para comenzar.
