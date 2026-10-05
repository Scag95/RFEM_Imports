import os
from collections import defaultdict
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.core.dxf_utils import collect_entity_geometries, flatten_segments_for_viewer
from src.core.geometry import EntityGeometry

DXF_INVALID_LAYER_CHAR_REPLACEMENTS = str.maketrans(
    {
        "<": "(",
        ">": ")",
        "/": "-",
        "\\": "-",
        '"': "",
        ":": " -",
        ";": " -",
        "?": "",
        "*": "x",
        "|": "-",
        "=": "-",
        "'": "",
    }
)

PREDEFINED_LAYERS_PALETTE: list[tuple[int, QColor]] = [
    (1, QColor(255, 63, 63)),      # Rojo
    (2, QColor(255, 255, 63)),     # Amarillo
    (3, QColor(63, 255, 63)),      # Verde
    (4, QColor(63, 255, 255)),     # Cian
    (5, QColor(63, 63, 255)),      # Azul
    (6, QColor(255, 63, 255)),     # Magenta
    (8, QColor(128, 128, 128)),    # Gris oscuro
    (9, QColor(192, 192, 192)),    # Gris claro
    (10, QColor(255, 0, 0)),       # Rojo puro
    (30, QColor(255, 127, 0)),     # Naranja
    (50, QColor(212, 175, 55)),    # Oro / Amarillo oscuro
    (70, QColor(127, 255, 0)),     # Verde lima
    (90, QColor(0, 192, 0)),       # Verde oscuro
    (110, QColor(0, 255, 127)),    # Verde primavera
    (130, QColor(0, 192, 192)),    # Cian oscuro
    (150, QColor(0, 127, 255)),    # Azul claro
    (170, QColor(0, 0, 128)),      # Azul marino
    (190, QColor(127, 0, 255)),    # Púrpura
    (210, QColor(192, 0, 192)),    # Violeta
    (230, QColor(255, 0, 127)),    # Rosa
    (14, QColor(150, 75, 0)),      # Marrón
    (24, QColor(255, 191, 0)),     # Ámbar
    (34, QColor(128, 128, 0)),     # Oliva
    (44, QColor(34, 139, 34)),     # Verde bosque
    (54, QColor(0, 128, 128)),     # Teal
    (64, QColor(135, 206, 235)),   # Cielo
    (74, QColor(75, 0, 130)),      # Índigo
    (84, QColor(238, 130, 238)),   # Violeta claro
    (104, QColor(46, 204, 113)),   # Esmeralda
    (124, QColor(52, 152, 219)),   # Azul belice
    (144, QColor(155, 89, 182)),   # Amatista
    (164, QColor(241, 196, 15)),   # Girasol
    (184, QColor(230, 126, 34)),   # Zanahoria
    (204, QColor(231, 76, 60)),    # Alizarina
]

PREDEFINED_LAYER_MAP: dict[str, tuple[int, QColor]] = {
    # DURMIENTES (Verdes)
    "DURMIENTE (PASANTE)": (90, QColor(0, 192, 0)),           # Verde oscuro
    "DURMIENTE (NO PASANTE)": (70, QColor(127, 255, 0)),      # Verde lima
    
    # TESTEROS (Rojos)
    "TESTERO (PASANTE)": (10, QColor(255, 0, 0)),             # Rojo puro
    "TESTERO (NO PASANTE)": (204, QColor(231, 76, 60)),        # Rojo coral / Alizarina
    
    # UNION (Púrpuras)
    "MONTANTES UNION (PASANTE)": (190, QColor(127, 0, 255)),   # Púrpura
    "MONTANTES UNION (NO PASANTE)": (210, QColor(192, 0, 192)), # Violeta
    
    # ESQUINAS (Naranjas/Amarillos)
    "MONTANTES DE ESQUINA (PASANTE)": (30, QColor(255, 127, 0)), # Naranja
    "MONTANTES DE ESQUINA (NO PASANTE)": (164, QColor(241, 196, 15)), # Girasol / Amarillo
    "MONTANTES ESQUINA (PASANTE)": (30, QColor(255, 127, 0)),
    "MONTANTES ESQUINA (NO PASANTE)": (164, QColor(241, 196, 15)),
    
    # SIMPLES / LARGOS (Azules)
    "MONTANTES SIMPLES (PASANTE)": (150, QColor(0, 127, 255)), # Azul claro
    "MONTANTES SIMPLES (NO PASANTE)": (170, QColor(0, 0, 128)),  # Azul marino
    "MONTANTES LARGOS (PASANTE)": (150, QColor(0, 127, 255)),
    "MONTANTES LARGOS (NO PASANTE)": (170, QColor(0, 0, 128)),
    
    # CORTOS (Grises)
    "MONTANTES CORTOS (PASANTE)": (8, QColor(128, 128, 128)),  # Gris oscuro
    "MONTANTES CORTOS (NO PASANTE)": (9, QColor(192, 192, 192)), # Gris claro
    
    # HUECOS (Verdes/Azules oscuros)
    "MONTANTES HUECOS (PASANTE)": (54, QColor(0, 128, 128)),   # Teal / Verde azulado
    "MONTANTES HUECOS (NO PASANTE)": (44, QColor(34, 139, 34)), # Verde bosque
    "MONTANTES SIMPLES HUECOS (PASANTE)": (54, QColor(0, 128, 128)),
    "MONTANTES SIMPLES HUECOS (NO PASANTE)": (44, QColor(34, 139, 34)),
    
    # DINTELES (Rosas/Crimson)
    "DINTEL (PASANTE)": (230, QColor(255, 0, 127)),            # Rosa fuerte
    "DINTEL (NO PASANTE)": (6, QColor(255, 63, 255)),          # Magenta
    
    # ALFEIZARES (Cian/Sky)
    "ALFEIZAR (PASANTE)": (64, QColor(135, 206, 235)),         # Cielo / Azul celeste
    "ALFEIZAR (NO PASANTE)": (4, QColor(63, 255, 255)),        # Cian
    
    # CAPIALZADOS (Violeta / Púrpura)
    "CAPIALZADO (PASANTE)": (190, QColor(127, 0, 255)),        # Púrpura
    "CAPIALZADO (NO PASANTE)": (210, QColor(192, 0, 192)),     # Violeta
    
    # COTAS DE NIVEL (Oro/Magenta)
    "LINEA COTA DE NIVEL (PASANTE)": (24, QColor(255, 191, 0)), # Ámbar / Oro
    "LINEA COTA DE NIVEL (NO PASANTE)": (84, QColor(238, 130, 238)), # Violeta claro
    
    # PETOS (Marrones)
    "LISTON PETOS": (14, QColor(150, 75, 0)),                  # Marrón
    "TESTERO (PETOS)": (14, QColor(150, 75, 0)),
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RFEM DXF Manager - 3D Viewer")
        self.resize(1400, 900)

        self.viewer = None
        self.layer_counts: dict[str, int] = {}
        self.current_file_path = ""
        self.entity_geometries: list[EntityGeometry] = []
        self.merge_z_drop = 125.0
        self.merge_union_tol = 100.0
        self.merge_esquina_tol = 145.0
        self.dxf_scale_factor = 1.0
        
        self.layer_list = QListWidget()
        self.layer_list.setMinimumWidth(320)
        self.layer_list.itemChanged.connect(self._on_layer_item_changed)

        self.info_label = QLabel("Usa Archivo > Cargar DXF para visualizar lineas en 3D.")
        self.info_label.setStyleSheet("font-size: 11px;")
        self.info_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.info_label.setMaximumHeight(22)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("Capas detectadas"))
        right_layout.addWidget(self.layer_list)

        central = QWidget()
        layout = QHBoxLayout(central)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        self.viewer_host = QWidget()
        self.viewer_host_layout = QVBoxLayout(self.viewer_host)
        self.viewer_host_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.viewer_host)
        left_layout.addWidget(self.info_label)
        left_layout.setStretch(0, 1)
        left_layout.setStretch(1, 0)

        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Listo")
        self._create_menu()

    def _ensure_viewer(self):
        if self.viewer is not None:
            return self.viewer

        from src.ui.viewer_3d import DXFViewer3D

        self.viewer = DXFViewer3D()
        self.viewer_host_layout.addWidget(self.viewer)
        return self.viewer

    def _create_menu(self) -> None:
        file_menu = self.menuBar().addMenu("Archivo")

        open_action = QAction("Cargar DXF", self)
        open_action.triggered.connect(self.load_dxf_file)
        file_menu.addAction(open_action)

        save_action = QAction("Guardar DXF", self)
        save_action.triggered.connect(self.save_dxf_file)
        file_menu.addAction(save_action)

        exit_action = QAction("Salir", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        tools_menu = self.menuBar().addMenu("Herramientas")

        preprocesar_cadwork_action = QAction("Preprocesar y clasificar (Cadwork)", self)
        preprocesar_cadwork_action.triggered.connect(self.preprocesar_cadwork)
        tools_menu.addAction(preprocesar_cadwork_action)

        # remove_layer_action = QAction("Eliminar capa", self)
        # remove_layer_action.triggered.connect(self.open_delete_layer_dialog)
        # tools_menu.addAction(remove_layer_action)

        # rename_layers_action = QAction("Renombrar capas", self)
        # rename_layers_action.triggered.connect(self.open_rename_layers_dialog)
        # tools_menu.addAction(rename_layers_action)

        # detect_connections_action = QAction("Detectar montantes-durmientes", self)
        # detect_connections_action.triggered.connect(self.detect_montantes_durmientes)
        # tools_menu.addAction(detect_connections_action)

        merge_action = QAction("Unir durmientes y testeros", self)
        merge_action.triggered.connect(self.merge_durmientes_testeros)
        tools_menu.addAction(merge_action)

        merge_contrapeados_action = QAction("Unir durmientes y testeros (Contrapeados)", self)
        merge_contrapeados_action.triggered.connect(self.merge_durmientes_testeros_contrapeados)
        tools_menu.addAction(merge_contrapeados_action)

        merge_union_action = QAction("Unir montantes union", self)
        merge_union_action.triggered.connect(self.merge_montantes_union)
        tools_menu.addAction(merge_union_action)

        merge_corner_action = QAction("Unir montantes esquinas", self)
        merge_corner_action.triggered.connect(self.merge_montantes_esquinas)
        tools_menu.addAction(merge_corner_action)

        fuse_cotas_action = QAction("Generar líneas de sobrecarga", self)
        fuse_cotas_action.triggered.connect(self.generate_level_lines)
        tools_menu.addAction(fuse_cotas_action)






    def _build_layer_colors(self, layers: list[str]) -> dict[str, QColor]:
        colors: dict[str, QColor] = {}
        for i, layer in enumerate(sorted(layers)):
            upper = layer.upper()
            if upper in PREDEFINED_LAYER_MAP:
                _, qcolor = PREDEFINED_LAYER_MAP[upper]
                colors[layer] = qcolor
            else:
                _, qcolor = PREDEFINED_LAYERS_PALETTE[i % len(PREDEFINED_LAYERS_PALETTE)]
                colors[layer] = qcolor
        return colors

    def _get_layer_aci(self, layer_name: str, layers: list[str]) -> int:
        upper = layer_name.upper()
        if upper in PREDEFINED_LAYER_MAP:
            aci, _ = PREDEFINED_LAYER_MAP[upper]
            return aci
        
        sorted_layers = sorted(layers)
        if layer_name in sorted_layers:
            idx = sorted_layers.index(layer_name)
            aci, _ = PREDEFINED_LAYERS_PALETTE[idx % len(PREDEFINED_LAYERS_PALETTE)]
            return aci
        return 7

    def _sanitize_layer_name(self, layer_name: str) -> str:
        sanitized = layer_name.translate(DXF_INVALID_LAYER_CHAR_REPLACEMENTS)
        sanitized = " ".join(sanitized.split())
        return sanitized or "Layer"

    def _build_safe_layer_map(self, layer_names: list[str]) -> tuple[dict[str, str], list[tuple[str, str]]]:
        used_names: set[str] = set()
        safe_names: dict[str, str] = {}
        adjusted_names: list[tuple[str, str]] = []

        for original_name in sorted(layer_names):
            base_name = self._sanitize_layer_name(original_name)
            safe_name = base_name
            suffix = 2
            while safe_name in used_names:
                safe_name = f"{base_name} ({suffix})"
                suffix += 1

            used_names.add(safe_name)
            safe_names[original_name] = safe_name
            if safe_name != original_name:
                adjusted_names.append((original_name, safe_name))

        return safe_names, adjusted_names

    def _refresh_from_entity_geometries(self) -> None:
        self.layer_counts = defaultdict(int)
        for entity in self.entity_geometries:
            self.layer_counts[entity.layer] += len(entity.segments)

        if not self.entity_geometries:
            if self.viewer is not None:
                self.viewer.clear()
            self.layer_list.clear()
            self._update_info_after_layer_delete()
            return

        layer_counts = dict(self.layer_counts)
        segments = flatten_segments_for_viewer(self.entity_geometries)
        layer_colors = self._build_layer_colors(list(layer_counts.keys()))
        self._ensure_viewer().draw_segments(segments, layer_colors)

        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for layer in sorted(layer_counts.keys()):
            item = QListWidgetItem(f"{layer} ({layer_counts[layer]} segmentos)")
            item.setForeground(layer_colors[layer])
            item.setData(Qt.ItemDataRole.UserRole, layer)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.layer_list.addItem(item)
        self.layer_list.blockSignals(False)
        self._update_info_after_layer_delete()

    def merge_montantes_union(self) -> None:
        from src.domain.structural_processor import StructuralProcessor
        from src.ui.dialogs import ResultTextDialog

        if not self.entity_geometries:
            QMessageBox.information(
                self,
                "Unir montantes union",
                "Carga primero un archivo DXF con geometria.",
            )
            return

        tol_union_horizontal, ok = QInputDialog.getDouble(
            self,
            "Unir montantes union",
            "Tolerancia horizontal:",
            self.merge_union_tol,
            0.001,
            1000000.0,
            3,
        )
        if not ok:
            return
        self.merge_union_tol = tol_union_horizontal

        processor = StructuralProcessor(self.entity_geometries)
        success, report = processor.execute_merge_montantes_union(tol_union_horizontal)
        
        if not success:
            QMessageBox.information(self, "Unir montantes union", report)
            return

        self.entity_geometries = processor.get_geometries()
        self._refresh_from_entity_geometries()

        ResultTextDialog(title="Resultado: unir montantes union", text=report, parent=self).exec()

    def merge_montantes_esquinas(self) -> None:
        from src.domain.structural_processor import StructuralProcessor
        from src.ui.dialogs import ResultTextDialog

        if not self.entity_geometries:
            QMessageBox.information(
                self,
                "Unir montantes esquinas",
                "Carga primero un archivo DXF con geometria.",
            )
            return

        tol_union_horizontal, ok = QInputDialog.getDouble(
            self,
            "Unir montantes esquinas",
            "Tolerancia horizontal:",
            self.merge_esquina_tol,
            0.001,
            1000000.0,
            3,
        )
        if not ok:
            return
        self.merge_esquina_tol = tol_union_horizontal

        processor = StructuralProcessor(self.entity_geometries)
        success, report = processor.execute_merge_montantes_esquinas(tol_union_horizontal)
        
        if not success:
            QMessageBox.information(self, "Unir montantes esquinas", report)
            return

        self.entity_geometries = processor.get_geometries()
        self._refresh_from_entity_geometries()

        ResultTextDialog(title="Resultado: unir montantes esquinas", text=report, parent=self).exec()

    def merge_durmientes_testeros(self) -> None:
        from src.domain.structural_processor import StructuralProcessor
        from src.ui.dialogs import ResultTextDialog

        if not self.entity_geometries:
            QMessageBox.information(
                self,
                "Unir durmientes y testeros",
                "Carga primero un archivo DXF con geometria.",
            )
            return

        z_drop, ok = QInputDialog.getDouble(
            self,
            "Unir durmientes y testeros",
            "Distancia vertical (Z):",
            self.merge_z_drop,
            0.001,
            1000000.0,
            3,
        )
        if not ok:
            return
        self.merge_z_drop = z_drop

        processor = StructuralProcessor(self.entity_geometries)
        success, report = processor.execute_merge_durmientes_testeros(z_drop)
        
        if not success:
            QMessageBox.information(self, "Unir durmientes y testeros", report)
            return

        self.entity_geometries = processor.get_geometries()
        self._refresh_from_entity_geometries()

        ResultTextDialog(title="Resultado: unir durmientes y testeros", text=report, parent=self).exec()

    def merge_durmientes_testeros_contrapeados(self) -> None:
        from src.domain.structural_processor import StructuralProcessor
        from src.ui.dialogs import ResultTextDialog

        if not self.entity_geometries:
            QMessageBox.information(
                self,
                "Unir durmientes y testeros (Contrapeados)",
                "Carga primero un archivo DXF con geometria.",
            )
            return

        z_drop, ok = QInputDialog.getDouble(
            self,
            "Unir durmientes y testeros (Contrapeados)",
            "Distancia vertical maxima (Z):",
            self.merge_z_drop,
            0.001,
            1000000.0,
            3,
        )
        if not ok:
            return
        self.merge_z_drop = z_drop

        processor = StructuralProcessor(self.entity_geometries)
        success, report = processor.execute_merge_durmientes_testeros_contrapeados(z_drop)
        
        if not success:
            QMessageBox.information(self, "Unir durmientes y testeros (Contrapeados)", report)
            return

        self.entity_geometries = processor.get_geometries()
        self._refresh_from_entity_geometries()

        ResultTextDialog(title="Resultado: unir durmientes y testeros (contrapeados)", text=report, parent=self).exec()

    def generate_level_lines(self) -> None:
        from src.domain.structural_processor import StructuralProcessor
        from src.ui.dialogs import ResultTextDialog

        if not self.entity_geometries:
            QMessageBox.information(
                self,
                "Generar líneas de sobrecarga",
                "Carga primero un archivo DXF con geometria.",
            )
            return

        processor = StructuralProcessor(self.entity_geometries)
        success, report = processor.generate_level_lines()
        
        if not success:
            QMessageBox.information(self, "Generar líneas de sobrecarga", report)
            return

        self.entity_geometries = processor.get_geometries()
        self._refresh_from_entity_geometries()

        ResultTextDialog(title="Resultado: generar líneas de sobrecarga", text=report, parent=self).exec()
        
    def detect_montantes_durmientes(self) -> None:
        from src.domain.structural_processor import StructuralProcessor
        from src.ui.dialogs import ResultTextDialog

        if not self.entity_geometries:
            QMessageBox.information(
                self,
                "Detectar montantes-durmientes",
                "Carga primero un archivo DXF con geometria.",
            )
            return

        tolerance = 1.0
        processor = StructuralProcessor(self.entity_geometries)
        connections, entity_lookup = processor.detect_montantes_to_durmientes(tolerance)
        
        if not connections:
            QMessageBox.information(
                self,
                "Detectar montantes-durmientes",
                "No se encontraron entidades en capas de durmiente.",
            )
            return

        lines: list[str] = [f"Tolerancia: {tolerance}", ""]
        total_links = 0
        for durmiente_handle in sorted(connections.keys()):
            durmiente = entity_lookup.get(durmiente_handle)
            if durmiente is None:
                continue

            montantes = sorted(connections[durmiente_handle])
            total_links += len(montantes)
            lines.append(f"DURMIENTE {durmiente.layer} [{durmiente.handle}]")
            if montantes:
                for montante_handle in montantes:
                    montante = entity_lookup.get(montante_handle)
                    if montante is None:
                        lines.append(f"  - MONTANTE [{montante_handle}]")
                    else:
                        lines.append(f"  - {montante.layer} [{montante.handle}]")
            else:
                lines.append("  - Sin montantes conectados")
            lines.append("")

        summary = (
            f"Durmientes analizados: {len(connections)} | "
            f"Conexiones detectadas: {total_links}"
        )
        result_text = summary + "\n\n" + "\n".join(lines)
        ResultTextDialog(
            title="Resultado: montantes por durmiente",
            text=result_text,
            parent=self,
        ).exec()

    def preprocesar_cadwork(self) -> None:
        from src.domain.structural_processor import StructuralProcessor
        from src.ui.dialogs import PreprocesarCadworkDialog, ResultTextDialog

        if not self.entity_geometries:
            QMessageBox.information(
                self,
                "Preprocesar y clasificar (Cadwork)",
                "Carga primero un archivo DXF con geometria.",
            )
            return

        dialog = PreprocesarCadworkDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        tol_esquinas, tol_extremo, tol_xy, tol_z, tol_hueco, tol_cota, isolated_as_pasante = dialog.get_tolerances()

        processor = StructuralProcessor(self.entity_geometries)
        success, report = processor.execute_preprocesar_cadwork(
            tol_corner_esquinas=tol_esquinas,
            tol_extremo_fachada=tol_extremo,
            tol_xy_proximity=tol_xy,
            tol_z_plate=tol_z,
            tol_hueco_xy=tol_hueco,
            tol_cota_nivel=tol_cota,
            default_isolated_as_pasante=isolated_as_pasante,
        )

        if not success:
            QMessageBox.warning(
                self,
                "Preprocesar y clasificar (Cadwork)",
                f"Error durante el preprocesamiento:\n\n{report}",
            )
            return

        self.entity_geometries = processor.get_geometries()
        self._refresh_from_entity_geometries()

        ResultTextDialog(
            title="Preprocesamiento Cadwork finalizado",
            text=report,
            parent=self,
        ).exec()

    def save_dxf_file(self) -> None:
        from ezdxf import new

        if not self.entity_geometries:
            QMessageBox.information(
                self,
                "Guardar DXF",
                "No hay geometria para guardar.",
            )
            return

        default_name = "resultado_uniones.dxf"
        if self.current_file_path.lower().endswith(".dxf"):
            base = os.path.splitext(os.path.basename(self.current_file_path))[0]
            default_name = f"{base}_editado.dxf"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar archivo DXF",
            default_name,
            "DXF Files (*.dxf);;All Files (*)",
        )
        if not file_path:
            return

        if not file_path.lower().endswith(".dxf"):
            file_path = f"{file_path}.dxf"

        try:
            layer_names = sorted({entity.layer for entity in self.entity_geometries})
            safe_layer_map, adjusted_names = self._build_safe_layer_map(layer_names)

            # Determinar nombre de capa efectivo sin mutar en memoria todavía
            effective_layer = {
                entity.handle: safe_layer_map[entity.layer] if adjusted_names else entity.layer
                for entity in self.entity_geometries
            }

            doc = new("R2010")
            msp = doc.modelspace()

            existing_layers = {layer.dxf.name for layer in doc.layers}
            for entity in self.entity_geometries:
                lname = effective_layer[entity.handle]
                if lname not in existing_layers:
                    aci = self._get_layer_aci(entity.layer, layer_names)
                    doc.layers.new(name=lname, dxfattribs={"color": aci})
                    existing_layers.add(lname)

            scale_down = getattr(self, "dxf_scale_factor", 1.0)
            for entity in self.entity_geometries:
                lname = effective_layer[entity.handle]
                for x1, y1, z1, x2, y2, z2 in entity.segments:
                    msp.add_line(
                        (float(x1 / scale_down), float(y1 / scale_down), float(z1 / scale_down)),
                        (float(x2 / scale_down), float(y2 / scale_down), float(z2 / scale_down)),
                        dxfattribs={"layer": lname},
                    )

            doc.saveas(file_path)

            # Mutar en memoria solo tras escritura exitosa
            if adjusted_names:
                for entity in self.entity_geometries:
                    entity.layer = effective_layer[entity.handle]
                self._refresh_from_entity_geometries()
                self.statusBar().showMessage(
                    f"DXF guardado: {file_path} | Se ajustaron nombres de capa para DXF"
                )
            else:
                self.statusBar().showMessage(f"DXF guardado: {file_path}")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Guardar DXF",
                f"No se pudo guardar el archivo DXF.\n\nDetalle: {exc}",
            )

    def load_dxf_file(self) -> None:
        from ezdxf import readfile

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo DXF",
            "",
            "DXF Files (*.dxf);;All Files (*)",
        )
        if not file_path:
            return

        try:
            doc = readfile(file_path)
            msp = doc.modelspace()

            entity_geometries, layer_counts = collect_entity_geometries(msp)
            if not entity_geometries:
                if self.viewer is not None:
                    self.viewer.clear()
                self.layer_counts.clear()
                self.current_file_path = file_path
                self.entity_geometries = []
                self.layer_list.clear()
                self.info_label.setText("El archivo no contiene lineas o polilineas visibles.")
                self.statusBar().showMessage("DXF cargado sin lineas")
                return

            self.layer_counts = dict(layer_counts)
            self.current_file_path = file_path
            
            # Auto-detect unit scaling (meters vs millimeters)
            max_coord = 0.0
            for entity in entity_geometries:
                for seg in entity.segments:
                    max_coord = max(max_coord, abs(seg[0]), abs(seg[1]), abs(seg[2]), abs(seg[3]), abs(seg[4]), abs(seg[5]))
            
            self.dxf_scale_factor = 1.0
            if max_coord < 50.0:
                self.dxf_scale_factor = 1000.0  # meters -> millimeters
            elif max_coord < 500.0:
                self.dxf_scale_factor = 10.0    # centimeters -> millimeters
                
            if self.dxf_scale_factor != 1.0:
                scaled_geometries = []
                for entity in entity_geometries:
                    scaled_segs = [
                        (
                            x1 * self.dxf_scale_factor,
                            y1 * self.dxf_scale_factor,
                            z1 * self.dxf_scale_factor,
                            x2 * self.dxf_scale_factor,
                            y2 * self.dxf_scale_factor,
                            z2 * self.dxf_scale_factor
                        )
                        for x1, y1, z1, x2, y2, z2 in entity.segments
                    ]
                    entity.segments = scaled_segs
                    scaled_geometries.append(entity)
                self.entity_geometries = scaled_geometries
            else:
                self.entity_geometries = entity_geometries

            segments = flatten_segments_for_viewer(self.entity_geometries)
            layer_colors = self._build_layer_colors(list(layer_counts.keys()))
            self._ensure_viewer().draw_segments(segments, layer_colors)

            self.layer_list.clear()
            for layer in sorted(layer_counts.keys()):
                item = QListWidgetItem(f"{layer} ({layer_counts[layer]} segmentos)")
                item.setForeground(layer_colors[layer])
                item.setData(Qt.ItemDataRole.UserRole, layer)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                self.layer_list.addItem(item)

            self.info_label.setText(
                f"Archivo: {file_path} | Segmentos: {len(segments)} | Capas: {len(layer_counts)}"
            )
            self.statusBar().showMessage("DXF cargado correctamente")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error al cargar DXF",
                f"No se pudo abrir o procesar el archivo.\n\nDetalle: {exc}",
            )
            self.statusBar().showMessage("Error al cargar DXF")

    def _on_layer_item_changed(self, item: QListWidgetItem) -> None:
        layer = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(layer, str):
            return
        visible = item.checkState() == Qt.CheckState.Checked
        if self.viewer is not None:
            self.viewer.set_layer_visible(layer, visible)

    def open_delete_layer_dialog(self) -> None:
        from src.ui.dialogs import DeleteLayerDialog

        layers = sorted(self.layer_counts.keys())
        if not layers:
            QMessageBox.information(
                self,
                "Eliminar capa",
                "No hay capas cargadas para eliminar.",
            )
            return

        dialog = DeleteLayerDialog(layers, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        layer_to_remove = dialog.selected_layer()
        if not layer_to_remove:
            return

        if self.viewer is None or not self.viewer.remove_layer(layer_to_remove):
            QMessageBox.warning(
                self,
                "Eliminar capa",
                f"No se pudo eliminar la capa '{layer_to_remove}'.",
            )
            return

        self.layer_counts.pop(layer_to_remove, None)
        self.entity_geometries = [
            entity for entity in self.entity_geometries if entity.layer != layer_to_remove
        ]
        self._remove_layer_item(layer_to_remove)
        self._update_info_after_layer_delete()
        self.statusBar().showMessage(f"Capa eliminada: {layer_to_remove}")

    def open_rename_layers_dialog(self) -> None:
        from src.ui.dialogs import RenameLayersDialog

        layers = sorted(self.layer_counts.keys())
        if not layers:
            QMessageBox.information(
                self,
                "Renombrar capas",
                "No hay capas cargadas para renombrar.",
            )
            return

        dialog = RenameLayersDialog(layers, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        renamed_layers = dialog.renamed_layers()
        if not renamed_layers:
            return

        invalid_names = [name for name in renamed_layers.values() if not name]
        if invalid_names:
            QMessageBox.warning(
                self,
                "Renombrar capas",
                "Todos los nombres de capa deben tener contenido.",
            )
            return

        new_names = list(renamed_layers.values())
        if len(new_names) != len(set(new_names)):
            QMessageBox.warning(
                self,
                "Renombrar capas",
                "No puede haber nombres de capa duplicados.",
            )
            return

        safe_layer_map, adjusted_names = self._build_safe_layer_map(new_names)
        safe_renamed_layers = {
            original_name: safe_layer_map[new_name]
            for original_name, new_name in renamed_layers.items()
        }

        safe_names = list(safe_renamed_layers.values())
        if len(safe_names) != len(set(safe_names)):
            QMessageBox.warning(
                self,
                "Renombrar capas",
                "Los nombres de capa resultan duplicados despues de ajustarlos a un formato valido para DXF.",
            )
            return

        changes = 0
        for entity in self.entity_geometries:
            updated_name = safe_renamed_layers.get(entity.layer)
            if updated_name is None or updated_name == entity.layer:
                continue
            entity.layer = updated_name
            changes += 1

        if changes == 0:
            self.statusBar().showMessage("No se realizaron cambios en los nombres de capas")
            return

        self._refresh_from_entity_geometries()
        if adjusted_names:
            self.statusBar().showMessage(
                "Capas renombradas; se ajustaron caracteres incompatibles con DXF"
            )
        else:
            self.statusBar().showMessage("Capas renombradas correctamente")

    def _remove_layer_item(self, layer: str) -> None:
        for index in range(self.layer_list.count()):
            item = self.layer_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == layer:
                self.layer_list.takeItem(index)
                return

    def _update_info_after_layer_delete(self) -> None:
        total_segments = sum(self.layer_counts.values())
        total_layers = len(self.layer_counts)
        file_part = self.current_file_path if self.current_file_path else "Sin archivo"
        self.info_label.setText(
            f"Archivo: {file_part} | Segmentos: {total_segments} | Capas: {total_layers}"
        )
