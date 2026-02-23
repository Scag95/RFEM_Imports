import os
from collections import defaultdict

import ezdxf
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import (
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
from src.domain.structural_processor import StructuralProcessor
from src.ui.dialogs import DeleteLayerDialog, ResultTextDialog
from src.ui.viewer_3d import DXFViewer3D


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RFEM DXF Manager - 3D Viewer")
        self.resize(1400, 900)

        self.viewer = DXFViewer3D()
        self.layer_counts: dict[str, int] = {}
        self.current_file_path = ""
        self.entity_geometries: list[EntityGeometry] = []
        self.merge_z_drop = 125.0
        self.merge_union_tol = 100.0
        self.merge_esquina_tol = 145.0
        
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
        left_layout.addWidget(self.viewer)
        left_layout.addWidget(self.info_label)
        left_layout.setStretch(0, 1)
        left_layout.setStretch(1, 0)

        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Listo")
        self._create_menu()

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
        remove_layer_action = QAction("Eliminar capa", self)
        remove_layer_action.triggered.connect(self.open_delete_layer_dialog)
        tools_menu.addAction(remove_layer_action)

        detect_connections_action = QAction("Detectar montantes-durmientes", self)
        detect_connections_action.triggered.connect(self.detect_montantes_durmientes)
        tools_menu.addAction(detect_connections_action)

        merge_action = QAction("Unir durmientes y testeros", self)
        merge_action.triggered.connect(self.merge_durmientes_testeros)
        tools_menu.addAction(merge_action)

        merge_union_action = QAction("Unir montantes union", self)
        merge_union_action.triggered.connect(self.merge_montantes_union)
        tools_menu.addAction(merge_union_action)

        merge_corner_action = QAction("Unir montantes esquinas", self)
        merge_corner_action.triggered.connect(self.merge_montantes_esquinas)
        tools_menu.addAction(merge_corner_action)

    def _build_layer_colors(self, layers: list[str]) -> dict[str, QColor]:
        colors: dict[str, QColor] = {}
        for i, layer in enumerate(sorted(layers)):
            hue = int((i * 137.5) % 360)
            colors[layer] = QColor.fromHsv(hue, 210, 230)
        return colors

    def _refresh_from_entity_geometries(self) -> None:
        self.layer_counts = defaultdict(int)
        for entity in self.entity_geometries:
            self.layer_counts[entity.layer] += len(entity.segments)

        if not self.entity_geometries:
            self.viewer.clear()
            self.layer_list.clear()
            self._update_info_after_layer_delete()
            return

        layer_counts = dict(self.layer_counts)
        segments = flatten_segments_for_viewer(self.entity_geometries)
        layer_colors = self._build_layer_colors(list(layer_counts.keys()))
        self.viewer.draw_segments(segments, layer_colors)

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
        
    def detect_montantes_durmientes(self) -> None:
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

    def save_dxf_file(self) -> None:
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
            doc = ezdxf.new("R2010")
            msp = doc.modelspace()

            existing_layers = {layer.dxf.name for layer in doc.layers}
            for entity in self.entity_geometries:
                if entity.layer not in existing_layers:
                    doc.layers.new(name=entity.layer)
                    existing_layers.add(entity.layer)

            for entity in self.entity_geometries:
                for x1, y1, z1, x2, y2, z2 in entity.segments:
                    msp.add_line(
                        (float(x1), float(y1), float(z1)),
                        (float(x2), float(y2), float(z2)),
                        dxfattribs={"layer": entity.layer},
                    )

            doc.saveas(file_path)
            self.statusBar().showMessage(f"DXF guardado: {file_path}")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Guardar DXF",
                f"No se pudo guardar el archivo DXF.\n\nDetalle: {exc}",
            )

    def load_dxf_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo DXF",
            "",
            "DXF Files (*.dxf);;All Files (*)",
        )
        if not file_path:
            return

        try:
            doc = ezdxf.readfile(file_path)
            msp = doc.modelspace()

            entity_geometries, layer_counts = collect_entity_geometries(msp)
            if not entity_geometries:
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
            self.entity_geometries = entity_geometries
            segments = flatten_segments_for_viewer(entity_geometries)
            layer_colors = self._build_layer_colors(list(layer_counts.keys()))
            self.viewer.draw_segments(segments, layer_colors)

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
        self.viewer.set_layer_visible(layer, visible)

    def open_delete_layer_dialog(self) -> None:
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

        if not self.viewer.remove_layer(layer_to_remove):
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
