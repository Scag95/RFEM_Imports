from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

DEFAULT_LAYER_RENAMES: dict[str, str] = {
    "TESTERO (PASANTE)": "R_M1 90-140 $C22 (140. Testero - durmiente)",
    "MONTANTES SIMPLES HUECOS (PASANTE)": "R_M1 90-140 $C22 (140. Montante hueco)",
    "MONTANTES LARGOS (PASANTE)": "R_M1 90-140 $C22 (140. Montante simple)",
    "MONTANTES CORTOS (PASANTE)": "R_M1 90-140 $C22 (140. Montante corto)",
    "DURMIENTE (PASANTE)": "R_M1 90-140 $C22 (140. Durmiente)",
    "ALFEIZAR (PASANTE)": "R_M1 90-140 $C22 (140. Alfeizar)",
    "DINTEL (PASANTE)": "R_M1 90-140 $C22 (140. Dintel)",
    "MONTANTES DE ESQUINA (PASANTE)": "R_M1 140-140 $C22 (140. Montantes esquina)",
    "LISTON PETOS": "R_M1 140-140 $C22 (140. Liston petos)",
    "TESTERO (PETOS)": "R_M1 140-140 $C22 (140. Testero-durmiente petos)",
    "TESTERO (NO PASANTE)": "R_M1 90-90 $C22 (90. Testero)",
    "MONTANTES LARGOS (NO PASANTE)": "R_M1 90-90 $C22 (90. Montante simple)",
    "MONTANTES SIMPLES HUECOS (NO PASANTE)": "R_M1 90-90 $C22 (90. Montante hueco (doble))",
    "MONTANTES CORTOS (NO PASANTE)": "R_M1 90-90 $C22 (90. Montante corto)",
    "DURMIENTE (NO PASANTE)": "R_M1 90-90 $C22 (90. Durmiente)",
    "ALFEIZAR (NO PASANTE)": "R_M1 90-90 $C22 (90. Alfeizar)",
    "DINTEL (NO PASANTE)": "R_M1 90-90 $C22 (90. Dintel)",
    "MONTANTES DE ESQUINA (NO PASANTE)": "R_M1 90-90 $C22 (90. Montantes esquina)",
    "MONTANTES UNION (PASANTE)": "R_M1 180-140 $C22 (140. Montantes Union)",
}


class ExcelPasteTableWidget(QTableWidget):
    def __init__(self, rows: int, columns: int, parent: QWidget | None = None) -> None:
        super().__init__(rows, columns, parent)
        self._paste_shortcut = QShortcut(QKeySequence.StandardKey.Paste, self)
        self._paste_shortcut.activated.connect(self.paste_from_clipboard)

    def paste_from_clipboard(self) -> None:
        text = QApplication.clipboard().text().strip()
        if not text:
            return

        start_row = max(self.currentRow(), 0)
        start_column = max(self.currentColumn(), 0)
        clipboard_rows = [row for row in text.splitlines() if row]
        if not clipboard_rows:
            return

        for row_offset, row_text in enumerate(clipboard_rows):
            target_row = start_row + row_offset
            if target_row >= self.rowCount():
                break

            for column_offset, value in enumerate(row_text.split("\t")):
                target_column = start_column + column_offset
                if target_column >= self.columnCount():
                    break

                item = self.item(target_row, target_column)
                if item is None:
                    item = QTableWidgetItem()
                    self.setItem(target_row, target_column, item)
                item.setText(value.strip())


# class DeleteLayerDialog(QDialog):
#     def __init__(self, layers: list[str], parent: QWidget | None = None) -> None:
#         super().__init__(parent)
#         self.setWindowTitle("Eliminar capa")
#         self.setModal(True)
#         self.setMinimumWidth(360)

#         self.layer_combo = QComboBox(self)
#         self.layer_combo.addItems(layers)

#         self.delete_button = QPushButton("Eliminar", self)
#         self.delete_button.clicked.connect(self.accept)

#         layout = QVBoxLayout(self)
#         layout.addWidget(QLabel("Selecciona la capa a eliminar:"))
#         layout.addWidget(self.layer_combo)
#         layout.addWidget(self.delete_button)

#     def selected_layer(self) -> str:
#         return self.layer_combo.currentText().strip()


class ResultTextDialog(QDialog):
    def __init__(self, title: str, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(760, 560)

        text_view = QPlainTextEdit(self)
        text_view.setPlainText(text)
        text_view.setReadOnly(True)

        close_button = QPushButton("Cerrar", self)
        close_button.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(text_view)
        layout.addWidget(close_button)




# class RenameLayersDialog(QDialog):
#     def __init__(self, layers: list[str], parent: QWidget | None = None) -> None:
#         super().__init__(parent)
#         self.setWindowTitle("Renombrar capas")
#         self.setModal(True)
#         self.resize(760, 480)

#         self.layer_table = ExcelPasteTableWidget(len(layers), 2, self)
#         self.layer_table.setHorizontalHeaderLabels(["Capa actual", "Nuevo nombre"])
#         self.layer_table.verticalHeader().setVisible(False)
#         self.layer_table.setAlternatingRowColors(True)
#         self.layer_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
#         self.layer_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
#         self.layer_table.setEditTriggers(
#             QAbstractItemView.EditTrigger.DoubleClicked
#             | QAbstractItemView.EditTrigger.EditKeyPressed
#             | QAbstractItemView.EditTrigger.AnyKeyPressed
#         )

#         header = self.layer_table.horizontalHeader()
#         header.setStretchLastSection(True)

#         for row, layer in enumerate(layers):
#             self.layer_table.setItem(row, 0, QTableWidgetItem(layer))
#             self.layer_table.setItem(
#                 row,
#                 1,
#                 QTableWidgetItem(DEFAULT_LAYER_RENAMES.get(layer, layer)),
#             )

#         if layers:
#             self.layer_table.setCurrentCell(0, 0)

#         self.accept_button = QPushButton("Aplicar", self)
#         self.accept_button.clicked.connect(self.accept)

#         self.cancel_button = QPushButton("Cancelar", self)
#         self.cancel_button.clicked.connect(self.reject)

#         buttons_layout = QHBoxLayout()
#         buttons_layout.addStretch(1)
#         buttons_layout.addWidget(self.cancel_button)
#         buttons_layout.addWidget(self.accept_button)

#         layout = QVBoxLayout(self)
#         layout.addWidget(QLabel("Edita o pega desde Excel en la tabla (Ctrl+V):"))
#         layout.addWidget(self.layer_table)
#         layout.addLayout(buttons_layout)

    # def renamed_layers(self) -> dict[str, str]:
    #     renamed: dict[str, str] = {}
    #     for row in range(self.layer_table.rowCount()):
    #         current_item = self.layer_table.item(row, 0)
    #         new_item = self.layer_table.item(row, 1)
    #         if current_item is None or new_item is None:
    #             continue
    #         current_name = current_item.text().strip()
    #         new_name = new_item.text().strip()
    #         if current_name:
    #             renamed[current_name] = new_name
    #     return renamed

class PreprocesarCadworkDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preprocesar y clasificar (Cadwork)")
        self.setModal(True)
        self.setMinimumWidth(480)

        from PyQt6.QtWidgets import QFormLayout, QDoubleSpinBox, QDialogButtonBox

        layout = QVBoxLayout(self)

        instructions = QLabel(
            "Configura las tolerancias geométricas (en milímetros) para el pipeline de Cadwork:\n"
            "- Las uniones de esquina se detectarán en paneles cruzados en ángulo.\n"
            "- Los extremos libres desclasifican montantes de unión aislados."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("margin-bottom: 15px; font-size: 11px; color: #444444; font-weight: bold;")
        layout.addWidget(instructions)

        form = QFormLayout()

        # Corner Esquinas Tolerance
        self.tol_esquinas_spin = QDoubleSpinBox(self)
        self.tol_esquinas_spin.setRange(0.0, 1000.0)
        self.tol_esquinas_spin.setSingleStep(10.0)
        self.tol_esquinas_spin.setValue(150.0)
        self.tol_esquinas_spin.setSuffix(" mm")
        form.addRow("Tolerancia de Esquinas:", self.tol_esquinas_spin)
        
        tol_esquinas_help = QLabel("Distancia máxima en planta para agrupar y fusionar montantes en esquinas cruzadas (ej. 150 mm representa el espesor del muro).")
        tol_esquinas_help.setWordWrap(True)
        tol_esquinas_help.setStyleSheet("color: #666666; font-size: 10px; font-style: italic; margin-bottom: 8px;")
        form.addRow("", tol_esquinas_help)

        # Facade Ends Isolation Tolerance
        self.tol_extremo_spin = QDoubleSpinBox(self)
        self.tol_extremo_spin.setRange(0.0, 1000.0)
        self.tol_extremo_spin.setSingleStep(10.0)
        self.tol_extremo_spin.setValue(200.0)
        self.tol_extremo_spin.setSuffix(" mm")
        form.addRow("Aislamiento Extremos Fachada:", self.tol_extremo_spin)
        
        tol_extremo_help = QLabel("Distancia mínima a otro panel. Si no hay ningún vecino en este radio, el extremo se considera final de fachada y el montante de unión se convierte a montante simple.")
        tol_extremo_help.setWordWrap(True)
        tol_extremo_help.setStyleSheet("color: #666666; font-size: 10px; font-style: italic; margin-bottom: 8px;")
        form.addRow("", tol_extremo_help)

        # XY Wall Proximity Tolerance
        self.tol_xy_spin = QDoubleSpinBox(self)
        self.tol_xy_spin.setRange(0.0, 1000.0)
        self.tol_xy_spin.setSingleStep(10.0)
        self.tol_xy_spin.setValue(150.0)
        self.tol_xy_spin.setSuffix(" mm")
        form.addRow("Proximidad Paneles (XY):", self.tol_xy_spin)
        
        tol_xy_help = QLabel("Tolerancia máxima horizontal en planta para asociar montantes, dinteles y alfeizares a la línea directriz del panel.")
        tol_xy_help.setWordWrap(True)
        tol_xy_help.setStyleSheet("color: #666666; font-size: 10px; font-style: italic; margin-bottom: 8px;")
        form.addRow("", tol_xy_help)

        # Z Plate Proximity Tolerance
        self.tol_z_spin = QDoubleSpinBox(self)
        self.tol_z_spin.setRange(0.0, 1000.0)
        self.tol_z_spin.setSingleStep(10.0)
        self.tol_z_spin.setValue(100.0)
        self.tol_z_spin.setSuffix(" mm")
        form.addRow("Tolerancia de Placas (Z):", self.tol_z_spin)
        
        tol_z_help = QLabel("Desviación vertical máxima permitida para considerar que la cabeza o el pie de un montante coincide con el testero o durmiente de su panel.")
        tol_z_help.setWordWrap(True)
        tol_z_help.setStyleSheet("color: #666666; font-size: 10px; font-style: italic; margin-bottom: 8px;")
        form.addRow("", tol_z_help)

        # Hueco (Windows) Proximity Tolerance
        self.tol_hueco_spin = QDoubleSpinBox(self)
        self.tol_hueco_spin.setRange(0.0, 1000.0)
        self.tol_hueco_spin.setSingleStep(10.0)
        self.tol_hueco_spin.setValue(100.0)
        self.tol_hueco_spin.setSuffix(" mm")
        form.addRow("Proximidad a Huecos (planta):", self.tol_hueco_spin)
        
        tol_hueco_help = QLabel("Tolerancia horizontal en planta para identificar los montantes laterales de ventanas/puertas (jambas) y extender dinteles y alfeizares hacia ellos.")
        tol_hueco_help.setWordWrap(True)
        tol_hueco_help.setStyleSheet("color: #666666; font-size: 10px; font-style: italic; margin-bottom: 8px;")
        form.addRow("", tol_hueco_help)

        # Cota de Nivel Z Height
        self.tol_cota_spin = QDoubleSpinBox(self)
        self.tol_cota_spin.setRange(0.0, 5000.0)
        self.tol_cota_spin.setSingleStep(25.0)
        self.tol_cota_spin.setValue(1125.0)
        self.tol_cota_spin.setSuffix(" mm")
        form.addRow("Altura Cota de Nivel (Z):", self.tol_cota_spin)
        
        tol_cota_help = QLabel("Distancia vertical en Z por encima de la base del durmiente para trazar la línea de cota de nivel continua (ej. 1125 mm).")
        tol_cota_help.setWordWrap(True)
        tol_cota_help.setStyleSheet("color: #666666; font-size: 10px; font-style: italic; margin-bottom: 12px;")
        form.addRow("", tol_cota_help)

        # Isolated Panels Checkbox
        self.isolated_as_pasante_check = QCheckBox("Considerar paneles aislados como Pasantes", self)
        self.isolated_as_pasante_check.setChecked(True)
        form.addRow("", self.isolated_as_pasante_check)
        
        isolated_help = QLabel("Si un panel no tiene vecinos arriba ni abajo (ej. plantas únicas), se clasificará como Pasante si está marcado, o No Pasante si se desmarca.")
        isolated_help.setWordWrap(True)
        isolated_help.setStyleSheet("color: #666666; font-size: 10px; font-style: italic; margin-bottom: 15px;")
        form.addRow("", isolated_help)

        layout.addLayout(form)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self
        )
        button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Aplicar")
        button_box.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancelar")
        
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_tolerances(self) -> tuple[float, float, float, float, float, float, bool]:
        return (
            self.tol_esquinas_spin.value(),
            self.tol_extremo_spin.value(),
            self.tol_xy_spin.value(),
            self.tol_z_spin.value(),
            self.tol_hueco_spin.value(),
            self.tol_cota_spin.value(),
            self.isolated_as_pasante_check.isChecked()
        )
