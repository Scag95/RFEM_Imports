from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
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


class DeleteLayerDialog(QDialog):
    def __init__(self, layers: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Eliminar capa")
        self.setModal(True)
        self.setMinimumWidth(360)

        self.layer_combo = QComboBox(self)
        self.layer_combo.addItems(layers)

        self.delete_button = QPushButton("Eliminar", self)
        self.delete_button.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Selecciona la capa a eliminar:"))
        layout.addWidget(self.layer_combo)
        layout.addWidget(self.delete_button)

    def selected_layer(self) -> str:
        return self.layer_combo.currentText().strip()


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


class RenameLayersDialog(QDialog):
    def __init__(self, layers: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Renombrar capas")
        self.setModal(True)
        self.resize(760, 480)

        self.layer_table = ExcelPasteTableWidget(len(layers), 2, self)
        self.layer_table.setHorizontalHeaderLabels(["Capa actual", "Nuevo nombre"])
        self.layer_table.verticalHeader().setVisible(False)
        self.layer_table.setAlternatingRowColors(True)
        self.layer_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.layer_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.layer_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
        )

        header = self.layer_table.horizontalHeader()
        header.setStretchLastSection(True)

        for row, layer in enumerate(layers):
            self.layer_table.setItem(row, 0, QTableWidgetItem(layer))
            self.layer_table.setItem(
                row,
                1,
                QTableWidgetItem(DEFAULT_LAYER_RENAMES.get(layer, layer)),
            )

        if layers:
            self.layer_table.setCurrentCell(0, 0)

        self.accept_button = QPushButton("Aplicar", self)
        self.accept_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton("Cancelar", self)
        self.cancel_button.clicked.connect(self.reject)

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(self.cancel_button)
        buttons_layout.addWidget(self.accept_button)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Edita o pega desde Excel en la tabla (Ctrl+V):"))
        layout.addWidget(self.layer_table)
        layout.addLayout(buttons_layout)

    def renamed_layers(self) -> dict[str, str]:
        renamed: dict[str, str] = {}
        for row in range(self.layer_table.rowCount()):
            current_item = self.layer_table.item(row, 0)
            new_item = self.layer_table.item(row, 1)
            if current_item is None or new_item is None:
                continue
            current_name = current_item.text().strip()
            new_name = new_item.text().strip()
            if current_name:
                renamed[current_name] = new_name
        return renamed
