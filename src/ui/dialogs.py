from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


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
