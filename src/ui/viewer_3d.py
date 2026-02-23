import struct
from collections import defaultdict

from PyQt6.Qt3DCore import QAttribute, QBuffer, QEntity, QGeometry
from PyQt6.Qt3DExtras import QPhongMaterial, Qt3DWindow
from PyQt6.Qt3DRender import QGeometryRenderer
from PyQt6.QtCore import QByteArray, QPoint, Qt
from PyQt6.QtGui import QColor, QMouseEvent, QQuaternion, QVector3D, QWheelEvent
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class Interactive3DWindow(Qt3DWindow):
    def __init__(self) -> None:
        super().__init__()
        self._last_mouse_pos = QPoint()
        self._active_button = Qt.MouseButton.NoButton
        self._world_up = QVector3D(0.0, 0.0, 1.0)
        self._min_distance = 1.0
        self._max_distance = 1_000_000.0

    def set_zoom_limits(self, min_distance: float, max_distance: float) -> None:
        self._min_distance = min_distance
        self._max_distance = max_distance

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta != 0:
            self._zoom(0.85 if delta > 0 else 1.15)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._active_button = event.button()
        self._last_mouse_pos = event.position().toPoint()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._active_button = Qt.MouseButton.NoButton
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        current = event.position().toPoint()
        delta = current - self._last_mouse_pos
        self._last_mouse_pos = current

        if self._active_button == Qt.MouseButton.LeftButton:
            self._orbit(delta.x(), delta.y())
        elif self._active_button in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan(delta.x(), delta.y())
        event.accept()

    def _zoom(self, factor: float) -> None:
        camera = self.camera()
        center = camera.viewCenter()
        offset = camera.position() - center
        distance = offset.length()
        if distance <= 0.0:
            return

        new_distance = max(self._min_distance, min(distance * factor, self._max_distance))
        offset.normalize()
        camera.setPosition(center + (offset * new_distance))

    def _orbit(self, dx: int, dy: int) -> None:
        camera = self.camera()
        center = camera.viewCenter()
        camera_pos = camera.position()
        offset = camera_pos - center
        if offset.lengthSquared() == 0.0:
            return

        yaw = QQuaternion.fromAxisAndAngle(self._world_up, -dx * 0.35)
        forward = (center - camera_pos).normalized()
        right = QVector3D.crossProduct(forward, self._world_up)
        if right.lengthSquared() < 1e-9:
            right = QVector3D(1.0, 0.0, 0.0)
        else:
            right.normalize()
        pitch = QQuaternion.fromAxisAndAngle(right, -dy * 0.35)

        rotated = (yaw * pitch).rotatedVector(offset)
        camera.setPosition(center + rotated)
        camera.setUpVector(self._world_up)

    def _pan(self, dx: int, dy: int) -> None:
        camera = self.camera()
        center = camera.viewCenter()
        camera_pos = camera.position()
        distance = (camera_pos - center).length()
        if distance <= 0.0:
            return

        forward = (center - camera_pos).normalized()
        right = QVector3D.crossProduct(forward, self._world_up)
        if right.lengthSquared() < 1e-9:
            right = QVector3D(1.0, 0.0, 0.0)
        else:
            right.normalize()
        up = QVector3D.crossProduct(right, forward).normalized()

        scale = distance * 0.0015
        shift = (-right * float(dx) + up * float(dy)) * scale
        camera.setPosition(camera_pos + shift)
        camera.setViewCenter(center + shift)


class DXFViewer3D(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.view = Interactive3DWindow()
        self.container = QWidget.createWindowContainer(self.view)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.container)

        self.root_entity = QEntity()
        self.model_entity = QEntity(self.root_entity)
        self.view.setRootEntity(self.root_entity)
        self.layer_entities: dict[str, QEntity] = {}
        self.view.defaultFrameGraph().setClearColor(QColor(250, 250, 250))

        self.camera = self.view.camera()
        self.camera.lens().setPerspectiveProjection(45.0, 16.0 / 9.0, 0.1, 1000000.0)
        self.camera.setPosition(QVector3D(0.0, -500.0, 500.0))
        self.camera.setViewCenter(QVector3D(0.0, 0.0, 0.0))
        self._world_up = QVector3D(0.0, 0.0, 1.0)

        self.container.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.container.setMouseTracking(True)

    def clear(self) -> None:
        self.model_entity.setParent(None)
        self.model_entity.deleteLater()
        self.model_entity = QEntity(self.root_entity)
        self.layer_entities.clear()

    def draw_segments(
        self,
        segments: list[tuple[float, float, float, float, float, float, str]],
        layer_colors: dict[str, QColor],
    ) -> None:
        self.clear()

        layer_vertices: dict[str, list[float]] = defaultdict(list)
        min_x = min_y = min_z = float("inf")
        max_x = max_y = max_z = float("-inf")

        for x1, y1, z1, x2, y2, z2, layer in segments:
            layer_vertices[layer].extend([x1, y1, z1, x2, y2, z2])

            min_x = min(min_x, x1, x2)
            min_y = min(min_y, y1, y2)
            min_z = min(min_z, z1, z2)
            max_x = max(max_x, x1, x2)
            max_y = max(max_y, y1, y2)
            max_z = max(max_z, z1, z2)

        for layer, vertices in layer_vertices.items():
            entity = self._add_layer_geometry(vertices, layer_colors[layer])
            if entity is not None:
                self.layer_entities[layer] = entity

        self._fit_camera(min_x, min_y, min_z, max_x, max_y, max_z)

    def _add_layer_geometry(self, vertices: list[float], color: QColor) -> QEntity | None:
        if not vertices:
            return None

        entity = QEntity(self.model_entity)
        geometry = QGeometry(entity)

        buffer = QBuffer(geometry)

        packed = struct.pack(f"<{len(vertices)}f", *vertices)
        buffer.setData(QByteArray(packed))

        position_attribute = QAttribute(geometry)
        position_attribute.setName(QAttribute.defaultPositionAttributeName())
        position_attribute.setVertexBaseType(QAttribute.VertexBaseType.Float)
        position_attribute.setVertexSize(3)
        position_attribute.setAttributeType(QAttribute.AttributeType.VertexAttribute)
        position_attribute.setBuffer(buffer)
        position_attribute.setByteStride(3 * 4)
        position_attribute.setCount(len(vertices) // 3)

        geometry.addAttribute(position_attribute)

        renderer = QGeometryRenderer(entity)
        renderer.setGeometry(geometry)
        renderer.setPrimitiveType(QGeometryRenderer.PrimitiveType.Lines)
        renderer.setVertexCount(len(vertices) // 3)

        material = QPhongMaterial(entity)
        material.setAmbient(color)
        material.setDiffuse(color)

        entity.addComponent(renderer)
        entity.addComponent(material)
        return entity

    def set_layer_visible(self, layer: str, visible: bool) -> None:
        entity = self.layer_entities.get(layer)
        if entity is not None:
            entity.setEnabled(visible)

    def remove_layer(self, layer: str) -> bool:
        entity = self.layer_entities.pop(layer, None)
        if entity is None:
            return False
        entity.setEnabled(False)
        entity.setParent(None)
        entity.deleteLater()
        return True

    def _fit_camera(
        self,
        min_x: float,
        min_y: float,
        min_z: float,
        max_x: float,
        max_y: float,
        max_z: float,
    ) -> None:
        center = QVector3D(
            float((min_x + max_x) / 2.0),
            float((min_y + max_y) / 2.0),
            float((min_z + max_z) / 2.0),
        )
        span_x = max_x - min_x
        span_y = max_y - min_y
        span_z = max_z - min_z
        span = max(span_x, span_y, span_z, 1.0)
        distance = float(span * 1.8)
        min_distance = max(span * 0.01, 0.1)
        max_distance = max(distance * 50.0, 1000.0)
        self.view.set_zoom_limits(min_distance, max_distance)

        self.camera.setViewCenter(center)
        self.camera.setPosition(
            QVector3D(center.x(), center.y() - distance, center.z() + distance)
        )
        self.camera.setUpVector(self._world_up)
        self.camera.lens().setPerspectiveProjection(
            45.0,
            max(self.width(), 1) / max(self.height(), 1),
            0.1,
            max(distance * 10.0, 1000.0),
        )
