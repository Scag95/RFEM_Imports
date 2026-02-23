import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

import ezdxf

# Workaround for Qt3D RHI pipeline failures on some Windows/D3D drivers.
os.environ.setdefault("QT3D_RENDERER", "opengl")
os.environ.setdefault("QT_OPENGL", "desktop")

from PyQt6.Qt3DCore import QAttribute, QBuffer, QEntity, QGeometry
from PyQt6.Qt3DExtras import QPhongMaterial, Qt3DWindow
from PyQt6.Qt3DRender import QGeometryRenderer
from PyQt6.QtCore import QByteArray, QPoint, Qt
from PyQt6.QtGui import QAction, QColor, QMouseEvent, QQuaternion, QVector3D, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QInputDialog,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

Point3 = tuple[float, float, float]
Segment3 = tuple[float, float, float, float, float, float]


@dataclass
class EntityGeometry:
    handle: str
    layer: str
    segments: list[Segment3]


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
        import struct

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
        self.save_dxf_button = QPushButton("Guardar DXF")
        self.save_dxf_button.clicked.connect(self.save_dxf_file)

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
        left_layout.addWidget(self.save_dxf_button)
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

    def _entity_segments(self, entity) -> list[Segment3]:
        entity_type = entity.dxftype()
        entity_segments: list[Segment3] = []

        if entity_type == "LINE":
            start = entity.dxf.start
            end = entity.dxf.end
            entity_segments.append((start.x, start.y, start.z, end.x, end.y, end.z))
            return entity_segments

        if entity_type == "LWPOLYLINE":
            elevation = float(getattr(entity.dxf, "elevation", 0.0))
            points = [(p[0], p[1], elevation) for p in entity.get_points("xy")]
            if len(points) < 2:
                return entity_segments

            for (x1, y1, z1), (x2, y2, z2) in zip(points, points[1:]):
                entity_segments.append((x1, y1, z1, x2, y2, z2))

            if entity.closed:
                x1, y1, z1 = points[-1]
                x2, y2, z2 = points[0]
                entity_segments.append((x1, y1, z1, x2, y2, z2))
            return entity_segments

        if entity_type == "POLYLINE":
            points = [
                (v.dxf.location.x, v.dxf.location.y, v.dxf.location.z) for v in entity.vertices
            ]
            if len(points) < 2:
                return entity_segments

            for (x1, y1, z1), (x2, y2, z2) in zip(points, points[1:]):
                entity_segments.append((x1, y1, z1, x2, y2, z2))

            if entity.is_closed:
                x1, y1, z1 = points[-1]
                x2, y2, z2 = points[0]
                entity_segments.append((x1, y1, z1, x2, y2, z2))
            return entity_segments

        return entity_segments

    def _collect_entity_geometries(self, msp) -> tuple[list[EntityGeometry], dict[str, int]]:
        entity_geometries: list[EntityGeometry] = []
        layer_counts: dict[str, int] = defaultdict(int)

        for index, entity in enumerate(msp):
            layer = entity.dxf.layer
            entity_segments = self._entity_segments(entity)
            if not entity_segments:
                continue

            handle = getattr(entity.dxf, "handle", None)
            handle_id = str(handle) if handle else f"NO_HANDLE_{index}"
            entity_geometries.append(
                EntityGeometry(
                    handle=handle_id,
                    layer=layer,
                    segments=entity_segments,
                )
            )
            layer_counts[layer] += len(entity_segments)

        return entity_geometries, layer_counts

    def _flatten_segments_for_viewer(
        self, entities: list[EntityGeometry]
    ) -> list[tuple[float, float, float, float, float, float, str]]:
        flat_segments: list[tuple[float, float, float, float, float, float, str]] = []
        for entity in entities:
            for x1, y1, z1, x2, y2, z2 in entity.segments:
                flat_segments.append((x1, y1, z1, x2, y2, z2, entity.layer))
        return flat_segments

    def _is_layer_match(self, layer_name: str, token: str) -> bool:
        return token.upper() in layer_name.upper()

    def _point_to_segment_distance(self, point: Point3, segment: Segment3) -> float:
        px, py, pz = point
        x1, y1, z1, x2, y2, z2 = segment
        vx, vy, vz = x2 - x1, y2 - y1, z2 - z1
        wx, wy, wz = px - x1, py - y1, pz - z1

        segment_len_sq = vx * vx + vy * vy + vz * vz
        if segment_len_sq <= 1e-12:
            dx, dy, dz = px - x1, py - y1, pz - z1
            return math.sqrt(dx * dx + dy * dy + dz * dz)

        t = (wx * vx + wy * vy + wz * vz) / segment_len_sq
        t = max(0.0, min(1.0, t))
        cx, cy, cz = x1 + t * vx, y1 + t * vy, z1 + t * vz
        dx, dy, dz = px - cx, py - cy, pz - cz
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _segments_are_connected(self, segment_a: Segment3, segment_b: Segment3, tol: float) -> bool:
        a_start: Point3 = (segment_a[0], segment_a[1], segment_a[2])
        a_end: Point3 = (segment_a[3], segment_a[4], segment_a[5])
        b_start: Point3 = (segment_b[0], segment_b[1], segment_b[2])
        b_end: Point3 = (segment_b[3], segment_b[4], segment_b[5])

        return (
            self._point_to_segment_distance(a_start, segment_b) <= tol
            or self._point_to_segment_distance(a_end, segment_b) <= tol
            or self._point_to_segment_distance(b_start, segment_a) <= tol
            or self._point_to_segment_distance(b_end, segment_a) <= tol
        )

    def _detect_montantes_to_durmientes(
        self, tolerance: float = 1.0
    ) -> tuple[dict[str, set[str]], dict[str, EntityGeometry]]:
        entity_lookup = {entity.handle: entity for entity in self.entity_geometries}
        montantes = [
            entity
            for entity in self.entity_geometries
            if self._is_layer_match(entity.layer, "MONTANTE")
        ]
        durmientes = [
            entity
            for entity in self.entity_geometries
            if self._is_layer_match(entity.layer, "DURMIENTE")
        ]

        connections: dict[str, set[str]] = {durmiente.handle: set() for durmiente in durmientes}
        for montante in montantes:
            for durmiente in durmientes:
                connected = False
                for montante_segment in montante.segments:
                    for durmiente_segment in durmiente.segments:
                        if self._segments_are_connected(
                            montante_segment, durmiente_segment, tolerance
                        ):
                            connections[durmiente.handle].add(montante.handle)
                            connected = True
                            break
                    if connected:
                        break

        return connections, entity_lookup

    def _layer_is_no_pasante(self, layer: str) -> bool:
        return "NO PASANTE" in layer.upper()

    def _entity_reference(self, entity: EntityGeometry) -> tuple[float, float, float]:
        points: list[Point3] = []
        for x1, y1, z1, x2, y2, z2 in entity.segments:
            points.append((x1, y1, z1))
            points.append((x2, y2, z2))
        if not points:
            return (0.0, 0.0, 0.0)

        x_avg = sum(p[0] for p in points) / len(points)
        y_avg = sum(p[1] for p in points) / len(points)
        z_avg = sum(p[2] for p in points) / len(points)
        return (x_avg, y_avg, z_avg)

    def _cluster_z_levels(self, values: list[float], tol_z: float) -> list[float]:
        if not values:
            return []
        ordered = sorted(values)
        groups: list[list[float]] = [[ordered[0]]]
        for value in ordered[1:]:
            if abs(value - groups[-1][-1]) <= tol_z:
                groups[-1].append(value)
            else:
                groups.append([value])
        return [sum(group) / len(group) for group in groups]

    def _find_connected_montantes(
        self, durmiente: EntityGeometry, montantes: list[EntityGeometry], tol_conn: float
    ) -> list[EntityGeometry]:
        result: list[EntityGeometry] = []
        for montante in montantes:
            connected = False
            for mont_seg in montante.segments:
                for dur_seg in durmiente.segments:
                    if self._segments_are_connected(mont_seg, dur_seg, tol_conn):
                        connected = True
                        break
                if connected:
                    break
            if connected:
                result.append(montante)
        return result

    def _move_montante_node_down(
        self, montante: EntityGeometry, durmiente: EntityGeometry, z_drop: float, tol_conn: float
    ) -> bool:
        durmiente_ref = self._entity_reference(durmiente)
        durmiente_z = durmiente_ref[2]
        endpoint_hits: list[Point3] = []

        for x1, y1, z1, x2, y2, z2 in montante.segments:
            p1: Point3 = (x1, y1, z1)
            p2: Point3 = (x2, y2, z2)
            if any(self._point_to_segment_distance(p1, dur_seg) <= tol_conn for dur_seg in durmiente.segments):
                endpoint_hits.append(p1)
            if any(self._point_to_segment_distance(p2, dur_seg) <= tol_conn for dur_seg in durmiente.segments):
                endpoint_hits.append(p2)

        if not endpoint_hits:
            return False

        # Use the node that best matches durmiente elevation.
        endpoint_hits.sort(key=lambda p: (abs(p[2] - durmiente_z), p[2]))
        old_point = endpoint_hits[0]
        new_point: Point3 = (old_point[0], old_point[1], old_point[2] - z_drop)

        updated = False
        new_segments: list[Segment3] = []
        point_tol = max(tol_conn, 1e-6)
        for x1, y1, z1, x2, y2, z2 in montante.segments:
            p1: Point3 = (x1, y1, z1)
            p2: Point3 = (x2, y2, z2)
            if math.dist(p1, old_point) <= point_tol:
                p1 = new_point
                updated = True
            if math.dist(p2, old_point) <= point_tol:
                p2 = new_point
                updated = True
            new_segments.append((p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]))

        if updated:
            montante.segments = new_segments
        return updated

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
        segments = self._flatten_segments_for_viewer(self.entity_geometries)
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

    def _entity_z_limits(self, entity: EntityGeometry) -> tuple[float, float]:
        z_values: list[float] = []
        for x1, y1, z1, x2, y2, z2 in entity.segments:
            z_values.append(z1)
            z_values.append(z2)
        if not z_values:
            return (0.0, 0.0)
        return (min(z_values), max(z_values))

    def _entity_xy_reference(self, entity: EntityGeometry) -> tuple[float, float]:
        points_count = 0
        sum_x = 0.0
        sum_y = 0.0
        for x1, y1, z1, x2, y2, z2 in entity.segments:
            sum_x += x1 + x2
            sum_y += y1 + y2
            points_count += 2
        if points_count == 0:
            return (0.0, 0.0)
        return (sum_x / points_count, sum_y / points_count)

    def _are_union_montantes_pairable(
        self,
        a: EntityGeometry,
        b: EntityGeometry,
        tol_union_horizontal: float,
        tol_z_equal: float,
        tol_shared_xy: float,
    ) -> bool:
        a_z_ini, a_z_fin = self._entity_z_limits(a)
        b_z_ini, b_z_fin = self._entity_z_limits(b)
        if abs(a_z_ini - b_z_ini) > tol_z_equal or abs(a_z_fin - b_z_fin) > tol_z_equal:
            return False

        a_x, a_y = self._entity_xy_reference(a)
        b_x, b_y = self._entity_xy_reference(b)
        dx = abs(a_x - b_x)
        dy = abs(a_y - b_y)
        numeric_eps = max(1e-6, tol_union_horizontal * 1e-4)

        share_x = dx <= (tol_shared_xy + numeric_eps) and dy <= (tol_union_horizontal + numeric_eps)
        share_y = dy <= (tol_shared_xy + numeric_eps) and dx <= (tol_union_horizontal + numeric_eps)
        return share_x or share_y

    def _build_union_groups(
        self,
        montantes_union: list[EntityGeometry],
        tol_union_horizontal: float,
        tol_z_equal: float,
        tol_shared_xy: float,
    ) -> list[list[EntityGeometry]]:
        if not montantes_union:
            return []

        parent: dict[int, int] = {i: i for i in range(len(montantes_union))}

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri = find(i)
            rj = find(j)
            if ri != rj:
                parent[rj] = ri

        for i in range(len(montantes_union)):
            for j in range(i + 1, len(montantes_union)):
                if self._are_union_montantes_pairable(
                    montantes_union[i],
                    montantes_union[j],
                    tol_union_horizontal,
                    tol_z_equal,
                    tol_shared_xy,
                ):
                    union(i, j)

        groups_idx: dict[int, list[int]] = defaultdict(list)
        for i in range(len(montantes_union)):
            groups_idx[find(i)].append(i)

        groups: list[list[EntityGeometry]] = []
        for idxs in groups_idx.values():
            if len(idxs) > 1:
                groups.append([montantes_union[i] for i in idxs])
        return groups

    def _build_corner_pair_groups(
        self,
        montantes_esquina: list[EntityGeometry],
        tol_horizontal: float,
        tol_z_equal: float,
    ) -> list[list[EntityGeometry]]:
        if len(montantes_esquina) < 2:
            return []

        candidates: list[tuple[float, int, int]] = []
        refs = [self._entity_xy_reference(e) for e in montantes_esquina]
        z_lims = [self._entity_z_limits(e) for e in montantes_esquina]

        for i in range(len(montantes_esquina)):
            for j in range(i + 1, len(montantes_esquina)):
                zi0, zf0 = z_lims[i]
                zi1, zf1 = z_lims[j]
                if abs(zi0 - zi1) > tol_z_equal or abs(zf0 - zf1) > tol_z_equal:
                    continue

                dx = abs(refs[i][0] - refs[j][0])
                dy = abs(refs[i][1] - refs[j][1])
                # Corner pairs are close in both horizontal axes.
                if dx <= tol_horizontal and dy <= tol_horizontal:
                    dist = math.hypot(dx, dy)
                    candidates.append((dist, i, j))

        candidates.sort(key=lambda x: x[0])
        used: set[int] = set()
        pair_groups: list[list[EntityGeometry]] = []
        for _, i, j in candidates:
            if i in used or j in used:
                continue
            used.add(i)
            used.add(j)
            pair_groups.append([montantes_esquina[i], montantes_esquina[j]])
        return pair_groups

    def _translate_entity_xy(self, entity: EntityGeometry, dx: float, dy: float) -> None:
        translated: list[Segment3] = []
        for x1, y1, z1, x2, y2, z2 in entity.segments:
            translated.append((x1 + dx, y1 + dy, z1, x2 + dx, y2 + dy, z2))
        entity.segments = translated

    def _is_montante_esquina_layer(self, layer: str) -> bool:
        upper = layer.upper()
        return "MONTANTE" in upper and "ESQUINA" in upper

    def _set_connector_nodes_to_xy(
        self,
        connector: EntityGeometry,
        reference_segments: list[Segment3],
        new_x: float,
        new_y: float,
        tol_conn: float,
    ) -> bool:
        changed = False
        updated_segments: list[Segment3] = []
        for x1, y1, z1, x2, y2, z2 in connector.segments:
            p1: Point3 = (x1, y1, z1)
            p2: Point3 = (x2, y2, z2)
            if any(self._point_to_segment_distance(p1, ref_seg) <= tol_conn for ref_seg in reference_segments):
                p1 = (new_x, new_y, z1)
                changed = True
            if any(self._point_to_segment_distance(p2, ref_seg) <= tol_conn for ref_seg in reference_segments):
                p2 = (new_x, new_y, z2)
                changed = True
            updated_segments.append((p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]))
        if changed:
            connector.segments = updated_segments
        return changed

    def _count_connector_hits(
        self,
        connector: EntityGeometry,
        reference_segments: list[Segment3],
        tol_conn: float,
    ) -> int:
        hits = 0
        for x1, y1, z1, x2, y2, z2 in connector.segments:
            p1: Point3 = (x1, y1, z1)
            p2: Point3 = (x2, y2, z2)
            if any(self._point_to_segment_distance(p1, ref_seg) <= tol_conn for ref_seg in reference_segments):
                hits += 1
            if any(self._point_to_segment_distance(p2, ref_seg) <= tol_conn for ref_seg in reference_segments):
                hits += 1
        return hits

    def _connector_dominant_axis(self, connector: EntityGeometry) -> str:
        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")
        for x1, y1, z1, x2, y2, z2 in connector.segments:
            min_x = min(min_x, x1, x2)
            max_x = max(max_x, x1, x2)
            min_y = min(min_y, y1, y2)
            max_y = max(max_y, y1, y2)

        if min_x == float("inf"):
            return "none"
        span_x = max_x - min_x
        span_y = max_y - min_y
        if span_x <= 1e-9 and span_y <= 1e-9:
            return "none"
        return "x" if span_x >= span_y else "y"

    def _count_locked_connectors_for_axis(
        self,
        corner_segments: list[Segment3],
        connectors: list[EntityGeometry],
        axis_to_move: str,
        tol_conn: float,
    ) -> tuple[int, int]:
        # locked: connector direction forbids moving this axis
        # if connector is dominant in X => Y must remain fixed
        # if connector is dominant in Y => X must remain fixed
        locked = 0
        total = 0
        for connector in connectors:
            hits = self._count_connector_hits(connector, corner_segments, tol_conn)
            if hits <= 0:
                continue
            total += 1
            dominant = self._connector_dominant_axis(connector)
            if axis_to_move == "y" and dominant == "x":
                locked += 1
            elif axis_to_move == "x" and dominant == "y":
                locked += 1
        return locked, total

    def _set_connector_nodes_axis(
        self,
        connector: EntityGeometry,
        reference_segments: list[Segment3],
        axis_to_move: str,
        target_x: float,
        target_y: float,
        tol_conn: float,
    ) -> bool:
        changed = False
        updated_segments: list[Segment3] = []
        dominant = self._connector_dominant_axis(connector)
        if dominant == "x":
            allowed_axis = "x"
        elif dominant == "y":
            allowed_axis = "y"
        else:
            allowed_axis = axis_to_move

        for x1, y1, z1, x2, y2, z2 in connector.segments:
            p1: Point3 = (x1, y1, z1)
            p2: Point3 = (x2, y2, z2)
            if any(self._point_to_segment_distance(p1, ref_seg) <= tol_conn for ref_seg in reference_segments):
                if allowed_axis == "x":
                    p1 = (target_x, p1[1], z1)
                elif allowed_axis == "y":
                    p1 = (p1[0], target_y, z1)
                else:
                    p1 = (target_x, target_y, z1)
                changed = True
            if any(self._point_to_segment_distance(p2, ref_seg) <= tol_conn for ref_seg in reference_segments):
                if allowed_axis == "x":
                    p2 = (target_x, p2[1], z2)
                elif allowed_axis == "y":
                    p2 = (p2[0], target_y, z2)
                else:
                    p2 = (target_x, target_y, z2)
                changed = True
            updated_segments.append((p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]))
        if changed:
            connector.segments = updated_segments
        return changed

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

        tol_z_equal = 1.0
        tol_shared_xy = 1.0
        tol_conn = 1.0

        montantes_union = [
            e
            for e in self.entity_geometries
            if self._is_layer_match(e.layer, "MONTANTES UNION")
        ]
        if not montantes_union:
            QMessageBox.information(
                self,
                "Unir montantes union",
                "No se encontraron elementos en capas MONTANTES UNION.",
            )
            return

        connector_candidates = [
            e
            for e in self.entity_geometries
            if (self._is_layer_match(e.layer, "DURMIENTE") or self._is_layer_match(e.layer, "TESTERO"))
        ]

        groups = self._build_union_groups(
            montantes_union,
            tol_union_horizontal,
            tol_z_equal,
            tol_shared_xy,
        )
        if not groups:
            QMessageBox.information(
                self,
                "Unir montantes union",
                "No se encontraron grupos para unir con la tolerancia dada.",
            )
            return

        handles_to_remove: set[str] = set()
        moved_connectors: set[str] = set()
        merged_groups = 0
        merged_elements = 0

        for group in groups:
            merged_groups += 1
            merged_elements += len(group)
            avg_x = sum(self._entity_xy_reference(e)[0] for e in group) / len(group)
            avg_y = sum(self._entity_xy_reference(e)[1] for e in group) / len(group)
            group_z_ini = min(self._entity_z_limits(e)[0] for e in group)
            group_z_fin = max(self._entity_z_limits(e)[1] for e in group)
            # Keep original geometry as reference, otherwise connectors attached to the
            # former anchor position may be missed after moving it to the average axis.
            original_group_segments: list[Segment3] = []
            for member in group:
                original_group_segments.extend(list(member.segments))

            anchor = group[0]
            anchor_x, anchor_y = self._entity_xy_reference(anchor)
            self._translate_entity_xy(anchor, avg_x - anchor_x, avg_y - anchor_y)

            ref_segments: list[Segment3] = []
            ref_segments.extend(original_group_segments)
            ref_segments.append((avg_x, avg_y, group_z_ini, avg_x, avg_y, group_z_fin))

            for connector in connector_candidates:
                if self._set_connector_nodes_to_xy(
                    connector,
                    ref_segments,
                    avg_x,
                    avg_y,
                    tol_conn,
                ):
                    moved_connectors.add(connector.handle)

            for member in group[1:]:
                handles_to_remove.add(member.handle)

        if handles_to_remove:
            self.entity_geometries = [
                e for e in self.entity_geometries if e.handle not in handles_to_remove
            ]

        self._refresh_from_entity_geometries()

        report_text = "\n".join(
            [
                f"Grupos unidos: {merged_groups}",
                f"Elementos de montantes union involucrados: {merged_elements}",
                f"Elementos de montantes union eliminados: {len(handles_to_remove)}",
                f"Durmientes/testeros modificados: {len(moved_connectors)}",
                f"Tolerancia horizontal usada: {tol_union_horizontal}",
            ]
        )
        ResultTextDialog(
            title="Resultado: unir montantes union",
            text=report_text,
            parent=self,
        ).exec()

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

        tol_z_equal = 1.0
        tol_conn = 1.0

        montantes_esquina = [
            e for e in self.entity_geometries if self._is_montante_esquina_layer(e.layer)
        ]
        if not montantes_esquina:
            QMessageBox.information(
                self,
                "Unir montantes esquinas",
                "No se encontraron elementos en capas MONTANTES ESQUINA.",
            )
            return

        corners_pasante = [
            e for e in montantes_esquina if not self._layer_is_no_pasante(e.layer)
        ]
        corners_no_pasante = [
            e for e in montantes_esquina if self._layer_is_no_pasante(e.layer)
        ]

        connector_candidates = [
            e
            for e in self.entity_geometries
            if (
                self._is_layer_match(e.layer, "DURMIENTE")
                or self._is_layer_match(e.layer, "TESTERO")
                or self._is_layer_match(e.layer, "DINTEL")
                or self._is_layer_match(e.layer, "ALFEIZAR")
                or self._is_layer_match(e.layer, "COTA DE NIVEL")
            )
        ]

        all_groups: list[list[EntityGeometry]] = []
        all_groups.extend(
            self._build_corner_pair_groups(
                corners_pasante,
                tol_union_horizontal,
                tol_z_equal,
            )
        )
        all_groups.extend(
            self._build_corner_pair_groups(
                corners_no_pasante,
                tol_union_horizontal,
                tol_z_equal,
            )
        )

        if not all_groups:
            QMessageBox.information(
                self,
                "Unir montantes esquinas",
                "No se encontraron grupos para unir con la tolerancia dada.",
            )
            return

        handles_to_remove: set[str] = set()
        moved_connectors: set[str] = set()
        merged_groups = 0
        merged_elements = 0
        merged_groups_pasante = 0
        merged_groups_no_pasante = 0

        for group in all_groups:
            merged_groups += 1
            merged_elements += len(group)
            if self._layer_is_no_pasante(group[0].layer):
                merged_groups_no_pasante += 1
            else:
                merged_groups_pasante += 1

            if len(group) < 2:
                continue

            first = group[0]
            second = group[1]
            fx, fy = self._entity_xy_reference(first)
            sx, sy = self._entity_xy_reference(second)
            dx = abs(fx - sx)
            dy = abs(fy - sy)

            if dx <= dy:
                axis_to_move = "y"
            else:
                axis_to_move = "x"

            first_segments = list(first.segments)
            second_segments = list(second.segments)
            first_hits = 0
            second_hits = 0
            for connector in connector_candidates:
                first_hits += self._count_connector_hits(connector, first_segments, tol_conn)
                second_hits += self._count_connector_hits(connector, second_segments, tol_conn)
            first_locked, first_total = self._count_locked_connectors_for_axis(
                first_segments,
                connector_candidates,
                axis_to_move,
                tol_conn,
            )
            second_locked, second_total = self._count_locked_connectors_for_axis(
                second_segments,
                connector_candidates,
                axis_to_move,
                tol_conn,
            )

            if first_locked > second_locked:
                anchor = first
                mover = second
            elif second_locked > first_locked:
                anchor = second
                mover = first
            elif first_hits > second_hits:
                anchor = first
                mover = second
            elif second_hits > first_hits:
                anchor = second
                mover = first
            else:
                if first_total > second_total:
                    anchor = first
                    mover = second
                elif second_total > first_total:
                    anchor = second
                    mover = first
                else:
                    if axis_to_move == "y":
                        anchor = first if fy >= sy else second
                    else:
                        anchor = first if fx >= sx else second
                    mover = second if anchor is first else first

            anchor_x, anchor_y = self._entity_xy_reference(anchor)
            mover_x, mover_y = self._entity_xy_reference(mover)
            target_x = anchor_x
            target_y = anchor_y
            group_z_ini = min(self._entity_z_limits(e)[0] for e in group)
            group_z_fin = max(self._entity_z_limits(e)[1] for e in group)

            original_group_segments: list[Segment3] = []
            for member in group:
                original_group_segments.extend(list(member.segments))

            if axis_to_move == "y":
                self._translate_entity_xy(mover, 0.0, target_y - mover_y)
            else:
                self._translate_entity_xy(mover, target_x - mover_x, 0.0)

            ref_segments: list[Segment3] = []
            ref_segments.extend(original_group_segments)
            ref_segments.append((target_x, target_y, group_z_ini, target_x, target_y, group_z_fin))

            for connector in connector_candidates:
                if self._set_connector_nodes_axis(
                    connector,
                    ref_segments,
                    axis_to_move,
                    target_x,
                    target_y,
                    tol_conn,
                ):
                    moved_connectors.add(connector.handle)

            handles_to_remove.add(mover.handle)

        if handles_to_remove:
            self.entity_geometries = [
                e for e in self.entity_geometries if e.handle not in handles_to_remove
            ]

        self._refresh_from_entity_geometries()

        report_text = "\n".join(
            [
                f"Grupos unidos: {merged_groups}",
                f"Grupos PASANTE: {merged_groups_pasante}",
                f"Grupos NO PASANTE: {merged_groups_no_pasante}",
                f"Elementos de montantes esquina involucrados: {merged_elements}",
                f"Elementos de montantes esquina eliminados: {len(handles_to_remove)}",
                f"Durmientes/testeros modificados: {len(moved_connectors)}",
                f"Tolerancia horizontal usada: {tol_union_horizontal}",
            ]
        )
        ResultTextDialog(
            title="Resultado: unir montantes esquinas",
            text=report_text,
            parent=self,
        ).exec()

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

        tol_xy = 1.0
        tol_z = 1.0
        tol_conn = 1.0

        durmientes = [
            e
            for e in self.entity_geometries
            if self._is_layer_match(e.layer, "DURMIENTE") and not self._layer_is_no_pasante(e.layer)
        ]
        testeros = [
            e
            for e in self.entity_geometries
            if self._is_layer_match(e.layer, "TESTERO") and not self._layer_is_no_pasante(e.layer)
        ]
        montantes = [
            e
            for e in self.entity_geometries
            if self._is_layer_match(e.layer, "MONTANTE") and not self._layer_is_no_pasante(e.layer)
        ]

        if not durmientes or not testeros:
            QMessageBox.information(
                self,
                "Unir durmientes y testeros",
                "No hay suficientes capas PASANTE de durmiente/testero para procesar.",
            )
            return

        testero_levels = self._cluster_z_levels(
            [self._entity_reference(testero)[2] for testero in testeros],
            tol_z,
        )

        dur_refs = {d.handle: self._entity_reference(d) for d in durmientes}
        tes_refs = {t.handle: self._entity_reference(t) for t in testeros}
        base_z = min(ref[2] for ref in dur_refs.values())

        removed_handles: set[str] = set()
        moved_montantes: set[str] = set()
        unresolved: list[str] = []
        matches_count = 0

        for durmiente in durmientes:
            dx, dy, dz = dur_refs[durmiente.handle]
            if abs(dz - base_z) <= tol_z:
                continue

            best_testero: EntityGeometry | None = None
            best_score = float("inf")
            for testero in testeros:
                tx, ty, tz = tes_refs[testero.handle]
                xy_dist = math.hypot(dx - tx, dy - ty)
                z_delta = dz - tz
                if xy_dist > tol_xy:
                    continue
                if tz >= dz:
                    continue
                if abs(z_delta - z_drop) > tol_z:
                    continue

                score = xy_dist + abs(z_delta - z_drop) * 0.1
                if score < best_score:
                    best_score = score
                    best_testero = testero

            if best_testero is None:
                unresolved.append(f"Durmiente [{durmiente.handle}] sin testero valido")
                continue

            connected_montantes = self._find_connected_montantes(durmiente, montantes, tol_conn)
            moved_this = 0
            for montante in connected_montantes:
                if self._move_montante_node_down(montante, durmiente, z_drop, tol_conn):
                    moved_montantes.add(montante.handle)
                    moved_this += 1

            removed_handles.add(durmiente.handle)
            matches_count += 1
            if moved_this == 0:
                unresolved.append(
                    f"Durmiente [{durmiente.handle}] pareado con testero [{best_testero.handle}] sin montantes movidos"
                )

        if removed_handles:
            self.entity_geometries = [
                e for e in self.entity_geometries if e.handle not in removed_handles
            ]
            self._refresh_from_entity_geometries()

        report_lines = [
            f"Niveles detectados por testero: {len(testero_levels)}",
            f"Durmientes pareados: {matches_count}",
            f"Durmientes eliminados: {len(removed_handles)}",
            f"Montantes modificados: {len(moved_montantes)}",
        ]
        if unresolved:
            report_lines.append("")
            report_lines.append("Observaciones:")
            report_lines.extend(f"- {line}" for line in unresolved[:30])

        ResultTextDialog(
            title="Resultado: unir durmientes y testeros",
            text="\n".join(report_lines),
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

            entity_geometries, layer_counts = self._collect_entity_geometries(msp)
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
            segments = self._flatten_segments_for_viewer(entity_geometries)
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

    def detect_montantes_durmientes(self) -> None:
        if not self.entity_geometries:
            QMessageBox.information(
                self,
                "Detectar montantes-durmientes",
                "Carga primero un archivo DXF con geometria.",
            )
            return

        tolerance = 1.0
        connections, entity_lookup = self._detect_montantes_to_durmientes(tolerance)
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


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
