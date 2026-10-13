import math
from collections import defaultdict

from src.core.geometry import EntityGeometry, Point3, Segment3, point_to_segment_distance, segments_are_connected

def point_to_segment_distance_xy(px, py, x1, y1, x2, y2):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    segment_len_sq = vx * vx + vy * vy
    if segment_len_sq <= 1e-12:
        return math.hypot(px - x1, py - y1)
    t = (wx * vx + wy * vy) / segment_len_sq
    t = max(0.0, min(1.0, t))
    cx, cy = x1 + t * vx, y1 + t * vy
    return math.hypot(px - cx, py - cy)

def point_to_line_distance_2d(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    vx, vy = x2 - x1, y2 - y1
    segment_len = math.hypot(vx, vy)
    if segment_len <= 1e-12:
        return math.hypot(px - x1, py - y1)
    return abs(vx * (y1 - py) - (x1 - px) * vy) / segment_len

def project_point_onto_line_2d(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    vx, vy = x2 - x1, y2 - y1
    segment_len_sq = vx * vx + vy * vy
    if segment_len_sq <= 1e-12:
        return (x1, y1)
    t = ((px - x1) * vx + (py - y1) * vy) / segment_len_sq
    return (x1 + t * vx, y1 + t * vy)

class StructuralProcessor:
    def __init__(self, geometries: list[EntityGeometry]):
        self.entity_geometries = geometries

    def get_geometries(self) -> list[EntityGeometry]:
        return self.entity_geometries

    def fuse_level_lines(self, tol_snap: float = 300.0) -> tuple[bool, str]:
        cota_entities = [e for e in self.entity_geometries if "COTA" in e.layer.upper()]
        other_entities = [e for e in self.entity_geometries if e not in cota_entities]

        if not cota_entities:
            return False, "No se encontraron líneas de cota de nivel para unir."

        all_zs = [e.segments[0][2] for e in cota_entities if e.segments]
        z_clusters = self._cluster_z_levels(all_zs, tol_z=150.0)

        def get_clustered_z(z: float) -> float:
            for cz in z_clusters:
                if abs(z - cz) <= 150.0:
                    return cz
            return round(z / 50.0) * 50.0

        floor_layer_groups = defaultdict(list)
        for e in cota_entities:
            if not e.segments:
                continue
            seg = e.segments[0]
            z_floor = get_clustered_z(seg[2])
            floor_layer_groups[(z_floor, e.layer)].append(e)

        fused_cotas = []

        def line_intersection_2d(p1, p2, p3, p4):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-9:
                return None
            num_x = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
            num_y = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
            return (num_x / denom, num_y / denom)

        for (z_floor, layer_name), group in floor_layer_groups.items():
            x_lines = []
            y_lines = []
            others = []

            for e in group:
                seg = e.segments[0]
                x1, y1, z1, x2, y2, z2 = seg
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dy <= 10.0:
                    y_val = (y1 + y2) / 2.0
                    min_x, max_x = min(x1, x2), max(x1, x2)
                    x_lines.append((y_val, min_x, max_x, z1))
                elif dx <= 10.0:
                    x_val = (x1 + x2) / 2.0
                    min_y, max_y = min(y1, y2), max(y1, y2)
                    y_lines.append((x_val, min_y, max_y, z1))
                else:
                    others.append(e)

            y_grouped = defaultdict(list)
            for y_val, min_x, max_x, z in x_lines:
                y_key = round(y_val / 50.0) * 50.0
                y_grouped[y_key].append((y_val, min_x, max_x, z))

            floor_merged_segments = []

            for y_key, intervals in y_grouped.items():
                avg_y = sum(item[0] for item in intervals) / len(intervals)
                intervals.sort(key=lambda x: x[1])
                merged = []
                for y_val, cur_min, cur_max, z in intervals:
                    if not merged:
                        merged.append([cur_min, cur_max, z])
                    else:
                        prev = merged[-1]
                        if cur_min <= prev[1] + tol_snap:
                            prev[1] = max(prev[1], cur_max)
                        else:
                            merged.append([cur_min, cur_max, z])
                for min_x, max_x, z in merged:
                    floor_merged_segments.append([min_x, avg_y, z, max_x, avg_y, z])

            x_grouped = defaultdict(list)
            for x_val, min_y, max_y, z in y_lines:
                x_key = round(x_val / 50.0) * 50.0
                x_grouped[x_key].append((x_val, min_y, max_y, z))

            for x_key, intervals in x_grouped.items():
                avg_x = sum(item[0] for item in intervals) / len(intervals)
                intervals.sort(key=lambda x: x[1])
                merged = []
                for x_val, cur_min, cur_max, z in intervals:
                    if not merged:
                        merged.append([cur_min, cur_max, z])
                    else:
                        prev = merged[-1]
                        if cur_min <= prev[1] + tol_snap:
                            prev[1] = max(prev[1], cur_max)
                        else:
                            merged.append([cur_min, cur_max, z])
                for min_y, max_y, z in merged:
                    floor_merged_segments.append([avg_x, min_y, z, avg_x, max_y, z])

            for i in range(len(floor_merged_segments)):
                ax1, ay1, az1, ax2, ay2, az2 = floor_merged_segments[i]
                for j in range(i + 1, len(floor_merged_segments)):
                    bx1, by1, bz1, bx2, by2, bz2 = floor_merged_segments[j]

                    endpoints = [
                        ((ax1, ay1), (bx1, by1), 'start_a', 'start_b'),
                        ((ax1, ay1), (bx2, by2), 'start_a', 'end_b'),
                        ((ax2, ay2), (bx1, by1), 'end_a', 'start_b'),
                        ((ax2, ay2), (bx2, by2), 'end_a', 'end_b'),
                    ]
                    for pt_a, pt_b, label_a, label_b in endpoints:
                        dist = math.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])
                        if dist <= tol_snap:
                            v_a = (ax2 - ax1, ay2 - ay1)
                            v_b = (bx2 - bx1, by2 - by1)
                            len_a = math.hypot(*v_a)
                            len_b = math.hypot(*v_b)
                            if len_a > 0 and len_b > 0:
                                dot = (v_a[0] * v_b[0] + v_a[1] * v_b[1]) / (len_a * len_b)
                                angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
                                if angle_deg <= 15.0 or angle_deg >= 165.0:
                                    snap_pt = ((pt_a[0] + pt_b[0]) / 2.0, (pt_a[1] + pt_b[1]) / 2.0)
                                else:
                                    snap_pt = line_intersection_2d((ax1, ay1), (ax2, ay2), (bx1, by1), (bx2, by2))
                                    if not snap_pt or math.hypot(snap_pt[0] - pt_a[0], snap_pt[1] - pt_a[1]) > 500.0:
                                        snap_pt = ((pt_a[0] + pt_b[0]) / 2.0, (pt_a[1] + pt_b[1]) / 2.0)

                                if snap_pt:
                                    if label_a == 'start_a':
                                        ax1, ay1 = snap_pt
                                    else:
                                        ax2, ay2 = snap_pt
                                    floor_merged_segments[i] = [ax1, ay1, az1, ax2, ay2, az2]

                                    if label_b == 'start_b':
                                        bx1, by1 = snap_pt
                                    else:
                                        bx2, by2 = snap_pt
                                    floor_merged_segments[j] = [bx1, by1, bz1, bx2, by2, bz2]
                            break

            # Clamp level line endpoints so they never protrude past the outermost montantes on this floor wall line
            montantes = [m for m in other_entities if "MONTANTE" in m.layer.upper()]
            floor_montantes = [
                m for m in montantes
                if m.segments and (min(m.segments[0][2], m.segments[0][5]) - 250.0 <= z_floor <= max(m.segments[0][2], m.segments[0][5]) + 250.0)
            ]

            for k in range(len(floor_merged_segments)):
                seg = floor_merged_segments[k]
                sx1, sy1, sz1, sx2, sy2, sz2 = seg
                sdx = abs(sx2 - sx1)
                sdy = abs(sy2 - sy1)
                s_is_x = sdx >= sdy

                wall_m_projs = []
                for m in floor_montantes:
                    mseg = m.segments[0]
                    mx = (mseg[0] + mseg[3]) / 2.0
                    my = (mseg[1] + mseg[4]) / 2.0
                    if s_is_x:
                        if abs(my - (sy1 + sy2) / 2.0) <= 150.0:
                            if min(sx1, sx2) - 300.0 <= mx <= max(sx1, sx2) + 300.0:
                                wall_m_projs.append(mx)
                    else:
                        if abs(mx - (sx1 + sx2) / 2.0) <= 150.0:
                            if min(sy1, sy2) - 300.0 <= my <= max(sy1, sy2) + 300.0:
                                wall_m_projs.append(my)

                if wall_m_projs:
                    min_m = min(wall_m_projs)
                    max_m = max(wall_m_projs)
                    if s_is_x:
                        nx1 = max(min_m, min(max_m, sx1))
                        nx2 = max(min_m, min(max_m, sx2))
                        floor_merged_segments[k] = [nx1, sy1, sz1, nx2, sy2, sz2]
                    else:
                        ny1 = max(min_m, min(max_m, sy1))
                        ny2 = max(min_m, min(max_m, sy2))
                        floor_merged_segments[k] = [sx1, ny1, sz1, sx2, ny2, sz2]

            for idx, seg in enumerate(floor_merged_segments):
                if math.hypot(seg[3] - seg[0], seg[4] - seg[1]) <= 1.0:
                    continue
                fused_cotas.append(
                    EntityGeometry(
                        handle=f"COTA_FUSED_{z_floor:.0f}_{idx}",
                        layer=layer_name,
                        segments=[(seg[0], seg[1], seg[2], seg[3], seg[4], seg[5])],
                    )
                )

        initial_count = len(cota_entities)
        self.entity_geometries = other_entities + fused_cotas
        final_count = len(fused_cotas)

        report = f"Unión de líneas de cota de nivel (sobrecarga) finalizada:\n  - Tramos de línea originales: {initial_count}\n  - Líneas de cota continuas unificadas: {final_count}"
        return True, report

    def generate_level_lines(
        self, tol_cota_nivel: float = 1125.0, tol_snap: float = 300.0
    ) -> tuple[bool, str]:
        # 1. Clear existing level line entities
        self.entity_geometries = [
            e for e in self.entity_geometries if "COTA" not in e.layer.upper()
        ]

        durmientes = [
            e for e in self.entity_geometries if self._is_layer_match(e.layer, "DURMIENTE")
        ]
        testeros = [
            e for e in self.entity_geometries if self._is_layer_match(e.layer, "TESTERO")
        ]
        montantes = [
            e for e in self.entity_geometries if self._is_layer_match(e.layer, "MONTANTE")
        ]

        if not durmientes and not testeros:
            return False, "No se encontraron placas DURMIENTE o TESTERO para generar líneas de sobrecarga."

        cota_entities: list[EntityGeometry] = []

        for d in durmientes:
            if not d.segments:
                continue
            x1, y1, z1, x2, y2, z2 = d.segments[0]
            z_base = min(z1, z2)
            z_cota = z_base + tol_cota_nivel
            cota_entities.append(
                EntityGeometry(
                    handle=f"COTA_GEN_{d.handle}",
                    layer="Linea Cota de Nivel",
                    segments=[(x1, y1, z_cota, x2, y2, z_cota)],
                )
            )

        # Collect local durmiente geometry info to avoid generating duplicate level lines from testeros
        # ONLY when a durmiente is locally present at the exact same wall segment and floor height.
        durmiente_info = []
        for d in durmientes:
            if d.segments:
                dx1, dy1, dz1, dx2, dy2, dz2 = d.segments[0]
                dz_min = min(dz1, dz2)
                durmiente_info.append((dx1, dy1, dx2, dy2, dz_min))

        # 2. Level lines ONLY for PASANTE TESTERO plates that serve as base plates for upper floor Pasante walls
        # (No Pasante panels never merge their durmientes into testeros and always retain their DURMIENTE)
        pasante_testeros = [t for t in testeros if not self._layer_is_no_pasante(t.layer)]
        pasante_montantes = [m for m in montantes if not self._layer_is_no_pasante(m.layer)]

        for t in pasante_testeros:
            if not t.segments:
                continue
            tx1, ty1, tz1, tx2, ty2, tz2 = t.segments[0]
            tz_base = min(tz1, tz2)
            tx_mid = (tx1 + tx2) / 2.0
            ty_mid = (ty1 + ty2) / 2.0

            # Skip testero ONLY if there is a local durmiente at the exact same wall location (<= 200mm XY) and floor height (<= 350mm Z)
            has_local_durmiente = False
            for dx1, dy1, dx2, dy2, dz in durmiente_info:
                if abs(dz - tz_base) <= 350.0:
                    dist_xy = point_to_segment_distance_xy(tx_mid, ty_mid, dx1, dy1, dx2, dy2)
                    if dist_xy <= 200.0:
                        has_local_durmiente = True
                        break

            if has_local_durmiente:
                continue

            has_montantes_above = False
            for m in pasante_montantes:
                if not m.segments:
                    continue
                mx1, my1, mz1, mx2, my2, mz2 = m.segments[0]
                mz_min = min(mz1, mz2)
                mz_max = max(mz1, mz2)
                if (tz_base - 100.0 <= mz_min <= tz_base + 350.0) and (mz_max >= tz_base + 1000.0):
                    mx_mid = (mx1 + mx2) / 2.0
                    my_mid = (my1 + my2) / 2.0
                    dist_xy = point_to_segment_distance_xy(mx_mid, my_mid, tx1, ty1, tx2, ty2)
                    if dist_xy <= 200.0:
                        has_montantes_above = True
                        break

            if has_montantes_above:
                z_cota = tz_base + tol_cota_nivel
                cota_entities.append(
                    EntityGeometry(
                        handle=f"COTA_GEN_{t.handle}",
                        layer="Linea Cota de Nivel",
                        segments=[(tx1, ty1, z_cota, tx2, ty2, z_cota)],
                    )
                )

        self.entity_geometries.extend(cota_entities)

        success, report = self.fuse_level_lines(tol_snap=tol_snap)
        if not success:
            return False, report

        report_lines = [
            "Generación y fusión de líneas de sobrecarga finalizada:",
            f"  - Tramos generados de placas: {len(cota_entities)}",
            f"  - {report.splitlines()[1] if len(report.splitlines()) > 1 else report}",
        ]
        return True, "\n".join(report_lines)

    def _is_layer_match(self, layer_name: str, token: str) -> bool:
        return token.upper() in layer_name.upper()

    def _layer_is_no_pasante(self, layer: str) -> bool:
        return "NO PASANTE" in layer.upper()

    def _is_montante_esquina_layer(self, layer: str) -> bool:
        upper = layer.upper()
        return "MONTANTE" in upper and "ESQUINA" in upper

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

    def _entity_z_limits(self, entity: EntityGeometry) -> tuple[float, float]:
        z_values: list[float] = []
        for x1, y1, z1, x2, y2, z2 in entity.segments:
            z_values.append(z1)
            z_values.append(z2)
        if not z_values:
            return (0.0, 0.0)
        return (min(z_values), max(z_values))

    def _translate_entity_xy(self, entity: EntityGeometry, dx: float, dy: float) -> None:
        translated: list[Segment3] = []
        for x1, y1, z1, x2, y2, z2 in entity.segments:
            translated.append((x1 + dx, y1 + dy, z1, x2 + dx, y2 + dy, z2))
        entity.segments = translated

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
                    if segments_are_connected(mont_seg, dur_seg, tol_conn):
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
            if any(point_to_segment_distance(p1, dur_seg) <= tol_conn for dur_seg in durmiente.segments):
                endpoint_hits.append(p1)
            if any(point_to_segment_distance(p2, dur_seg) <= tol_conn for dur_seg in durmiente.segments):
                endpoint_hits.append(p2)

        if not endpoint_hits:
            return False

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

    def detect_montantes_to_durmientes(
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
                        if segments_are_connected(montante_segment, durmiente_segment, tolerance):
                            connections[durmiente.handle].add(montante.handle)
                            connected = True
                            break
                    if connected:
                        break

        return connections, entity_lookup

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
            if any(point_to_segment_distance(p1, ref_seg) <= tol_conn for ref_seg in reference_segments):
                p1 = (new_x, new_y, z1)
                changed = True
            if any(point_to_segment_distance(p2, ref_seg) <= tol_conn for ref_seg in reference_segments):
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
            if any(point_to_segment_distance(p1, ref_seg) <= tol_conn for ref_seg in reference_segments):
                hits += 1
            if any(point_to_segment_distance(p2, ref_seg) <= tol_conn for ref_seg in reference_segments):
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
            if any(point_to_segment_distance(p1, ref_seg) <= tol_conn for ref_seg in reference_segments):
                if allowed_axis == "x":
                    p1 = (target_x, p1[1], z1)
                elif allowed_axis == "y":
                    p1 = (p1[0], target_y, z1)
                else:
                    p1 = (target_x, target_y, z1)
                changed = True
            if any(point_to_segment_distance(p2, ref_seg) <= tol_conn for ref_seg in reference_segments):
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

    def execute_merge_montantes_union(self, tol_union_horizontal: float) -> tuple[bool, str]:
        tol_z_equal = 1.0
        tol_shared_xy = 1.0
        tol_conn = 1.0

        montantes_union = [
            e
            for e in self.entity_geometries
            if self._is_layer_match(e.layer, "MONTANTES UNION")
        ]
        if not montantes_union:
            return False, "No se encontraron elementos en capas MONTANTES UNION."

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
            return False, "No se encontraron grupos para unir con la tolerancia dada."

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

        report_text = "\n".join(
            [
                f"Grupos unidos: {merged_groups}",
                f"Elementos de montantes union involucrados: {merged_elements}",
                f"Elementos de montantes union eliminados: {len(handles_to_remove)}",
                f"Durmientes/testeros modificados: {len(moved_connectors)}",
                f"Tolerancia horizontal usada: {tol_union_horizontal}",
            ]
        )
        return True, report_text

    def execute_unir_dobles_montantes(self, distance_threshold: float = 90.0) -> tuple[bool, str]:
        montantes_hueco = [
            e
            for e in self.entity_geometries
            if "HUECO" in e.layer.upper() and "MONTANTE" in e.layer.upper()
        ]
        if not montantes_hueco:
            return False, "No se encontraron elementos en capas de Montante Hueco."

        montantes_simples = [
            e
            for e in self.entity_geometries
            if "MONTANTE" in e.layer.upper()
            and "SIMPLE" in e.layer.upper()
            and "HUECO" not in e.layer.upper()
            and "UNION" not in e.layer.upper()
            and "ESQUINA" not in e.layer.upper()
        ]
        if not montantes_simples:
            return False, "No se encontraron elementos en capas de Montante Simple para evaluar."

        handles_to_remove: set[str] = set()
        matched_hueco_count = 0

        for h in montantes_hueco:
            hx, hy = self._entity_xy_reference(h)
            hz_min, hz_max = self._entity_z_limits(h)
            h_matched = False

            for s in montantes_simples:
                if s.handle in handles_to_remove:
                    continue

                sz_min, sz_max = self._entity_z_limits(s)
                if abs(hz_min - sz_min) > 300.0 and abs(hz_max - sz_max) > 300.0:
                    continue

                sx, sy = self._entity_xy_reference(s)
                dist_xy = math.hypot(hx - sx, hy - sy)

                if dist_xy <= distance_threshold:
                    handles_to_remove.add(s.handle)
                    h_matched = True

            if h_matched:
                matched_hueco_count += 1

        if not handles_to_remove:
            return False, f"No se encontraron montantes simples a menos de {distance_threshold} mm de ningún montante hueco."

        self.entity_geometries = [
            e for e in self.entity_geometries if e.handle not in handles_to_remove
        ]

        report_lines = [
            "Unión de dobles montantes finalizada:",
            f"  - Montantes huecos analizados: {len(montantes_hueco)}",
            f"  - Montantes huecos con doble montante simple detectado: {matched_hueco_count}",
            f"  - Montantes simples eliminados: {len(handles_to_remove)}",
            f"  - Distancia límite usada: {distance_threshold} mm",
        ]
        return True, "\n".join(report_lines)

    def execute_round_coordinates(self, decimals: int = 0) -> tuple[bool, str]:
        if not self.entity_geometries:
            return False, "No hay geometrías para redondear."

        modified_entities = 0
        total_segments = 0

        for entity in self.entity_geometries:
            new_segments = []
            entity_modified = False
            for x1, y1, z1, x2, y2, z2 in entity.segments:
                total_segments += 1
                if decimals == 0:
                    rx1, ry1, rz1 = float(round(x1)), float(round(y1)), float(round(z1))
                    rx2, ry2, rz2 = float(round(x2)), float(round(y2)), float(round(z2))
                else:
                    rx1, ry1, rz1 = round(x1, decimals), round(y1, decimals), round(z1, decimals)
                    rx2, ry2, rz2 = round(x2, decimals), round(y2, decimals), round(z2, decimals)

                if (rx1, ry1, rz1, rx2, ry2, rz2) != (x1, y1, z1, x2, y2, z2):
                    entity_modified = True

                new_segments.append((rx1, ry1, rz1, rx2, ry2, rz2))

            entity.segments = new_segments
            if entity_modified:
                modified_entities += 1

        report_lines = [
            "Redondeo de coordenadas finalizado:",
            f"  - Total de entidades procesadas: {len(self.entity_geometries)}",
            f"  - Entidades modificadas: {modified_entities}",
            f"  - Total de segmentos redondeados: {total_segments}",
            f"  - Decimales aplicados: {decimals}",
        ]
        return True, "\n".join(report_lines)

    def execute_merge_montantes_esquinas(self, tol_union_horizontal: float) -> tuple[bool, str]:
        tol_z_equal = 1.0
        tol_conn = 1.0

        montantes_esquina = [
            e for e in self.entity_geometries if self._is_montante_esquina_layer(e.layer)
        ]
        if not montantes_esquina:
            return False, "No se encontraron elementos en capas MONTANTES ESQUINA."

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
            return False, "No se encontraron grupos para unir con la tolerancia dada."

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
        return True, report_text

    def execute_merge_durmientes_testeros(self, z_drop: float) -> tuple[bool, str]:
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
            return False, "No hay suficientes capas PASANTE de durmiente/testero para procesar."

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
            best_z_delta = 0.0
            for testero in testeros:
                tx, ty, tz = tes_refs[testero.handle]
                xy_dist = math.hypot(dx - tx, dy - ty)
                z_delta = dz - tz
                if xy_dist > tol_xy:
                    continue
                if tz >= dz or z_delta <= 0:
                    continue
                if z_delta > (z_drop + tol_z):
                    continue

                score = xy_dist + z_delta * 0.1
                if score < best_score:
                    best_score = score
                    best_testero = testero
                    best_z_delta = z_delta

            if best_testero is None:
                unresolved.append(f"Durmiente [{durmiente.handle}] sin testero valido")
                continue

            connected_montantes = self._find_connected_montantes(durmiente, montantes, tol_conn)
            moved_this = 0
            for montante in connected_montantes:
                if self._move_montante_node_down(montante, durmiente, best_z_delta, tol_conn):
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

        return True, "\n".join(report_lines)

    def execute_merge_durmientes_testeros_contrapeados(
        self, z_drop: float, tol_xy: float = 150.0
    ) -> tuple[bool, str]:
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
            return False, "No hay suficientes capas PASANTE de durmiente/testero para procesar."

        dur_refs = {d.handle: self._entity_reference(d) for d in durmientes}
        base_z = min(ref[2] for ref in dur_refs.values())

        removed_handles: set[str] = set()
        moved_montantes: set[str] = set()
        unresolved: list[str] = []
        matches_count = 0

        for durmiente in durmientes:
            dx, dy, dz = dur_refs[durmiente.handle]
            if abs(dz - base_z) <= tol_z:
                continue

            connected_montantes = self._find_connected_montantes(durmiente, montantes, tol_conn)
            if not connected_montantes:
                continue

            moved_for_this_durmiente = 0
            for montante in connected_montantes:
                durmiente_z = dz
                endpoint_hits: list[Point3] = []
                for x1, y1, z1, x2, y2, z2 in montante.segments:
                    p1: Point3 = (x1, y1, z1)
                    p2: Point3 = (x2, y2, z2)
                    if any(point_to_segment_distance(p1, dur_seg) <= tol_conn for dur_seg in durmiente.segments):
                        endpoint_hits.append(p1)
                    if any(point_to_segment_distance(p2, dur_seg) <= tol_conn for dur_seg in durmiente.segments):
                        endpoint_hits.append(p2)

                if not endpoint_hits:
                    continue

                endpoint_hits.sort(key=lambda p: (abs(p[2] - durmiente_z), p[2]))
                old_point = endpoint_hits[0]
                m_x, m_y, m_z = old_point

                best_testero: EntityGeometry | None = None
                best_z_delta = 0.0
                best_score = float("inf")

                for testero in testeros:
                    for tx1, ty1, tz1, tx2, ty2, tz2 in testero.segments:
                        tz_mid = (tz1 + tz2) / 2.0
                        z_delta = m_z - tz_mid
                        if z_delta <= 0 or z_delta > (z_drop + tol_z):
                            continue
                        
                        dist_segment = point_to_segment_distance_xy(m_x, m_y, tx1, ty1, tx2, ty2)
                        if dist_segment <= tol_xy:
                            score = dist_segment + z_delta * 0.1
                            if score < best_score:
                                best_score = score
                                best_testero = testero
                                best_z_delta = z_delta

                if best_testero is not None and best_z_delta > 0:
                    new_point: Point3 = (old_point[0], old_point[1], old_point[2] - best_z_delta)
                    point_tol = max(tol_conn, 1e-6)
                    updated = False
                    new_segments: list[Segment3] = []
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
                        moved_montantes.add(montante.handle)
                        moved_for_this_durmiente += 1

            if moved_for_this_durmiente > 0:
                removed_handles.add(durmiente.handle)
                matches_count += 1
            else:
                unresolved.append(f"Durmiente [{durmiente.handle}] (contrapeado) sin testero inferior coincidente")

        if removed_handles:
            self.entity_geometries = [
                e for e in self.entity_geometries if e.handle not in removed_handles
            ]

        report_lines = [
            f"Fusión de durmientes/testeros contrapeados finalizada:",
            f"  - Durmientes procesados y eliminados: {len(removed_handles)}",
            f"  - Montantes desfasados extendidos a cota testero: {len(moved_montantes)}",
            f"  - Tolerancia de desalineación XY usada: {tol_xy} mm",
            f"  - Caída máxima Z usada: {z_drop} mm",
        ]
        if unresolved:
            report_lines.append("")
            report_lines.append("Observaciones:")
            report_lines.extend(f"- {line}" for line in unresolved[:30])

        return True, "\n".join(report_lines)

    def execute_preprocesar_cadwork(
        self,
        tol_corner_esquinas: float = 150.0,
        tol_extremo_fachada: float = 200.0,
        tol_xy_proximity: float = 150.0,
        tol_z_plate: float = 100.0,
        tol_hueco_xy: float = 100.0,
        tol_cota_nivel: float = 1125.0,
        default_isolated_as_pasante: bool = True,
        max_panel_height: float = 10000.0,
    ) -> tuple[bool, str]:
        # Collect elements from self.entity_geometries
        plates = []
        montantes = []
        dinteles = []
        alfeizares = []
        capialzados = []
        cadenetas = []

        for entity in self.entity_geometries:
            if not entity.segments:
                continue
            x1, y1, z1, x2, y2, z2 = entity.segments[0]
            data = {
                'entity': entity,
                'handle': entity.handle,
                'start': [x1, y1, z1],
                'end': [x2, y2, z2],
                'orig_layer': entity.layer
            }
            layer_upper = entity.layer.upper()
            if "TESTERO" in layer_upper or "DURMIENTE" in layer_upper:
                plates.append(data)
            elif "MONTANTE" in layer_upper:
                montantes.append(data)
            elif "DINTEL" in layer_upper:
                dinteles.append(data)
            elif "ALFEIZAR" in layer_upper:
                alfeizares.append(data)
            elif "CAPIALZADO" in layer_upper:
                capialzados.append(data)
            elif "CADENETA" in layer_upper:
                cadenetas.append(data)

        if not plates:
            return False, "No se encontraron elementos en capas TESTERO o DURMIENTE."

        # 1. ROBUST PANEL MATCHING
        panels = []
        used_plates = set()
        
        # Separate plates into testeros and durmientes to avoid incorrect vertical pairing
        testeros = [p for p in plates if "TESTERO" in p['orig_layer'].upper()]
        durmientes = [p for p in plates if "DURMIENTE" in p['orig_layer'].upper()]
        
        # Sort testeros by Z height (highest first)
        testeros.sort(key=lambda p: max(p['start'][2], p['end'][2]), reverse=True)
        
        for t in testeros:
            if t['handle'] in used_plates:
                continue
                
            t_start_xy = (t['start'][0], t['start'][1])
            t_end_xy = (t['end'][0], t['end'][1])
            t_z = max(t['start'][2], t['end'][2])
            
            best_d = None
            min_score = (True, float('inf'), float('inf'))
            
            for d in durmientes:
                if d['handle'] in used_plates:
                    continue
                    
                d_start_xy = (d['start'][0], d['start'][1])
                d_end_xy = (d['end'][0], d['end'][1])
                d_z = min(d['start'][2], d['end'][2])
                
                # Height difference check (1000 to max_panel_height mm to support 8m double-height panels)
                z_height = t_z - d_z
                if not (1000.0 <= z_height <= max_panel_height):
                    continue
                    
                # Proximity check in XY
                dist_start = math.hypot(t_start_xy[0] - d_start_xy[0], t_start_xy[1] - d_start_xy[1])
                dist_end = math.hypot(t_end_xy[0] - d_end_xy[0], t_end_xy[1] - d_end_xy[1])
                dist_start_rev = math.hypot(t_start_xy[0] - d_end_xy[0], t_start_xy[1] - d_end_xy[1])
                dist_end_rev = math.hypot(t_end_xy[0] - d_start_xy[0], t_end_xy[1] - d_start_xy[1])
                
                d_dist = min(dist_start + dist_end, dist_start_rev + dist_end_rev)
                if d_dist <= tol_xy_proximity * 2:
                    is_outside = d_dist > 150.0
                    score = (is_outside, z_height, d_dist)
                    
                    if score < min_score:
                        min_score = score
                        best_d = d
                    
            if best_d:
                used_plates.add(t['handle'])
                used_plates.add(best_d['handle'])
                panels.append({
                    'id': len(panels),
                    'testero': t,
                    'durmiente': best_d,
                    'start_xy': t_start_xy,
                    'end_xy': t_end_xy,
                    'start_xy_trimmed': t_start_xy,
                    'end_xy_trimmed': t_end_xy,
                    'z_top': t_z,
                    'z_bottom': min(best_d['start'][2], best_d['end'][2])
                })

        if not panels:
            return False, "No se pudieron emparejar paneles a partir de las placas (comprueba tolerancias)."

        # Associate montantes to panels
        panel_montantes = {i: [] for i in range(len(panels))}
        unassigned_montantes = []
        
        for m in montantes:
            mx = (m['start'][0] + m['end'][0]) / 2.0
            my = (m['start'][1] + m['end'][1]) / 2.0
            m_z_mid = (m['start'][2] + m['end'][2]) / 2.0
            
            best_panel_idx = -1
            min_proj_dist = float('inf')
            
            for idx, panel in enumerate(panels):
                if not (panel['z_bottom'] - 200.0 <= m_z_mid <= panel['z_top'] + 200.0):
                    continue
                t = panel['testero']
                dist = point_to_segment_distance_xy(mx, my, t['start'][0], t['start'][1], t['end'][0], t['end'][1])
                if dist < min_proj_dist and dist <= tol_xy_proximity:
                    min_proj_dist = dist
                    best_panel_idx = idx
                    
            if best_panel_idx != -1:
                panel_montantes[best_panel_idx].append(m)
            else:
                unassigned_montantes.append(m)

        # Associate cadenetas to panels
        panel_cadenetas = {i: [] for i in range(len(panels))}
        unassigned_cadenetas = []
        for c in cadenetas:
            cmx = (c['start'][0] + c['end'][0]) / 2.0
            cmy = (c['start'][1] + c['end'][1]) / 2.0
            c_z = c['start'][2]
            best_panel_idx = -1
            min_proj_dist = float('inf')
            for idx, panel in enumerate(panels):
                if not (panel['z_bottom'] - 50.0 <= c_z <= panel['z_top'] + 50.0):
                    continue
                t = panel['testero']
                dist = point_to_segment_distance_xy(cmx, cmy, t['start'][0], t['start'][1], t['end'][0], t['end'][1])
                if dist < min_proj_dist and dist <= tol_xy_proximity:
                    min_proj_dist = dist
                    best_panel_idx = idx
            if best_panel_idx != -1:
                panel_cadenetas[best_panel_idx].append(c)
            else:
                unassigned_cadenetas.append(c)

        # 2. Determine classifications (Pasante / No Pasante) using TESTERO separation
        panel_classes = {} # panel_id -> "Pasante" or "No Pasante"

        for p in panels:
            p_idx = p['id']
            t_z = p['z_top']
            d_z = p['z_bottom']
            p_mid_xy = ((p['start_xy'][0] + p['end_xy'][0])/2.0, (p['start_xy'][1] + p['end_xy'][1])/2.0)
            
            # Check panel above
            best_above = None
            min_dist_above = float('inf')
            for other in panels:
                if other['id'] == p_idx:
                    continue
                if other['z_bottom'] <= t_z:
                    continue
                dist = point_to_segment_distance_xy(p_mid_xy[0], p_mid_xy[1], other['start_xy'][0], other['start_xy'][1], other['end_xy'][0], other['end_xy'][1])
                if dist <= tol_xy_proximity:
                    z_dist = other['z_bottom'] - t_z
                    if z_dist < min_dist_above:
                        min_dist_above = z_dist
                        best_above = other

            # Check panel below (only for top-most panels that have no panel above)
            best_below = None
            min_dist_below = float('inf')
            for other in panels:
                if other['id'] == p_idx:
                    continue
                if other['z_top'] >= d_z:
                    continue
                dist = point_to_segment_distance_xy(p_mid_xy[0], p_mid_xy[1], other['start_xy'][0], other['start_xy'][1], other['end_xy'][0], other['end_xy'][1])
                if dist <= tol_xy_proximity:
                    z_dist = d_z - other['z_top']
                    if z_dist < min_dist_below:
                        min_dist_below = z_dist
                        best_below = other

            if min_dist_above != float('inf'):
                is_pasante = min_dist_above <= 150.0
            elif min_dist_below != float('inf'):
                is_pasante = min_dist_below <= 150.0
            else:
                # Isolated panel fallback determined by user checkbox
                is_pasante = default_isolated_as_pasante
                
            panel_classes[p_idx] = "Pasante" if is_pasante else "No Pasante"

        # STEP 1: Process each panel independently
        classified = {}
        union_studs_per_panel = {}

        for idx, panel in enumerate(panels):
            t_data = panel['testero']
            d_data = panel['durmiente']
            z_top = panel['z_top']
            z_bottom = panel['z_bottom']
            associated = panel_montantes[idx]
            
            # STRICT Z CONSTRAINT: Dinteles, alfeizares and capialzados must lie within the panel Z range!
            panel_dinteles = [d for d in dinteles if (z_bottom - 50.0 <= d['start'][2] <= z_top + 50.0) and point_to_segment_distance_xy((d['start'][0]+d['end'][0])/2.0, (d['start'][1]+d['end'][1])/2.0, t_data['start'][0], t_data['start'][1], t_data['end'][0], t_data['end'][1]) <= tol_xy_proximity]
            panel_alfeizares = [a for a in alfeizares if (z_bottom - 50.0 <= a['start'][2] <= z_top + 50.0) and point_to_segment_distance_xy((a['start'][0]+a['end'][0])/2.0, (a['start'][1]+a['end'][1])/2.0, t_data['start'][0], t_data['start'][1], t_data['end'][0], t_data['end'][1]) <= tol_xy_proximity]
            
            # Capialzados can be out-of-plane, so check perpendicular distance to panel line
            panel_capialzados = []
            for c in capialzados:
                c_z_min = min(c['start'][2], c['end'][2])
                c_z_max = max(c['start'][2], c['end'][2])
                if (z_bottom - 50.0 <= c_z_min <= z_top + 50.0) or (z_bottom - 50.0 <= c_z_max <= z_top + 50.0):
                    cmx, cmy = (c['start'][0] + c['end'][0]) / 2.0, (c['start'][1] + c['end'][1]) / 2.0
                    perp_dist = point_to_line_distance_2d(cmx, cmy, t_data['start'][0], t_data['start'][1], t_data['end'][0], t_data['end'][1])
                    if perp_dist <= tol_xy_proximity * 2.0:
                        panel_capialzados.append(c)

            panel['dinteles'] = panel_dinteles
            panel['alfeizares'] = panel_alfeizares
            panel['capialzados'] = panel_capialzados
            panel['cadenetas'] = panel_cadenetas[idx]

            # Snap cadenetas to containing montantes (BEFORE short montantes Z are modified)
            for cad in panel['cadenetas']:
                cs = [cad['start'][0], cad['start'][1], cad['start'][2]]
                ce = [cad['end'][0], cad['end'][1], cad['end'][2]]
                cz = cs[2]
                
                cands = []
                for m in associated:
                    mz_min, mz_max = min(m['start'][2], m['end'][2]), max(m['start'][2], m['end'][2])
                    if mz_min - 50.0 <= cz <= mz_max + 50.0:
                        cands.append(((m['start'][0] + m['end'][0]) / 2.0, (m['start'][1] + m['end'][1]) / 2.0, m))
                        
                best_s, min_ds = None, float('inf')
                best_e, min_de = None, float('inf')
                for mx, my, m in cands:
                    ds = math.hypot(mx - cs[0], my - cs[1])
                    de = math.hypot(mx - ce[0], my - ce[1])
                    if ds < min_ds:
                        min_ds = ds
                        best_s = (mx, my, m)
                    if de < min_de:
                        min_de = de
                        best_e = (mx, my, m)
                        
                # If either endpoint didn't find a close stud in this panel, search all montantes
                if min_ds > tol_xy_proximity * 2 or min_de > tol_xy_proximity * 2 or (best_s and best_e and best_s[2]['handle'] == best_e[2]['handle']):
                    for m in montantes:
                        if m in associated:
                            continue
                        mz_min, mz_max = min(m['start'][2], m['end'][2]), max(m['start'][2], m['end'][2])
                        if mz_min - 50.0 <= cz <= mz_max + 50.0:
                            mx = (m['start'][0] + m['end'][0]) / 2.0
                            my = (m['start'][1] + m['end'][1]) / 2.0
                            ds = math.hypot(mx - cs[0], my - cs[1])
                            de = math.hypot(mx - ce[0], my - ce[1])
                            if ds < min_ds:
                                min_ds = ds
                                best_s = (mx, my, m)
                            if de < min_de:
                                min_de = de
                                best_e = (mx, my, m)
                                
                if best_s and best_e and best_s[2]['handle'] != best_e[2]['handle'] and min_ds <= tol_xy_proximity * 2 and min_de <= tol_xy_proximity * 2:
                    cad['start'][0], cad['start'][1] = best_s[0], best_s[1]
                    cad['end'][0], cad['end'][1] = best_e[0], best_e[1]
                else:
                    cmx = (cs[0] + ce[0]) / 2.0
                    cmy = (cs[1] + ce[1]) / 2.0
                    near_center = []
                    for mx, my, m in cands:
                        d = math.hypot(mx - cmx, my - cmy)
                        if d <= tol_xy_proximity * 2:
                            near_center.append((d, mx, my, m))
                    near_center.sort()
                    if len(near_center) >= 2:
                        cad['start'][0], cad['start'][1] = near_center[0][1], near_center[0][2]
                        cad['end'][0], cad['end'][1] = near_center[1][1], near_center[1][2]
                        
                cad['entity'].layer = f"CADENETA{suffix}"
                cad['entity'].segments = [(cad['start'][0], cad['start'][1], cad['start'][2], cad['end'][0], cad['end'][1], cad['end'][2])]

            full_height = []
            short_montantes = []
            for m in associated:
                m_z_min = min(m['start'][2], m['end'][2])
                m_z_max = max(m['start'][2], m['end'][2])
                
                bottom_diff = abs(m_z_min - z_bottom)
                top_diff = abs(m_z_max - z_top)
                
                if bottom_diff <= tol_z_plate and top_diff <= tol_z_plate:
                    full_height.append(m)
                else:
                    short_montantes.append(m)

            # Detect Huecos (checking dinteles, alfeizares and projected capialzados)
            hueco_montantes = []
            remaining_full_height = []
            for m in full_height:
                mx = (m['start'][0] + m['end'][0]) / 2.0
                my = (m['start'][1] + m['end'][1]) / 2.0
                is_hueco = False
                for d in panel_dinteles:
                    for p in [(d['start'][0], d['start'][1]), (d['end'][0], d['end'][1])]:
                        if math.hypot(mx - p[0], my - p[1]) <= tol_hueco_xy:
                            is_hueco = True
                            break
                    if is_hueco:
                        break
                if not is_hueco:
                    for a in panel_alfeizares:
                        for p in [(a['start'][0], a['start'][1]), (a['end'][0], a['end'][1])]:
                            if math.hypot(mx - p[0], my - p[1]) <= tol_hueco_xy:
                                is_hueco = True
                                break
                        if is_hueco:
                            break
                if not is_hueco:
                    for c in panel_capialzados:
                        c_start_proj = project_point_onto_line_2d(c['start'][0], c['start'][1], t_data['start'][0], t_data['start'][1], t_data['end'][0], t_data['end'][1])
                        c_end_proj = project_point_onto_line_2d(c['end'][0], c['end'][1], t_data['start'][0], t_data['start'][1], t_data['end'][0], t_data['end'][1])
                        for p in [c_start_proj, c_end_proj]:
                            if math.hypot(mx - p[0], my - p[1]) <= tol_hueco_xy:
                                is_hueco = True
                                break
                        if is_hueco:
                            break
                if is_hueco:
                    hueco_montantes.append(m)
                    classified[m['handle']] = "MONTANTES HUECOS"
                else:
                    remaining_full_height.append(m)

            # Detect Left and Right Union Studs
            left_union = None
            right_union = None
            if full_height:
                p_start = panel['start_xy']
                p_end = panel['end_xy']
                
                min_dist_start = float('inf')
                min_dist_end = float('inf')
                
                for m in full_height:
                    mx = (m['start'][0] + m['end'][0]) / 2.0
                    my = (m['start'][1] + m['end'][1]) / 2.0
                    
                    d_start = math.hypot(mx - p_start[0], my - p_start[1])
                    d_end = math.hypot(mx - p_end[0], my - p_end[1])
                    
                    if d_start < min_dist_start:
                        min_dist_start = d_start
                        left_union = m
                    if d_end < min_dist_end:
                        min_dist_end = d_end
                        right_union = m
                
                for m in full_height:
                    is_union = False
                    if m['handle'] == left_union['handle']:
                        is_union = True
                    elif m['handle'] == right_union['handle']:
                        is_union = True
                        
                    if is_union:
                        classified[m['handle']] = "MONTANTES UNION"
                    elif classified.get(m['handle']) == "MONTANTES HUECOS":
                        pass
                    else:
                        classified[m['handle']] = "MONTANTES SIMPLES"
                        
                union_studs_per_panel[idx] = {'left': left_union, 'right': right_union}
                
                # Snap/Trim Testero and Durmiente to union studs
                left_xy = ((left_union['start'][0] + left_union['end'][0])/2.0, (left_union['start'][1] + left_union['end'][1])/2.0)
                right_xy = ((right_union['start'][0] + right_union['end'][0])/2.0, (right_union['start'][1] + right_union['end'][1])/2.0)
                
                t_data['start'][0], t_data['start'][1] = left_xy[0], left_xy[1]
                t_data['end'][0], t_data['end'][1] = right_xy[0], right_xy[1]
                
                d_data['start'][0], d_data['start'][1] = left_xy[0], left_xy[1]
                d_data['end'][0], d_data['end'][1] = right_xy[0], right_xy[1]
                
                panel['start_xy_trimmed'] = left_xy
                panel['end_xy_trimmed'] = right_xy

            # Get the suffix (Pasante / No Pasante) for this panel
            p_class = panel_classes[idx]
            suffix = f" ({p_class})"

            # Extend dinteles and alfeizares and save with correct layer name suffix
            for d in panel_dinteles:
                d_start_xy = (d['start'][0], d['start'][1])
                d_end_xy = (d['end'][0], d['end'][1])
                
                best_start_m = None
                min_ds = float('inf')
                best_end_m = None
                min_de = float('inf')
                
                for hm in hueco_montantes:
                    hm_xy = ((hm['start'][0] + hm['end'][0]) / 2.0,
                             (hm['start'][1] + hm['end'][1]) / 2.0)
                    
                    ds = math.hypot(hm_xy[0] - d_start_xy[0], hm_xy[1] - d_start_xy[1])
                    de = math.hypot(hm_xy[0] - d_end_xy[0], hm_xy[1] - d_end_xy[1])
                    
                    if ds < min_ds and ds <= tol_hueco_xy * 2:
                        min_ds = ds
                        best_start_m = hm
                    if de < min_de and de <= tol_hueco_xy * 2:
                        min_de = de
                        best_end_m = hm
                        
                if best_start_m:
                    hm_xy = ((best_start_m['start'][0] + best_start_m['end'][0]) / 2.0,
                             (best_start_m['start'][1] + best_start_m['end'][1]) / 2.0)
                    d['start'][0], d['start'][1] = hm_xy[0], hm_xy[1]
                if best_end_m:
                    hm_xy = ((best_end_m['start'][0] + best_end_m['end'][0]) / 2.0,
                             (best_end_m['start'][1] + best_end_m['end'][1]) / 2.0)
                    d['end'][0], d['end'][1] = hm_xy[0], hm_xy[1]
                
                d['entity'].layer = f"DINTEL{suffix}"
                d['entity'].segments = [(d['start'][0], d['start'][1], d['start'][2], d['end'][0], d['end'][1], d['end'][2])]

            for a in panel_alfeizares:
                a_start_xy = (a['start'][0], a['start'][1])
                a_end_xy = (a['end'][0], a['end'][1])
                
                best_start_m = None
                min_ds = float('inf')
                best_end_m = None
                min_de = float('inf')
                
                for hm in hueco_montantes:
                    hm_xy = ((hm['start'][0] + hm['end'][0]) / 2.0,
                             (hm['start'][1] + hm['end'][1]) / 2.0)
                    
                    ds = math.hypot(hm_xy[0] - a_start_xy[0], hm_xy[1] - a_start_xy[1])
                    de = math.hypot(hm_xy[0] - a_end_xy[0], hm_xy[1] - a_end_xy[1])
                    
                    if ds < min_ds and ds <= tol_hueco_xy * 2:
                        min_ds = ds
                        best_start_m = hm
                    if de < min_de and de <= tol_hueco_xy * 2:
                        min_de = de
                        best_end_m = hm
                        
                if best_start_m:
                    hm_xy = ((best_start_m['start'][0] + best_start_m['end'][0]) / 2.0,
                             (best_start_m['start'][1] + best_start_m['end'][1]) / 2.0)
                    a['start'][0], a['start'][1] = hm_xy[0], hm_xy[1]
                if best_end_m:
                    hm_xy = ((best_end_m['start'][0] + best_end_m['end'][0]) / 2.0,
                             (best_end_m['start'][1] + best_end_m['end'][1]) / 2.0)
                    a['end'][0], a['end'][1] = hm_xy[0], hm_xy[1]
                
                a['entity'].layer = f"ALFEIZAR{suffix}"
                a['entity'].segments = [(a['start'][0], a['start'][1], a['start'][2], a['end'][0], a['end'][1], a['end'][2])]

            for c in panel_capialzados:
                c_start_proj = project_point_onto_line_2d(c['start'][0], c['start'][1], t_data['start'][0], t_data['start'][1], t_data['end'][0], t_data['end'][1])
                c_end_proj = project_point_onto_line_2d(c['end'][0], c['end'][1], t_data['start'][0], t_data['start'][1], t_data['end'][0], t_data['end'][1])
                
                best_start_m = None
                min_ds = float('inf')
                best_end_m = None
                min_de = float('inf')
                
                for hm in hueco_montantes:
                    hm_xy = ((hm['start'][0] + hm['end'][0]) / 2.0,
                             (hm['start'][1] + hm['end'][1]) / 2.0)
                    
                    ds = math.hypot(hm_xy[0] - c_start_proj[0], hm_xy[1] - c_start_proj[1])
                    de = math.hypot(hm_xy[0] - c_end_proj[0], hm_xy[1] - c_end_proj[1])
                    
                    if ds < min_ds and ds <= tol_hueco_xy * 2:
                        min_ds = ds
                        best_start_m = hm
                    if de < min_de and de <= tol_hueco_xy * 2:
                        min_de = de
                        best_end_m = hm
                        
                if best_start_m:
                    hm_xy = ((best_start_m['start'][0] + best_start_m['end'][0]) / 2.0,
                             (best_start_m['start'][1] + best_start_m['end'][1]) / 2.0)
                    c['start'][0], c['start'][1] = hm_xy[0], hm_xy[1]
                if best_end_m:
                    hm_xy = ((best_end_m['start'][0] + best_end_m['end'][0]) / 2.0,
                             (best_end_m['start'][1] + best_end_m['end'][1]) / 2.0)
                    c['end'][0], c['end'][1] = hm_xy[0], hm_xy[1]
                
                c['entity'].layer = f"CAPIALZADO{suffix}"
                c['entity'].segments = [(c['start'][0], c['start'][1], c['start'][2], c['end'][0], c['end'][1], c['end'][2])]

            # Snapped full-height montantes vertically
            for m in full_height:
                m['start'][2] = z_bottom
                m['end'][2] = z_top
                
            # Snapped short montantes vertically
            for m in short_montantes:
                classified[m['handle']] = "MONTANTES CORTOS"
                mx = (m['start'][0] + m['end'][0]) / 2.0
                my = (m['start'][1] + m['end'][1]) / 2.0
                m_z_mid = (m['start'][2] + m['end'][2]) / 2.0
                
                if m_z_mid < (z_bottom + z_top) / 2.0:
                    m['start'][2] = z_bottom
                    best_a = None
                    min_dz = float('inf')
                    for a in panel_alfeizares:
                        dist = point_to_segment_distance_xy(mx, my, a['start'][0], a['start'][1], a['end'][0], a['end'][1])
                        if dist <= tol_xy_proximity:
                            dz = abs(a['start'][2] - m['end'][2])
                            if dz < min_dz:
                                min_dz = dz
                                best_a = a
                    if best_a and min_dz <= 500.0:
                        m['end'][2] = best_a['start'][2]
                else:
                    m['end'][2] = z_top
                    best_d = None
                    min_dz = float('inf')
                    for d in panel_dinteles:
                        dist = point_to_segment_distance_xy(mx, my, d['start'][0], d['start'][1], d['end'][0], d['end'][1])
                        if dist <= tol_xy_proximity:
                            min_x = min(d['start'][0], d['end'][0])
                            max_x = max(d['start'][0], d['end'][0])
                            if min_x - 10 <= mx <= max_x + 10:
                                dz = abs(d['start'][2] - m['start'][2])
                                if dz < min_dz:
                                    min_dz = dz
                                    best_d = d
                    if best_d and min_dz <= 500.0:
                        m['start'][2] = best_d['start'][2]

        # STEP 1.5: Facade Ends Demoted to MONTANTES SIMPLES (isolated union studs, tol = 200mm)
        demoted_count = 0
        for idx, studs in list(union_studs_per_panel.items()):
            p = panels[idx]
            for side in ['left', 'right']:
                m = studs[side]
                if not m:
                    continue
                mx = (m['start'][0] + m['end'][0]) / 2.0
                my = (m['start'][1] + m['end'][1]) / 2.0
                
                has_neighbor = False
                for other_idx, other_studs in union_studs_per_panel.items():
                    if other_idx == idx:
                        continue
                    other_p = panels[other_idx]
                    
                    # Check same Z level (within 200mm)
                    if abs(p['z_top'] - other_p['z_top']) > 200.0:
                        continue
                        
                    for other_side in ['left', 'right']:
                        other_m = other_studs[other_side]
                        if not other_m:
                            continue
                        omx = (other_m['start'][0] + other_m['end'][0]) / 2.0
                        omy = (other_m['start'][1] + other_m['end'][1]) / 2.0
                        
                        if math.hypot(mx - omx, my - omy) <= tol_extremo_fachada:
                            has_neighbor = True
                            break
                    if has_neighbor:
                        break
                        
                if not has_neighbor:
                    classified[m['handle']] = "MONTANTES SIMPLES"
                    demoted_count += 1

        # STEP 2: Corner Promoted to MONTANTES ESQUINA
        corner_count = 0
        for i in range(len(panels)):
            for j in range(i + 1, len(panels)):
                p_a = panels[i]
                p_b = panels[j]
                
                if abs(p_a['z_top'] - p_b['z_top']) > 200.0:
                    continue
                    
                v_a = (p_a['end_xy'][0] - p_a['start_xy'][0], p_a['end_xy'][1] - p_a['start_xy'][1])
                v_b = (p_b['end_xy'][0] - p_b['start_xy'][0], p_b['end_xy'][1] - p_b['start_xy'][1])
                len_a = math.hypot(*v_a)
                len_b = math.hypot(*v_b)
                if len_a > 0 and len_b > 0:
                    dot = (v_a[0]*v_b[0] + v_a[1]*v_b[1]) / (len_a * len_b)
                    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
                    
                    if 15.0 <= angle_deg <= 165.0 or 195.0 <= angle_deg <= 345.0:
                        for end_a_is_start in [True, False]:
                            pt_a = p_a['start_xy_trimmed'] if end_a_is_start else p_a['end_xy_trimmed']
                            for end_b_is_start in [True, False]:
                                pt_b = p_b['start_xy_trimmed'] if end_b_is_start else p_b['end_xy_trimmed']
                                
                                dist = math.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])
                                if dist <= tol_corner_esquinas:
                                    if i in union_studs_per_panel and j in union_studs_per_panel:
                                        stud_a = union_studs_per_panel[i]['left'] if end_a_is_start else union_studs_per_panel[i]['right']
                                        stud_b = union_studs_per_panel[j]['left'] if end_b_is_start else union_studs_per_panel[j]['right']
                                        
                                        if classified.get(stud_a['handle']) != "MONTANTES ESQUINA":
                                            classified[stud_a['handle']] = "MONTANTES ESQUINA"
                                            corner_count += 1
                                        if classified.get(stud_b['handle']) != "MONTANTES ESQUINA":
                                            classified[stud_b['handle']] = "MONTANTES ESQUINA"
                                            corner_count += 1

        # Rebuild geometries list and save to EntityGeometry objects
        new_geometries = []
        appended_handles = set()
        
        # Processed plates
        for idx, panel in enumerate(panels):
            p_class = panel_classes[idx]
            suffix = f" ({p_class})"
            
            t = panel['testero']
            t['entity'].layer = f"TESTERO{suffix}"
            t['entity'].segments = [(t['start'][0], t['start'][1], t['start'][2], t['end'][0], t['end'][1], t['end'][2])]
            if t['handle'] not in appended_handles:
                new_geometries.append(t['entity'])
                appended_handles.add(t['handle'])
            
            d = panel['durmiente']
            d['entity'].layer = f"DURMIENTE{suffix}"
            d['entity'].segments = [(d['start'][0], d['start'][1], d['start'][2], d['end'][0], d['end'][1], d['end'][2])]
            if d['handle'] not in appended_handles:
                new_geometries.append(d['entity'])
                appended_handles.add(d['handle'])
            
            # Processed montantes
            for m in panel_montantes[idx]:
                layer_base = classified.get(m['handle'], "MONTANTES SIMPLES")
                m['entity'].layer = f"{layer_base}{suffix}"
                m['entity'].segments = [(m['start'][0], m['start'][1], m['start'][2], m['end'][0], m['end'][1], m['end'][2])]
                if m['handle'] not in appended_handles:
                    new_geometries.append(m['entity'])
                    appended_handles.add(m['handle'])

            # Processed dinteles, alfeizares, capialzados and cadenetas of this panel
            for dintel_data in panel['dinteles']:
                if dintel_data['handle'] not in appended_handles:
                    new_geometries.append(dintel_data['entity'])
                    appended_handles.add(dintel_data['handle'])
            for alfeizar_data in panel['alfeizares']:
                if alfeizar_data['handle'] not in appended_handles:
                    new_geometries.append(alfeizar_data['entity'])
                    appended_handles.add(alfeizar_data['handle'])
            for capialzado_data in panel.get('capialzados', []):
                if capialzado_data['handle'] not in appended_handles:
                    new_geometries.append(capialzado_data['entity'])
                    appended_handles.add(capialzado_data['handle'])
            for cadeneta_data in panel.get('cadenetas', []):
                if cadeneta_data['handle'] not in appended_handles:
                    new_geometries.append(cadeneta_data['entity'])
                    appended_handles.add(cadeneta_data['handle'])

        # Unassigned montantes
        for m in unassigned_montantes:
            m['entity'].layer = "MONTANTES CORTOS (No Pasante)"
            m['entity'].segments = [(m['start'][0], m['start'][1], m['start'][2], m['end'][0], m['end'][1], m['end'][2])]
            if m['handle'] not in appended_handles:
                new_geometries.append(m['entity'])
                appended_handles.add(m['handle'])

        # Fallback for unmatched dinteles
        for d in dinteles:
            if d['handle'] not in appended_handles:
                d['entity'].layer = "DINTEL (No Pasante)"
                d['entity'].segments = [(d['start'][0], d['start'][1], d['start'][2], d['end'][0], d['end'][1], d['end'][2])]
                new_geometries.append(d['entity'])
                appended_handles.add(d['handle'])

        # Fallback for unmatched alfeizares
        for a in alfeizares:
            if a['handle'] not in appended_handles:
                a['entity'].layer = "ALFEIZAR (No Pasante)"
                a['entity'].segments = [(a['start'][0], a['start'][1], a['start'][2], a['end'][0], a['end'][1], a['end'][2])]
                new_geometries.append(a['entity'])
                appended_handles.add(a['handle'])

        # Fallback for unmatched capialzados
        for c in capialzados:
            if c['handle'] not in appended_handles:
                c['entity'].layer = "CAPIALZADO (No Pasante)"
                c['entity'].segments = [(c['start'][0], c['start'][1], c['start'][2], c['end'][0], c['end'][1], c['end'][2])]
                new_geometries.append(c['entity'])
                appended_handles.add(c['handle'])

        # Fallback for unassigned cadenetas
        for c in unassigned_cadenetas:
            if c['handle'] not in appended_handles:
                c['entity'].layer = "CADENETA (No Pasante)"
                c['entity'].segments = [(c['start'][0], c['start'][1], c['start'][2], c['end'][0], c['end'][1], c['end'][2])]
                new_geometries.append(c['entity'])
                appended_handles.add(c['handle'])

        # Fallback for unmatched plates (prevent plate loss)
        for p in plates:
            if p['handle'] not in appended_handles:
                p['entity'].layer = f"{p['orig_layer']} (No Pasante)"
                p['entity'].segments = [(p['start'][0], p['start'][1], p['start'][2], p['end'][0], p['end'][1], p['end'][2])]
                new_geometries.append(p['entity'])
                appended_handles.add(p['handle'])

        # Non-processed entities
        processed_handles = set()
        for p in plates:
            processed_handles.add(p['handle'])
        for m in montantes:
            processed_handles.add(m['handle'])
        for d in dinteles:
            processed_handles.add(d['handle'])
        for a in alfeizares:
            processed_handles.add(a['handle'])
        for c in capialzados:
            processed_handles.add(c['handle'])
        for c in cadenetas:
            processed_handles.add(c['handle'])
            
        for entity in self.entity_geometries:
            if entity.handle not in processed_handles and "COTA" not in entity.layer.upper():
                new_geometries.append(entity)
                
        self.entity_geometries = new_geometries

        # Prepare details report
        report_lines = [
            f"Preprocesamiento Cadwork finalizado:",
            f"  - Paneles formados: {len(panels)}",
            f"  - Capas de placas emparejadas: {len(plates)}",
            f"  - Montantes procesados: {len(montantes)}",
            f"  - Dinteles procesados: {len(dinteles)}",
            f"  - Alfeizares procesados: {len(alfeizares)}",
            f"  - Capialzados procesados: {len(capialzados)}",
            f"  - Cadenetas procesadas: {len(cadenetas)}",
            f"  - Montantes desclasificados por extremo libre (aislamiento): {demoted_count}",
            f"  - Montantes promovidos a esquina: {corner_count}",
        ]
        
        return True, "\n".join(report_lines)
