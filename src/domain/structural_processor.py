import math
from collections import defaultdict

from src.core.geometry import EntityGeometry, Point3, Segment3, point_to_segment_distance, segments_are_connected

class StructuralProcessor:
    def __init__(self, geometries: list[EntityGeometry]):
        self.entity_geometries = geometries

    def get_geometries(self) -> list[EntityGeometry]:
        return self.entity_geometries

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
