import ezdxf
import sys
import math

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

def main():
    dxf_path = r"C:\PythonProjects\RFEM_Imports\Panel No Pasante.dxf"
    output_path = r"C:\PythonProjects\RFEM_Imports\Panel No Pasante_unido.dxf"
    
    try:
        doc = ezdxf.readfile(dxf_path)
    except Exception as e:
        print(f"Error reading input DXF: {e}")
        sys.exit(1)

    msp = doc.modelspace()
    
    # Collect elements
    plates = []
    montantes = []
    dinteles = []
    alfeizares = []

    for entity in msp:
        if entity.dxftype() == 'LINE':
            layer = entity.dxf.layer.upper()
            start = entity.dxf.start
            end = entity.dxf.end
            handle = entity.dxf.handle
            data = {
                'handle': handle,
                'start': [start.x, start.y, start.z],
                'end': [end.x, end.y, end.z],
                'orig_layer': entity.dxf.layer
            }
            if "TESTERO" in layer or "DURMIENTE" in layer:
                plates.append(data)
            elif "MONTANTE" in layer:
                montantes.append(data)
            elif "DINTEL" in layer:
                dinteles.append(data)
            elif "ALFEIZAR" in layer:
                alfeizares.append(data)

    # 1. ROBUST PANEL MATCHING: Combine TESTERO and DURMIENTE and match purely by Z-height & XY projection
    panels = []
    tol_xy = 150.0
    
    used_plates = set()
    
    # Sort plates by Z height (highest first)
    plates.sort(key=lambda p: max(p['start'][2], p['end'][2]), reverse=True)
    
    for i, t in enumerate(plates):
        if t['handle'] in used_plates:
            continue
            
        t_start_xy = (t['start'][0], t['start'][1])
        t_end_xy = (t['end'][0], t['end'][1])
        t_z = max(t['start'][2], t['end'][2])
        
        best_d = None
        min_score = (True, float('inf'), float('inf'))
        
        for j, d in enumerate(plates):
            if i == j:
                continue
            if d['handle'] in used_plates:
                continue
                
            d_start_xy = (d['start'][0], d['start'][1])
            d_end_xy = (d['end'][0], d['end'][1])
            d_z = min(d['start'][2], d['end'][2])
            
            # Height difference check (1500 to 4000 mm)
            z_height = t_z - d_z
            if not (1500.0 <= z_height <= 4000.0):
                continue
                
            # Proximity check in XY
            dist_start = math.hypot(t_start_xy[0] - d_start_xy[0], t_start_xy[1] - d_start_xy[1])
            dist_end = math.hypot(t_end_xy[0] - d_end_xy[0], t_end_xy[1] - d_end_xy[1])
            dist_start_rev = math.hypot(t_start_xy[0] - d_end_xy[0], t_start_xy[1] - d_end_xy[1])
            dist_end_rev = math.hypot(t_end_xy[0] - d_start_xy[0], t_end_xy[1] - d_start_xy[1])
            
            d_dist = min(dist_start + dist_end, dist_start_rev + dist_end_rev)
            if d_dist <= tol_xy * 2:
                # Scoring: (is_outside_tight_xy, z_height, d_dist)
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
                'z_top': t_z,
                'z_bottom': min(best_d['start'][2], best_d['end'][2])
            })

    print(f"Formed {len(panels)} panels using robust matching with composite scoring!")

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
            if dist < min_proj_dist and dist <= tol_xy:
                min_proj_dist = dist
                best_panel_idx = idx
                
        if best_panel_idx != -1:
            panel_montantes[best_panel_idx].append(m)
        else:
            unassigned_montantes.append(m)

    # 2. Determine classifications using the TESTERO (separation to the panel above)
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
            if dist <= tol_xy:
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
            if dist <= tol_xy:
                z_dist = d_z - other['z_top']
                if z_dist < min_dist_below:
                    min_dist_below = z_dist
                    best_below = other

        if min_dist_above != float('inf'):
            is_pasante = min_dist_above <= 150.0
        else:
            is_pasante = min_dist_below <= 150.0
            
        panel_classes[p_idx] = "Pasante" if is_pasante else "No Pasante"

    # STEP 1: Process each panel INDEPENDENTLY first
    tol_z_plate = 100.0
    tol_hueco_xy = 100.0

    classified = {}
    union_studs_per_panel = {}
    
    # Track final geometries to write
    final_lines = []

    for idx, panel in enumerate(panels):
        t_entity = panel['testero']
        d_entity = panel['durmiente']
        z_top = panel['z_top']
        z_bottom = panel['z_bottom']
        associated = panel_montantes[idx]
        
        # STRICT Z CONSTRAINT: Dinteles and alfeizares must lie strictly within the panel Z range!
        panel_dinteles = [d for d in dinteles if (z_bottom - 50.0 <= d['start'][2] <= z_top + 50.0) and point_to_segment_distance_xy((d['start'][0]+d['end'][0])/2.0, (d['start'][1]+d['end'][1])/2.0, panel['testero']['start'][0], panel['testero']['start'][1], panel['testero']['end'][0], panel['testero']['end'][1]) <= tol_xy]
        panel_alfeizares = [a for a in alfeizares if (z_bottom - 50.0 <= a['start'][2] <= z_top + 50.0) and point_to_segment_distance_xy((a['start'][0]+a['end'][0])/2.0, (a['start'][1]+a['end'][1])/2.0, panel['testero']['start'][0], panel['testero']['start'][1], panel['testero']['end'][0], panel['testero']['end'][1]) <= tol_xy]

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

        # Detect Huecos
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
            if is_hueco:
                hueco_montantes.append(m)
                classified[m['handle']] = "MONTANTES HUECOS"
            else:
                remaining_full_height.append(m)

        # Detect Left and Right Union Studs
        left_union = None
        right_union = None
        if remaining_full_height:
            p_start = panel['start_xy']
            p_end = panel['end_xy']
            
            min_dist_start = float('inf')
            min_dist_end = float('inf')
            
            for m in remaining_full_height:
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
            
            for m in remaining_full_height:
                is_union = False
                if m['handle'] == left_union['handle']:
                    is_union = True
                elif m['handle'] == right_union['handle']:
                    is_union = True
                    
                if is_union:
                    classified[m['handle']] = "MONTANTES UNION"
                else:
                    classified[m['handle']] = "MONTANTES SIMPLES"
                    
            union_studs_per_panel[idx] = {'left': left_union, 'right': right_union}
            
            # Snap/Trim Testero and Durmiente to union studs
            left_xy = ((left_union['start'][0] + left_union['end'][0])/2.0, (left_union['start'][1] + left_union['end'][1])/2.0)
            right_xy = ((right_union['start'][0] + right_union['end'][0])/2.0, (right_union['start'][1] + right_union['end'][1])/2.0)
            
            t_entity['start'][0], t_entity['start'][1] = left_xy[0], left_xy[1]
            t_entity['end'][0], t_entity['end'][1] = right_xy[0], right_xy[1]
            
            d_entity['start'][0], d_entity['start'][1] = left_xy[0], left_xy[1]
            d_entity['end'][0], d_entity['end'][1] = right_xy[0], right_xy[1]
            
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
            
            final_lines.append((f"DINTEL{suffix}", d['start'], d['end']))

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
            
            final_lines.append((f"ALFEIZAR{suffix}", a['start'], a['end']))

        # Snapped full-height montantes vertically
        for m in full_height:
            m['start'][2] = z_bottom
            m['end'][2] = z_top
            
        # Snapped short montantes vertically
        for m in short_montantes:
            mx = (m['start'][0] + m['end'][0]) / 2.0
            my = (m['start'][1] + m['end'][1]) / 2.0
            m_z_mid = (m['start'][2] + m['end'][2]) / 2.0
            
            if m_z_mid < (z_bottom + z_top) / 2.0:
                m['start'][2] = z_bottom
                best_a = None
                for a in panel_alfeizares:
                    dist = point_to_segment_distance_xy(mx, my, a['start'][0], a['start'][1], a['end'][0], a['end'][1])
                    if dist <= tol_xy:
                        best_a = a
                        break
                if best_a:
                    m['end'][2] = best_a['start'][2]
            else:
                m['end'][2] = z_top
                best_d = None
                for d in panel_dinteles:
                    dist = point_to_segment_distance_xy(mx, my, d['start'][0], d['start'][1], d['end'][0], d['end'][1])
                    if dist <= tol_xy:
                        min_x = min(d['start'][0], d['end'][0])
                        max_x = max(d['start'][0], d['end'][0])
                        if min_x - 10 <= mx <= max_x + 10:
                            best_d = d
                            break
                if best_d:
                    m['start'][2] = best_d['start'][2]

    # STEP 1.5: Facade Ends Demoted to MONTANTES SIMPLES (isolated union studs, tol = 200mm)
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
                
                # Check if panels are at the same Z level (within 200 mm)
                if abs(p['z_top'] - other_p['z_top']) > 200.0:
                    continue
                    
                for other_side in ['left', 'right']:
                    other_m = other_studs[other_side]
                    if not other_m:
                        continue
                    omx = (other_m['start'][0] + other_m['end'][0]) / 2.0
                    omy = (other_m['start'][1] + other_m['end'][1]) / 2.0
                    
                    if math.hypot(mx - omx, my - omy) <= 200.0:
                        has_neighbor = True
                        break
                if has_neighbor:
                    break
                    
            if not has_neighbor:
                classified[m['handle']] = "MONTANTES SIMPLES"

    # STEP 2: Corner Promoted to MONTANTES ESQUINA
    tol_corner_trimmed = 150.0
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
                            if dist <= tol_corner_trimmed:
                                # Safe check in case union studs were not found for either panel
                                if i in union_studs_per_panel and j in union_studs_per_panel:
                                    stud_a = union_studs_per_panel[i]['left'] if end_a_is_start else union_studs_per_panel[i]['right']
                                    stud_b = union_studs_per_panel[j]['left'] if end_b_is_start else union_studs_per_panel[j]['right']
                                    
                                    classified[stud_a['handle']] = "MONTANTES ESQUINA"
                                    classified[stud_b['handle']] = "MONTANTES ESQUINA"

    # Add remaining entities with their finalized layers
    for idx, panel in enumerate(panels):
        p_class = panel_classes[idx]
        suffix = f" ({p_class})"
        
        final_lines.append((f"TESTERO{suffix}", panel['testero']['start'], panel['testero']['end']))
        final_lines.append((f"DURMIENTE{suffix}", panel['durmiente']['start'], panel['durmiente']['end']))
        
        for m in panel_montantes[idx]:
            layer_base = classified.get(m['handle'], "MONTANTES SIMPLES")
            final_lines.append((f"{layer_base}{suffix}", m['start'], m['end']))

    for m in unassigned_montantes:
        final_lines.append(("MONTANTES CORTOS (No Pasante)", m['start'], m['end']))

    # Write output DXF (with fallback if open in another program)
    out_doc = ezdxf.new("R2010")
    out_msp = out_doc.modelspace()
    
    existing_layers = set()
    for layer_name, start, end in final_lines:
        if layer_name not in existing_layers:
            out_doc.layers.new(name=layer_name)
            existing_layers.add(layer_name)
            
        out_msp.add_line(start, end, dxfattribs={"layer": layer_name})
        
    try:
        out_doc.saveas(output_path)
        print(f"\n[Success] Joined No Pasante DXF saved successfully to: {output_path}")
    except PermissionError:
        n = 2
        saved = False
        while not saved:
            fallback_path = f"C:\\PythonProjects\\RFEM_Imports\\Panel No Pasante_unido_v{n}.dxf"
            try:
                out_doc.saveas(fallback_path)
                print(f"\n[Warning] Permission denied on {output_path} (file is probably open). Saved to: {fallback_path}")
                saved = True
            except PermissionError:
                n += 1
                if n > 100:
                    print("\n[Error] Could not save DXF. All files from v2 to v100 are locked by other programs.")
                    sys.exit(1)

if __name__ == "__main__":
    main()
