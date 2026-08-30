import numpy as np
import cv2

def compute_spatial_fingerprint(search_img: np.ndarray, cx: float, cy: float, 
                              scale: float, family_members: list) -> dict:
    """
    Computes a structural spatial fingerprint G(candidate) for candidate location:
    1. nearest_edge_distance: Distance to nearest Canny edge pixel.
    2. nearest_defect_distance: Distance to nearest low-intensity gate cut / dummy fill boundary.
    3. row_spacing / column_spacing: Distances to nearest periodic neighbors in same family.
    4. local_density: Average pixel value in concentric neighborhood.
    """
    sh, sw = search_img.shape[:2]
    
    # 1. Edge Detection & Distance
    edges = cv2.Canny(search_img, 50, 150)
    y_coords, x_coords = np.where(edges > 0)
    
    nearest_edge_dist = 1000.0
    if len(x_coords) > 0:
        dists = np.hypot(x_coords - cx, y_coords - cy)
        nearest_edge_dist = float(np.min(dists))
        
    # 2. Defect / Cut detection (Low intensity structures inside high intensity grids)
    cuts = (search_img < 40).astype(np.uint8) * 255
    cy_coords, cx_coords = np.where(cuts > 0)
    
    nearest_cut_dist = 1000.0
    if len(cx_coords) > 0:
        dists = np.hypot(cx_coords - cx, cy_coords - cy)
        nearest_cut_dist = float(np.min(dists))
        
    # 3. Row & Column Spacing to Family Neighbors
    min_dx = 1000.0
    min_dy = 1000.0
    for m in family_members:
        if m["cx"] != cx or m["cy"] != cy:
            dx = abs(cx - m["cx"])
            dy = abs(cy - m["cy"])
            if dx > 1.0:
                min_dx = min(min_dx, dx)
            if dy > 1.0:
                min_dy = min(min_dy, dy)
                
    row_spacing = float(min_dx) if min_dx < 900.0 else 0.0
    col_spacing = float(min_dy) if min_dy < 900.0 else 0.0
    
    # 4. Local Density (Concentric mean intensity)
    y1, y2 = max(0, int(cy - 32)), min(sh, int(cy + 32))
    x1, x2 = max(0, int(cx - 32)), min(sw, int(cx + 32))
    local_patch = search_img[y1:y2, x1:x2]
    local_density = float(np.mean(local_patch)) if local_patch.size > 0 else 0.0
    
    return {
        "nearest_edge_dist": nearest_edge_dist,
        "nearest_cut_dist": nearest_cut_dist,
        "row_spacing": row_spacing,
        "col_spacing": col_spacing,
        "local_density": local_density
    }
