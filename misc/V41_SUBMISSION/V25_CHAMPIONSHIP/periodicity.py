import numpy as np
import cv2
from scipy.signal import find_peaks

def estimate_periodicity_from_corr(corr_plane: np.ndarray, min_pitch=10, max_pitch=200):
    # Find local maxima in the corr_plane
    # We can just threshold the corr_plane to top 1% and look at distances between them
    
    from scipy.ndimage import maximum_filter
    threshold = np.max(corr_plane) * 0.7
    local_max = maximum_filter(corr_plane, size=5) == corr_plane
    y_coords, x_coords = np.where((corr_plane > threshold) & local_max)
    
    if len(x_coords) > 500:
        # Take the top 500 by correlation value
        vals = corr_plane[y_coords, x_coords]
        top_idx = np.argsort(vals)[-500:]
        y_coords = y_coords[top_idx]
        x_coords = x_coords[top_idx]
        
    if len(x_coords) < 2:
        return {'pitch_x': 0, 'pitch_y': 0, 'strength_x': 0.0, 'strength_y': 0.0, 'mode': 'NON_PERIODIC'}
        
    # Compute pairwise distances
    dx = np.abs(x_coords[:, None] - x_coords[None, :])
    dy = np.abs(y_coords[:, None] - y_coords[None, :])
    
    # Flatten and filter out 0
    dx = dx.flatten()
    dy = dy.flatten()
    
    mask_x = (dx >= min_pitch) & (dx <= max_pitch) & (dy <= 5) # purely horizontal neighbors
    mask_y = (dy >= min_pitch) & (dy <= max_pitch) & (dx <= 5) # purely vertical neighbors
    
    pitch_x = 0
    strength_x = 0.0
    if np.any(mask_x):
        hist, bin_edges = np.histogram(dx[mask_x], bins=np.arange(min_pitch, max_pitch+2))
        best_bin = np.argmax(hist)
        pitch_x = int(bin_edges[best_bin])
        strength_x = hist[best_bin] / len(x_coords)
        
    pitch_y = 0
    strength_y = 0.0
    if np.any(mask_y):
        hist, bin_edges = np.histogram(dy[mask_y], bins=np.arange(min_pitch, max_pitch+2))
        best_bin = np.argmax(hist)
        pitch_y = int(bin_edges[best_bin])
        strength_y = hist[best_bin] / len(y_coords)
        
    mode = "NON_PERIODIC"
    if strength_x > 0.5 or strength_y > 0.5:
        mode = "WEAK"
    if strength_x > 1.5 or strength_y > 1.5:
        mode = "STRONG"
        
    return {
        "pitch_x": pitch_x,
        "pitch_y": pitch_y,
        "strength_x": strength_x,
        "strength_y": strength_y,
        "mode": mode
    }

if __name__ == '__main__':
    import sys
    sys.path.append('fallbacks')
    from pose_fallback import perform_pose_fallback_search
    ref = cv2.imread('data/phase2_dev/reference/pair_000.png', 0)
    search = cv2.imread('data/phase2_dev/search/pair_000.png', 0)
    pose = perform_pose_fallback_search(ref, search)
    print(estimate_periodicity_from_corr(pose['corr_plane']))
