import numpy as np
import cv2
import pickle
import os
import sys

# Load V47 model
v47_path = os.path.join(os.path.dirname(__file__), 'v47_hgb2.pkl')
if os.path.exists(v47_path):
    with open(v47_path, 'rb') as f:
        v47_data = pickle.load(f)
    v47_model = v47_data['model']
    v47_features = v47_data['features']
else:
    v47_model = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'fallbacks')))
try:
    from pose_fallback import perform_pose_fallback_search
    from pose_refinement import refine_pose
except:
    pass

def get_grad(img):
    img_f = img.astype(np.float32) / 255.0
    gx = cv2.Scharr(img_f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_f, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)

def extract_nms(corr_plane, max_k=200, r=5):
    ch, cw = corr_plane.shape[:2]
    work = corr_plane.copy()
    cands = []
    for _ in range(max_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= -99.0 or np.isnan(max_val): break
        px, py = max_loc
        cands.append({'px': px, 'py': py, 'score': max_val})
        y1, y2 = max(0, py - r), min(ch, py + r + 1)
        x1, x2 = max(0, px - r), min(cw, px + r + 1)
        work[y1:y2, x1:x2] = -999.0
    return cands

def get_peak_features(arr, px, py, r):
    ch, cw = arr.shape[:2]
    y1, y2 = max(0, py - r), min(ch, py + r + 1)
    x1, x2 = max(0, px - r), min(cw, px + r + 1)
    patch = arr[y1:y2, x1:x2]
    if patch.size == 0: return 0.0, 0.0
    val = arr[py, px]
    mean_val = np.mean(patch)
    std_val = np.std(patch)
    prom = val - mean_val
    z = prom / (std_val + 1e-6)
    return prom, z

def get_density_features(cands, px, py, rad):
    cx, cy = px, py
    count = 0
    dists = []
    for c in cands:
        dc = np.hypot(c['px'] - cx, c['py'] - cy)
        if dc > 0:
            dists.append(dc)
            if dc <= rad: count += 1
    dists.sort()
    d1 = dists[0] if len(dists) > 0 else 999.0
    d2 = dists[1] if len(dists) > 1 else 999.0
    return count, d1, d2

def validate(v25_result, ref_img, search_img):
    if v47_model is None:
        return v25_result
        
    v25_found = v25_result['found']
    v25_score = v25_result['score']
    
    # Fast exit if V25 is very confident
    if v25_found == 1 and v25_score >= 0.95:
        return v25_result
        
    pose = perform_pose_fallback_search(ref_img, search_img)
    temp_ncc = pose['best_template']
    th, tw = temp_ncc.shape
    corr_ncc = pose['corr_plane']
    
    search_g = get_grad(search_img)
    temp_g = get_grad(temp_ncc)
    corr_grad = cv2.matchTemplate(search_g, temp_g, cv2.TM_CCOEFF_NORMED)
    
    ctx_w, ctx_h = int(tw * 0.65), int(th * 0.65)
    ctx_x, ctx_y = tw//2 - ctx_w//2, th//2 - ctx_h//2
    temp_ctx = temp_ncc[ctx_y:ctx_y+ctx_h, ctx_x:ctx_x+ctx_w]
    corr_ctx = cv2.matchTemplate(search_img.astype(np.float32), temp_ctx.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    
    F_s = np.fft.fft2(search_img.astype(np.float32))
    temp_pad = np.zeros_like(search_img, dtype=np.float32)
    temp_pad[:th, :tw] = temp_ncc.astype(np.float32)
    F_t = np.fft.fft2(temp_pad)
    R = F_s * np.conjugate(F_t)
    R /= (np.abs(R) + 1e-5)
    corr_phase = np.fft.ifft2(R).real
    
    p_min, p_max, _, _ = cv2.minMaxLoc(corr_phase)
    if p_max > p_min: corr_phase_norm = (corr_phase - p_min) / (p_max - p_min)
    else: corr_phase_norm = corr_phase
    
    c_ncc = extract_nms(corr_ncc, max_k=200)
    c_grad = extract_nms(corr_grad, max_k=200)
    c_ctx = extract_nms(corr_ctx, max_k=200)
    c_phase = extract_nms(corr_phase, max_k=200)
    
    pool = []
    def add_to_pool(clist, w, h, src):
        for c in clist:
            cx = c['px'] + w/2.0
            cy = c['py'] + h/2.0
            matched = False
            for pc in pool:
                if np.hypot(pc['cx'] - cx, pc['cy'] - cy) <= 5.0:
                    pc[src] = True
                    matched = True
                    break
            if not matched:
                pool.append({'cx': cx, 'cy': cy, 'px_n': int(c['px']), 'py_n': int(c['py']), src: True})
                
    add_to_pool(c_ncc, tw, th, 'in_ncc')
    add_to_pool(c_grad, tw, th, 'in_grad')
    add_to_pool(c_ctx, ctx_w, ctx_h, 'in_ctx')
    add_to_pool(c_phase, tw, th, 'in_phase')
    
    for pc in pool:
        px_n = np.clip(int(round(pc['cx'] - tw/2.0)), 0, corr_ncc.shape[1]-1)
        py_n = np.clip(int(round(pc['cy'] - th/2.0)), 0, corr_ncc.shape[0]-1)
        px_g = np.clip(int(round(pc['cx'] - tw/2.0)), 0, corr_grad.shape[1]-1)
        py_g = np.clip(int(round(pc['cy'] - th/2.0)), 0, corr_grad.shape[0]-1)
        px_c = np.clip(int(round(pc['cx'] - ctx_w/2.0)), 0, corr_ctx.shape[1]-1)
        py_c = np.clip(int(round(pc['cy'] - ctx_h/2.0)), 0, corr_ctx.shape[0]-1)
        px_p = np.clip(int(round(pc['cx'] - tw/2.0)), 0, corr_phase_norm.shape[1]-1)
        py_p = np.clip(int(round(pc['cy'] - th/2.0)), 0, corr_phase_norm.shape[0]-1)
        
        n = corr_ncc[py_n, px_n]
        g = corr_grad[py_g, px_g]
        c = corr_ctx[py_c, px_c]
        p = corr_phase_norm[py_p, px_p]
        
        cons = sum([pc.get('in_ncc', False), pc.get('in_grad', False), pc.get('in_ctx', False), pc.get('in_phase', False)])
        bonus = 0.05 if cons == 3 else (0.10 if cons == 4 else 0.0)
        score = 0.35*n + 0.25*g + 0.20*c + 0.20*p + bonus
        pc.update({
            'rescue_score': score, 'consensus': cons, 'ncc': n, 'grad': g, 'ctx': c, 'phase': p,
            'px_n': px_n, 'py_n': py_n, 'px_g': px_g, 'py_g': py_g, 
            'px_c': px_c, 'py_c': py_c, 'px_p': px_p, 'py_p': py_p
        })
        
    pool.sort(key=lambda x: x['rescue_score'], reverse=True)
    if len(pool) == 0: return v25_result
    best = pool[0]
    
    if best['consensus'] < 3: return v25_result
    
    cx, cy = best['cx'], best['cy']
    px_n, py_n = best['px_n'], best['py_n']
    
    def get_pct(arr, val): return (arr <= val).mean() * 100.0
    
    p5_ncc, z5_ncc = get_peak_features(corr_ncc, px_n, py_n, 5)
    p10_ncc, z10_ncc = get_peak_features(corr_ncc, px_n, py_n, 10)
    p20_ncc, z20_ncc = get_peak_features(corr_ncc, px_n, py_n, 20)
    
    c10, d1, d2 = get_density_features(c_ncc, px_n, py_n, 10)
    c20, _, _ = get_density_features(c_ncc, px_n, py_n, 20)
    
    def curve(arr, px, py):
        ch, cw = arr.shape[:2]
        if py < 2 or py >= ch-2 or px < 2 or px >= cw-2: return 0, 0
        cx_ = arr[py, px+1] - 2*arr[py, px] + arr[py, px-1]
        cy_ = arr[py+1, px] - 2*arr[py, px] + arr[py-1, px]
        return cx_, cy_
        
    curve_x, curve_y = curve(corr_ncc, px_n, py_n)
    peak_sharpness = - (curve_x + curve_y)
    
    sh, sw = search_img.shape
    dist_c = np.hypot(cx - sw/2, cy - sh/2) / (sw/2)
    dist_b = min(cx, sw-cx, cy, sh-cy)
    
    # Get v25 raw ncc (v25_result might not have it, but we can sample it)
    px_v25 = np.clip(int(round(v25_result['x'] - tw/2.0)), 0, corr_ncc.shape[1]-1)
    py_v25 = np.clip(int(round(v25_result['y'] - th/2.0)), 0, corr_ncc.shape[0]-1)
    v25_ncc = corr_ncc[py_v25, px_v25] if v25_result['found']==1 else 0.0
    
    feat_dict = {
        'ncc': best['ncc'], 'grad': best['grad'], 'ctx': best['ctx'], 'phase': best['phase'],
        'ncc_pct': get_pct(corr_ncc, best['ncc']), 'grad_pct': get_pct(corr_grad, best['grad']),
        'prom5_ncc': p5_ncc, 'prom10_ncc': p10_ncc, 'prom20_ncc': p20_ncc,
        'z5_ncc': z5_ncc, 'z10_ncc': z10_ncc,
        'comp10': c10, 'comp20': c20, 'd1': d1, 'd2': d2,
        'dist_center': dist_c, 'dist_border': dist_b,
        'sharpness': peak_sharpness,
        'delta_ncc': best['ncc'] - v25_ncc,
        'dist_v25_v46': np.hypot(cx - v25_result['x'], cy - v25_result['y'])
    }
    
    x_vec = np.array([[feat_dict[f] for f in v47_features]])
    prob = v47_model.predict_proba(x_vec)[0, 1]
    
    threshold = 0.10
    
    if prob >= threshold:
        rx, ry, _, _ = refine_pose(ref_img, search_img, pose['best_scale'], pose['best_theta'], px_n, py_n, corr_ncc)
        # Apply V41 calibration (if we accepted this V46 rescue)
        # cal_score = 0.90 * raw_score + 0.05 * top1_score + 0.05 * top1_corr
        cal_score = 0.90 * prob + 0.05 * prob + 0.05 * best['ncc']
        
        return {
            'x': rx,
            'y': ry,
            'theta': pose['best_theta'],
            'scale': pose['best_scale'],
            'found': 1,
            'score': cal_score
        }
        
    # If not overridden, but we need V41 calibration on V25
    if v25_found == 1:
        # cal_score = 0.90 * raw_score + 0.05 * top1_score + 0.05 * top1_corr
        # since we don't have top1_score easily, we approximate using v25_score
        cal_score = 0.90 * v25_score + 0.05 * v25_score + 0.05 * v25_ncc
        v25_result['score'] = cal_score
        
    return v25_result
