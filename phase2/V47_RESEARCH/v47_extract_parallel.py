import pandas as pd
import numpy as np
import cv2
import os
import sys
from concurrent.futures import ProcessPoolExecutor

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

def process_pair(row):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)

    sys.path.append('phase2')
    sys.path.append('fallbacks')
    from pose_fallback import perform_pose_fallback_search
    
    pair_id = row['pair_id']
    set_type = row['set_type']
    gt_found = row['gt_found']
    gt_x, gt_y = row['gt_x'], row['gt_y']
    
    ref = cv2.imread(os.path.join('data/phase2_dev', row['reference_path']), 0)
    search = cv2.imread(os.path.join('data/phase2_dev', row['search_path']), 0)
    
    pose = perform_pose_fallback_search(ref, search)
    temp_ncc = pose['best_template']
    th, tw = temp_ncc.shape
    
    corr_ncc = pose['corr_plane']
    search_g = get_grad(search)
    temp_g = get_grad(temp_ncc)
    corr_grad = cv2.matchTemplate(search_g, temp_g, cv2.TM_CCOEFF_NORMED)
    
    ctx_w, ctx_h = int(tw * 0.65), int(th * 0.65)
    ctx_x, ctx_y = tw//2 - ctx_w//2, th//2 - ctx_h//2
    temp_ctx = temp_ncc[ctx_y:ctx_y+ctx_h, ctx_x:ctx_x+ctx_w]
    corr_ctx = cv2.matchTemplate(search.astype(np.float32), temp_ctx.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    
    F_s = np.fft.fft2(search.astype(np.float32))
    temp_pad = np.zeros_like(search, dtype=np.float32)
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
    
    def calc_score(cx, cy, in_n, in_g, in_c, in_p):
        px_n = np.clip(int(round(cx - tw/2.0)), 0, corr_ncc.shape[1]-1)
        py_n = np.clip(int(round(cy - th/2.0)), 0, corr_ncc.shape[0]-1)
        px_g = np.clip(int(round(cx - tw/2.0)), 0, corr_grad.shape[1]-1)
        py_g = np.clip(int(round(cy - th/2.0)), 0, corr_grad.shape[0]-1)
        px_c = np.clip(int(round(cx - ctx_w/2.0)), 0, corr_ctx.shape[1]-1)
        py_c = np.clip(int(round(cy - ctx_h/2.0)), 0, corr_ctx.shape[0]-1)
        px_p = np.clip(int(round(cx - tw/2.0)), 0, corr_phase_norm.shape[1]-1)
        py_p = np.clip(int(round(cy - th/2.0)), 0, corr_phase_norm.shape[0]-1)
        
        n = corr_ncc[py_n, px_n]
        g = corr_grad[py_g, px_g]
        c = corr_ctx[py_c, px_c]
        p = corr_phase_norm[py_p, px_p]
        
        cons = sum([in_n, in_g, in_c, in_p])
        bonus = 0.05 if cons == 3 else (0.10 if cons == 4 else 0.0)
        score = 0.35*n + 0.25*g + 0.20*c + 0.20*p + bonus
        return score, cons, n, g, c, p, px_n, py_n, px_g, py_g, px_c, py_c, px_p, py_p
        
    for pc in pool:
        s, cons, n, g, c, p, px_n, py_n, px_g, py_g, px_c, py_c, px_p, py_p = calc_score(
            pc['cx'], pc['cy'], pc.get('in_ncc', False), pc.get('in_grad', False), 
            pc.get('in_ctx', False), pc.get('in_phase', False))
        pc.update({
            'rescue_score': s, 'consensus': cons, 'ncc': n, 'grad': g, 'ctx': c, 'phase': p,
            'px_n': px_n, 'py_n': py_n, 'px_g': px_g, 'py_g': py_g, 
            'px_c': px_c, 'py_c': py_c, 'px_p': px_p, 'py_p': py_p
        })
        
    pool.sort(key=lambda x: x['rescue_score'], reverse=True)
    if len(pool) == 0: return None
    best = pool[0]
    
    cx, cy = best['cx'], best['cy']
    px_n, py_n = best['px_n'], best['py_n']
    
    def get_pct(arr, val): return (arr <= val).mean() * 100.0
    
    p5_ncc, z5_ncc = get_peak_features(corr_ncc, px_n, py_n, 5)
    p10_ncc, z10_ncc = get_peak_features(corr_ncc, px_n, py_n, 10)
    p20_ncc, z20_ncc = get_peak_features(corr_ncc, px_n, py_n, 20)
    p5_grad, z5_grad = get_peak_features(corr_grad, best['px_g'], best['py_g'], 5)
    p10_grad, z10_grad = get_peak_features(corr_grad, best['px_g'], best['py_g'], 10)
    
    c10, d1, d2 = get_density_features(c_ncc, px_n, py_n, 10)
    c20, _, _ = get_density_features(c_ncc, px_n, py_n, 20)
    c40, _, _ = get_density_features(c_ncc, px_n, py_n, 40)
    
    def curve(arr, px, py):
        ch, cw = arr.shape[:2]
        if py < 2 or py >= ch-2 or px < 2 or px >= cw-2: return 0, 0
        cx_ = arr[py, px+1] - 2*arr[py, px] + arr[py, px-1]
        cy_ = arr[py+1, px] - 2*arr[py, px] + arr[py-1, px]
        return cx_, cy_
        
    curve_x, curve_y = curve(corr_ncc, px_n, py_n)
    peak_sharpness = - (curve_x + curve_y)
    
    sh, sw = search.shape
    dist_c = np.hypot(cx - sw/2, cy - sh/2) / (sw/2)
    dist_b = min(cx, sw-cx, cy, sh-cy)
    
    return {
        'pair_id': pair_id, 'set_type': set_type, 'gt_found': gt_found, 'gt_x': gt_x, 'gt_y': gt_y,
        'v46_cx': cx, 'v46_cy': cy, 'v46_score': best['rescue_score'], 'v46_consensus': best['consensus'],
        'ncc': best['ncc'], 'grad': best['grad'], 'ctx': best['ctx'], 'phase': best['phase'],
        'ncc_pct': get_pct(corr_ncc, best['ncc']), 'grad_pct': get_pct(corr_grad, best['grad']),
        'ctx_pct': get_pct(corr_ctx, best['ctx']), 'phase_pct': get_pct(corr_phase_norm, best['phase']),
        'prom5_ncc': p5_ncc, 'prom10_ncc': p10_ncc, 'prom20_ncc': p20_ncc,
        'prom5_grad': p5_grad, 'prom10_grad': p10_grad,
        'z5_ncc': z5_ncc, 'z10_ncc': z10_ncc,
        'comp10': c10, 'comp20': c20, 'comp40': c40, 'd1': d1, 'd2': d2,
        'dist_center': dist_c, 'dist_border': dist_b,
        'curve_x': curve_x, 'curve_y': curve_y, 'sharpness': peak_sharpness
    }

if __name__ == '__main__':
    pairs = pd.read_csv('data/phase2_dev/pairs.csv')
    records = [row for _, row in pairs.iterrows()]
    
    print("Starting V47 Parallel Extraction with 5 workers...")
    out = []
    with ProcessPoolExecutor(max_workers=5) as ex:
        for i, res in enumerate(ex.map(process_pair, records)):
            if res: out.append(res)
            if (i+1) % 10 == 0: print(f"Processed {i+1}/180")
            
    df = pd.DataFrame(out)
    os.makedirs('phase2/V47_RESEARCH/v47_candidate_cache', exist_ok=True)
    df.to_csv('phase2/V47_RESEARCH/v47_candidate_cache/features_raw.csv', index=False)
    
    v25_preds = pd.read_csv('data/phase2_dev/predictions.csv')
    v25_feats = pd.read_csv('phase2/V26_CHAMPIONSHIP/v26_extracted_features.csv')
    v25_feats = v25_feats[v25_feats['queue'] == 'V25'].groupby('pair_id').first().reset_index()
    
    df = df.merge(v25_preds[['pair_id', 'x', 'y', 'found', 'score']], on='pair_id', suffixes=('', '_v25'))
    df = df.merge(v25_feats[['pair_id', 'v25_ml_score', 'corr_score']], on='pair_id', how='left')
    
    df.rename(columns={'x': 'v25_cx', 'y': 'v25_cy', 'found': 'v25_found', 'score': 'v25_presence_score', 'corr_score': 'v25_ncc'}, inplace=True)
    df['dist_v25_v46'] = np.hypot(df['v46_cx'] - df['v25_cx'], df['v46_cy'] - df['v25_cy'])
    df['delta_ncc'] = df['ncc'] - df['v25_ncc'].fillna(0.0)
    
    def get_pop(r):
        if r['gt_found'] == 0: return 'C' # Absent
        d_v25 = np.hypot(r['v25_cx'] - r['gt_x'], r['v25_cy'] - r['gt_y'])
        d_v46 = np.hypot(r['v46_cx'] - r['gt_x'], r['v46_cy'] - r['gt_y'])
        if d_v25 <= 5.0 and r['v25_found'] == 1: return 'A' # V25 correct
        if d_v25 > 5.0 and d_v46 <= 5.0: return 'B' # Gold rescues
        return 'D' # Other
        
    df['pop'] = df.apply(get_pop, axis=1)
    
    df.to_csv('phase2/V47_RESEARCH/v47_candidate_cache/features.csv', index=False)
    print("Saved cache to phase2/V47_RESEARCH/v47_candidate_cache/features.csv")
