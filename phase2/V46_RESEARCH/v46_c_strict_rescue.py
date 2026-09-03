import pandas as pd
import numpy as np
import cv2
import os
import sys

sys.path.append('phase2')
sys.path.append('fallbacks')
from pose_fallback import perform_pose_fallback_search

pairs_df = pd.read_csv('data/phase2_dev/pairs.csv')
pairs_df = pairs_df[pairs_df['gt_found'] == 1]

v25_preds = pd.read_csv('data/phase2_dev/predictions.csv')
v25_features = pd.read_csv('phase2/V26_CHAMPIONSHIP/v26_extracted_features.csv')

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

def eval_loc(cx, cy, gt_x, gt_y):
    d = np.hypot(cx - gt_x, cy - gt_y)
    if d <= 1.0: return 1
    if d <= 2.0: return 2
    if d <= 5.0: return 5
    return 99

gates = {
    'A_3_of_4': lambda c, v25, m: c['consensus'] >= 3,
    'B_3_of_4_strength': lambda c, v25, m: c['consensus'] >= 3 and c['strong_signals'] >= 2,
    'C_margin_0.00': lambda c, v25, m: c['consensus'] >= 3 and c['rescue_score'] > v25['rescue_score'] + 0.00,
    'C_margin_0.02': lambda c, v25, m: c['consensus'] >= 3 and c['rescue_score'] > v25['rescue_score'] + 0.02,
    'C_margin_0.05': lambda c, v25, m: c['consensus'] >= 3 and c['rescue_score'] > v25['rescue_score'] + 0.05,
    'C_margin_0.10': lambda c, v25, m: c['consensus'] >= 3 and c['rescue_score'] > v25['rescue_score'] + 0.10,
    'C_margin_0.15': lambda c, v25, m: c['consensus'] >= 3 and c['rescue_score'] > v25['rescue_score'] + 0.15,
    'D_4_of_4': lambda c, v25, m: c['consensus'] == 4,
    'Protected_Tiers': lambda c, v25, m: evaluate_protected(c, v25, m)
}

def evaluate_protected(c, v25, m):
    if v25['ml'] >= 0.95: return False
    if v25['ml'] >= 0.90: return c['consensus'] == 4
    if v25['found'] == 0:
        return c['consensus'] >= 3 and c['rescue_score'] > v25['rescue_score'] + 0.10
    return c['consensus'] >= 3 and c['strong_signals'] >= 2

results = {g: {'rescued': 0, 'broken': 0, 'unchanged': 0, 'A_gain': 0, 'B_gain': 0} for g in gates}

print("Running V46-C Strict Rescue Evaluation on 140 pairs...")
count = 0

for idx, row in pairs_df.iterrows():
    pair_id = row['pair_id']
    set_type = row['set_type']
    ref = cv2.imread(os.path.join('data/phase2_dev', row['reference_path']), 0)
    search = cv2.imread(os.path.join('data/phase2_dev', row['search_path']), 0)
    gt_x, gt_y = row['gt_x'], row['gt_y']
    
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
            px, py = c['px'], c['py']
            cx = px + w/2.0
            cy = py + h/2.0
            matched = False
            for pc in pool:
                if np.hypot(pc['cx'] - cx, pc['cy'] - cy) <= 5.0:
                    pc[src] = True
                    matched = True
                    break
            if not matched:
                pool.append({'cx': cx, 'cy': cy, src: True})
                
    add_to_pool(c_ncc, tw, th, 'in_ncc')
    add_to_pool(c_grad, tw, th, 'in_grad')
    add_to_pool(c_ctx, ctx_w, ctx_h, 'in_ctx')
    add_to_pool(c_phase, tw, th, 'in_phase')
    
    v25_cands = v25_features[(v25_features['pair_id'] == pair_id) & (v25_features['queue'] == 'V25')]
    v25_pred = v25_preds[v25_preds['pair_id'] == pair_id].iloc[0]
    if len(v25_cands) > 0:
        ml_score = v25_cands.iloc[0]['v25_ml_score']
    else:
        ml_score = 0.0
        
    v25_found = v25_pred['found']
    v25_cx = v25_pred['x']
    v25_cy = v25_pred['y']
    
    def calc_score(cx, cy, in_n=False, in_g=False, in_c=False, in_p=False):
        px_n = np.clip(int(round(cx - tw/2.0)), 0, corr_ncc.shape[1]-1)
        py_n = np.clip(int(round(cy - th/2.0)), 0, corr_ncc.shape[0]-1)
        n = corr_ncc[py_n, px_n]
        
        px_g = np.clip(int(round(cx - tw/2.0)), 0, corr_grad.shape[1]-1)
        py_g = np.clip(int(round(cy - th/2.0)), 0, corr_grad.shape[0]-1)
        g = corr_grad[py_g, px_g]
        
        px_c = np.clip(int(round(cx - ctx_w/2.0)), 0, corr_ctx.shape[1]-1)
        py_c = np.clip(int(round(cy - ctx_h/2.0)), 0, corr_ctx.shape[0]-1)
        c = corr_ctx[py_c, px_c]
        
        px_p = np.clip(int(round(cx - tw/2.0)), 0, corr_phase_norm.shape[1]-1)
        py_p = np.clip(int(round(cy - th/2.0)), 0, corr_phase_norm.shape[0]-1)
        p = corr_phase_norm[py_p, px_p]
        
        cons = sum([in_n, in_g, in_c, in_p])
        bonus = 0.05 if cons == 3 else (0.10 if cons == 4 else 0.0)
        score = 0.35*n + 0.25*g + 0.20*c + 0.20*p + bonus
        
        strong = sum(1 for val in [n, g, c, p] if val >= 0.50)
        return score, cons, strong
        
    v25_score, v25_cons, _ = calc_score(v25_cx, v25_cy)
    v25_info = {'rescue_score': v25_score, 'ml': ml_score, 'found': v25_found}
    
    for pc in pool:
        score, cons, strong = calc_score(pc['cx'], pc['cy'], 
                                         pc.get('in_ncc', False), 
                                         pc.get('in_grad', False), 
                                         pc.get('in_ctx', False), 
                                         pc.get('in_phase', False))
        pc['rescue_score'] = score
        pc['consensus'] = cons
        pc['strong_signals'] = strong
        
    pool.sort(key=lambda x: x['rescue_score'], reverse=True)
    best_v46 = pool[0] if len(pool) > 0 else None
    
    v25_loc = eval_loc(v25_cx, v25_cy, gt_x, gt_y)
    v25_is_correct = v25_loc <= 5
    
    for g_name, g_func in gates.items():
        v46_used = False
        if best_v46 is not None:
            if g_name != 'Protected_Tiers':
                if ml_score < 0.90 or v25_found == 0:
                    if g_func(best_v46, v25_info, ml_score):
                        v46_used = True
            else:
                if g_func(best_v46, v25_info, ml_score):
                    v46_used = True
                    
        if v46_used:
            v46_loc = eval_loc(best_v46['cx'], best_v46['cy'], gt_x, gt_y)
            v46_is_correct = v46_loc <= 5
            
            if v46_is_correct and not v25_is_correct:
                results[g_name]['rescued'] += 1
                if set_type == 'Set A': results[g_name]['A_gain'] += 1
                if set_type == 'Set B': results[g_name]['B_gain'] += 1
            elif not v46_is_correct and v25_is_correct:
                results[g_name]['broken'] += 1
                if set_type == 'Set A': results[g_name]['A_gain'] -= 1
                if set_type == 'Set B': results[g_name]['B_gain'] -= 1
            else:
                results[g_name]['unchanged'] += 1
        else:
            results[g_name]['unchanged'] += 1
            
    count += 1
    if count % 10 == 0:
        print(f"Processed {count}/{len(pairs_df)}")

print("\n--- V46-C RESULTS ---")
rep = "Gate\tRescued\tBroken\tNet\tA gain\tB gain\n"
for g in gates:
    r = results[g]
    net = r['rescued'] - r['broken']
    rep += f"{g}\t{r['rescued']}\t{r['broken']}\t{net}\t{r['A_gain']}\t{r['B_gain']}\n"
    
print(rep)
with open('phase2/V46_RESEARCH/V46_C_GATE_REPORT.md', 'w') as f:
    f.write(rep)

pd.DataFrame(results).T.to_csv('phase2/V46_RESEARCH/v46_gate_results.csv')
