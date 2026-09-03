import pandas as pd
import numpy as np
import cv2
import os
import sys

sys.path.append('phase2')
sys.path.append('fallbacks')
from pose_fallback import perform_pose_fallback_search

pairs_df = pd.read_csv('data/phase2_dev/pairs.csv')
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
    if np.isnan(gt_x) or np.isnan(gt_y): return 99
    d = np.hypot(cx - gt_x, cy - gt_y)
    return d

def categorize_loc(d):
    if d <= 1.0: return '<=1'
    if d <= 2.0: return '<=2'
    if d <= 5.0: return '<=5'
    return '>5'

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

results = {g: {'rescued': 0, 'broken': 0, 'unchanged': 0, 'A_gain': 0, 'B_gain': 0, 'new_absent_fp': 0, 'absent_fp': 0} for g in gates}
detailed_audit = []

print("Running V46-D FULL 180 CLEAN-ROOM Evaluation...")
count = 0

for idx, row in pairs_df.iterrows():
    pair_id = row['pair_id']
    set_type = row['set_type'] # 'SetA', 'SetB', 'SetC'
    ref = cv2.imread(os.path.join('data/phase2_dev', row['reference_path']), 0)
    search = cv2.imread(os.path.join('data/phase2_dev', row['search_path']), 0)
    gt_x, gt_y, gt_found = row['gt_x'], row['gt_y'], row['gt_found']
    
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
                pool.append({'cx': cx, 'cy': cy, 'px': px, 'py': py, src: True})
                
    add_to_pool(c_ncc, tw, th, 'in_ncc')
    add_to_pool(c_grad, tw, th, 'in_grad')
    add_to_pool(c_ctx, ctx_w, ctx_h, 'in_ctx')
    add_to_pool(c_phase, tw, th, 'in_phase')
    
    v25_cands = v25_features[(v25_features['pair_id'] == pair_id) & (v25_features['queue'] == 'V25')]
    v25_pred = v25_preds[v25_preds['pair_id'] == pair_id].iloc[0]
    if len(v25_cands) > 0: ml_score = v25_cands.iloc[0]['v25_ml_score']
    else: ml_score = 0.0
        
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
                                         pc.get('in_ncc', False), pc.get('in_grad', False), 
                                         pc.get('in_ctx', False), pc.get('in_phase', False))
        pc['rescue_score'] = score
        pc['consensus'] = cons
        pc['strong_signals'] = strong
        
    pool.sort(key=lambda x: x['rescue_score'], reverse=True)
    best_v46 = pool[0] if len(pool) > 0 else None
    
    v25_dist = eval_loc(v25_cx, v25_cy, gt_x, gt_y)
    v25_is_correct = v25_dist <= 5.0 and gt_found == 1
    
    gate_outcomes = {}
    for g_name, g_func in gates.items():
        v46_used = False
        v46_found = v25_found
        v46_cx, v46_cy = v25_cx, v25_cy
        
        if best_v46 is not None:
            if g_name != 'Protected_Tiers':
                if ml_score < 0.90 or v25_found == 0:
                    if g_func(best_v46, v25_info, ml_score): v46_used = True
            else:
                if g_func(best_v46, v25_info, ml_score): v46_used = True
                    
        if v46_used:
            v46_cx, v46_cy = best_v46['cx'], best_v46['cy']
            v46_found = 1 # if we rescue, we assert presence
            
        gate_outcomes[g_name] = {'cx': v46_cx, 'cy': v46_cy, 'found': v46_found, 'used': v46_used}
        
        if gt_found == 1:
            v46_dist = eval_loc(v46_cx, v46_cy, gt_x, gt_y)
            v46_is_correct = v46_dist <= 5.0
            
            if v46_used:
                if v46_is_correct and not v25_is_correct:
                    results[g_name]['rescued'] += 1
                    if set_type == 'SetA': results[g_name]['A_gain'] += 1
                    if set_type == 'SetB': results[g_name]['B_gain'] += 1
                elif not v46_is_correct and v25_is_correct:
                    results[g_name]['broken'] += 1
                    if set_type == 'SetA': results[g_name]['A_gain'] -= 1
                    if set_type == 'SetB': results[g_name]['B_gain'] -= 1
                else: results[g_name]['unchanged'] += 1
            else: results[g_name]['unchanged'] += 1
        else:
            # Absent pair
            # FP occurs if found == 1. 
            if v46_found == 1:
                results[g_name]['absent_fp'] += 1
                if v25_found == 0:
                    results[g_name]['new_absent_fp'] += 1
                    
    # Log detailed audit for C_margin_0.10
    g_res = gate_outcomes['C_margin_0.10']
    v46_dist = eval_loc(g_res['cx'], g_res['cy'], gt_x, gt_y)
    detailed_audit.append({
        'pair_id': pair_id, 'set_type': set_type, 'gt_found': gt_found,
        'v25_found': v25_found, 'v25_ml': ml_score, 'v25_score': v25_score, 'v25_dist': v25_dist,
        'v46_triggered': g_res['used'],
        'v46_consensus': best_v46['consensus'] if best_v46 else 0,
        'v46_score': best_v46['rescue_score'] if best_v46 else 0,
        'v46_dist': v46_dist if gt_found == 1 else 99,
        'final_found': g_res['found']
    })
            
    count += 1
    if count % 10 == 0: print(f"Processed {count}/{len(pairs_df)}")

# Generate Detailed Metrics
df_audit = pd.DataFrame(detailed_audit)
df_audit.to_csv('phase2/V46_RESEARCH/v46_d_pair_audit.csv', index=False)
pd.DataFrame(results).T.to_csv('phase2/V46_RESEARCH/v46_d_gate_results.csv')

def summarize_loc(df, target_found=1):
    sub = df[df['gt_found'] == target_found]
    v25_cats = sub['v25_dist'].apply(categorize_loc).value_counts()
    v46_cats = sub['v46_dist'].apply(categorize_loc).value_counts()
    return v25_cats, v46_cats

v25_loc, v46_loc = summarize_loc(df_audit, 1)
v25_loc_A, v46_loc_A = summarize_loc(df_audit[df_audit['set_type']=='SetA'], 1)
v25_loc_B, v46_loc_B = summarize_loc(df_audit[df_audit['set_type']=='SetB'], 1)

# Rejection stats for C_margin_0.10
tp = len(df_audit[(df_audit['gt_found'] == 1) & (df_audit['final_found'] == 1)])
fn = len(df_audit[(df_audit['gt_found'] == 1) & (df_audit['final_found'] == 0)])
fp = len(df_audit[(df_audit['gt_found'] == 0) & (df_audit['final_found'] == 1)])
tn = len(df_audit[(df_audit['gt_found'] == 0) & (df_audit['final_found'] == 0)])

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

v25_tp = len(df_audit[(df_audit['gt_found'] == 1) & (df_audit['v25_found'] == 1)])
v25_fp = len(df_audit[(df_audit['gt_found'] == 0) & (df_audit['v25_found'] == 1)])
v25_fn = len(df_audit[(df_audit['gt_found'] == 1) & (df_audit['v25_found'] == 0)])
v25_p = v25_tp / (v25_tp + v25_fp) if (v25_tp + v25_fp) > 0 else 0
v25_r = v25_tp / (v25_tp + v25_fn) if (v25_tp + v25_fn) > 0 else 0
v25_f1 = 2 * v25_p * v25_r / (v25_p + v25_r) if (v25_p + v25_r) > 0 else 0

rep = f"# V46-D CLEAN-ROOM REPORT\n\n"
rep += "## 1. Localization Tiers (Primary Gate: C_margin_0.10)\n"
rep += "| Tier | V25 | V46 |\n|---|---|---|\n"
for t in ['<=1', '<=2', '<=5', '>5']:
    rep += f"| {t} | {v25_loc.get(t,0)} | {v46_loc.get(t,0)} |\n"
rep += "\n## 2. Set A Localization\n"
for t in ['<=1', '<=2', '<=5', '>5']:
    rep += f"| {t} | {v25_loc_A.get(t,0)} | {v46_loc_A.get(t,0)} |\n"
rep += "\n## 3. Set B Localization\n"
for t in ['<=1', '<=2', '<=5', '>5']:
    rep += f"| {t} | {v25_loc_B.get(t,0)} | {v46_loc_B.get(t,0)} |\n"
    
rep += f"\n## 4. Rejection (Primary Gate: C_margin_0.10)\n"
rep += f"- V25: TP={v25_tp} FP={v25_fp} FN={v25_fn} TN={40-v25_fp} | F1={v25_f1:.3f}\n"
rep += f"- V46: TP={tp} FP={fp} FN={fn} TN={tn} | F1={f1:.3f}\n"
rep += f"- New Absent False Accepts: {results['C_margin_0.10']['new_absent_fp']}\n"

rep += "\n## 5. Gate Ablations\n"
rep += "Gate | Rescued | Broken | Net | A gain | B gain | New Absent FP | Total Absent FP\n"
rep += "---|---|---|---|---|---|---|---\n"
for g in gates:
    r = results[g]
    net = r['rescued'] - r['broken']
    rep += f"{g} | {r['rescued']} | {r['broken']} | {net} | {r['A_gain']} | {r['B_gain']} | {r['new_absent_fp']} | {r['absent_fp']}\n"

with open('phase2/V46_RESEARCH/V46_D_CLEANROOM_REPORT.md', 'w') as f: f.write(rep)
print(rep)
