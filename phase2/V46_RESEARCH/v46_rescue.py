import pandas as pd
import numpy as np
import cv2
import os
import sys

sys.path.append('phase2')
sys.path.append('fallbacks')

from pose_fallback import perform_pose_fallback_search

# Load V25 Predictions
v25_preds = pd.read_csv('data/phase2_dev/predictions.csv')
v25_features = pd.read_csv('phase2/V26_CHAMPIONSHIP/v26_extracted_features.csv')
pairs_df = pd.read_csv('data/phase2_dev/pairs.csv')

# Only focus on present pairs
pairs_df = pairs_df[pairs_df['gt_found'] == 1]

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

def evaluate_cands(cands_list, w, h, gt_x, gt_y):
    gt_rank = 999
    gt_score = 0.0
    for i, c in enumerate(cands_list):
        cx = c['px'] + w/2.0
        cy = c['py'] + h/2.0
        if np.hypot(cx - gt_x, cy - gt_y) <= 8.0:
            gt_rank = i + 1
            gt_score = c['score']
            break
    return gt_rank, gt_score

rescued_cases = []
broken_cases = []
new_retrievals = []
all_results = []

print(f"Evaluating {len(pairs_df)} pairs for V46-B Rescue...")

for idx, row in pairs_df.iterrows():
    pair_id = row['pair_id']
    ref = cv2.imread(os.path.join('data/phase2_dev', row['reference_path']), 0)
    search = cv2.imread(os.path.join('data/phase2_dev', row['search_path']), 0)
    gt_x, gt_y = row['gt_x'], row['gt_y']
    
    pose = perform_pose_fallback_search(ref, search)
    temp_ncc = pose['best_template']
    th, tw = temp_ncc.shape
    
    # 1. NCC
    corr_ncc = pose['corr_plane']
    c_ncc = extract_nms(corr_ncc, max_k=200)
    
    # 2. Gradient
    search_g = get_grad(search)
    temp_g = get_grad(temp_ncc)
    corr_grad = cv2.matchTemplate(search_g, temp_g, cv2.TM_CCOEFF_NORMED)
    c_grad = extract_nms(corr_grad, max_k=200)
    
    # 3. Context (65% center crop of template)
    ctx_w, ctx_h = int(tw * 0.65), int(th * 0.65)
    ctx_x, ctx_y = tw//2 - ctx_w//2, th//2 - ctx_h//2
    temp_ctx = temp_ncc[ctx_y:ctx_y+ctx_h, ctx_x:ctx_x+ctx_w]
    corr_ctx = cv2.matchTemplate(search.astype(np.float32), temp_ctx.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    c_ctx = extract_nms(corr_ctx, max_k=200)
    
    # 4. Phase
    F_s = np.fft.fft2(search.astype(np.float32))
    temp_pad = np.zeros_like(search, dtype=np.float32)
    temp_pad[:th, :tw] = temp_ncc.astype(np.float32)
    F_t = np.fft.fft2(temp_pad)
    R = F_s * np.conjugate(F_t)
    R /= (np.abs(R) + 1e-5)
    corr_phase = np.fft.ifft2(R).real
    c_phase = extract_nms(corr_phase, max_k=200)
    
    # Assess GT retrieval
    rank_ncc, _ = evaluate_cands(c_ncc, tw, th, gt_x, gt_y)
    rank_grad, _ = evaluate_cands(c_grad, tw, th, gt_x, gt_y)
    rank_ctx, _ = evaluate_cands(c_ctx, ctx_w, ctx_h, gt_x, gt_y)
    rank_phase, _ = evaluate_cands(c_phase, tw, th, gt_x, gt_y)
    
    has_union = (rank_ncc <= 200) or (rank_grad <= 200) or (rank_ctx <= 200) or (rank_phase <= 200)
    
    # Check V25 Top200
    v25_cands = v25_features[(v25_features['pair_id'] == pair_id) & (v25_features['queue'] == 'V25')]
    in_v25 = any(c['is_correct'] == 1 for _, c in v25_cands.iterrows())
    
    if has_union and not in_v25:
        new_retrievals.append({
            'pair_id': pair_id,
            'rank_ncc': rank_ncc,
            'rank_grad': rank_grad,
            'rank_ctx': rank_ctx,
            'rank_phase': rank_phase
        })
        
    # --- RESCUE LOGIC ---
    v25_pred = v25_preds[v25_preds['pair_id'] == pair_id].iloc[0]
    v25_is_correct = (np.hypot(v25_pred['x'] - gt_x, v25_pred['y'] - gt_y) <= 8.0)
    
    v25_uncertain = False
    if v25_pred['found'] == 0:
        v25_uncertain = True
    else:
        top1_f = v25_cands.iloc[0]
        # Low margin or low V25 score = uncertain
        if top1_f['v25_ml_score'] < 0.90:
            v25_uncertain = True
            
    rescued = False
    if v25_uncertain:
        # Build union candidate pool
        pool = []
        for src, clist, cw, ch in [('ncc', c_ncc, tw, th), ('grad', c_grad, tw, th), 
                                   ('ctx', c_ctx, ctx_w, ctx_h), ('phase', c_phase, tw, th)]:
            for c in clist:
                cx, cy = c['px'] + cw/2.0, c['py'] + ch/2.0
                # Find if close to existing
                found_match = False
                for pc in pool:
                    if np.hypot(cx - pc['cx'], cy - pc['cy']) <= 5.0:
                        pc[src] = c['score']
                        found_match = True
                        break
                if not found_match:
                    pool.append({'cx': cx, 'cy': cy, src: c['score']})
                    
        # Verification Gate
        # candidate must have >= 2 signals
        for pc in pool:
            sigs = sum(1 for src in ['ncc', 'grad', 'ctx', 'phase'] if src in pc)
            if sigs >= 2:
                # Need to beat V25 winner?
                # For now, just say if it has >= 3 signals it overrides!
                if sigs >= 3:
                    rescued = True
                    is_correct_rescue = (np.hypot(pc['cx'] - gt_x, pc['cy'] - gt_y) <= 8.0)
                    if is_correct_rescue and not v25_is_correct:
                        rescued_cases.append(pair_id)
                    elif not is_correct_rescue and v25_is_correct:
                        broken_cases.append(pair_id)
                    break
                    
    all_results.append({
        'pair_id': pair_id,
        'in_v25': in_v25,
        'has_union': has_union
    })

pd.DataFrame(new_retrievals).to_csv('phase2/V46_RESEARCH/new_retrievals.csv', index=False)

rep = f"V46-B RESCUE RESULTS\n"
rep += f"New +10 Retrievals:\n"
for n in new_retrievals:
    rep += f"  {n['pair_id']}: NCC={n['rank_ncc']} Grad={n['rank_grad']} Ctx={n['rank_ctx']} Phase={n['rank_phase']}\n"

rep += f"\nRescued cases: {len(rescued_cases)}\n"
rep += f"Broken cases: {len(broken_cases)}\n"
rep += f"NET: {len(rescued_cases) - len(broken_cases)}\n"

with open('phase2/V46_RESEARCH/V46_RESCUE_REPORT.md', 'w') as f:
    f.write(rep)
