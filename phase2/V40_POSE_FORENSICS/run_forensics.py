
import os, sys, cv2, time
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.append('.')
sys.path.append('phase2')

WORKERS = max(1, os.cpu_count() - 1)

def compute_scharr_gradient(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32)
    gx = cv2.Scharr(img_f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_f, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)

def compute_local_gradient_ncc(search_crop_grad: np.ndarray, template_grad: np.ndarray, max_loc: tuple[int, int]) -> float:
    th, tw = template_grad.shape[:2]
    px, py = max_loc
    if py + th > search_crop_grad.shape[0] or px + tw > search_crop_grad.shape[1] or py < 0 or px < 0:
        return 0.0
    s_patch = search_crop_grad[py:py+th, px:px+tw]
    s_norm = s_patch - np.mean(s_patch)
    t_norm = template_grad - np.mean(template_grad)
    s_std = np.std(s_norm)
    t_std = np.std(t_norm)
    if s_std < 1e-6 or t_std < 1e-6:
        return 0.0
    ncc = np.mean(s_norm * t_norm) / (s_std * t_std + 1e-8)
    return float(np.clip(ncc, -1.0, 1.0))

def process_forensic_pair(item: dict) -> dict:
    cv2.setNumThreads(1)
    pid = item['pair_id']
    st = item['set_type']
    found = item['found']
    gt_found = item['gt_found']
    
    if found == 0 or gt_found == 0:
        return {'pair_id': pid, 'valid': False}
        
    x_v39 = item['x']
    y_v39 = item['y']
    th_v39 = item['theta']
    sc_v39 = item['scale']
    gt_th = item['gt_theta']
    gt_sc = item['gt_scale']
    
    ref_path = os.path.join('data/phase2_dev', item['reference_path'].replace('\\', '/'))
    srch_path = os.path.join('data/phase2_dev', item['search_path'].replace('\\', '/'))
    
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)
    
    if ref_img is None or search_img is None:
        return {'pair_id': pid, 'valid': False}
        
    ref_h, ref_w = ref_img.shape[:2]
    sh, sw = search_img.shape[:2]
    
    # Pre-crop search ROI around V39 anchor
    tw = max(16, int(round(ref_w / sc_v39)))
    th = max(16, int(round(ref_h / sc_v39)))
    
    pad = 4
    y1 = int(round(y_v39 - th / 2.0)) - pad
    x1 = int(round(x_v39 - tw / 2.0)) - pad
    y2 = int(round(y_v39 + th / 2.0)) + pad
    x2 = int(round(x_v39 + tw / 2.0)) + pad
    
    if y1 < 0 or x1 < 0 or y2 > sh or x2 > sw:
        return {'pair_id': pid, 'valid': False}
        
    search_crop = search_img[y1:y2, x1:x2]
    search_crop_grad = compute_scharr_gradient(search_crop)
    ref_center = (ref_w / 2.0, ref_h / 2.0)
    
    # Mission 2 & 4: Theta sweep (-0.30 to +0.30 in 0.05 step)
    theta_offsets = np.arange(-0.30, 0.3001, 0.05) # 13 points
    # Objectives to test:
    # Obj A: intensity NCC
    # Obj B: gradient NCC
    # Obj C: 0.70 int + 0.30 grad (V39 default)
    # Obj D: 0.50 int + 0.50 grad
    # Obj E: gradient only
    # Interpolations to test for Obj C: LINEAR, CUBIC, AREA
    
    interps = {
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA
    }
    
    sweep_results = []
    
    for d_th in theta_offsets:
        cand_th = th_v39 + d_th
        M = cv2.getRotationMatrix2D(ref_center, cand_th, 1.0)
        
        # We test linear warpAffine
        rot_lin = cv2.warpAffine(ref_img, M, (ref_w, ref_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        tpl_area = cv2.resize(rot_lin, (tw, th), interpolation=cv2.INTER_AREA)
        tpl_grad_area = compute_scharr_gradient(tpl_area)
        
        res_int = cv2.matchTemplate(search_crop, tpl_area, cv2.TM_CCOEFF_NORMED)
        _, max_val_int, _, max_loc_int = cv2.minMaxLoc(res_int)
        
        res_grad = cv2.matchTemplate(search_crop_grad, tpl_grad_area, cv2.TM_CCOEFF_NORMED)
        _, max_val_grad_peak, _, _ = cv2.minMaxLoc(res_grad)
        
        # Local grad NCC at max_loc_int
        grad_ncc_at_int = compute_local_gradient_ncc(search_crop_grad, tpl_grad_area, max_loc_int)
        
        # Objective values:
        score_A = float(max_val_int)
        score_B = float(max_val_grad_peak)
        score_C = float(0.70 * max_val_int + 0.30 * grad_ncc_at_int)
        score_D = float(0.50 * max_val_int + 0.50 * grad_ncc_at_int)
        score_E = float(grad_ncc_at_int)
        
        # Interpolation test
        # CUBIC
        tpl_cubic = cv2.resize(rot_lin, (tw, th), interpolation=cv2.INTER_CUBIC)
        tpl_grad_cubic = compute_scharr_gradient(tpl_cubic)
        res_cubic = cv2.matchTemplate(search_crop, tpl_cubic, cv2.TM_CCOEFF_NORMED)
        _, max_val_c, _, max_loc_c = cv2.minMaxLoc(res_cubic)
        grad_c = compute_local_gradient_ncc(search_crop_grad, tpl_grad_cubic, max_loc_c)
        score_interp_cubic = float(0.70 * max_val_c + 0.30 * grad_c)
        
        # LINEAR resize
        tpl_lin = cv2.resize(rot_lin, (tw, th), interpolation=cv2.INTER_LINEAR)
        tpl_grad_lin = compute_scharr_gradient(tpl_lin)
        res_lin = cv2.matchTemplate(search_crop, tpl_lin, cv2.TM_CCOEFF_NORMED)
        _, max_val_l, _, max_loc_l = cv2.minMaxLoc(res_lin)
        grad_l = compute_local_gradient_ncc(search_crop_grad, tpl_grad_lin, max_loc_l)
        score_interp_linear = float(0.70 * max_val_l + 0.30 * grad_l)
        
        sweep_results.append({
            'd_th': d_th,
            'cand_th': cand_th,
            'gt_err': abs(cand_th - gt_th),
            'score_A': score_A,
            'score_B': score_B,
            'score_C': score_C,
            'score_D': score_D,
            'score_E': score_E,
            'score_interp_area': score_C,
            'score_interp_cubic': score_interp_cubic,
            'score_interp_linear': score_interp_linear
        })
        
    df_sw = pd.DataFrame(sweep_results)
    
    # Pick best for each objective
    best_A = df_sw.loc[df_sw['score_A'].idxmax()]
    best_B = df_sw.loc[df_sw['score_B'].idxmax()]
    best_C = df_sw.loc[df_sw['score_C'].idxmax()]
    best_D = df_sw.loc[df_sw['score_D'].idxmax()]
    best_E = df_sw.loc[df_sw['score_E'].idxmax()]
    best_cubic = df_sw.loc[df_sw['score_interp_cubic'].idxmax()]
    best_linear = df_sw.loc[df_sw['score_interp_linear'].idxmax()]
    
    # Mission 4: Curvature & Confidence on default Objective C
    scores_C = df_sw['score_C'].values
    th_vals = df_sw['cand_th'].values
    sorted_idx = np.argsort(scores_C)[::-1]
    peak_score = scores_C[sorted_idx[0]]
    second_peak_score = scores_C[sorted_idx[1]] if len(scores_C) > 1 else peak_score
    th_margin = peak_score - second_peak_score
    
    # Curvature around peak index (2nd derivative d^2(score)/d(theta)^2)
    p_idx = df_sw['score_C'].idxmax()
    if 0 < p_idx < len(scores_C) - 1:
        curvature = float((scores_C[p_idx-1] - 2*scores_C[p_idx] + scores_C[p_idx+1]) / (0.05**2))
    else:
        curvature = 0.0
        
    return {
        'pair_id': pid,
        'set_type': st,
        'valid': True,
        'v39_theta': th_v39,
        'v39_scale': sc_v39,
        'gt_theta': gt_th,
        'gt_scale': gt_sc,
        'v39_rot_err': abs(th_v39 - gt_th),
        'v39_scale_err': abs(sc_v39 - gt_sc),
        # Best choices per objective
        'th_A': best_A['cand_th'], 'err_A': best_A['gt_err'],
        'th_B': best_B['cand_th'], 'err_B': best_B['gt_err'],
        'th_C': best_C['cand_th'], 'err_C': best_C['gt_err'],
        'th_D': best_D['cand_th'], 'err_D': best_D['gt_err'],
        'th_E': best_E['cand_th'], 'err_E': best_E['gt_err'],
        # Interpolation best
        'th_cubic': best_cubic['cand_th'], 'err_cubic': best_cubic['gt_err'],
        'th_linear': best_linear['cand_th'], 'err_linear': best_linear['gt_err'],
        'th_area': best_C['cand_th'], 'err_area': best_C['gt_err'],
        # Confidence metrics
        'peak_score': peak_score,
        'th_margin': th_margin,
        'th_curvature': curvature,
        'full_sweep': df_sw.to_dict('records')
    }

if __name__ == '__main__':
    pairs_df = pd.read_csv('data/phase2_dev/pairs.csv')
    v39_df = pd.read_csv('phase2/V39_POSE/v39_predictions.csv')
    merged = pd.merge(pairs_df, v39_df, on='pair_id', suffixes=('_gt', '_v39'))
    items = merged.to_dict('records')
    
    print(f'Running forensic experiments across {len(items)} pairs with {WORKERS} workers...')
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for res in executor.map(process_forensic_pair, items):
            if res.get('valid', False):
                results.append(res)
    print(f'Done in {time.time()-t0:.2f}s! Valid present pairs: {len(results)}')
    
    df_all = pd.DataFrame(results)
    
    # 1. Pose Error Distribution
    err_dist = df_all[['pair_id', 'set_type', 'v39_theta', 'gt_theta', 'v39_rot_err', 'v39_scale', 'gt_scale', 'v39_scale_err']]
    err_dist.to_csv('phase2/V40_POSE_FORENSICS/pose_error_distribution.csv', index=False)
    
    # 2. Theta Objective Ablation
    records_abl = []
    for method, col in [
        ('V39_Baseline', 'v39_rot_err'),
        ('Obj_A_Intensity_NCC', 'err_A'),
        ('Obj_B_Gradient_Peak', 'err_B'),
        ('Obj_C_0.7Int_0.3Grad', 'err_C'),
        ('Obj_D_0.5Int_0.5Grad', 'err_D'),
        ('Obj_E_Gradient_Only', 'err_E'),
        ('Interp_AREA', 'err_area'),
        ('Interp_CUBIC', 'err_cubic'),
        ('Interp_LINEAR', 'err_linear')
    ]:
        for st in ['SetA', 'SetB', 'Overall']:
            sub = df_all if st == 'Overall' else df_all[df_all['set_type'] == st]
            mae = sub[col].mean()
            median = sub[col].median()
            p90 = sub[col].quantile(0.90)
            max_err = sub[col].max()
            
            # Count improved / worsened vs V39
            deltas = sub[col] - sub['v39_rot_err']
            improved = (deltas < -1e-4).sum()
            worsened = (deltas > 1e-4).sum()
            max_reg = deltas.max()
            
            records_abl.append({
                'method': method,
                'set_type': st,
                'mae': round(mae, 5),
                'median': round(median, 5),
                'p90': round(p90, 5),
                'max_err': round(max_err, 5),
                'improved': int(improved),
                'worsened': int(worsened),
                'max_regression': round(max_reg, 5)
            })
    df_abl = pd.DataFrame(records_abl)
    df_abl.to_csv('phase2/V40_POSE_FORENSICS/theta_objective_ablation.csv', index=False)
    
    # 3. Pose Confidence
    conf_df = df_all[['pair_id', 'set_type', 'v39_rot_err', 'v39_scale_err', 'peak_score', 'th_margin', 'th_curvature']]
    conf_df.to_csv('phase2/V40_POSE_FORENSICS/pose_confidence.csv', index=False)
    
    # 4. Detailed theta sweep
    sweep_rows = []
    for r in results:
        for sw in r['full_sweep']:
            sw_entry = {'pair_id': r['pair_id'], 'set_type': r['set_type']}
            sw_entry.update(sw)
            sweep_rows.append(sw_entry)
    pd.DataFrame(sweep_rows).to_csv('phase2/V40_POSE_FORENSICS/theta_sweep.csv', index=False)
    
    print('All forensic data saved successfully to phase2/V40_POSE_FORENSICS/')
