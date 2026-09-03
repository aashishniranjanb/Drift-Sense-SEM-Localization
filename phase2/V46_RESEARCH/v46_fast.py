import pandas as pd
import numpy as np
import cv2
import os
import sys

df = pd.read_csv('phase2/V26_CHAMPIONSHIP/v26_extracted_features.csv')
df = df[df['gt_found'] == 1]
df = df[df['queue'] == 'V25']
df = df.sort_values(by=['pair_id', 'v25_ml_score'], ascending=[True, False])

pairs_df = pd.read_csv('data/phase2_dev/pairs.csv')
pairs_df = pairs_df[pairs_df['gt_found'] == 1]

def get_grad(img):
    img_f = img.astype(np.float32) / 255.0
    gx = cv2.Scharr(img_f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_f, cv2.CV_32F, 0, 1)
    g = cv2.magnitude(gx, gy)
    return g

def extract_nms(corr_plane, max_k=200, r=5):
    ch, cw = corr_plane.shape[:2]
    work = corr_plane.copy()
    cands = []
    for _ in range(max_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= -99.0 or np.isnan(max_val):
            break
        px, py = max_loc
        cands.append((px, py, max_val))
        y1, y2 = max(0, py - r), min(ch, py + r + 1)
        x1, x2 = max(0, px - r), min(cw, px + r + 1)
        work[y1:y2, x1:x2] = -999.0
    return cands

def check_gt(cands, tw, th, gt_x, gt_y, px_offset=0, py_offset=0):
    for px, py, _ in cands:
        cx = px + tw/2.0 + px_offset
        cy = py + th/2.0 + py_offset
        if np.hypot(cx - gt_x, cy - gt_y) <= 8.0:
            return True
    return False

results = []
print(f"Evaluating {len(pairs_df)} pairs for V46 Fast Retrieval Union...")

for idx, row in pairs_df.iterrows():
    pair_id = row['pair_id']
    ref = cv2.imread(os.path.join('data/phase2_dev', row['reference_path']), 0)
    search = cv2.imread(os.path.join('data/phase2_dev', row['search_path']), 0)
    gt_x, gt_y = row['gt_x'], row['gt_y']
    
    cands = df[df['pair_id'] == pair_id].to_dict('records')
    if len(cands) == 0: continue
    top1 = cands[0]
    
    scale = top1.get('est_scale', 10.0)
    theta = top1.get('est_theta', 0.0)
    if np.isnan(scale): scale = 10.0
    if np.isnan(theta): theta = 0.0
    
    c = 500
    M = cv2.getRotationMatrix2D((c, c), theta, scale)
    ref_warped = cv2.warpAffine(ref, M, (1000, 1000), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    tw = int(215 / scale) if scale > 0 else 215
    th = tw
    px_c = 500 - tw//2
    py_c = 500 - th//2
    temp_ncc = ref_warped[py_c:py_c+th, px_c:px_c+tw]
    
    # 1. NCC
    corr_ncc = cv2.matchTemplate(search.astype(np.float32), temp_ncc.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    cands_ncc = extract_nms(corr_ncc, max_k=200)
    has_ncc = check_gt(cands_ncc, tw, th, gt_x, gt_y)
    
    # 2. Gradient
    search_g = get_grad(search)
    temp_g = get_grad(temp_ncc)
    corr_grad = cv2.matchTemplate(search_g, temp_g, cv2.TM_CCOEFF_NORMED)
    cands_grad = extract_nms(corr_grad, max_k=200)
    has_grad = check_gt(cands_grad, tw, th, gt_x, gt_y)
    
    # 3. Context (2.0x template)
    tw_ctx = int(tw * 2.0)
    th_ctx = int(th * 2.0)
    px_ctx = 500 - tw_ctx//2
    py_ctx = 500 - th_ctx//2
    temp_ctx = ref_warped[py_ctx:py_ctx+th_ctx, px_ctx:px_ctx+tw_ctx]
    
    corr_ctx = cv2.matchTemplate(search.astype(np.float32), temp_ctx.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    cands_ctx = extract_nms(corr_ctx, max_k=200)
    has_ctx = check_gt(cands_ctx, tw_ctx, th_ctx, gt_x, gt_y)
    
    # 4. Phase Correlation
    sh, sw = search.shape
    F_s = np.fft.fft2(search.astype(np.float32))
    temp_pad = np.zeros_like(search, dtype=np.float32)
    temp_pad[:th, :tw] = temp_ncc.astype(np.float32)
    F_t = np.fft.fft2(temp_pad)
    
    R = F_s * np.conjugate(F_t)
    R /= (np.abs(R) + 1e-5)
    corr_phase = np.fft.ifft2(R).real
    
    cands_phase = extract_nms(corr_phase, max_k=200)
    has_phase = check_gt(cands_phase, tw, th, gt_x, gt_y)
    
    # UNION
    all_cxcy = []
    def add_cands(cands_list, w, h, px_off=0, py_off=0):
        for px, py, _ in cands_list:
            cx = px + w/2.0 + px_off
            cy = py + h/2.0 + py_off
            if not any(np.hypot(cx - ecx, cy - ecy) < 5.0 for ecx, ecy in all_cxcy):
                all_cxcy.append((cx, cy))
                
    add_cands(cands_ncc, tw, th)
    add_cands(cands_grad, tw, th)
    add_cands(cands_ctx, tw_ctx, th_ctx)
    add_cands(cands_phase, tw, th)
    
    has_union = any(np.hypot(cx - gt_x, cy - gt_y) <= 8.0 for cx, cy in all_cxcy)
    
    results.append({
        'pair_id': pair_id,
        'ncc': has_ncc,
        'grad': has_grad,
        'ctx': has_ctx,
        'phase': has_phase,
        'union': has_union,
        'union_size': len(all_cxcy)
    })
    
    if len(results) % 10 == 0:
        print(f"Processed {len(results)}/{len(pairs_df)}")

res_df = pd.DataFrame(results)
res_df.to_csv('phase2/V46_RESEARCH/v46_results.csv', index=False)

n_pairs = len(res_df)
ncc_n = res_df['ncc'].sum()
grad_n = res_df['grad'].sum()
ctx_n = res_df['ctx'].sum()
phase_n = res_df['phase'].sum()
union_n = res_df['union'].sum()

print("\n--- V46-A RESULT ---")
print(f"NCC                  {ncc_n}/{n_pairs}")
print(f"Gradient             {grad_n}/{n_pairs}")
print(f"Phase                {phase_n}/{n_pairs}")
print(f"Context              {ctx_n}/{n_pairs}")
print(f"UNION                {union_n}/{n_pairs}")
print(f"\nAverage UNION pool size: {res_df['union_size'].mean():.1f} candidates")

report = f"""# V46-A MULTI-HYPOTHESIS RETRIEVAL

## Coverage across Top-200 queues:
NCC                  {ncc_n}/{n_pairs}
Gradient             {grad_n}/{n_pairs}
Phase                {phase_n}/{n_pairs}
Context              {ctx_n}/{n_pairs}
UNION                {union_n}/{n_pairs}

## Verdict
"""

if union_n <= 105:
    report += "KILL"
elif union_n <= 110:
    report += "INTERESTING"
elif union_n <= 116:
    report += "STRONG"
else:
    report += "VERY STRONG"

with open("phase2/V46_RESEARCH/V46_REPORT.md", "w") as f:
    f.write(report)

