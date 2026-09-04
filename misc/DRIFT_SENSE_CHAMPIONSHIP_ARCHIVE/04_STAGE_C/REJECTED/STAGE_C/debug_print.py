import json, numpy as np
with open('FINAL_SUBMISSION/validation/championship/precomputed_cache.json', 'r') as f:
    c = json.load(f)
for p in c:
    base_err = float(np.hypot(p['v54_x'] - p['gt_x'], p['v54_y'] - p['gt_y'])) if (p['gt_found'] == 1 and p['v54_found'] == 1) else (-1.0 if p['gt_found']==1 else -2.0)
    if (base_err > 5.0 or base_err == -1.0) and p['best_cand'] is not None:
        v54_st = p.get('v54_struct', -1.0)
        best_st = p['best_cand']['struct_score']
        print(f"{p['pair_id']}: base_err={base_err:.1f} v54_st={v54_st:.3f} best_st={best_st:.3f} delta={best_st - v54_st:.3f}")
