import pandas as pd
import numpy as np

gt = pd.read_csv('data/phase2_dev/pairs.csv')
pred = pd.read_csv('FINAL_SUBMISSION/validation/scale_only.csv')
cache = pd.read_csv('FINAL_SUBMISSION/runtime/models/v25_stage_cache.csv')

m = pd.merge(gt, pred, on='pair_id', suffixes=('_gt', '_pred'))
m = pd.merge(m, cache, on='pair_id', suffixes=('', '_cache'))

tp = np.sum((m['gt_found'] == 0) & (m['found'] == 0))
fp = np.sum((m['gt_found'] == 1) & (m['found'] == 0))
fn = np.sum((m['gt_found'] == 0) & (m['found'] == 1))
tn = np.sum((m['gt_found'] == 1) & (m['found'] == 1))

print(f"=== Baseline Rejection Confusion (Absent=Positive) ===")
print(f"TP (Absent correctly rejected): {tp} / 40")
print(f"FN (Absent incorrectly accepted): {fn} / 40")
print(f"FP (Present incorrectly rejected): {fp} / 140")
print(f"TN (Present correctly accepted): {tn} / 140")

prec = tp / (tp + fp) if (tp + fp) > 0 else 0
rec = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, Rej Points: {f1*15:.2f}")

# Now analyze the 78 accepted pairs (found == 1):
acc = m[m['found'] == 1].copy()
acc['err'] = np.hypot(acc['x'] - acc['gt_x'], acc['y'] - acc['gt_y'])
print('\n=== Accepted Pairs (found==1, n=78) ===')
print(f"gt_found==1: {np.sum(acc['gt_found']==1)}, gt_found==0: {np.sum(acc['gt_found']==0)}")
print(f"Error <= 1px: {np.sum(acc['err'] <= 1)}")
print(f"Error <= 2px: {np.sum(acc['err'] <= 2)}")
print(f"Error <= 3px: {np.sum(acc['err'] <= 3)}")
print(f"Error <= 5px: {np.sum(acc['err'] <= 5)}")
print(f"Error > 5px: {np.sum(acc['err'] > 5)}")

# For the 2 false accepts (gt_found == 0, found == 1):
if fn > 0:
    print('\nFalse accepts (gt_found==0, found==1):')
    print(m[(m['gt_found'] == 0) & (m['found'] == 1)][['pair_id', 'set_type', 'v25_score', 'margin', 'top1_corr', 'top1_ctx', 'top1_neigh']])

# Now analyze the 62 rejected present pairs (gt_found == 1, found == 0):
rej_pres = m[(m['gt_found'] == 1) & (m['found'] == 0)].copy()
rej_pres['v25_err'] = np.hypot(rej_pres['v25_x'] - rej_pres['gt_x'], rej_pres['v25_y'] - rej_pres['gt_y'])
print(f'\n=== Rejected Present Pairs (gt_found==1, found==0, n={len(rej_pres)}) ===')
print(f"v25_err <= 1px: {np.sum(rej_pres['v25_err'] <= 1)}")
print(f"v25_err <= 2px: {np.sum(rej_pres['v25_err'] <= 2)}")
print(f"v25_err <= 3px: {np.sum(rej_pres['v25_err'] <= 3)}")
print(f"v25_err <= 5px: {np.sum(rej_pres['v25_err'] <= 5)}")
print(f"v25_err > 5px: {np.sum(rej_pres['v25_err'] > 5)}")

good_rej = rej_pres[rej_pres['v25_err'] <= 5]
print(f'\nBreakdown of <=5px candidates among rejected present (n={len(good_rej)}):')
print(good_rej[['pair_id', 'set_type', 'v25_score', 'margin', 'top1_corr', 'top1_ctx', 'top1_neigh', 'top1_grad', 'v25_err']])

print('\nSummary stats of v25_err for all rejected present:')
print(rej_pres['v25_err'].describe())
