import pandas as pd
import numpy as np

out_dir = 'phase2/V25_SCORE_MAX'
df_gt = pd.read_csv(f'{out_dir}/gt_relative_gaps.csv')
df_res = pd.read_csv(f'{out_dir}/rescue_potential.csv')

# Let's count groups directly from df_gt
print('TOTAL PRESENT', 140)
print('GT IN TOP10  ', len(df_gt[df_gt['gt_rank'] <= 10]))
print('GT IN TOP20  ', len(df_gt[df_gt['gt_rank'] <= 20]))
print('GT IN TOP50  ', len(df_gt[df_gt['gt_rank'] <= 50]))
print('GT IN TOP100 ', len(df_gt[df_gt['gt_rank'] <= 100]))
print('GT IN TOP200 ', len(df_gt[df_gt['gt_rank'] <= 200]))
print('GT OUTSIDE TOP200', 140 - len(df_gt[df_gt['gt_rank'] <= 200]))

print('\nMedian winner-GT score gap:')
gaps = df_gt[df_gt['gt_rank'] > 1]['winner_minus_gt'].dropna().values
if len(gaps) > 0:
    print(f"P25: {np.percentile(gaps, 25):.4f}")
    print(f"P50: {np.percentile(gaps, 50):.4f}")
    print(f"P75: {np.percentile(gaps, 75):.4f}")
    print(f"P90: {np.percentile(gaps, 90):.4f}")
    print(f"P95: {np.percentile(gaps, 95):.4f}")

print('\nHow many GT candidates require:')
req = df_res[df_res['required_ranker_delta'] > 0]['required_ranker_delta'].values
print(f"<0.005: {len(req[req < 0.005])}")
print(f"<0.010: {len(req[req < 0.010])}")
print(f"<0.020: {len(req[req < 0.020])}")
print(f"<0.050: {len(req[req < 0.050])}")
print(f"<0.100: {len(req[req < 0.100])}")
print(f">=0.100: {len(req[req >= 0.100])}")

if len(req[req >= 0.050]) / len(req) > 0.5:
    print('\nCONCLUSION: C) gap mostly non-discriminative (large gaps dominate)')
elif len(req[req < 0.010]) / len(req) > 0.5:
    print('\nCONCLUSION: A) gap appears highly discriminative (small gaps dominate)')
else:
    print('\nCONCLUSION: B) gap partially discriminative')
