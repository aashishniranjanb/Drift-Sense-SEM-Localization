import pandas as pd, numpy as np

df_22 = pd.read_csv('FINAL_SUBMISSION/validation/CHAMPIONSHIP_FINAL/lattice_rescue_22_cases.csv')
v25 = pd.read_csv('data/phase2_dev/v25_predictions.csv')
audit = pd.read_csv('FINAL_SUBMISSION/validation/candidate_pool_audit_140.csv')

broken_pids = ['pair_059', 'pair_067', 'pair_092']
print('BROKEN SUCCESS PAIRS PROFILE')
for pid in broken_pids:
    r = v25[v25['pair_id'] == pid].iloc[0]
    score_val = float(r['score'])
    found_val = int(r['found'])
    print(f'  {pid}: v25_score={score_val:.4f}  found={found_val}')

print()
success_ids = audit[audit['category'] == 'SUCCESS_ACCEPTED']['pair_id'].tolist()
v25_succ = v25[v25['pair_id'].isin(success_ids)]
print('V25 score distribution for 76 successes:')
mn = v25_succ['score'].min()
me = v25_succ['score'].mean()
mx = v25_succ['score'].max()
print(f'  Min={mn:.4f}  Mean={me:.4f}  Max={mx:.4f}')

print()
print('Score percentiles for successes:')
for q in [0.05, 0.10, 0.25, 0.50]:
    pct = v25_succ['score'].quantile(q)
    print(f'  P{int(q*100):02d}: {pct:.4f}')

print()
print('22-case table (base_err, final_err, lat_conf, n_rescue, was_rescued):')
cols = ['pair_id','base_err','final_err','lattice_confidence','rescue_candidates_generated','was_rescued']
print(df_22[cols].to_string())
