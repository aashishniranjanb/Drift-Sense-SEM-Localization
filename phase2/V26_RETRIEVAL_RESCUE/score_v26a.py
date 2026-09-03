import sys
import json
import pandas as pd

sys.path.append('phase2')
from benchmark_phase2 import evaluate_phase2

v25_csv = 'data/phase2_dev/v25_predictions.csv'
v26_csv = 'phase2/V26_RETRIEVAL_RESCUE/v26a_predictions.csv'
gt_csv = 'data/phase2_dev/pairs.csv'

s25 = evaluate_phase2(gt_csv, v25_csv)
s26 = evaluate_phase2(gt_csv, v26_csv)

print('\n' + '='*50)
print('V25 vs V26-A OFFICIAL SCORE COMPARISON')
print('='*50)

def fmt(k, d):
    return f"{d.get(k, 0.0):.4f}"

print(f"{'Metric':<25} | {'V25':<10} | {'V26-A':<10} | {'Delta':<10}")
print('-'*60)
print(f"{'Overall Score':<25} | {fmt('total_score', s25):<10} | {fmt('total_score', s26):<10} | {s26['total_score'] - s25['total_score']:.4f}")
print(f"{'Localization':<25} | {fmt('localization_score', s25):<10} | {fmt('localization_score', s26):<10} | {s26['localization_score'] - s25['localization_score']:.4f}")
print(f"{'Pose':<25} | {fmt('pose_score', s25):<10} | {fmt('pose_score', s26):<10} | {s26['pose_score'] - s25['pose_score']:.4f}")
print(f"{'Rejection F1':<25} | {fmt('rejection_score', s25):<10} | {fmt('rejection_score', s26):<10} | {s26['rejection_score'] - s25['rejection_score']:.4f}")
print(f"{'Calibration':<25} | {fmt('calibration_score', s25):<10} | {fmt('calibration_score', s26):<10} | {s26['calibration_score'] - s25['calibration_score']:.4f}")

# Runtime
v26_preds = pd.read_csv(v26_csv)
med_time = v26_preds['runtime'].median()
print(f"{'Runtime Median (s)':<25} | {'~3.2000':<10} | {med_time:<10.4f} |")

print('\n' + '='*50)
print('PROMOTION GATE CHECK')
print('='*50)
loc_diff = s26['localization_score'] - s25['localization_score']
total_diff = s26['total_score'] - s25['total_score']

promoted = True
if total_diff <= 0:
    print('? FAILED: Total score did not increase.')
    promoted = False
if loc_diff < 0:
    print('? FAILED: Localization score decreased.')
    promoted = False
if med_time > 5.0:
    print('? FAILED: Runtime median > 5.0s.')
    promoted = False

if promoted:
    print('? PASSED: V26-A is promoted.')
else:
    print('?? REJECTED: Keeping V25 as baseline.')

with open('phase2/V26_RETRIEVAL_RESCUE/v26a_results.json', 'w') as f:
    json.dump({'v25': s25, 'v26a': s26}, f, indent=4)
