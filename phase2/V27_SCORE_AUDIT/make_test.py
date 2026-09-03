import pandas as pd
import numpy as np

# Ground Truth for synthetic tests
gt = []
# Predictions for synthetic tests
pred = []

# Tiers check:
# Set A <= 1px, Set A <= 5px, Set B <= 1px, Set B <= 5px, >5px
gt.append({'pair_id': 'A1', 'set_type': 'SetA', 'gt_found': 1, 'gt_x': 100, 'gt_y': 100, 'gt_scale': 1.0, 'gt_theta': 0.0})
pred.append({'pair_id': 'A1', 'found': 1, 'x': 100, 'y': 100, 'scale': 1.0, 'theta': 0.0, 'score': 0.99}) # <= 1px

gt.append({'pair_id': 'A2', 'set_type': 'SetA', 'gt_found': 1, 'gt_x': 100, 'gt_y': 100, 'gt_scale': 1.0, 'gt_theta': 0.0})
pred.append({'pair_id': 'A2', 'found': 1, 'x': 103, 'y': 100, 'scale': 1.0, 'theta': 0.0, 'score': 0.95}) # <= 5px

gt.append({'pair_id': 'A3', 'set_type': 'SetA', 'gt_found': 1, 'gt_x': 100, 'gt_y': 100, 'gt_scale': 1.0, 'gt_theta': 0.0})
pred.append({'pair_id': 'A3', 'found': 1, 'x': 110, 'y': 100, 'scale': 1.0, 'theta': 0.0, 'score': 0.90}) # > 5px

gt.append({'pair_id': 'B1', 'set_type': 'SetB', 'gt_found': 1, 'gt_x': 100, 'gt_y': 100, 'gt_scale': 1.0, 'gt_theta': 0.0})
pred.append({'pair_id': 'B1', 'found': 1, 'x': 100, 'y': 100.5, 'scale': 1.0, 'theta': 0.0, 'score': 0.88}) # <= 1px

gt.append({'pair_id': 'C1', 'set_type': 'SetC', 'gt_found': 0, 'gt_x': 0, 'gt_y': 0, 'gt_scale': 0.0, 'gt_theta': 0.0})
pred.append({'pair_id': 'C1', 'found': 0, 'x': 0, 'y': 0, 'scale': 0.0, 'theta': 0.0, 'score': 0.10}) # Correct rejection

gt.append({'pair_id': 'C2', 'set_type': 'SetC', 'gt_found': 0, 'gt_x': 0, 'gt_y': 0, 'gt_scale': 0.0, 'gt_theta': 0.0})
pred.append({'pair_id': 'C2', 'found': 1, 'x': 50, 'y': 50, 'scale': 1.0, 'theta': 0.0, 'score': 0.80}) # False Positive

pd.DataFrame(gt).to_csv('phase2/V27_SCORE_AUDIT/gt.csv', index=False)
pd.DataFrame(pred).to_csv('phase2/V27_SCORE_AUDIT/pred.csv', index=False)
