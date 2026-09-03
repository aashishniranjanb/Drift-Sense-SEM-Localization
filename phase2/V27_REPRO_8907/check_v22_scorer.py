import pandas as pd
import sys
import os

sys.path.append('phase2/V22_CHAMPIONSHIP')
from scorer import score_predictions

# Score V25 baseline under V22 scorer
res_v25 = score_predictions('data/phase2_dev/v25_predictions_thresh.csv', 'data/phase2_dev/pairs.csv', runtime_median=3.2)
print("=== V25 UNDER V22 SCORER ===")
for k, v in res_v25.items():
    print(f"{k}: {v}")
