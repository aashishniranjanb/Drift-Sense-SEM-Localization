import pandas as pd
import numpy as np

# Let's inspect data/phase2_dev/v25_predictions.csv vs v25_predictions_thresh.csv
p_all = pd.read_csv('data/phase2_dev/v25_predictions.csv')
p_thresh = pd.read_csv('data/phase2_dev/v25_predictions_thresh.csv')
gt = pd.read_csv('data/phase2_dev/pairs.csv')

print("Unthresholded rows:", len(p_all))
print("Thresholded rows:", len(p_thresh))

# Check agreement between coordinates where p_thresh found == 1:
m = pd.merge(p_all, p_thresh, on='pair_id', suffixes=('_raw', '_thresh'))
m_found = m[m['found_thresh'] == 1]
diff_x = np.abs(m_found['x_raw'] - m_found['x_thresh']).max()
diff_y = np.abs(m_found['y_raw'] - m_found['y_thresh']).max()
print(f"Max coordinate diff on found==1: dx={diff_x}, dy={diff_y}")
print(f"Max score diff: {np.abs(m['score_raw'] - m['score_thresh']).max()}")

# Save the raw unthresholded coordinates to phase2/V27_FINAL/v25_unthresholded.csv
p_all.to_csv('phase2/V27_FINAL/v25_unthresholded.csv', index=False)
print("Saved phase2/V27_FINAL/v25_unthresholded.csv")
