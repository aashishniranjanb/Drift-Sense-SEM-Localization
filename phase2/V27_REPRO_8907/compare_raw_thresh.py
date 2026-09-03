import pandas as pd
import numpy as np

p_raw = pd.read_csv('data/phase2_dev/v25_predictions.csv')
p_thresh = pd.read_csv('data/phase2_dev/v25_predictions_thresh.csv')

print("Raw found values count:")
print(p_raw['found'].value_counts())

print("\nThresh found values count:")
print(p_thresh['found'].value_counts())

print("\nAre the score columns identical?")
print(np.allclose(p_raw['score'], p_thresh['score']))
print("Max abs difference between scores:", np.max(np.abs(p_raw['score'] - p_thresh['score'])))
