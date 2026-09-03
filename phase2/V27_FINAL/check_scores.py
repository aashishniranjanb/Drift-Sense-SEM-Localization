import pandas as pd
import numpy as np

unthresh = pd.read_csv('phase2/V27_FINAL/v25_unthresholded.csv')
p98 = unthresh[unthresh['pair_id'] == 'pair_098'].iloc[0]
p116 = unthresh[unthresh['pair_id'] == 'pair_116'].iloc[0]

print(f"pair_098 score: {p98['score']:.6f}")
print(f"pair_116 score: {p116['score']:.6f}")
