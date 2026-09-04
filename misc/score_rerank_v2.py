import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

def eval_scores(pred_csv, gt_csv="data/phase2_dev/pairs.csv"):
    gt = pd.read_csv(gt_csv)
    pred = pd.read_csv(pred_csv)
    m = pd.merge(gt, pred, on="pair_id", suffixes=("_gt", "_pred"))

    # Localization
    set_a = m[(m['set_type'] == 'SetA') & (m['gt_found'] == 1)]
    set_b = m[(m['set_type'] == 'SetB') & (m['gt_found'] == 1)]

    def loc_pct(df):
        loc = df[df['found'] == 1].copy()
        if len(loc) == 0: return 0.0
        loc['err'] = np.hypot(loc['x'] - loc['gt_x'], loc['y'] - loc['gt_y'])
        return np.mean(loc['err'] <= 5.0) * 100.0

    a_le5 = loc_pct(set_a)
    b_le5 = loc_pct(set_b)
    loc_pts = (0.45 * a_le5 + 0.55 * b_le5) * 0.40

    # Rejection
    tp = np.sum((m["gt_found"] == 0) & (m["found"] == 0))
    fp = np.sum((m["gt_found"] == 1) & (m["found"] == 0))
    fn = np.sum((m["gt_found"] == 0) & (m["found"] == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    rej_pts = f1 * 15.0

    # Calibration
    correctness = []
    for _, row in m.iterrows():
        gt_f = row['gt_found']
        pr_f = row['found']
        if gt_f == 1 and pr_f == 1:
            err = np.hypot(row['x'] - row['gt_x'], row['y'] - row['gt_y'])
            correctness.append(1 if err <= 5.0 else 0)
        elif gt_f == 0 and pr_f == 0:
            correctness.append(1)
        else:
            correctness.append(0)

    auc = roc_auc_score(correctness, m['score']) if len(set(correctness)) > 1 else 0.0
    spearman, _ = spearmanr(m['score'], correctness)

    return {
        "loc_pts": loc_pts,
        "rej_pts": rej_pts,
        "auc": auc,
        "spearman": spearman,
        "total": loc_pts + 19.743 + rej_pts + 8.269 + 5.0 + 10.0
    }

print("GOLDEN BASELINE:")
print(eval_scores("FINAL_SUBMISSION_GOLDEN/validation/scale_only.csv"))

print("\nRERANK-V2 SHADOW (T=0.005):")
print(eval_scores("FINAL_SUBMISSION/validation/rerank_v2_shadow_predictions.csv"))
