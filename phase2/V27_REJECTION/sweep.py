import pandas as pd
import numpy as np

def score_at_threshold(thresh):
    df = pd.read_csv('phase2/V27_REJECTION/oof_predictions.csv')
    df['pred_found'] = (df['oof_combined'] > thresh).astype(int)
    
    v25_preds = pd.read_csv('data/phase2_dev/v25_predictions_thresh.csv')
    df = pd.merge(df, v25_preds[['pair_id', 'x', 'y']], on='pair_id')
    
    tp_rej = np.sum((df["gt_found"] == 0) & (df["pred_found"] == 0))
    fp_rej = np.sum((df["gt_found"] == 1) & (df["pred_found"] == 0))
    fn_rej = np.sum((df["gt_found"] == 0) & (df["pred_found"] == 1))
    
    prec_rej = tp_rej / (tp_rej + fp_rej) if (tp_rej + fp_rej) > 0 else 0.0
    rec_rej = tp_rej / (tp_rej + fn_rej) if (tp_rej + fn_rej) > 0 else 0.0
    f1_rej = 2 * prec_rej * rec_rej / (prec_rej + rec_rej) if (prec_rej + rec_rej) > 0 else 0.0
    rej_points = f1_rej * 15.0
    
    pairs = pd.read_csv('data/phase2_dev/pairs.csv')
    merged = pd.merge(pairs, df, on='pair_id', suffixes=('', '_df'))
    
    merged['loc_err'] = np.hypot(merged['x'] - merged['gt_x'], merged['y'] - merged['gt_y'])
    
    set_a = merged[(merged['set_type'] == 'SetA') & (merged['gt_found'] == 1)]
    set_b = merged[(merged['set_type'] == 'SetB') & (merged['gt_found'] == 1)]
    
    def loc_score(d):
        loc = d[d['pred_found'] == 1]
        if len(loc) == 0: return 0.0
        return np.mean(loc['loc_err'] <= 5.0) * 100.0
        
    a_le5 = loc_score(set_a)
    b_le5 = loc_score(set_b)
    loc_points = (0.45 * a_le5 + 0.55 * b_le5) * 0.40
    
    total = loc_points + rej_points
    return total, loc_points, rej_points, f1_rej

def sweep():
    results = []
    for t in np.linspace(0.01, 0.99, 99):
        tot, loc, rej, f1 = score_at_threshold(t)
        results.append({'threshold': t, 'total': tot, 'loc': loc, 'rej': rej, 'f1': f1})
        
    res_df = pd.DataFrame(results)
    best = res_df.loc[res_df['total'].idxmax()]
    print(f"Best threshold: {best['threshold']:.2f}")
    print(f"Total points (Loc+Rej): {best['total']:.2f}")
    print(f"Loc points: {best['loc']:.2f}")
    print(f"Rej points: {best['rej']:.2f} (F1: {best['f1']:.4f})")
    res_df.to_csv('phase2/V27_REJECTION/threshold_sweep.csv', index=False)

if __name__ == '__main__':
    sweep()
