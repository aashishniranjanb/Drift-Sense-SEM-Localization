import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score

def localization_credit(e):
    if e <= 1.0: return 1.0
    if e <= 2.0: return 0.8
    if e <= 3.0: return 0.6
    if e <= 5.0: return 0.4
    return 0.0

def pose_credit(s_pct, r_deg):
    s = 1.0 if s_pct<=1 else (0.75 if s_pct<=2 else (0.5 if s_pct<=5 else 0.0))
    r = 1.0 if r_deg<=0.25 else (0.75 if r_deg<=0.5 else (0.5 if r_deg<=1.0 else 0.0))
    return (s+r)/2.0

def competition_score(df):
    la, lb, pc = [], [], []
    for _, r in df.iterrows():
        gf, pf, st = int(r.gt_found), int(r.found), r.set_type
        if gf==1 and pf==1:
            e = float(np.hypot(r.pred_x - r.gt_x, r.pred_y - r.gt_y))
            lc = localization_credit(e)
            (la if st=="SetA" else lb).append(lc)
            if lc > 0:
                sp = abs(r.pred_scale - r.gt_scale) / max(abs(r.gt_scale), 1e-6) * 100.0
                rd = abs(r.pred_theta - r.gt_theta)
                pc.append(pose_credit(sp, rd))
        elif gf==1 and pf==0:
            (la if st=="SetA" else lb).append(0.0)

    avg_a = np.mean(la) if la else 0.0
    avg_b = np.mean(lb) if lb else 0.0
    loc_s = (0.45*avg_a + 0.55*avg_b) * 40.0
    pose_s = np.mean(pc)*20.0 if pc else 0.0

    gf_arr = df.gt_found.values
    pf_arr = df.found.values
    tp = np.sum((gf_arr==0)&(pf_arr==0))
    fp = np.sum((gf_arr==1)&(pf_arr==0))
    fn = np.sum((gf_arr==0)&(pf_arr==1))
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    rej_s = f1 * 15.0

    correctness, scores = [], []
    for _, r in df.iterrows():
        gf, pf = int(r.gt_found), int(r.found)
        if gf==1 and pf==1:
            e = float(np.hypot(r.pred_x - r.gt_x, r.pred_y - r.gt_y))
            correctness.append(1 if e<=5.0 else 0)
        elif gf==0 and pf==0:
            correctness.append(1)
        else:
            correctness.append(0)
        scores.append(float(r.score))
    cal_auc = roc_auc_score(correctness, scores) if len(set(correctness))>1 else 0.5
    cal_s = cal_auc * 10.0

    total = loc_s + pose_s + rej_s + cal_s + 5.0 + 10.0

    return dict(loc_s=loc_s, set_a=avg_a*100, set_b=avg_b*100,
                pose_s=pose_s, f1=f1, rej_s=rej_s,
                cal_auc=cal_auc, cal_s=cal_s,
                total=total)

def run_evaluation():
    # We evaluate on TEST pairs: 144-179
    # Read the full ranking results which contains all candidates with 'final_score' (only computed for test_mask)
    df = pd.read_csv("phase2/V22_CHAMPIONSHIP/results/blast2_ranking_results.csv")
    full_pairs = pd.read_csv("data/phase2_dev/pairs.csv")
    
    test_pairs = set(full_pairs.loc[144:179, 'pair_id'])
    test_df = df[df['pair_id'].isin(test_pairs)].copy()
    
    with open("phase2/V22_CHAMPIONSHIP/results/chosen_model.txt", "r") as f:
        best_model = f.read().strip()
        
    print(f"Applying model {best_model} on TEST pairs")
    
    # We also need the original prediction for scale, theta (since ranker only changes x,y)
    # The baseline prediction is in data/phase2_dev/predictions.csv (V21 predictions)
    pred_v21 = pd.read_csv("data/phase2_dev/predictions.csv")
    
    new_preds = []
    
    for pid in test_pairs:
        group = test_df[test_df['pair_id'] == pid]
        
        base_pred = pred_v21[pred_v21['pair_id'] == pid].iloc[0]
        
        if len(group) == 0:
            new_preds.append({
                "pair_id": pid,
                "pred_x": base_pred["x"],
                "pred_y": base_pred["y"],
                "pred_theta": base_pred["theta"],
                "pred_scale": base_pred["scale"],
                "found": base_pred["found"],
                "score": base_pred["score"]
            })
            continue
            
        best_idx = group['final_score'].idxmax()
        best_cand = group.loc[best_idx]
        
        score_combined = 0.50 * best_cand["corr_score"] + 0.50 * best_cand["context_128"] - best_cand["phase_penalty"]
        
        found = 1 if score_combined >= 0.58 else 0
        
        new_preds.append({
            "pair_id": pid,
            "pred_x": best_cand["cx"] if found else 0.0,
            "pred_y": best_cand["cy"] if found else 0.0,
            "pred_theta": base_pred["theta"] if found else 0.0,
            "pred_scale": base_pred["scale"] if found else 0.0,
            "found": found,
            "score": score_combined
        })
        
    pred_df = pd.DataFrame(new_preds)
    
    # We want to replace the TEST pair predictions in the full pred_v21 and compute score on all 180
    full_preds = pred_v21.copy()
    for _, r in pred_df.iterrows():
        idx = full_preds[full_preds['pair_id'] == r['pair_id']].index
        if len(idx) > 0:
            full_preds.loc[idx[0], 'x'] = r['pred_x']
            full_preds.loc[idx[0], 'y'] = r['pred_y']
            full_preds.loc[idx[0], 'theta'] = r['pred_theta']
            full_preds.loc[idx[0], 'scale'] = r['pred_scale']
            full_preds.loc[idx[0], 'found'] = r['found']
            full_preds.loc[idx[0], 'score'] = r['score']
            
    # rename x,y to pred_x, pred_y for competition_score
    full_preds = full_preds.rename(columns={'x': 'pred_x', 'y': 'pred_y', 'theta': 'pred_theta', 'scale': 'pred_scale'})
    gt_df = pd.read_csv("data/phase2_dev/pairs.csv")
    merged = gt_df.merge(full_preds, on="pair_id")
    
    m_test = competition_score(merged)
    
    print(f"=== TEST SCORE with {best_model} Ranker ===")
    print(f"Total Competition Score: {m_test['total']:.2f}")
    print(f"Localization: {m_test['loc_s']:.2f} / 40.0")
    print(f"Pose: {m_test['pose_s']:.2f} / 20.0")
    print(f"Rejection: {m_test['rej_s']:.2f} / 15.0")
    print(f"Calibration: {m_test['cal_s']:.2f} / 10.0")
    
    # Write to md
    with open("phase2/V22_CHAMPIONSHIP/results/BLAST2_DECISION.md", "w") as f:
        f.write(f"# Blast 2 Replica Killer Ranker Evaluation\n\n")
        f.write(f"Chosen Variant: {best_model}\n\n")
        f.write(f"## Test Set Performance (Pairs 144-179)\n")
        f.write(f"- Total Score: **{m_test['total']:.2f}**\n")
        f.write(f"- Localization Score: {m_test['loc_s']:.2f}\n")
        f.write(f"- Pose Score: {m_test['pose_s']:.2f}\n")
        f.write(f"- Rejection Score: {m_test['rej_s']:.2f}\n")
        f.write(f"- Calibration Score: {m_test['cal_s']:.2f}\n")
        
        if m_test['total'] > 50.65:
            f.write(f"\nDECISION: KEEP {best_model}\n")
        else:
            f.write(f"\nDECISION: REJECT (Score <= 50.65, Revert to V18-C)\n")

if __name__ == "__main__":
    run_evaluation()
