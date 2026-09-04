"""
V22 Appendix: Rejection Threshold Sweep on V21 Production System.
Sweeps the V14 composite gate threshold to find T* maximising competition score.
"""
import sys, os, pandas as pd, numpy as np
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
    """
    df must have columns: gt_x, gt_y, gt_theta, gt_scale, gt_found, set_type,
                          pred_x, pred_y, pred_theta, pred_scale, found, score
    """
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

    present_recall = np.mean(pf_arr[gf_arr==1]==1) if (gf_arr==1).sum()>0 else 0.0
    total = loc_s + pose_s + rej_s + cal_s + 5.0 + 10.0

    return dict(loc_s=loc_s, set_a=avg_a*100, set_b=avg_b*100,
                pose_s=pose_s, f1=f1, rej_s=rej_s,
                cal_auc=cal_auc, cal_s=cal_s,
                present_recall=present_recall,
                n_found_1=int(pf_arr.sum()), n_found_0=int((pf_arr==0).sum()),
                total=total)


def build_merged(gt_df, feat_df, threshold):
    """Build a merged df with pred columns from features at given threshold."""
    corr    = feat_df["corr_score"].values
    ctx     = feat_df["context_128"].values
    psr_v   = feat_df["psr"].values
    margin  = feat_df["peak_margin"].values
    ph_res  = feat_df["phase_residual"].values
    composite = np.clip(0.35*corr + 0.40*ctx + 0.15*(psr_v/10.0) + 0.10*margin - 0.20*ph_res, 0, 1)

    pred_part = pd.DataFrame({
        "pair_id": feat_df["pair_id"],
        "pred_x":  feat_df["x"],
        "pred_y":  feat_df["y"],
        "pred_theta": feat_df["theta"],
        "pred_scale": feat_df["scale"],
        "found":   (composite >= threshold).astype(int),
        "score":   composite,
    })
    merged = gt_df.merge(pred_part, on="pair_id")
    return merged


def main():
    os.makedirs("phase2/V22_CHAMPIONSHIP/results", exist_ok=True)

    gt_df   = pd.read_csv("data/phase2_dev/pairs.csv")
    feat_df = pd.read_csv("results/v14/presence_features.csv")
    pred_v21 = pd.read_csv("data/phase2_dev/predictions.csv")
    feat_df = feat_df.merge(pred_v21[["pair_id","x","y","theta","scale"]], on="pair_id", how="left")

    print("=== V22 APPENDIX: V14 Gate Threshold Sweep ===")
    print(f"Pairs: {len(feat_df)} | GT absent: {(feat_df.gt_found==0).sum()} | GT present: {(feat_df.gt_found==1).sum()}")

    # Baseline at T=0.58
    merged_base = build_merged(gt_df, feat_df, 0.58)
    m = competition_score(merged_base)
    print(f"\nBaseline T=0.58: found=1={m['n_found_1']}, found=0={m['n_found_0']}")
    print(f"  Total={m['total']:.2f} | Loc={m['loc_s']:.2f}/40 | Pose={m['pose_s']:.2f}/20")
    print(f"  Rej F1={m['f1']:.4f} ({m['rej_s']:.2f}/15) | Cal AUC={m['cal_auc']:.4f} | PR={m['present_recall']:.3f}")

    # Fixed validation split: rows 108-143 (60%-80% of 180)
    pair_ids = gt_df["pair_id"].tolist()
    val_ids  = set(pair_ids[108:144])
    test_ids = set(pair_ids[144:])

    val_feat  = feat_df[feat_df["pair_id"].isin(val_ids)].reset_index(drop=True)
    test_feat = feat_df[feat_df["pair_id"].isin(test_ids)].reset_index(drop=True)
    val_gt    = gt_df[gt_df["pair_id"].isin(val_ids)].reset_index(drop=True)
    test_gt   = gt_df[gt_df["pair_id"].isin(test_ids)].reset_index(drop=True)

    print(f"\nVal set: {len(val_feat)} pairs | Test set: {len(test_feat)} pairs")

    # Threshold sweep on validation
    thresholds = np.round(np.arange(0.10, 0.75, 0.02), 2)
    sweep = []
    best_t, best_val = 0.58, -1.0

    for t in thresholds:
        mv = build_merged(val_gt, val_feat, t)
        m = competition_score(mv)
        sweep.append({"threshold": t, "val_total": m["total"], "val_loc": m["loc_s"],
                      "val_rej_f1": m["f1"], "val_PR": m["present_recall"],
                      "val_n1": m["n_found_1"]})
        if m["total"] > best_val:
            best_val = m["total"]
            best_t   = t

    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv("phase2/V22_CHAMPIONSHIP/results/V22_THRESHOLD_SWEEP_V14.csv", index=False)
    print(f"\nBest val threshold: T*={best_t:.2f} (val_total={best_val:.2f})")
    print(sweep_df.to_string(index=False))

    # Final test evaluation at T*
    mt = build_merged(test_gt, test_feat, best_t)
    m_test = competition_score(mt)
    print(f"\n=== TEST at T*={best_t:.2f} ===")
    print(f"  Total={m_test['total']:.2f} | Loc={m_test['loc_s']:.2f}/40 (SetA={m_test['set_a']:.1f}%, SetB={m_test['set_b']:.1f}%)")
    print(f"  Pose={m_test['pose_s']:.2f}/20 | Rej F1={m_test['f1']:.4f} ({m_test['rej_s']:.2f}/15)")
    print(f"  Cal AUC={m_test['cal_auc']:.4f} ({m_test['cal_s']:.2f}/10) | PR={m_test['present_recall']:.3f}")
    print(f"  found=1={m_test['n_found_1']}, found=0={m_test['n_found_0']}")

    # Save summary
    summary = pd.DataFrame([
        {"model": "V21_T0.58", "threshold": 0.58,
         **{f"test_{k}": v for k,v in competition_score(build_merged(gt_df, feat_df, 0.58)).items()}},
        {"model": f"V22_T{best_t:.2f}", "threshold": best_t,
         **{f"test_{k}": v for k,v in m_test.items()}},
    ])
    summary.to_csv("phase2/V22_CHAMPIONSHIP/results/V22_THRESHOLD_APPENDIX.csv", index=False)
    print("\nSaved V22_THRESHOLD_APPENDIX.csv and V22_THRESHOLD_SWEEP_V14.csv")

    # Decision
    full_v21  = competition_score(build_merged(gt_df, feat_df, 0.58))
    full_best = competition_score(build_merged(gt_df, feat_df, best_t))
    print(f"\n=== FULL 180-CASE COMPARISON ===")
    print(f"  V21  T=0.58:  Total={full_v21['total']:.2f} | Loc={full_v21['loc_s']:.2f} | PR={full_v21['present_recall']:.3f}")
    print(f"  Best T={best_t:.2f}: Total={full_best['total']:.2f} | Loc={full_best['loc_s']:.2f} | PR={full_best['present_recall']:.3f}")
    decision = "KEEP NEW THRESHOLD" if (full_best["total"] > full_v21["total"] and full_best["present_recall"] >= full_v21["present_recall"] - 0.02) else "KEEP T=0.58"
    print(f"\nDECISION: {decision}")
    print(f"Optimal T to update fallback rejection: {best_t:.2f}")

if __name__ == "__main__":
    main()
