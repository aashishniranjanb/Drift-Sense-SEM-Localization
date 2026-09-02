import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def sweep_rejection_thresholds(pairs_csv, raw_preds_csv):
    gt = pd.read_csv(pairs_csv)
    pred = pd.read_csv(raw_preds_csv)
    
    merged = pd.merge(gt, pred, on="pair_id", suffixes=("_gt", "_pred"))
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75]
    results = []
    
    funnel = {"R1": 0, "R2": 0, "R3": 0, "R4": 0, "R5": 0}
    funnel_setA = {"R1": 0, "R2": 0, "R3": 0, "R4": 0, "R5": 0}
    funnel_setB = {"R1": 0, "R2": 0, "R3": 0, "R4": 0, "R5": 0}
    
    for T in thresholds:
        sets_data = {"SetA": [], "SetB": [], "SetC": []}
        tax = []
        for _, row in merged.iterrows():
            gt_found = int(row["gt_found"])
            conf = row["score"]
            pred_found = 1 if conf >= T else 0
            set_type = row.get("set_type", "SetA" if gt_found == 1 else "SetC")
            
            if gt_found == 1 and pred_found == 1:
                loc_err = float(np.hypot(row["x"] - row["gt_x"], row["y"] - row["gt_y"]))
            else:
                loc_err = -1.0
                
            rec = {"gt_found": gt_found, "pred_found": pred_found, "loc_err": loc_err, "score": conf}
            tax.append(rec)
            if set_type in sets_data:
                sets_data[set_type].append(rec)
                
            if T == 0.58 and gt_found == 1 and pred_found == 0:
                raw_err = float(np.hypot(row["x"] - row["gt_x"], row["y"] - row["gt_y"]))
                r_cat = "R5"
                if raw_err <= 1.0: r_cat = "R1"
                elif raw_err <= 2.0: r_cat = "R2"
                elif raw_err <= 5.0: r_cat = "R3"
                else: r_cat = "R4"
                funnel[r_cat] += 1
                if set_type == "SetA": funnel_setA[r_cat] += 1
                elif set_type == "SetB": funnel_setB[r_cat] += 1
                    
        df_tax = pd.DataFrame(tax)
        tp_rej = np.sum((df_tax["gt_found"] == 0) & (df_tax["pred_found"] == 0))
        fp_rej = np.sum((df_tax["gt_found"] == 1) & (df_tax["pred_found"] == 0))
        fn_rej = np.sum((df_tax["gt_found"] == 0) & (df_tax["pred_found"] == 1))
        
        prec_rej = tp_rej / (tp_rej + fp_rej) if (tp_rej + fp_rej) > 0 else 0.0
        rec_rej = tp_rej / (tp_rej + fn_rej) if (tp_rej + fn_rej) > 0 else 0.0
        f1_rej = 2 * prec_rej * rec_rej / (prec_rej + rec_rej) if (prec_rej + rec_rej) > 0 else 0.0
        
        def get_loc_score(recs):
            d = pd.DataFrame(recs)
            if len(d) == 0: return 0.0, 0
            p_gt = d[d["gt_found"] == 1]
            if len(p_gt) == 0: return 0.0, 0
            locs = p_gt[p_gt["pred_found"] == 1]
            errs = locs["loc_err"].values
            if len(errs) > 0:
                return np.mean(errs <= 5.0) * 100.0, len(errs)
            return 0.0, 0
            
        le5_A, acc_A = get_loc_score(sets_data["SetA"])
        le5_B, acc_B = get_loc_score(sets_data["SetB"])
        weighted_loc = 0.45 * le5_A + 0.55 * le5_B
        loc_points = (weighted_loc / 100.0) * 40.0
        
        correctness = [1 if ((r["gt_found"]==1 and r["pred_found"]==1 and r["loc_err"]<=5.0) or (r["gt_found"]==0 and r["pred_found"]==0)) else 0 for r in tax]
        spearman_corr, _ = spearmanr([r["score"] for r in tax], correctness)
        if np.isnan(spearman_corr): spearman_corr = 0.0
        cal_points = max(0, spearman_corr) * 10.0
        rej_points = f1_rej * 15.0
        
        total_est = loc_points + rej_points + cal_points
        results.append({"T": T, "Loc_A": le5_A, "Loc_B": le5_B, "Loc_Pts": loc_points, "Rej_F1": f1_rej, "Rej_Pts": rej_points, "Cal_Pts": cal_points, "Total_Est": total_est, "Acc_A": acc_A, "Acc_B": acc_B})
        
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False, float_format="%.3f"))
    
    print("\n--- REJECTED PRESENT OPPORTUNITY FUNNEL (at T=0.58) ---")
    print(f"R1 (<=1px) : {funnel['R1']} (Set A: {funnel_setA['R1']}, Set B: {funnel_setB['R1']})")
    print(f"R2 (<=2px) : {funnel['R2']} (Set A: {funnel_setA['R2']}, Set B: {funnel_setB['R2']})")
    print(f"R3 (<=5px) : {funnel['R3']} (Set A: {funnel_setA['R3']}, Set B: {funnel_setB['R3']})")
    print(f"R4 (>5px)  : {funnel['R4']} (Set A: {funnel_setA['R4']}, Set B: {funnel_setB['R4']})")
    print(f"R5 (None)  : {funnel['R5']} (Set A: {funnel_setA['R5']}, Set B: {funnel_setB['R5']})")

if __name__ == '__main__':
    sweep_rejection_thresholds('data/phase2_dev/pairs.csv', 'data/phase2_dev/v24_predictions_raw.csv')
