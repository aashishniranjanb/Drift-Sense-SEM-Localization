"""
V24-D0 Falsification: Global Architecture Consistency
Tests whether 1D/2D lattice residuals and row/col consistency can separate
GT from periodic replicas without training a model.
"""
import pandas as pd
import numpy as np

def get_pitch_and_conf(d_array):
    if len(d_array) == 0: return 0.0, 0.0
    counts, bins = np.histogram(d_array, bins=np.arange(0, 800, 5))
    if len(counts) == 0: return 0.0, 0.0
    best_idx = np.argmax(counts)
    pitch = bins[best_idx] + 2.5
    conf = counts[best_idx] / len(d_array)
    return pitch, conf

def get_global_phase(vals, pitch, weights=None):
    if pitch == 0: return 0.0
    angles = 2 * np.pi * vals / pitch
    if weights is None:
        mean_cos = np.mean(np.cos(angles))
        mean_sin = np.mean(np.sin(angles))
    else:
        mean_cos = np.average(np.cos(angles), weights=weights)
        mean_sin = np.average(np.sin(angles), weights=weights)
    phase = np.arctan2(mean_sin, mean_cos) * pitch / (2 * np.pi)
    if phase < 0: phase += pitch
    return phase

def get_residual(val, pitch, phase):
    if pitch == 0: return 0.0
    diff = abs(val - phase) % pitch
    return min(diff, pitch - diff)

def get_consistency(cx, cy, px, py, cx_vals, cy_vals, corr_vals, tol=15):
    row_score = 0.0
    col_score = 0.0
    
    if px > 0:
        for i in range(len(cx_vals)):
            if abs(cy_vals[i] - cy) < tol and abs(cx_vals[i] - cx) > 15:
                rem = abs(cx_vals[i] - cx) % px
                if rem < tol or rem > px - tol:
                    row_score += corr_vals[i]
                    
    if py > 0:
        for i in range(len(cy_vals)):
            if abs(cx_vals[i] - cx) < tol and abs(cy_vals[i] - cy) > 15:
                rem = abs(cy_vals[i] - cy) % py
                if rem < tol or rem > py - tol:
                    col_score += corr_vals[i]
                    
    return row_score, col_score

def falsify_lattice():
    df = pd.read_csv("phase2/V22_CHAMPIONSHIP/results/candidate_pool_features.csv")
    
    results = []
    
    # We'll evaluate at K=200 for maximum structure
    for pair_id, group in df.groupby("pair_id"):
        pool = group.head(200).copy()
        
        gt_cand = pool[pool.is_correct == 1]
        if len(gt_cand) == 0: continue
        gt_cand = gt_cand.iloc[0]
        
        # False cand: highest corr score that is NOT GT
        false_cands = pool[pool.is_correct == 0].sort_values("corr_score", ascending=False)
        if len(false_cands) == 0: continue
        false_cand = false_cands.iloc[0]
        
        # Only care if false_cand is a strong replica (e.g. corr > 0.9 * max)
        if false_cand.corr_score < 0.9 * pool.corr_score.max():
            continue
            
        cx_vals = pool["cx"].values
        cy_vals = pool["cy"].values
        corr_vals = pool["corr_score"].values
        
        # Pairwise diffs
        dx_mat = np.abs(cx_vals[:, None] - cx_vals[None, :])
        dy_mat = np.abs(cy_vals[:, None] - cy_vals[None, :])
        dx = dx_mat[dx_mat > 15]
        dy = dy_mat[dy_mat > 15]
        
        px, conf_x = get_pitch_and_conf(dx)
        py, conf_y = get_pitch_and_conf(dy)
        
        # Weight phase by corr_score^3 to prioritize strong peaks
        w = corr_vals ** 3
        phase_x = get_global_phase(cx_vals, px, w)
        phase_y = get_global_phase(cy_vals, py, w)
        
        def extract_feats(cand):
            rx = get_residual(cand.cx, px, phase_x)
            ry = get_residual(cand.cy, py, phase_y)
            res = np.hypot(rx, ry)
            r_c, c_c = get_consistency(cand.cx, cand.cy, px, py, cx_vals, cy_vals, corr_vals)
            return {
                "corr": cand.corr_score,
                "psr": cand.psr,
                "rx": rx,
                "ry": ry,
                "lattice_residual": res,
                "row_consistency": r_c,
                "col_consistency": c_c
            }
            
        gt_feats = extract_feats(gt_cand)
        fc_feats = extract_feats(false_cand)
        
        results.append({
            "pair_id": pair_id,
            "px": px, "py": py,
            "conf_x": conf_x, "conf_y": conf_y,
            "gt_corr": gt_feats["corr"], "fc_corr": fc_feats["corr"],
            "gt_res": gt_feats["lattice_residual"], "fc_res": fc_feats["lattice_residual"],
            "gt_row": gt_feats["row_consistency"], "fc_row": fc_feats["row_consistency"],
            "gt_col": gt_feats["col_consistency"], "fc_col": fc_feats["col_consistency"]
        })
        
    res_df = pd.DataFrame(results)
    print(f"Evaluated {len(res_df)} competitive pairs.")
    
    # Win Rates (GT better than False Cand)
    # For residual, lower is better. For consistency, higher is better.
    res_win = (res_df["gt_res"] < res_df["fc_res"]).mean() * 100
    row_win = (res_df["gt_row"] > res_df["fc_row"]).mean() * 100
    col_win = (res_df["gt_col"] > res_df["fc_col"]).mean() * 100
    corr_win = (res_df["gt_corr"] > res_df["fc_corr"]).mean() * 100
    
    # Combined Consistency
    gt_tot_c = res_df["gt_row"] + res_df["gt_col"]
    fc_tot_c = res_df["fc_row"] + res_df["fc_col"]
    tot_c_win = (gt_tot_c > fc_tot_c).mean() * 100
    
    print("\\n=== FORENSIC PAIRWISE WIN RATES (GT > Replica) ===")
    print(f"Local NCC          : {corr_win:.1f}%")
    print(f"Lattice Residual   : {res_win:.1f}%")
    print(f"Row Consistency    : {row_win:.1f}%")
    print(f"Col Consistency    : {col_win:.1f}%")
    print(f"Total Consistency  : {tot_c_win:.1f}%")
    
    # Means and Deltas
    print("\\n=== FEATURE DELTAS (GT - Replica) ===")
    print(f"Mean Lattice Res Delta : {(res_df['gt_res'] - res_df['fc_res']).mean():.2f} px")
    print(f"Mean Row Consist Delta : {(res_df['gt_row'] - res_df['fc_row']).mean():.2f}")
    print(f"Mean Col Consist Delta : {(res_df['gt_col'] - res_df['fc_col']).mean():.2f}")

if __name__ == "__main__":
    falsify_lattice()
