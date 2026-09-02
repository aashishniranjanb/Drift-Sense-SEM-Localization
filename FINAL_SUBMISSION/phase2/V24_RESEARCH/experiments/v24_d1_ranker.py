"""
V24-D1 Controlled Architecture-Aware Ranker
Evaluates the integration of global lattice and architectural consistency.
"""
import pandas as pd
import numpy as np

def get_pitch_and_conf(d_array, K):
    if len(d_array) == 0: return 0.0, 0.0
    counts, bins = np.histogram(d_array, bins=np.arange(0, 800, 5))
    if len(counts) == 0: return 0.0, 0.0
    best_idx = np.argmax(counts)
    pitch = bins[best_idx] + 2.5
    conf = counts[best_idx] / len(d_array) if len(d_array) > 0 else 0.0
    return pitch, conf

def get_global_phase(vals, pitch, weights=None):
    if pitch == 0: return 0.0
    angles = 2 * np.pi * vals / pitch
    mean_cos = np.average(np.cos(angles), weights=weights)
    mean_sin = np.average(np.sin(angles), weights=weights)
    phase = np.arctan2(mean_sin, mean_cos) * pitch / (2 * np.pi)
    if phase < 0: phase += pitch
    return phase

def build_features(df_pool, pairs_df, K=200):
    df_sorted = df_pool.sort_values(["pair_id", "corr_score"], ascending=[True, False])
    
    out_rows = []
    pitch_stats = []
    
    for pair_id, group in df_sorted.groupby("pair_id"):
        pool = group.head(K).copy()
        if pool["is_correct"].sum() == 0:
            continue
            
        set_type = pairs_df[pairs_df.pair_id == pair_id].iloc[0].set_type
            
        cx_vals = pool["cx"].values
        cy_vals = pool["cy"].values
        corr_vals = pool["corr_score"].values
        
        dx_mat = np.abs(cx_vals[:, None] - cx_vals[None, :])
        dy_mat = np.abs(cy_vals[:, None] - cy_vals[None, :])
        dx = dx_mat[dx_mat > 15]
        dy = dy_mat[dy_mat > 15]
        
        px, conf_x = get_pitch_and_conf(dx, K)
        py, conf_y = get_pitch_and_conf(dy, K)
        p_conf = max(conf_x, conf_y)
        
        pitch_stats.append({"pair_id": pair_id, "K": K, "px": px, "py": py, "p_conf": p_conf})
        
        w = corr_vals ** 3
        ph_x = get_global_phase(cx_vals, px, w)
        ph_y = get_global_phase(cy_vals, py, w)
        
        pool["px"] = px; pool["py"] = py
        pool["p_conf"] = p_conf
        pool["set_type"] = set_type
        
        res_raw = []
        res_norm = []
        tot_c = []
        
        for i, c in pool.iterrows():
            rx = 0.0; ry = 0.0
            if px > 0:
                diff = abs(c["cx"] - ph_x) % px
                rx = min(diff, px - diff)
            if py > 0:
                diff = abs(c["cy"] - ph_y) % py
                ry = min(diff, py - diff)
                
            r_raw = np.hypot(rx, ry)
            r_norm = np.hypot(rx/px if px>0 else 0, ry/py if py>0 else 0)
            
            # Consistency
            r_c = 0.0; c_c = 0.0
            tol = 15
            if px > 0:
                mask = (np.abs(cy_vals - c["cy"]) < tol) & (np.abs(cx_vals - c["cx"]) > 15)
                rems = np.abs(cx_vals[mask] - c["cx"]) % px
                valid = (rems < tol) | (rems > px - tol)
                r_c = np.sum(corr_vals[mask][valid])
            if py > 0:
                mask = (np.abs(cx_vals - c["cx"]) < tol) & (np.abs(cy_vals - c["cy"]) > 15)
                rems = np.abs(cy_vals[mask] - c["cy"]) % py
                valid = (rems < tol) | (rems > py - tol)
                c_c = np.sum(corr_vals[mask][valid])
                
            res_raw.append(r_raw)
            res_norm.append(r_norm)
            tot_c.append(r_c + c_c)
            
        pool["raw_res"] = res_raw
        pool["norm_res"] = res_norm
        pool["tot_c"] = tot_c
        
        # Normalize consistency within pool
        if pool["tot_c"].max() > 0:
            pool["tot_c_norm"] = pool["tot_c"] / pool["tot_c"].max()
        else:
            pool["tot_c_norm"] = 0.0
            
        out_rows.append(pool)
        
    return pd.concat(out_rows), pd.DataFrame(pitch_stats)

def evaluate_ranker(pool_df, score_col):
    res = {"gt_in_pool": 0, "correct": 0, "setA_gt": 0, "setA_corr": 0, "setB_gt": 0, "setB_corr": 0, "per_gt": 0, "per_corr": 0}
    
    for pair_id, group in pool_df.groupby("pair_id"):
        res["gt_in_pool"] += 1
        is_setA = (group.iloc[0].set_type == "SetA")
        is_periodic = (group.iloc[0].p_conf > 0.06) # Dynamic periodicity threshold
        
        if is_setA: res["setA_gt"] += 1
        else: res["setB_gt"] += 1
        
        if is_periodic: res["per_gt"] += 1
        
        best_idx = group[score_col].idxmax()
        if group.loc[best_idx, "is_correct"] == 1:
            res["correct"] += 1
            if is_setA: res["setA_corr"] += 1
            else: res["setB_corr"] += 1
            if is_periodic: res["per_corr"] += 1
            
    # Calculate percents
    def p(c, t): return (c/t)*100 if t>0 else 0
    return {
        "Cond_Top1": p(res["correct"], res["gt_in_pool"]),
        "SetA_Cond": p(res["setA_corr"], res["setA_gt"]),
        "SetB_Cond": p(res["setB_corr"], res["setB_gt"]),
        "Per_Cond": p(res["per_corr"], res["per_gt"])
    }

def run_d1_experiments():
    print("Loading data...")
    df = pd.read_csv("phase2/V22_CHAMPIONSHIP/results/candidate_pool_features.csv")
    pairs = pd.read_csv("data/phase2_dev/pairs.csv")
    
    print("Building K=200 features...")
    pool200, stats200 = build_features(df, pairs, K=200)
    
    # Pre-calculate base scores
    # Base = corr + 0.15*ctx - 0.20*phase_pen
    pool200["base_score"] = pool200["corr_score"] + 0.15 * pool200["context_128"].fillna(0) - 0.20 * pool200["phase_penalty"].fillna(0)
    
    # D1-0: Control V24-A+ (adaptive center on p_conf)
    def calc_center_pen(row, base_w=0.04):
        if row["p_conf"] > 0.06:
            w = min(0.15, 0.04 + 0.8 * (row["p_conf"] - 0.06))
        else:
            w = 0.02
        return w * ((row["dist_to_center"] / 250.0) ** 2)
        
    pool200["D1_0"] = pool200["base_score"] - pool200.apply(calc_center_pen, axis=1)
    
    print("\\n=== D1-0: Control (V24-A+) ===")
    r0 = evaluate_ranker(pool200, "D1_0")
    print(f"Cond Top-1: {r0['Cond_Top1']:.2f}% | Set A: {r0['SetA_Cond']:.2f}% | Set B: {r0['SetB_Cond']:.2f}% | Per: {r0['Per_Cond']:.2f}%")
    
    print("\\n=== D1-1: Lattice Residual Sweep ===")
    for lam in [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]:
        col = f"D1_1_{lam}"
        pool200[col] = pool200["D1_0"] - lam * pool200["norm_res"]
        r = evaluate_ranker(pool200, col)
        print(f"lam={lam:<4} -> Cond Top-1: {r['Cond_Top1']:.2f}% (SetA: {r['SetA_Cond']:.1f}%, SetB: {r['SetB_Cond']:.1f}%)")
        
    # Pick best lambda (e.g. 0.20) for next step
    best_lam = 0.20
    pool200["D1_1_best"] = pool200["D1_0"] - best_lam * pool200["norm_res"]
    
    print("\\n=== D1-2: Adaptive Center Prior Sweep ===")
    # D1-2 varies the heavy center weight for periodic cases
    for cw in [0.04, 0.08, 0.12, 0.15, 0.20]:
        col = f"D1_2_{cw}"
        def dyn_center(row):
            w = cw if row["p_conf"] > 0.06 else 0.02
            return w * ((row["dist_to_center"] / 250.0) ** 2)
        pool200[col] = pool200["base_score"] - best_lam * pool200["norm_res"] - pool200.apply(dyn_center, axis=1)
        r = evaluate_ranker(pool200, col)
        print(f"cw={cw:<4} -> Cond Top-1: {r['Cond_Top1']:.2f}% (SetA: {r['SetA_Cond']:.1f}%, SetB: {r['SetB_Cond']:.1f}%)")
        
    best_cw = 0.12
        
    print("\\n=== D1-3: Architecture Composite ===")
    # Add tot_c_norm
    for w_consist in [0.05, 0.10, 0.20]:
        col = f"D1_3_{w_consist}"
        pool200[col] = pool200["base_score"] - best_lam * pool200["norm_res"] + w_consist * pool200["tot_c_norm"] - pool200.apply(lambda r: (best_cw if r["p_conf"]>0.06 else 0.02)*((r["dist_to_center"]/250.0)**2), axis=1)
        r = evaluate_ranker(pool200, col)
        print(f"w_consist={w_consist:<4} -> Cond Top-1: {r['Cond_Top1']:.2f}% (SetB: {r['SetB_Cond']:.1f}%)")
        
    print("\\n=== D1-4: LOFO Ablation (from Composite) ===")
    # Best composite
    comp = pool200["base_score"] - 0.20 * pool200["norm_res"] + 0.10 * pool200["tot_c_norm"] - pool200.apply(lambda r: (0.12 if r["p_conf"]>0.06 else 0.02)*((r["dist_to_center"]/250.0)**2), axis=1)
    pool200["D1_comp"] = comp
    print(f"All Features:      {evaluate_ranker(pool200, 'D1_comp')['Cond_Top1']:.2f}%")
    
    pool200["D1_no_lat"] = comp + 0.20 * pool200["norm_res"]
    print(f"- Lattice Res:     {evaluate_ranker(pool200, 'D1_no_lat')['Cond_Top1']:.2f}%")
    
    pool200["D1_no_cons"] = comp - 0.10 * pool200["tot_c_norm"]
    print(f"- Consistency:     {evaluate_ranker(pool200, 'D1_no_cons')['Cond_Top1']:.2f}%")
    
    pool200["D1_no_cent"] = pool200["base_score"] - 0.20 * pool200["norm_res"] + 0.10 * pool200["tot_c_norm"]
    print(f"- Center Prior:    {evaluate_ranker(pool200, 'D1_no_cent')['Cond_Top1']:.2f}%")
    
    print("\\n=== Pitch Estimator Stability (K=50 vs 100 vs 200) ===")
    p50, s50 = build_features(df, pairs, K=50)
    p100, s100 = build_features(df, pairs, K=100)
    
    for k, s in [(50, s50), (100, s100), (200, stats200)]:
        print(f"K={k:<3} -> Avg px={s.px.mean():.1f}, py={s.py.mean():.1f}, conf={s.p_conf.mean():.3f}")

if __name__ == "__main__":
    run_d1_experiments()
