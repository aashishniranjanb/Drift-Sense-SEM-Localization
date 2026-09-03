import os
import pandas as pd
import numpy as np

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "phase2", "candidate_features.csv")

def log_candidate_features(pair_id: str, candidates: list, gt_x: float, gt_y: float, gt_found: int):
    """
    Logs comprehensive spatial fingerprint and family features for candidates.
    """
    records = []
    
    for idx, c in enumerate(candidates):
        correct = 0
        if gt_found == 1:
            err = np.hypot(c["cx"] - gt_x, c["cy"] - gt_y)
            if err <= 5.0:
                correct = 1
                
        record = {
            "pair_id": pair_id,
            "candidate_id": idx,
            "cx": float(c["cx"]),
            "cy": float(c["cy"]),
            "corr_score": float(c["corr_score"]),
            "psr": float(c["psr"]),
            "peak_margin": float(c["peak_margin"]),
            "context_64": float(c["context_64"]),
            "context_128": float(c["context_128"]),
            "phase_residual": float(c["phase_residual"]),
            "family_id": int(c.get("family_id", 0)),
            "family_population": int(c.get("family_population", 1)),
            "family_score_variance": float(c.get("family_score_variance", 0.0)),
            "nearest_edge_dist": float(c.get("nearest_edge_dist", 0.0)),
            "nearest_cut_dist": float(c.get("nearest_cut_dist", 0.0)),
            "row_spacing": float(c.get("row_spacing", 0.0)),
            "col_spacing": float(c.get("col_spacing", 0.0)),
            "local_density": float(c.get("local_density", 0.0)),
            "center_prior": float(c["center_prior"]),
            "correct": correct
        }
        records.append(record)
        
    df_new = pd.DataFrame(records)
    
    if os.path.exists(CSV_PATH):
        try:
            df_old = pd.read_csv(CSV_PATH)
            df_old = df_old[df_old["pair_id"] != pair_id]
            df_final = pd.concat([df_old, df_new], ignore_index=True)
            df_final.to_csv(CSV_PATH, index=False)
        except Exception:
            df_new.to_csv(CSV_PATH, index=False)
    else:
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        df_new.to_csv(CSV_PATH, index=False)

def rank_candidates(candidates: list) -> list:
    """
    Ranks candidates using a lightweight structural evidence equation:
    score = 0.40 * corr_score + 0.35 * context_128 + 0.25 * phase_residual - 0.05 * (center_prior / 500.0)
    
    Also factors in the Replica Family size (boosting isolated candidates or penalty based on variance).
    """
    if len(candidates) == 0:
        return candidates
        
    for c in candidates:
        corr_val = c["corr_score"]
        ctx_val = c["context_128"]
        phase_val = c["phase_residual"]
        dist = c["center_prior"]
        
        # Base rank score from empirical delta findings
        rank_score = 0.40 * corr_val + 0.35 * ctx_val + 0.25 * phase_val - 0.05 * (dist / 500.0)
        
        # Penalize members of large highly-variable replica families (confused field)
        pop = c.get("family_population", 1)
        var = c.get("family_score_variance", 0.0)
        if pop > 3 and var < 0.002:
            # High consistency, high ambiguity family penalty
            rank_score -= 0.03
            
        c["rank_score"] = float(rank_score)
        
    # Re-sort candidates by rank_score
    candidates.sort(key=lambda x: x["rank_score"], reverse=True)
    return candidates
