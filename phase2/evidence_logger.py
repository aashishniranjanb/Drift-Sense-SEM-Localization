import os
import pandas as pd
import numpy as np

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "phase2", "candidate_evidence.csv")

def log_candidate_evidence(pair_id: str, candidates: list, gt_x: float, gt_y: float, gt_found: int):
    """
    Logs comprehensive feature evidence for all candidates to candidate_evidence.csv.
    """
    records = []
    
    for idx, c in enumerate(candidates):
        # Calculate if candidate is correct
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
            "corr_score": float(c.get("corr_score", 0.0)),
            "fft_intensity_score": float(c.get("corr_score", 0.0)),
            "fft_gradient_score": float(c.get("fft_gradient_score", 0.0)),
            "peak_margin": float(c.get("peak_margin", 0.0)),
            "psr": float(c.get("psr", 0.0)),
            "context_32": float(c.get("context_32", 0.0)),
            "context_64": float(c.get("context_64", 0.0)),
            "context_128": float(c.get("context_128", 0.0)),
            "phase_dx": float(c.get("phase_dx", 0.0)),
            "phase_dy": float(c.get("phase_dy", 0.0)),
            "phase_residual": float(c.get("phase_residual", 0.0)),
            "scale_consistency": float(c.get("scale_consistency", 0.0)),
            "rotation_consistency": float(c.get("rotation_consistency", 0.0)),
            "periodicity_distance": float(c.get("periodicity_distance", 0.0)),
            "periodicity_index": float(c.get("periodicity_index", 0.0)),
            "template_residual": float(c.get("template_residual", 0.0)),
            "edge_similarity": float(c.get("edge_similarity", 0.0)),
            "center_prior": float(c.get("center_prior", 0.0)),
            "PACE_score": float(c.get("pace_score", 0.0)),
            "score_combined": float(c.get("score_combined", 0.0)),
            "correct": correct
        }
        records.append(record)
        
    df_new = pd.DataFrame(records)
    
    # Append or create new CSV
    if os.path.exists(CSV_PATH):
        try:
            df_old = pd.read_csv(CSV_PATH)
            # Remove existing rows for this pair to avoid duplication
            df_old = df_old[df_old["pair_id"] != pair_id]
            df_final = pd.concat([df_old, df_new], ignore_index=True)
            df_final.to_csv(CSV_PATH, index=False)
        except Exception:
            df_new.to_csv(CSV_PATH, index=False)
    else:
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        df_new.to_csv(CSV_PATH, index=False)
