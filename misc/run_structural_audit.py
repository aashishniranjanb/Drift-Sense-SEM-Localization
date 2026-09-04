import os, sys, cv2, json, time, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, 'FINAL_SUBMISSION/runtime/src')
sys.path.insert(0, 'FINAL_SUBMISSION/validation/retrieval')
from utils import rotate_image
from candidate_extractor import extract_candidates_akhilesh, cluster_replica_families
from context_matcher import verify_candidate_context
from phase_verifier import verify_phase_consistency
from matcher import compute_neighborhood_consistency, compute_gradient_ncc
import rerank_features
from build_retrieval_v2 import extract_multi_source_union


def process_pair_audit(args):
    pid, ref_p, srch_p, gt_x, gt_y, est_scale, est_theta, fail_cat = args
    ref = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(srch_p, cv2.IMREAD_GRAYSCALE)
    if ref is None or srch is None: return None

    sh, sw = srch.shape[:2]
    if est_scale <= 0.01: est_scale = 10.0
    tw = max(16, int(round(ref.shape[1] / est_scale)))
    th = max(16, int(round(ref.shape[0] / est_scale)))
    tpl = cv2.resize(ref.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
    tpl_rot = rotate_image(tpl, est_theta) if abs(est_theta) > 0.01 else tpl
    corr = cv2.matchTemplate(srch.astype(np.float32), tpl_rot, cv2.TM_CCOEFF_NORMED)

    union = extract_multi_source_union(ref, srch, est_scale, est_theta, max_total_k=300)
    if not union: return None

    def get_struct_score(c):
        cx, cy = float(c["cx"]), float(c["cy"])
        px, py = c["peak_x"], c["peak_y"]
        
        # Require candidate to be inside valid image bounds
        if cx - tw/2.0 < 5.0 or cx + tw/2.0 > sw - 5.0 or cy - th/2.0 < 5.0 or cy + th/2.0 > sh - 5.0:
            return -999.0

        ctx = verify_candidate_context(ref, srch, cx, cy, est_scale, est_theta)
        gncc = float(compute_gradient_ncc(srch, tpl_rot, px, py))
        neigh = float(compute_neighborhood_consistency(srch, tpl_rot, px, py, 20.0, 20.0))
        phase_pen = float(verify_phase_consistency(srch, tpl_rot, px, py))
        
        # Use primary correlation score + gradient NCC + context + neighborhood
        corr_val = float(corr[int(round(py)), int(round(px))]) if (0 <= int(round(py)) < corr.shape[0] and 0 <= int(round(px)) < corr.shape[1]) else c["score"]
        
        return (0.35 * corr_val +
                0.30 * ctx["combined"] +
                0.25 * gncc +
                0.15 * neigh -
                0.50 * phase_pen)

    scored = []
    for idx, c in enumerate(union):
        s = get_struct_score(c)
        err = float(np.hypot(c['cx'] - gt_x, c['cy'] - gt_y))
        scored.append({"struct_score": float(s), "err": err, "cx": float(c['cx']), "cy": float(c['cy']), "cand_idx": idx})

    scored.sort(key=lambda x: x["struct_score"], reverse=True)
    best_s = scored[0]["struct_score"]
    best_err = scored[0]["err"]

    gt_win = (best_err <= 10.0)

    return {
        "pair_id": pid,
        "category": fail_cat,
        "best_err": best_err,
        "best_struct_score": best_s,
        "gt_win": gt_win
    }


def main():
    gt_df = pd.read_csv('data/phase2_dev/pairs.csv')
    v54_pred = pd.read_csv('FINAL_SUBMISSION_GOLDEN/predictions.csv')
    rf = pd.read_csv('FINAL_SUBMISSION/validation/candidate_pool_audit_140.csv')

    fails_df = rf[rf['category'].isin(['RANKING_FAILURE', 'RETRIEVAL_FAILURE'])]
    print(f"Parallel analyzing {len(fails_df)} failure pairs with interior bounds guard (8 workers)...")

    tasks = []
    for _, row in fails_df.iterrows():
        pid = row['pair_id']
        gt_row = gt_df[gt_df['pair_id'] == pid].iloc[0]
        v54_r = v54_pred[v54_pred['pair_id'] == pid].iloc[0]
        ref_p = os.path.join('data/phase2_dev', gt_row['reference_path'].replace('\\', '/'))
        srch_p = os.path.join('data/phase2_dev', gt_row['search_path'].replace('\\', '/'))
        tasks.append((
            pid, ref_p, srch_p,
            float(gt_row['gt_x']), float(gt_row['gt_y']),
            float(v54_r['scale']), float(v54_r['theta']),
            row['category']
        ))

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(process_pair_audit, tasks):
            if res is not None:
                results.append(res)

    print(f"Completed in {time.time()-t0:.1f}s.")
    df_out = pd.DataFrame(results)
    os.makedirs("FINAL_SUBMISSION/validation/reranking", exist_ok=True)
    df_out.to_csv("FINAL_SUBMISSION/validation/reranking/structural_score_audit.csv", index=False)

    print("Saved audit CSV to FINAL_SUBMISSION/validation/reranking/structural_score_audit.csv")
    print("GT Wins Count (err <= 10px):", int(df_out["gt_win"].sum()), "/", len(df_out))
    print(df_out['best_err'].describe())


if __name__ == "__main__":
    main()
