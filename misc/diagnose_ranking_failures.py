import os, sys, cv2, numpy as np, pandas as pd
sys.path.insert(0, 'FINAL_SUBMISSION/runtime/src')
from utils import rotate_image
from candidate_extractor import extract_candidates_akhilesh, cluster_replica_families
from context_matcher import verify_candidate_context
from phase_verifier import verify_phase_consistency
from matcher import compute_neighborhood_consistency, compute_gradient_ncc
import rerank_features

gt_df = pd.read_csv('data/phase2_dev/pairs.csv')
v54_pred = pd.read_csv('FINAL_SUBMISSION_GOLDEN/predictions.csv')
rf = pd.read_csv('FINAL_SUBMISSION/validation/candidate_pool_audit_140.csv')

rank_fails = rf[rf['category'] == 'RANKING_FAILURE']['pair_id'].tolist()
print(f'Analyzing feature signatures for {len(rank_fails)} ranking failures...')

for pid in rank_fails[:10]:
    row = gt_df[gt_df['pair_id'] == pid].iloc[0]
    v54_r = v54_pred[v54_pred['pair_id'] == pid].iloc[0]
    ref_p = os.path.join('data/phase2_dev', row['reference_path'].replace('\\', '/'))
    srch_p = os.path.join('data/phase2_dev', row['search_path'].replace('\\', '/'))
    gt_x, gt_y = float(row['gt_x']), float(row['gt_y'])
    est_scale, est_theta = float(v54_r['scale']), float(v54_r['theta'])
    if est_scale <= 0.01: est_scale = 10.0

    ref = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(srch_p, cv2.IMREAD_GRAYSCALE)
    tw = max(16, int(round(ref.shape[1] / est_scale)))
    th = max(16, int(round(ref.shape[0] / est_scale)))
    tpl = cv2.resize(ref.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
    tpl_rot = rotate_image(tpl, est_theta) if abs(est_theta) > 0.01 else tpl
    corr = cv2.matchTemplate(srch.astype(np.float32), tpl_rot, cv2.TM_CCOEFF_NORMED)

    cands = extract_candidates_akhilesh(corr, tw, th, ref, srch, est_scale, est_theta, max_final_k=200)
    cands = cluster_replica_families(cands, est_scale)

    top1 = cands[0]
    gt_cand = None
    gt_rank = -1
    for rank_idx, c in enumerate(cands):
        if np.hypot(c['cx'] - gt_x, c['cy'] - gt_y) <= 5.0:
            gt_cand = c
            gt_rank = rank_idx + 1
            break

    if gt_cand is not None:
        def get_score(c):
            px, py = c['peak_x'], c['peak_y']
            ctx = verify_candidate_context(ref, srch, float(c['cx']), float(c['cy']), est_scale, est_theta)
            gncc = float(compute_gradient_ncc(srch, tpl_rot, px, py))
            neigh = float(compute_neighborhood_consistency(srch, tpl_rot, px, py, 20.0, 20.0))
            morph = rerank_features.compute_candidate_morphology(corr, px, py)
            phase_pen = float(verify_phase_consistency(srch, tpl_rot, px, py))
            return {
                'corr': c['corr_score'], 'ctx': ctx['combined'], 'gncc': gncc,
                'neigh': neigh, 'prom': morph['prominence'], 'sharp': morph['sharpness'],
                'phase_pen': phase_pen
            }

        s1 = get_score(top1)
        sg = get_score(gt_cand)
        c1, cg = s1['corr'], sg['corr']
        x1, xg = s1['ctx'], sg['ctx']
        g1, gg = s1['gncc'], sg['gncc']
        n1, ng = s1['neigh'], sg['neigh']
        p1, pg = s1['phase_pen'], sg['phase_pen']
        print(f'{pid} (GT Rank {gt_rank}):')
        print(f'  Top1: corr={c1:.4f} ctx={x1:.4f} gncc={g1:.4f} neigh={n1:.4f} phase_pen={p1:.4f}')
        print(f'  GT:   corr={cg:.4f} ctx={xg:.4f} gncc={gg:.4f} neigh={ng:.4f} phase_pen={pg:.4f}')
