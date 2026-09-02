import cv2, sys
sys.path.append('production_engine')
from production_runner import run_production_localization
ref_img = cv2.imread('data/phase2_dev/reference/pair_000.png', 0)
search_img = cv2.imread('data/phase2_dev/search/pair_000.png', 0)

def mock_rank(cands, *args):
    for c in cands:
        if c['cy'] > 600 or (c['cy'] > 390 and c['cy'] < 410):
            fam = c.get('family_population', -1)
            fam_ratio = fam / max(1, len(cands))
            d_center = c['center_prior']
            w_c = 0.12 if fam_ratio > 0.08 else 0.04
            cp = (d_center / 250)**2
            score = c['corr_score'] + 0.15 * c['context_score'] - 0.20 * c['phase_penalty'] - w_c * cp
            print(f"cy={c['cy']:.1f}, fam={fam}, fr={fam_ratio:.3f}, w_c={w_c}, cp={cp:.3f}, pen={w_c*cp:.3f}, scr={score:.4f}, c_corr={c['corr_score']:.3f}, c_ctx={c['context_score']:.3f}")
    import sys
    sys.path.append('team/akhilesh-localization')
    import replica_ranker
    return replica_ranker.rank_candidates_akhilesh(cands, *args)

import sys
sys.path.append('team/akhilesh-localization')
import replica_ranker
replica_ranker.rank_candidates_akhilesh = mock_rank
res = run_production_localization(ref_img, search_img, verbose=False)
print("FINAL RESULT:", res)
