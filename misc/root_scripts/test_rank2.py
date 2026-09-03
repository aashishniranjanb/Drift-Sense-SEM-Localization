import cv2, sys
sys.path.append('production_engine')
from production_runner import run_production_localization
ref_img = cv2.imread('data/phase2_dev/reference/pair_000.png', 0)
search_img = cv2.imread('data/phase2_dev/search/pair_000.png', 0)

def mock_rank(cands, *args):
    import sys
    sys.path.append('team/akhilesh-localization')
    import replica_ranker
    ranked = replica_ranker.rank_candidates_akhilesh(cands, *args)
    for c in ranked[:3]:
        print(f"Ranked: cx={c['cx']:.1f}, cy={c['cy']:.1f}, d_score={c.get('discriminator_score', -1):.3f}")
    return ranked

import sys
sys.path.append('team/akhilesh-localization')
import replica_ranker
replica_ranker.rank_candidates_akhilesh = mock_rank
res = run_production_localization(ref_img, search_img, verbose=False)
print("FINAL RESULT:", res)
