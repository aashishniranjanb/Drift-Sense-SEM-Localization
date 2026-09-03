import cv2, sys
sys.path.append('production_engine')
from production_runner import run_production_localization
ref_img = cv2.imread('data/phase2_dev/reference/pair_000.png', 0)
search_img = cv2.imread('data/phase2_dev/search/pair_000.png', 0)

import builtins
orig_print = builtins.print
def mock_rank(cands, *args, **kwargs):
    import sys
    sys.path.append('team/akhilesh-localization')
    import replica_ranker
    ranked = replica_ranker.rank_candidates_akhilesh(cands, *args, **kwargs)
    for i, c in enumerate(ranked[:3]):
        orig_print(f"Rank {i}: cx={c['cx']:.1f}, cy={c['cy']:.1f}, score={c.get('discriminator_score', -1):.4f}, fam_pop={c.get('family_population', -1)}")
    return ranked

import sys
sys.path.append('team/akhilesh-localization')
import replica_ranker
replica_ranker.rank_candidates_akhilesh = mock_rank

res = run_production_localization(ref_img, search_img, verbose=False)
