"""Phase 6 retune — extract candidate-level features from the multi-hypothesis
pool for every dev pair, labelled by ground-truth proximity.

out: phase2/V48_MAX/mh_pool_features.csv  (one row per candidate)
     phase2/V48_MAX/mh_pair_features.csv   (one row per pair: presence-model inputs)
"""
import os, sys, time
import numpy as np, pandas as pd, cv2
from concurrent.futures import ProcessPoolExecutor

_SRC = os.path.join("FINAL_SUBMISSION", "runtime", "src")
OUT_C = "phase2/V48_MAX/mh_pool_features.csv"
OUT_P = "phase2/V48_MAX/mh_pair_features.csv"

FEATS = ["corr_score", "psr", "context_128", "context_combined", "phase_penalty",
         "dist_to_center", "neigh_cons", "grad_ncc"]


def process(row):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    cv2.setNumThreads(1)
    sys.path.insert(0, _SRC)
    from matcher import (multi_hypothesis_search, compute_neighborhood_consistency,
                         compute_gradient_ncc)
    from candidate_extractor import extract_nms_fast, cluster_replica_families
    from context_matcher import verify_candidate_context
    from phase_verifier import verify_phase_consistency
    from periodicity_detector import estimate_periodicity_from_corr
    import pipeline as P

    pid = row["pair_id"]
    gtf = int(row["gt_found"]); gx, gy = float(row["gt_x"]), float(row["gt_y"])
    ref = cv2.imread(os.path.join("data/phase2_dev", row["reference_path"].replace("\\", "/")), 0)
    srch = cv2.imread(os.path.join("data/phase2_dev", row["search_path"].replace("\\", "/")), 0)

    hyps = multi_hypothesis_search(ref, srch, k_scale=P._K_SCALE, m_rot=P._M_ROT)
    sh, sw = srch.shape
    scx, scy = sw / 2.0, sh / 2.0
    pools = []
    for hi, h in enumerate(hyps):
        tw, th = h["best_template"].shape[::-1]
        ch = extract_nms_fast(h["corr_plane"], tw, th, max_k=P._PER_HYP_K, r=P._NMS_R)
        for c in ch:
            c["dist_to_center"] = float(np.hypot(c["cx"] - scx, c["cy"] - scy))
        pools.append((hi, h, ch))
    cands = P._merge_pools(pools)
    cands = cluster_replica_families(cands, float(hyps[0]["best_scale"]))
    if not cands:
        return [], None

    per = estimate_periodicity_from_corr(hyps[0]["corr_plane"])
    px_, py_ = per["pitch_x"], per["pitch_y"]
    mode_strong = 1 if per["mode"] == "STRONG" else 0

    rows = []
    for c in cands:
        cx, cy = c["cx"], c["cy"]
        pkx, pky = c["peak_x"], c["peak_y"]
        hs = c.get("hscale", float(hyps[0]["best_scale"]))
        ht = c.get("htheta", float(hyps[0]["best_theta"]))
        tmpl = c.get("htemplate", hyps[0]["best_template"])
        ctx = verify_candidate_context(ref, srch, cx, cy, hs, ht)
        pen = verify_phase_consistency(srch, tmpl, pkx, pky)
        neigh = compute_neighborhood_consistency(srch, tmpl, pkx, pky, px_, py_)
        g = compute_gradient_ncc(srch, tmpl, pkx, pky)
        gterr = float(np.hypot(cx - gx, cy - gy)) if gtf == 1 else 1e9
        rows.append(dict(pair_id=pid, set_type=row["set_type"], gt_found=gtf,
                         cx=cx, cy=cy, h_idx=int(c.get("h_idx", 0)), hscale=hs, htheta=ht,
                         corr_score=c["corr_score"], psr=c.get("psr", 0.0),
                         context_128=ctx["s128"], context_combined=ctx["combined"],
                         phase_penalty=pen, dist_to_center=c["dist_to_center"],
                         neigh_cons=neigh, grad_ncc=g,
                         family_population=c.get("family_population", 1),
                         gterr=gterr, is_correct=int(gtf == 1 and gterr <= 5.0)))
    dfp = pd.DataFrame(rows)
    for col in FEATS:
        dfp[col + "_rel"] = dfp[col] - dfp[col].median()
    dfp["family_ratio"] = dfp["family_population"] / len(dfp)
    dfp["rank_corr"] = dfp["corr_score"].rank(ascending=False).astype(int)
    dfp["pool_size"] = len(dfp)

    # pair-level presence features: computed both for the raw-#1 candidate (frozen
    # V25 indexing) and, as a diagnostic, for the true best candidate by is_correct
    top = dfp.iloc[0]
    pairrow = dict(pair_id=pid, set_type=row["set_type"], gt_found=gtf,
                   pool_size=len(dfp), n_hyp=len(hyps), mode_strong=mode_strong,
                   pool_has_correct=int(dfp["is_correct"].max() == 1),
                   best_gterr=float(dfp["gterr"].min()),
                   raw1_corr=float(top["corr_score"]), raw1_ctx=float(top["context_combined"]),
                   raw1_neigh=float(top["neigh_cons"]), raw1_grad=float(top["grad_ncc"]))
    return rows, pairrow


if __name__ == "__main__":
    t0 = time.time()
    pairs = pd.read_csv("data/phase2_dev/pairs.csv")
    recs = [r for _, r in pairs.iterrows()]
    all_c, all_p = [], []
    with ProcessPoolExecutor(max_workers=5) as ex:
        for i, (rc, pr) in enumerate(ex.map(process, recs)):
            all_c.extend(rc)
            if pr:
                all_p.append(pr)
            if (i + 1) % 20 == 0:
                print(f"{i+1}/180  {time.time()-t0:.0f}s  cand rows={len(all_c)}", flush=True)
    dc = pd.DataFrame(all_c)
    dp = pd.DataFrame(all_p)
    dc.to_csv(OUT_C, index=False)
    dp.to_csv(OUT_P, index=False)
    print(f"\nwrote {OUT_C} ({len(dc)} rows) and {OUT_P} ({len(dp)} rows)  {time.time()-t0:.0f}s")
    pres = dp[dp.gt_found == 1]
    print(f"present pairs whose multi-hyp pool contains a <=5px candidate: "
          f"{int(pres.pool_has_correct.sum())}/{len(pres)}   "
          f"(SetA {int(pres[pres.set_type=='SetA'].pool_has_correct.sum())}/{sum(pres.set_type=='SetA')}, "
          f"SetB {int(pres[pres.set_type=='SetB'].pool_has_correct.sum())}/{sum(pres.set_type=='SetB')})")
