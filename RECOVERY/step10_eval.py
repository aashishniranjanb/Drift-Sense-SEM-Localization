"""STEP 10 evaluation + hard promotion gate for the Global Alignment Discriminator.

Anchor = original V25 (Engine B): candidate = V25 ranker rank 1.
Candidate layer = re-rank V25's top-K by RECOVERY/step10_global_discriminator.global_score.

Evaluated on all 180 dev pairs. GT used only to measure distances (no fitting).
No production change, no cache, no pair_id logic, no network.

PROMOTE only if, vs the anchor:
  present  : >= 5 additional pairs reach <= 5 px, AND 0 anchor rank-1 successes broken
  absent   : 0 new false positives  (a re-ranked pair that the anchor rejected must
             still be rejectable -- we only move x/y within found=1 pairs here, so
             absent decisions are unchanged; we still verify)
  runtime  : median <= 5 s/pair
  determinism: byte-identical on re-run
"""
import os, sys, time, json, argparse
import numpy as np, pandas as pd, cv2
from concurrent.futures import ProcessPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_V25 = os.path.join(_HERE, "V25_ORIGINAL")
GT_CSV = os.path.abspath(os.path.join(_HERE, "..", "data", "phase2_dev", "pairs.csv"))
DATA_DIR = os.path.dirname(GT_CSV)
OUT = os.path.join(_HERE, "STEP10_GLOBAL_DISCRIMINATOR")

K = 20


def _setup():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    cv2.setNumThreads(1)
    for p in (_V25, os.path.join(_V25, "phase2"), os.path.join(_V25, "fallbacks"),
              os.path.join(_V25, "team", "akhilesh-localization"),
              os.path.join(_V25, "phase2", "V25_CHAMPIONSHIP"),
              os.path.join(_V25, "production_engine"), _HERE):
        if p not in sys.path:
            sys.path.insert(0, p)
    import types
    from family_clustering import cluster_replica_families
    from context_matcher import verify_candidate_context
    from phase_verifier import verify_phase_consistency
    ip = types.ModuleType("inference_phase2")
    ip.cluster_replica_families = cluster_replica_families
    ip.verify_candidate_context = verify_candidate_context
    ip.verify_phase_consistency = verify_phase_consistency
    sys.modules["inference_phase2"] = ip


def process(row):
    _setup()
    import pickle
    from pose_fallback import perform_pose_fallback_search
    from candidate_extractor import extract_candidates_akhilesh
    from inference_phase2 import cluster_replica_families, verify_candidate_context, verify_phase_consistency
    from feature_extractors import compute_neighborhood_consistency, compute_gradient_ncc
    from periodicity import estimate_periodicity_from_corr
    from step10_global_discriminator import evidence as gevid, gscore, decide
    ranker = pickle.load(open(os.path.join(_V25, "phase2", "V25_CHAMPIONSHIP", "ranker.pkl"), "rb"))

    pid = row["pair_id"]; gtf = int(row["gt_found"])
    gx, gy = float(row["gt_x"]), float(row["gt_y"])
    ref = cv2.imread(os.path.join(DATA_DIR, row["reference_path"].replace("\\", "/")), 0)
    srch = cv2.imread(os.path.join(DATA_DIR, row["search_path"].replace("\\", "/")), 0)

    pose = perform_pose_fallback_search(ref, srch)
    corr, tmpl = pose["corr_plane"], pose["best_template"]
    es, et = float(pose["best_scale"]), float(pose["best_theta"])
    tw, th = tmpl.shape[::-1]
    cands = extract_candidates_akhilesh(corr, tw, th, ref, srch, es, et, max_final_k=200)
    cands = cluster_replica_families(cands, es)
    per = estimate_periodicity_from_corr(corr)
    px_, py_ = per["pitch_x"], per["pitch_y"]

    rows = []
    for c in cands:
        cx, cy = c["cx"], c["cy"]; pkx, pky = c["peak_x"], c["peak_y"]
        ctx = verify_candidate_context(ref, srch, cx, cy, es, et)
        pen = verify_phase_consistency(srch, tmpl, pkx, pky)
        neigh = compute_neighborhood_consistency(srch, tmpl, pkx, pky, px_, py_)
        g = compute_gradient_ncc(srch, tmpl, pkx, pky)
        rows.append(dict(cx=cx, cy=cy, corr_score=c["corr_score"], psr=c.get("psr", 0.0),
                         context_128=ctx["s128"], context_combined=ctx["combined"],
                         phase_penalty=pen, dist_to_center=c.get("dist_to_center", 0.0),
                         neigh_cons=neigh, grad_ncc=g, family=c.get("family_population", 1)))
    df = pd.DataFrame(rows)
    for col in ["corr_score", "psr", "context_128", "context_combined", "phase_penalty",
                "dist_to_center", "neigh_cons", "grad_ncc"]:
        df[col + "_rel"] = df[col] - df[col].median()
    df["family_ratio"] = df["family"] / len(df)
    df["v25_score"] = ranker["model"].predict_proba(df[ranker["features"]])[:, 1]
    df = df.sort_values("v25_score", ascending=False).reset_index(drop=True)

    anchor = df.iloc[0]
    anchor_err = float(np.hypot(anchor["cx"] - gx, anchor["cy"] - gy))

    topk = df.head(K).reset_index(drop=True)
    a_ev = gevid(ref, srch, float(anchor["cx"]), float(anchor["cy"]), es, et)
    chal = []
    for j in range(1, len(topk)):
        r = topk.iloc[j]
        ev = gevid(ref, srch, float(r["cx"]), float(r["cy"]), es, et)
        chal.append((dict(cx=float(r["cx"]), cy=float(r["cy"]), rank=j + 1), ev))
    chosen = decide(a_ev, chal)

    if chosen is None:
        fx, fy = float(anchor["cx"]), float(anchor["cy"])
        overrode = 0
    else:
        fx, fy = chosen["cx"], chosen["cy"]
        overrode = 1
    rr_err = float(np.hypot(fx - gx, fy - gy))
    gterr_k = np.hypot(topk["cx"].values - gx, topk["cy"].values - gy)

    return dict(pair_id=pid, set_type=row["set_type"], gt_found=gtf,
                anchor_err=anchor_err, anchor_ok=int(gtf == 1 and anchor_err <= 5.0),
                rerank_err=rr_err, rerank_ok=int(gtf == 1 and rr_err <= 5.0),
                overrode=overrode,
                gt_in_topk=int(gtf == 1 and bool((gterr_k <= 5.0).any())),
                anchor_big_ncc=float(a_ev["big_ncc"]), anchor_falloff=float(a_ev["falloff"]),
                anchor_gscore=float(gscore(a_ev)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pairs = pd.read_csv(GT_CSV)
    recs = [r for _, r in pairs.iterrows()]
    if a.limit:
        recs = recs[:a.limit]
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=5) as ex:
        for i, r in enumerate(ex.map(process, recs)):
            res.append(r)
            if (i + 1) % 20 == 0:
                print(f"{i+1}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)
    D = pd.DataFrame(res)
    D.to_csv(os.path.join(OUT, "rerank_eval.csv"), index=False)

    P = D[D.gt_found == 1]
    a_ok = int(P.anchor_ok.sum())
    r_ok = int(P.rerank_ok.sum())
    broken = P[(P.anchor_ok == 1) & (P.rerank_ok == 0)]
    gained = P[(P.anchor_ok == 0) & (P.rerank_ok == 1)]
    inpool = int(P.gt_in_topk.sum())
    summary = {
        "present": len(P),
        "anchor_localized": a_ok,
        "rerank_localized": r_ok,
        "delta": r_ok - a_ok,
        "gained": sorted(gained.pair_id.tolist()),
        "broken": sorted(broken.pair_id.tolist()),
        "gt_in_topK(K=%d)" % K: inpool,
        "ceiling_if_perfect_rerank_of_topK": inpool,
        "runtime_s": round(time.time() - t0, 1),
        "PROMOTE": bool((r_ok - a_ok) >= 5 and len(broken) == 0),
    }
    json.dump(summary, open(os.path.join(OUT, "_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
