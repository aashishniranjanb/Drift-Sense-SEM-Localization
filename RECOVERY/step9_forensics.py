"""STEP 9 -- Engine-B (original V25, cache-free) candidate forensics.

For every PRESENT dev pair: reproduce Engine B's candidate pipeline exactly
(perform_pose_fallback_search -> extract_candidates_akhilesh 200 ->
cluster_replica_families -> per-candidate V25 evidence -> V25 ranker ml_score),
dump the top-20 by ml_score with full feature vectors and distance to GT, record
the GT rank across the ranked pool, and probe a deeper retrieval pool to
separate ranking failures from retrieval failures.

No training. No production change. No cache. No pair_id logic. No organizer
labels used to fit anything -- GT is used only to measure distances.

Outputs -> RECOVERY/STEP9_CANDIDATE_FORENSICS/
"""
import os, sys, time, json
import numpy as np, pandas as pd, cv2
from concurrent.futures import ProcessPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_V25 = os.path.join(_HERE, "V25_ORIGINAL")
OUT = os.path.join(_HERE, "STEP9_CANDIDATE_FORENSICS")
GT_CSV = os.path.abspath(os.path.join(_HERE, "..", "data", "phase2_dev", "pairs.csv"))
DATA_DIR = os.path.dirname(GT_CSV)


def _setup():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    cv2.setNumThreads(1)
    os.chdir(_V25)
    for p in (_V25, os.path.join(_V25, "phase2"), os.path.join(_V25, "fallbacks"),
              os.path.join(_V25, "team", "akhilesh-localization"),
              os.path.join(_V25, "phase2", "V25_CHAMPIONSHIP"),
              os.path.join(_V25, "production_engine")):
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
    from pose_fallback import perform_pose_fallback_search
    from candidate_extractor import extract_candidates_akhilesh, extract_nms_fast
    from inference_phase2 import cluster_replica_families, verify_candidate_context, verify_phase_consistency
    import pickle
    from feature_extractors import compute_neighborhood_consistency, compute_gradient_ncc
    from periodicity import estimate_periodicity_from_corr
    ck = os.path.join(_V25, "phase2", "V25_CHAMPIONSHIP")
    ranker = pickle.load(open(os.path.join(ck, "ranker.pkl"), "rb"))

    pid = row["pair_id"]; gx, gy = float(row["gt_x"]), float(row["gt_y"])
    ref = cv2.imread(os.path.join(DATA_DIR, row["reference_path"].replace("\\", "/")), 0)
    srch = cv2.imread(os.path.join(DATA_DIR, row["search_path"].replace("\\", "/")), 0)

    pose = perform_pose_fallback_search(ref, srch)
    corr = pose["corr_plane"]; tmpl = pose["best_template"]
    es, et = float(pose["best_scale"]), float(pose["best_theta"])
    tw, th = tmpl.shape[::-1]

    cands = extract_candidates_akhilesh(corr, tw, th, ref, srch, es, et, max_final_k=200)
    cands = cluster_replica_families(cands, es)
    per = estimate_periodicity_from_corr(corr)
    px_, py_ = per["pitch_x"], per["pitch_y"]
    mode_strong = 1 if per["mode"] == "STRONG" else 0

    rows = []
    for c in cands:
        cx, cy = c["cx"], c["cy"]; pkx, pky = c["peak_x"], c["peak_y"]
        ctx = verify_candidate_context(ref, srch, cx, cy, es, et)
        pen = verify_phase_consistency(srch, tmpl, pkx, pky)
        neigh = compute_neighborhood_consistency(srch, tmpl, pkx, pky, px_, py_)
        g = compute_gradient_ncc(srch, tmpl, pkx, pky)
        rows.append(dict(cx=cx, cy=cy, raw_ncc=c["corr_score"], psr=c.get("psr", 0.0),
                         context_128=ctx["s128"], context=ctx["combined"], phase=pen,
                         family=c.get("family_population", 1), center=c.get("dist_to_center", 0.0),
                         neighbor=neigh, gradient=g,
                         dist_to_gt=float(np.hypot(cx - gx, cy - gy))))
    df = pd.DataFrame(rows)
    for col in ["raw_ncc", "psr", "context_128", "context", "phase", "center", "neighbor", "gradient"]:
        df[col + "_rel"] = df[col] - df[col].median()
    df["family_ratio"] = df["family"] / len(df)
    # V25 ranker feature order
    RF = ["corr_score", "psr", "context_128", "context_combined", "phase_penalty",
          "dist_to_center", "neigh_cons", "grad_ncc", "corr_score_rel", "psr_rel",
          "context_128_rel", "context_combined_rel", "phase_penalty_rel",
          "dist_to_center_rel", "neigh_cons_rel", "grad_ncc_rel", "family_ratio"]
    ren = {"raw_ncc": "corr_score", "context": "context_combined", "phase": "phase_penalty",
           "center": "dist_to_center", "neighbor": "neigh_cons", "gradient": "grad_ncc",
           "raw_ncc_rel": "corr_score_rel", "context_rel": "context_combined_rel",
           "phase_rel": "phase_penalty_rel", "center_rel": "dist_to_center_rel",
           "neighbor_rel": "neigh_cons_rel", "gradient_rel": "grad_ncc_rel"}
    Xdf = df.rename(columns=ren)
    df["score"] = ranker["model"].predict_proba(Xdf[ranker["features"]])[:, 1]
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    nxt = df["score"].shift(-1).fillna(df["score"].iloc[-1] if len(df) else 0.0)
    df["margin"] = df["score"] - nxt

    # GT rank across ranked pool
    hits = df.index[df["dist_to_gt"] <= 5.0].tolist()
    gt_rank = int(df.loc[hits[0], "rank"]) if hits else None
    best_gterr = float(df["dist_to_gt"].min())

    # deeper retrieval pool: NMS depth 600 r=3 on the same corr_plane
    deep = extract_nms_fast(corr, tw, th, max_k=600, r=3)
    deep_gterr = min((np.hypot(d["cx"] - gx, d["cy"] - gy) for d in deep), default=1e9)
    gt_in_deep = bool(deep_gterr <= 5.0)

    def topN(n): return bool(any(df["dist_to_gt"].iloc[:min(n, len(df))] <= 5.0))
    flags = {f"gt_in_top{n}": topN(n) for n in (1, 5, 10, 20, 50, 100, 200)}
    flags["gt_in_deep_pool"] = gt_in_deep

    if gt_rank == 1:
        cls = "R1"
    elif gt_rank is not None:
        cls = "R2"
    elif gt_in_deep:
        cls = "R3"
    else:
        cls = "R4"

    top = df.head(20).copy()
    top.insert(0, "pair_id", pid)
    top["set_type"] = row["set_type"]
    atlas_cols = ["pair_id", "set_type", "rank", "cx", "cy", "dist_to_gt", "raw_ncc", "psr",
                  "context", "gradient", "phase", "neighbor", "family", "center", "score", "margin"]
    atlas = top.rename(columns={"cx": "x", "cy": "y"})
    atlas = atlas.rename(columns={"x": "x", "y": "y"})[["pair_id", "set_type", "rank", "raw_ncc"]].assign() if False else atlas
    atlas = top[["pair_id", "set_type", "rank", "cx", "cy", "dist_to_gt", "raw_ncc", "psr",
                 "context", "gradient", "phase", "neighbor", "family", "center", "score", "margin"]]
    atlas = atlas.rename(columns={"cx": "x", "cy": "y"})

    # selected (rank1) vs GT-candidate feature comparison for ranking failures
    cmp = None
    if cls == "R2":
        sel = df.iloc[0]; gtc = df.loc[hits[0]]
        cmp = {"pair_id": pid, "set_type": row["set_type"], "gt_rank": gt_rank,
               "sel_dist_to_gt": float(sel["dist_to_gt"]), "gt_dist_to_gt": float(gtc["dist_to_gt"])}
        for f in ["raw_ncc", "psr", "context", "gradient", "phase", "neighbor", "family", "center", "score", "margin"]:
            cmp[f"sel_{f}"] = float(sel[f]); cmp[f"gt_{f}"] = float(gtc[f])

    pair = dict(pair_id=pid, set_type=row["set_type"], cls=cls, gt_rank=gt_rank,
                best_gterr=best_gterr, deep_gterr=float(deep_gterr), n_cands=len(df),
                mode_strong=mode_strong, **flags)
    return atlas.to_dict("records"), pair, cmp


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pairs = pd.read_csv(GT_CSV)
    present = [r for _, r in pairs.iterrows() if int(r["gt_found"]) == 1]
    print(f"{len(present)} present pairs")
    t0 = time.time()
    atlas_rows, pair_rows, cmp_rows = [], [], []
    with ProcessPoolExecutor(max_workers=5) as ex:
        for i, (a, p, c) in enumerate(ex.map(process, present)):
            atlas_rows.extend(a); pair_rows.append(p)
            if c: cmp_rows.append(c)
            if (i + 1) % 20 == 0:
                print(f"{i+1}/{len(present)}  {time.time()-t0:.0f}s", flush=True)

    A = pd.DataFrame(atlas_rows)
    P = pd.DataFrame(pair_rows)
    A.to_csv(os.path.join(OUT, "candidate_atlas_140.csv"), index=False)
    P.to_csv(os.path.join(OUT, "pair_classification_140.csv"), index=False)
    if cmp_rows:
        pd.DataFrame(cmp_rows).to_csv(os.path.join(OUT, "ranking_failures.csv"), index=False)
    P[P.cls == "R4"].to_csv(os.path.join(OUT, "retrieval_failures.csv"), index=False)

    # recall curve
    rec = {}
    for n in (1, 5, 10, 20, 50, 100, 200):
        rec[f"top{n}"] = int(P[f"gt_in_top{n}"].sum())
    rec["deep_pool"] = int(P["gt_in_deep_pool"].sum())
    pd.DataFrame([{"level": k, "gt_recall": v, "of": len(P)} for k, v in rec.items()]).to_csv(
        os.path.join(OUT, "gt_rank_distribution.csv"), index=False)

    cls_counts = P["cls"].value_counts().to_dict()
    ranks = P.dropna(subset=["gt_rank"])["gt_rank"].astype(int)
    rank_hist = {b: int(((ranks > lo) & (ranks <= hi)).sum())
                 for b, (lo, hi) in {"1": (0, 1), "2-5": (1, 5), "6-10": (5, 10),
                                     "11-20": (10, 20), "21-50": (20, 50),
                                     "51-100": (50, 100), "101-200": (100, 200)}.items()}

    summary = {"present_pairs": len(P), "recall": rec, "classes": cls_counts,
               "gt_rank_hist_in_ranked_pool": rank_hist,
               "R1_gt_rank1": cls_counts.get("R1", 0),
               "R2_ranking_failure_gt_buried": cls_counts.get("R2", 0),
               "R3_retrieval_only_deep": cls_counts.get("R3", 0),
               "R4_absent_no_5px_candidate": cls_counts.get("R4", 0),
               "runtime_s": round(time.time() - t0, 1)}
    json.dump(summary, open(os.path.join(OUT, "_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
