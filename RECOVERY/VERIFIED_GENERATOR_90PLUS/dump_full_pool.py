"""Dump Engine B's FULL 200-candidate pool with every V25 feature for a dataset.

Expensive part (pose search + per-candidate evidence) runs once; downstream
selector/discriminator experiments then reuse this artifact for free.

    python dump_full_pool.py --data DIR --out POOL.csv [--workers 5]

GT is used only to record each candidate's distance to it. Nothing is fitted.
No cache, no pair_id logic, no network.
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, cv2
from concurrent.futures import ProcessPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_V25 = os.path.abspath(os.path.join(_HERE, "..", "V25_ORIGINAL"))


def _setup():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    cv2.setNumThreads(1)
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


def process(job):
    row, data_dir = job
    _setup()
    import pickle
    from pose_fallback import perform_pose_fallback_search
    from candidate_extractor import extract_candidates_akhilesh
    from inference_phase2 import cluster_replica_families, verify_candidate_context, verify_phase_consistency
    from feature_extractors import compute_neighborhood_consistency, compute_gradient_ncc
    from periodicity import estimate_periodicity_from_corr
    from pose_refinement import refine_pose
    ck = os.path.join(_V25, "phase2", "V25_CHAMPIONSHIP")
    ranker = pickle.load(open(os.path.join(ck, "ranker.pkl"), "rb"))

    pid = row["pair_id"]; gtf = int(row["present"])
    gx, gy = float(row["x"]), float(row["y"])
    ref = cv2.imread(os.path.join(data_dir, row["reference_path"]), 0)
    srch = cv2.imread(os.path.join(data_dir, row["search_path"]), 0)
    t0 = time.time()

    pose = perform_pose_fallback_search(ref, srch)
    corr, tmpl = pose["corr_plane"], pose["best_template"]
    es, et = float(pose["best_scale"]), float(pose["best_theta"])
    tw, th = tmpl.shape[::-1]
    cands = extract_candidates_akhilesh(corr, tw, th, ref, srch, es, et, max_final_k=200)
    cands = cluster_replica_families(cands, es)
    if not cands:
        return []
    per = estimate_periodicity_from_corr(corr)
    px_, py_ = per["pitch_x"], per["pitch_y"]
    mode_strong = 1 if per["mode"] == "STRONG" else 0

    rows = []
    for c in cands:
        cx, cy = c["cx"], c["cy"]; pkx, pky = c["peak_x"], c["peak_y"]
        ctx = verify_candidate_context(ref, srch, cx, cy, es, et)
        pen = verify_phase_consistency(srch, tmpl, pkx, pky)
        ng = compute_neighborhood_consistency(srch, tmpl, pkx, pky, px_, py_)
        g = compute_gradient_ncc(srch, tmpl, pkx, pky)
        rows.append(dict(pair_id=pid, present=gtf, est_scale=es, est_theta=et,
                         mode_strong=mode_strong, cx=cx, cy=cy, peak_x=pkx, peak_y=pky,
                         corr_score=c["corr_score"], psr=c.get("psr", 0.0),
                         context_32=ctx["s32"], context_64=ctx["s64"], context_128=ctx["s128"],
                         context_combined=ctx["combined"], phase_penalty=pen,
                         dist_to_center=c.get("dist_to_center", 0.0), neigh_cons=ng, grad_ncc=g,
                         family=c.get("family_population", 1),
                         gterr=float(np.hypot(cx - gx, cy - gy)) if gtf else -1.0))
    df = pd.DataFrame(rows)
    for col in ["corr_score", "psr", "context_128", "context_combined", "phase_penalty",
                "dist_to_center", "neigh_cons", "grad_ncc"]:
        df[col + "_rel"] = df[col] - df[col].median()
    df["family_ratio"] = df["family"] / len(df)
    df["v25"] = ranker["model"].predict_proba(df[ranker["features"]])[:, 1]
    df["pool_size"] = len(df)
    df["pair_runtime"] = time.time() - t0

    # subpixel-refined coordinates for every candidate would be too slow; refine
    # only the ones any selector could plausibly pick (top-30 by each key signal)
    keep = set()
    for col, asc in (("v25", False), ("corr_score", False), ("context_combined", False),
                     ("grad_ncc", False)):
        keep |= set(df.sort_values(col, ascending=asc).head(30).index)
    df["refined_x"] = df["cx"]; df["refined_y"] = df["cy"]
    for i in keep:
        rx, ry, _, _ = refine_pose(ref, srch, es, et, int(df.at[i, "peak_x"]), int(df.at[i, "peak_y"]), corr)
        df.at[i, "refined_x"] = rx; df.at[i, "refined_y"] = ry
    df["refined_gterr"] = np.hypot(df["refined_x"] - gx, df["refined_y"] - gy) if gtf else -1.0
    return df.to_dict("records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    data = os.path.abspath(a.data)
    pairs = pd.read_csv(os.path.join(data, "pairs.csv"))
    gt = pd.read_csv(os.path.join(data, "ground_truth.csv"))
    m = pairs.merge(gt, on="pair_id")
    jobs = [(r, data) for _, r in m.iterrows()]
    t0 = time.time(); out = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, rr in enumerate(ex.map(process, jobs)):
            out.extend(rr)
            if (i + 1) % 20 == 0:
                print(f"{i+1}/{len(jobs)}  {time.time()-t0:.0f}s  rows={len(out)}", flush=True)
    D = pd.DataFrame(out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    D.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(D)} candidate rows, {D.pair_id.nunique()} pairs, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
