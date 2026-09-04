"""Run Engine B (original V25, recovered verbatim) on ANY dataset, cache-free,
and report retrieval / ranking / localization decomposition.

    python run_engineB_on_dataset.py --data DIR --out DIR [--workers 5]

DIR must contain pairs.csv, ground_truth.csv, reference/, search/.

No cache. No pair_id logic. No historical predictions. GT is used only to
measure distances -- nothing is fitted here.

Emits: predictions.csv (competition schema), candidate_atlas.csv (top-20 per
present pair), decomposition.json (Top-K recall, R1-R4 classes, <=1/2/5 px).
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, cv2
from concurrent.futures import ProcessPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_V25 = os.path.abspath(os.path.join(_HERE, "..", "V25_ORIGINAL"))
K_ATLAS = 20


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
    from candidate_extractor import extract_candidates_akhilesh, extract_nms_fast
    from inference_phase2 import cluster_replica_families, verify_candidate_context, verify_phase_consistency
    from feature_extractors import compute_neighborhood_consistency, compute_gradient_ncc
    from periodicity import estimate_periodicity_from_corr
    ck = os.path.join(_V25, "phase2", "V25_CHAMPIONSHIP")
    ranker = pickle.load(open(os.path.join(ck, "ranker.pkl"), "rb"))
    presence = pickle.load(open(os.path.join(ck, "presence.pkl"), "rb"))

    pid = row["pair_id"]
    gtf = int(row["present"]); gx, gy = float(row["x"]), float(row["y"])
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
        return dict(pair_id=pid, x=0.0, y=0.0, theta=0.0, scale=0.0, found=0, score=0.0), [], \
               dict(pair_id=pid, cls="R4", gt_rank=None, runtime=time.time() - t0)
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
        rows.append(dict(cx=cx, cy=cy, peak_x=pkx, peak_y=pky,
                         corr_score=c["corr_score"], psr=c.get("psr", 0.0),
                         context_128=ctx["s128"], context_combined=ctx["combined"],
                         phase_penalty=pen, dist_to_center=c.get("dist_to_center", 0.0),
                         neigh_cons=ng, grad_ncc=g, family=c.get("family_population", 1)))
    df = pd.DataFrame(rows)
    for col in ["corr_score", "psr", "context_128", "context_combined", "phase_penalty",
                "dist_to_center", "neigh_cons", "grad_ncc"]:
        df[col + "_rel"] = df[col] - df[col].median()
    df["family_ratio"] = df["family"] / len(df)
    df["v25"] = ranker["model"].predict_proba(df[ranker["features"]])[:, 1]
    df = df.sort_values("v25", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["gterr"] = np.hypot(df["cx"] - gx, df["cy"] - gy) if gtf else np.nan

    best = df.iloc[0]
    second = df.iloc[1] if len(df) > 1 else best
    prow = pd.DataFrame([{ "top1_score": best["v25"], "margin": best["v25"] - second["v25"],
                           "top1_corr": df.iloc[0]["corr_score"], "top1_ctx": df.iloc[0]["context_combined"],
                           "top1_neigh": df.iloc[0]["neigh_cons"], "top1_grad": df.iloc[0]["grad_ncc"],
                           "mode_strong": mode_strong }])
    pres = float(presence["model"].predict_proba(prow[presence["features"]])[0, 1])
    found = 1 if pres > 0.843 else 0

    if found:
        from pose_refinement import refine_pose
        rx, ry, _, _ = refine_pose(ref, srch, es, et, int(best["peak_x"]), int(best["peak_y"]), corr)
        pred = dict(pair_id=pid, x=float(rx), y=float(ry), theta=es * 0 + et, scale=es,
                    found=1, score=pres)
    else:
        pred = dict(pair_id=pid, x=0.0, y=0.0, theta=0.0, scale=0.0, found=0, score=pres)

    # decomposition
    dec = dict(pair_id=pid, runtime=time.time() - t0, presence=pres, found=found)
    if gtf:
        hits = df.index[df["gterr"] <= 5.0].tolist()
        gt_rank = int(df.loc[hits[0], "rank"]) if hits else None
        deep = extract_nms_fast(corr, tw, th, max_k=600, r=3)
        ddist = min((np.hypot(d["cx"] - gx, d["cy"] - gy) for d in deep), default=1e9)
        for n in (1, 5, 10, 20, 50, 100, 200):
            dec[f"top{n}"] = int(bool((df["gterr"].iloc[:min(n, len(df))] <= 5.0).any()))
        dec["deep_pool"] = int(ddist <= 5.0)
        dec["gt_rank"] = gt_rank
        dec["best_gterr"] = float(df["gterr"].min())
        dec["sel_err"] = float(np.hypot(pred["x"] - gx, pred["y"] - gy)) if found else None
        dec["cls"] = ("R1" if gt_rank == 1 else "R2" if gt_rank else "R3" if ddist <= 5.0 else "R4")
    else:
        dec["cls"] = "ABSENT"

    atlas = df.head(K_ATLAS)[["rank", "cx", "cy", "gterr", "corr_score", "psr", "context_combined",
                              "grad_ncc", "phase_penalty", "neigh_cons", "family",
                              "dist_to_center", "v25"]].copy()
    atlas.insert(0, "pair_id", pid)
    return pred, atlas.to_dict("records"), dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    data = os.path.abspath(a.data); out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    pairs = pd.read_csv(os.path.join(data, "pairs.csv"))
    gt = pd.read_csv(os.path.join(data, "ground_truth.csv"))
    m = pairs.merge(gt, on="pair_id")
    jobs = [(r, data) for _, r in m.iterrows()]
    t0 = time.time()
    preds, atlas, decs = [], [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (p, at, d) in enumerate(ex.map(process, jobs)):
            preds.append(p); atlas.extend(at); decs.append(d)
            if (i + 1) % 20 == 0:
                print(f"{i+1}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    P = pd.DataFrame(preds)[["pair_id", "x", "y", "theta", "scale", "found", "score"]]
    P.to_csv(os.path.join(out, "predictions.csv"), index=False)
    pd.DataFrame(atlas).to_csv(os.path.join(out, "candidate_atlas.csv"), index=False)
    D = pd.DataFrame(decs)
    D.to_csv(os.path.join(out, "decomposition.csv"), index=False)

    pres = D[D.cls != "ABSENT"]
    rt = D["runtime"].values
    rep = {"n_pairs": len(D), "n_present": len(pres), "n_absent": int((D.cls == "ABSENT").sum()),
           "recall": {f"top{n}": int(pres[f"top{n}"].sum()) for n in (1, 5, 10, 20, 50, 100, 200)},
           "deep_pool": int(pres["deep_pool"].sum()),
           "classes": D.cls.value_counts().to_dict(),
           "localized_le5px": int((pres["sel_err"] <= 5.0).sum()),
           "localized_le2px": int((pres["sel_err"] <= 2.0).sum()),
           "localized_le1px": int((pres["sel_err"] <= 1.0).sum()),
           "runtime_median_s": float(np.median(rt)), "runtime_max_s": float(rt.max()),
           "total_s": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(out, "engineB_report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
