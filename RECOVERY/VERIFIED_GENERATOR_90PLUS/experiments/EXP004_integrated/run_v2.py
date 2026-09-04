"""EXP004 -- integrated candidate engine: pose_v2 + context_combined selection.

Combines the two confirmed findings:
  EXP002  argmax(context_combined) over the candidate pool beats the V25 learned
          ranker by +16.7 localization points on verified labels.
  EXP003  joint scale/rotation search raises pose credit 0.853 -> 0.962 and
          removes the |theta| dependence that caused 3 of the 4 residual
          localization failures.

Emits competition-schema predictions plus a per-pair diagnostic. Presence is
carried over from Engine B unchanged so this experiment isolates localization and
pose; the presence/calibration stage is a separate hypothesis.

No cache, no pair_id logic, no network, nothing fitted on organizer data.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, cv2
from concurrent.futures import ProcessPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_V25 = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "V25_ORIGINAL"))
_POSE = os.path.abspath(os.path.join(_HERE, "..", "EXP003_pose"))


def _setup():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    cv2.setNumThreads(1)
    for p in (_POSE, _V25, os.path.join(_V25, "phase2"), os.path.join(_V25, "fallbacks"),
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
    from pose_v2 import estimate_pose_v2
    from candidate_extractor import extract_candidates_akhilesh
    from inference_phase2 import cluster_replica_families, verify_candidate_context
    from pose_refinement import refine_pose
    ck = os.path.join(_V25, "phase2", "V25_CHAMPIONSHIP")
    presence = pickle.load(open(os.path.join(ck, "presence.pkl"), "rb"))

    pid = row["pair_id"]
    ref = cv2.imread(os.path.join(data_dir, row["reference_path"]), 0)
    srch = cv2.imread(os.path.join(data_dir, row["search_path"]), 0)
    t0 = time.time()

    pose = estimate_pose_v2(ref, srch)
    if pose is None or pose["corr_plane"] is None:
        return dict(pair_id=pid, x=0.0, y=0.0, theta=0.0, scale=0.0, found=0,
                    score=0.0, runtime=time.time() - t0, sel_ctx=np.nan)
    corr, tmpl = pose["corr_plane"], pose["best_template"]
    es, et = pose["best_scale"], pose["best_theta"]
    tw, th = tmpl.shape[::-1]

    cands = extract_candidates_akhilesh(corr, tw, th, ref, srch, es, et, max_final_k=200)
    cands = cluster_replica_families(cands, es)
    if not cands:
        return dict(pair_id=pid, x=0.0, y=0.0, theta=0.0, scale=0.0, found=0,
                    score=0.0, runtime=time.time() - t0, sel_ctx=np.nan)

    # EXP002 selector: raw multi-radius context, argmax over the whole pool
    best, best_ctx = None, -1e9
    scored = []
    for c in cands:
        ctx = verify_candidate_context(ref, srch, c["cx"], c["cy"], es, et)
        scored.append((ctx["combined"], c))
        if ctx["combined"] > best_ctx:
            best_ctx, best = ctx["combined"], c
    ctx_vals = np.array([s for s, _ in scored])
    runner = float(np.sort(ctx_vals)[-2]) if len(ctx_vals) > 1 else best_ctx

    prow = pd.DataFrame([{"top1_score": best_ctx, "margin": best_ctx - runner,
                          "top1_corr": best["corr_score"],
                          "top1_ctx": best_ctx, "top1_neigh": 0.0, "top1_grad": 0.0,
                          "mode_strong": 0}])
    feats = [f for f in presence["features"] if f in prow.columns]
    pres = float(presence["model"].predict_proba(prow.reindex(columns=presence["features"],
                                                              fill_value=0.0))[0, 1]) \
        if len(feats) else 0.0
    found = 1 if pres > 0.843 else 0

    if found:
        rx, ry, _, _ = refine_pose(ref, srch, es, et, int(best["peak_x"]), int(best["peak_y"]), corr)
        return dict(pair_id=pid, x=float(rx), y=float(ry), theta=float(et), scale=float(es),
                    found=1, score=pres, runtime=time.time() - t0,
                    sel_ctx=best_ctx, ctx_margin=best_ctx - runner)
    return dict(pair_id=pid, x=0.0, y=0.0, theta=0.0, scale=0.0, found=0, score=pres,
                runtime=time.time() - t0, sel_ctx=best_ctx, ctx_margin=best_ctx - runner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_HERE, "..", "EXP001_pilot100", "data"))
    ap.add_argument("--out", default=_HERE)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    data = os.path.abspath(a.data)
    pairs = pd.read_csv(os.path.join(data, "pairs.csv"))
    jobs = [(r, data) for _, r in pairs.iterrows()]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        out = list(ex.map(process, jobs))
    R = pd.DataFrame(out)
    R[["pair_id", "x", "y", "theta", "scale", "found", "score"]].to_csv(
        os.path.join(a.out, "predictions.csv"), index=False)
    R.round(5).to_csv(os.path.join(a.out, "diagnostic.csv"), index=False)
    print(f"{len(R)} pairs, wall {time.time()-t0:.0f}s, "
          f"runtime median {R.runtime.median():.2f}s max {R.runtime.max():.2f}s")


if __name__ == "__main__":
    main()
