"""EXP005 -- engine_v3: EXP004's accuracy at a runtime that fits the 5 s budget.

EXP004 measured 5.65 s/pair median against a 5 s budget. Its two costs are the
joint pose search (2.67 s) and multi-radius context evaluation on all 200
candidates.

The shortlist sweep in EXP002 already showed the second cost is unnecessary:
selecting by context among the top-8 candidates by raw correlation gives exactly
the same result as searching all 200 (<=1px 72, <=5px 74, loc 37.74/40). K=8 is
the floor -- K=5 loses one pair. So context is evaluated on 8 candidates, not 200.

Every stage is timed, so this doubles as the profiler.

Nothing is fitted. No cache, no pair_id logic, no network.
"""
import os, sys, time
import numpy as np, cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_V25 = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "V25_ORIGINAL"))
_POSE = os.path.abspath(os.path.join(_HERE, "..", "EXP003_pose"))

K_SHORTLIST = 8


def setup():
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


def _nms(corr, tw, th, k, r=3):
    """Cheap non-maximum suppression straight off the correlation plane."""
    work = corr.copy()
    H, W = work.shape[:2]
    out = []
    for _ in range(k):
        _, mv, _, ml = cv2.minMaxLoc(work)
        if not np.isfinite(mv) or mv <= -50:
            break
        px, py = ml
        out.append(dict(peak_x=px, peak_y=py, cx=px + tw / 2.0, cy=py + th / 2.0,
                        corr_score=float(mv)))
        work[max(0, py - r):min(H, py + r + 1), max(0, px - r):min(W, px + r + 1)] = -99.0
    return out


def run_pair(ref, srch, k_shortlist=K_SHORTLIST):
    from pose_v2 import estimate_pose_v2
    from inference_phase2 import verify_candidate_context
    from pose_refinement import refine_pose
    T = {}
    t = time.time()

    pose = estimate_pose_v2(ref, srch)
    T["pose"] = time.time() - t; t = time.time()
    if pose is None or pose["corr_plane"] is None:
        return None, T
    corr, tmpl = pose["corr_plane"], pose["best_template"]
    es, et = pose["best_scale"], pose["best_theta"]
    tw, th = tmpl.shape[::-1]

    cands = _nms(corr, tw, th, k_shortlist)
    T["nms"] = time.time() - t; t = time.time()
    if not cands:
        return None, T

    best, best_ctx, vals = None, -1e9, []
    for c in cands:
        v = verify_candidate_context(ref, srch, c["cx"], c["cy"], es, et)["combined"]
        vals.append(v)
        if v > best_ctx:
            best_ctx, best = v, c
    T["context"] = time.time() - t; t = time.time()

    rx, ry, _, _ = refine_pose(ref, srch, es, et, int(best["peak_x"]), int(best["peak_y"]), corr)
    T["refine"] = time.time() - t
    vals = np.sort(np.array(vals))
    return dict(x=float(rx), y=float(ry), theta=float(et), scale=float(es),
                ctx=float(best_ctx), ctx_margin=float(best_ctx - (vals[-2] if len(vals) > 1 else 0.0)),
                corr=float(best["corr_score"]),
                corr_margin=float(best["corr_score"] - min(c["corr_score"] for c in cands)),
                n_cands=len(cands)), T
