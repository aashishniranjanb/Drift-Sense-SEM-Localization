"""STEP 10 calibration -- dump global-alignment evidence for the anchor (V25
rank-1) and, on R2 pairs, for the GT candidate too. Reveals whether any
threshold rule separates 'anchor is a replica, override to GT' from 'anchor is
correct, keep it' with ZERO baseline breakage. No model fitting.
"""
import os, sys, time, json
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
    from step10_global_discriminator import evidence as gevid, gscore
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
    df["v25"] = ranker["model"].predict_proba(df[ranker["features"]])[:, 1]
    df = df.sort_values("v25", ascending=False).reset_index(drop=True)
    df["gterr"] = np.hypot(df["cx"].values - gx, df["cy"].values - gy)

    anchor = df.iloc[0]
    a_ev = gevid(ref, srch, float(anchor["cx"]), float(anchor["cy"]), es, et)
    rec = dict(pair_id=pid, set_type=row["set_type"], gt_found=gtf,
               anchor_err=float(anchor["gterr"]), anchor_ok=int(gtf == 1 and anchor["gterr"] <= 5.0),
               a_big=a_ev["big_ncc"], a_grad=a_ev["big_grad"], a_rout=a_ev["r_out"],
               a_fall=a_ev["falloff"], a_lminl=a_ev["lm_inliers"], a_lmrms=a_ev["lm_rms"],
               a_g=gscore(a_ev))
    topk = df.head(K)
    gt_hits = topk.index[topk["gterr"] <= 5.0].tolist()
    if gtf == 1 and gt_hits and not rec["anchor_ok"]:
        gtc = df.loc[gt_hits[0]]
        e = gevid(ref, srch, float(gtc["cx"]), float(gtc["cy"]), es, et)
        rec.update(gt_rank=int(gt_hits[0] + 1),
                   g_big=e["big_ncc"], g_grad=e["big_grad"], g_rout=e["r_out"],
                   g_fall=e["falloff"], g_lminl=e["lm_inliers"], g_lmrms=e["lm_rms"],
                   g_g=gscore(e), g_minus_a=gscore(e) - gscore(a_ev))
    return rec


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pairs = pd.read_csv(GT_CSV)
    recs = [r for _, r in pairs.iterrows()]
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(max_workers=5) as ex:
        for i, r in enumerate(ex.map(process, recs)):
            out.append(r)
            if (i + 1) % 20 == 0:
                print(f"{i+1}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)
    D = pd.DataFrame(out)
    D.to_csv(os.path.join(OUT, "calibration_evidence.csv"), index=False)

    pres = D[D.gt_found == 1]
    r1 = pres[pres.anchor_ok == 1]                       # 43 baseline successes -- must NOT break
    r2 = pres[(pres.anchor_ok == 0) & pres.g_big.notna()]  # GT-in-topK, anchor wrong
    ab = D[D.gt_found == 0]                              # absent

    print(f"\nR1 baseline successes (n={len(r1)}): anchor global match")
    print(f"  a_big   min {r1.a_big.min():.2f}  q25 {r1.a_big.quantile(.25):.2f}  median {r1.a_big.median():.2f}")
    print(f"  a_fall  max {r1.a_fall.max():.2f}  q75 {r1.a_fall.quantile(.75):.2f}  median {r1.a_fall.median():.2f}")
    print(f"\nR2 recoverable (n={len(r2)}): anchor(replica) vs GT candidate")
    print(f"  anchor a_big median {r2.a_big.median():.2f}   GT g_big median {r2.g_big.median():.2f}")
    print(f"  anchor a_fall median {r2.a_fall.median():.2f}  GT g_fall median {r2.g_fall.median():.2f}")
    print(f"  g_minus_a  >0: {(r2.g_minus_a>0).sum()}/{len(r2)}   >0.2: {(r2.g_minus_a>0.2).sum()}   >0.35: {(r2.g_minus_a>0.35).sum()}")
    # try override rules, measure recoveries vs baseline breaks
    print(f"\n{'rule':48s} {'recover':>8s} {'break_R1':>9s}")
    for gm in (0.10, 0.20, 0.30, 0.40):
        for gbig in (0.30, 0.40, 0.50):
            for gf in (0.20, 0.30, 0.40):
                # would override an R1 pair? (fires when anchor not clearly strong AND some challenger passes)
                # here approximated on R2 only for recover; R1 break needs full topK -- flag if anchor would be replaced
                rec_n = int(((r2.g_minus_a >= gm) & (r2.g_big >= gbig) & (r2.g_fall <= gf)).sum())
                # R1 break proxy: an R1 pair where a *non-anchor* topK cand beats these thresholds vs anchor.
                # conservative proxy: R1 anchor is safe if a_big>=0.5 & a_fall<=0.2 (decide() short-circuits).
                unsafe_r1 = int(((r1.a_big < 0.50) | (r1.a_fall > 0.20)).sum())
                print(f"g_minus_a>={gm:.2f} g_big>={gbig:.2f} g_fall<={gf:.2f}       {rec_n:8d} {unsafe_r1:9d}")
                break
            break
        break
    # focused: print the R2 rows
    cols = ["pair_id", "set_type", "gt_rank", "anchor_err", "a_big", "a_fall", "a_g", "g_big", "g_fall", "g_g", "g_minus_a"]
    print("\nR2 detail:\n", r2[cols].round(3).to_string(index=False))
    print(f"\nruntime {time.time()-t0:.0f}s")
