"""V25 localization pipeline (frozen). Pose search -> candidate extraction ->
replica-family clustering -> per-candidate evidence -> ML ranker -> ML presence
gate -> subpixel refinement. Returns the prediction AND the top-1 evidence
vector the calibrator needs (the frozen pipeline computes it internally; here it
is also returned)."""
import os
import pickle
import numpy as np
import pandas as pd

from matcher import (perform_pose_fallback_search, compute_neighborhood_consistency,
                     compute_gradient_ncc)
from candidate_extractor import extract_candidates_akhilesh, cluster_replica_families
from context_matcher import verify_candidate_context
from phase_verifier import verify_phase_consistency
from periodicity_detector import estimate_periodicity_from_corr
from pose_estimator import refine_pose

_MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
with open(os.path.join(_MODELS, "ranker.pkl"), "rb") as f:
    _RANKER = pickle.load(f)
with open(os.path.join(_MODELS, "presence.pkl"), "rb") as f:
    _PRESENCE = pickle.load(f)

_PRESENCE_THRESHOLD = 0.843   # V25 native gate


def localize_grayscale(ref_img, search_img):
    pose = perform_pose_fallback_search(ref_img, search_img)
    corr_plane = pose["corr_plane"]
    best_template = pose["best_template"]
    est_scale = float(pose["best_scale"])
    est_theta = float(pose["best_theta"])
    tw, th = best_template.shape[::-1]

    cands = extract_candidates_akhilesh(corr_plane, tw, th, ref_img, search_img,
                                        est_scale, est_theta, max_final_k=200)
    cands = cluster_replica_families(cands, est_scale)
    if not cands:
        return _null_result()

    per = estimate_periodicity_from_corr(corr_plane)
    pitch_x, pitch_y = per["pitch_x"], per["pitch_y"]
    mode_strong = 1 if per["mode"] == "STRONG" else 0

    rows = []
    for c in cands:
        cx, cy = c["cx"], c["cy"]
        px, py = c["peak_x"], c["peak_y"]
        ctx = verify_candidate_context(ref_img, search_img, cx, cy, est_scale, est_theta)
        phase_pen = verify_phase_consistency(search_img, best_template, px, py)
        neigh = compute_neighborhood_consistency(search_img, best_template, px, py, pitch_x, pitch_y)
        gncc = compute_gradient_ncc(search_img, best_template, px, py)
        rows.append({"corr_score": c["corr_score"], "psr": c.get("psr", 0),
                     "context_128": ctx["s128"], "context_combined": ctx["combined"],
                     "phase_penalty": phase_pen, "family_population": c.get("family_population", 1),
                     "dist_to_center": c.get("dist_to_center", 0.0),
                     "neigh_cons": neigh, "grad_ncc": gncc})

    df = pd.DataFrame(rows)
    fcols = ["corr_score", "psr", "context_128", "context_combined", "phase_penalty",
             "dist_to_center", "neigh_cons", "grad_ncc"]
    for col in fcols:
        df[col + "_rel"] = df[col] - df[col].median()
    df["family_ratio"] = df["family_population"] / len(cands)

    rank_scores = _RANKER["model"].predict_proba(df[_RANKER["features"]])[:, 1]
    for i, c in enumerate(cands):
        c["ml_score"] = rank_scores[i]
    # frozen V25 behaviour: sort the candidate list in place; the presence-model
    # feature row is then read from df.iloc[best_idx] where best_idx == 0 (the
    # first *extracted* candidate), NOT the ML-top row. presence.pkl was trained
    # against exactly this indexing, so it is preserved verbatim here.
    cands.sort(key=lambda c: c["ml_score"], reverse=True)
    best_cand = cands[0]
    second_cand = cands[1] if len(cands) > 1 else best_cand
    best_idx = cands.index(best_cand)  # == 0

    pres_row = pd.DataFrame([{
        "top1_score": best_cand["ml_score"],
        "margin": best_cand["ml_score"] - second_cand["ml_score"],
        "top1_corr": df.iloc[best_idx]["corr_score"],
        "top1_ctx": df.iloc[best_idx]["context_combined"],
        "top1_neigh": df.iloc[best_idx]["neigh_cons"],
        "top1_grad": df.iloc[best_idx]["grad_ncc"],
        "mode_strong": mode_strong,
    }])
    pres_score = float(_PRESENCE["model"].predict_proba(pres_row[_PRESENCE["features"]])[0, 1])
    found = 1 if pres_score > _PRESENCE_THRESHOLD else 0

    # gate-independent best-candidate localization (used by the standalone
    # inference.py, which always returns a coordinate)
    raw_x, raw_y, _, _ = refine_pose(ref_img, search_img, est_scale, est_theta,
                                     best_cand["peak_x"], best_cand["peak_y"], corr_plane)

    if found == 1:
        rx, ry = raw_x, raw_y
        theta_out, scale_out = est_theta, est_scale
    else:
        rx = ry = 0.0
        theta_out = scale_out = 0.0

    ev = pres_row.iloc[0].to_dict()
    return {"x": float(rx), "y": float(ry), "theta": float(theta_out), "scale": float(scale_out),
            "found": int(found), "score": float(pres_score),
            "raw_x": float(raw_x), "raw_y": float(raw_y),
            "evidence": {k: float(v) for k, v in ev.items()},
            "corr_plane": corr_plane, "template": best_template,
            "est_scale": est_scale, "est_theta": est_theta,
            "best_peak": (int(best_cand["peak_x"]), int(best_cand["peak_y"]))}


def _null_result():
    return {"x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0, "found": 0, "score": 0.0,
            "raw_x": 0.0, "raw_y": 0.0,
            "evidence": {k: 0.0 for k in ["top1_score", "margin", "top1_corr", "top1_ctx",
                                          "top1_neigh", "top1_grad", "mode_strong"]},
            "corr_plane": None, "template": None, "est_scale": 10.0, "est_theta": 0.0,
            "best_peak": (0, 0)}
