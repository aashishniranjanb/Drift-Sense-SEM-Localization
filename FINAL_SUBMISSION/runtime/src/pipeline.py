"""V25 localization pipeline.

Pose search -> candidate extraction -> replica-family clustering ->
per-candidate evidence -> ML ranker -> ML presence gate -> subpixel refinement.

Multi-hypothesis retrieval: instead of collapsing to the single global-max
(scale, theta) before extracting candidates -- which on a periodic layout drops
the true site out of the pool entirely -- candidates are extracted from the top
few (scale, theta) hypotheses and merged. Each candidate keeps the hypothesis it
came from, so its structural evidence is verified against the matching template.

Returns the prediction AND the top-1 evidence vector the calibrator needs.
"""
import os
import pickle
import numpy as np
import pandas as pd

from matcher import (multi_hypothesis_search, perform_pose_fallback_search,
                     compute_neighborhood_consistency, compute_gradient_ncc)
from candidate_extractor import extract_nms_fast, cluster_replica_families
from context_matcher import verify_candidate_context
from phase_verifier import verify_phase_consistency
from periodicity_detector import estimate_periodicity_from_corr
from pose_estimator import refine_pose

_MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
with open(os.path.join(_MODELS, "ranker.pkl"), "rb") as f:
    _RANKER = pickle.load(f)
with open(os.path.join(_MODELS, "presence.pkl"), "rb") as f:
    _PRESENCE = pickle.load(f)

_PRESENCE_THRESHOLD = 0.843     # V25 native gate (register.py applies V28-C on top)
_K_SCALE = 3                    # scale hypotheses carried into extraction
_M_ROT = 2                      # rotation hypotheses per scale
_PER_HYP_K = 400               # NMS peaks drawn from each hypothesis before merge
_NMS_R = 3                      # NMS suppression radius (px) -- tighter to keep near-neighbours
_POOL_CAP = 280                 # merged pool size sent through full structural verification
_MERGE_PX = 4.0                 # dedup radius when merging pools across hypotheses


def _merge_pools(hyp_pools):
    """hyp_pools: list of (hyp_index, hyp, [candidates]) in score-desc hyp order.
    Returns one deduplicated candidate list; each candidate carries h_idx / hscale
    / htheta / htemplate / and keeps the strongest corr_score seen at that site."""
    merged = []
    for h_idx, hyp, cands in hyp_pools:
        for c in cands:
            cx, cy = c["cx"], c["cy"]
            hit = None
            for m in merged:
                if np.hypot(m["cx"] - cx, m["cy"] - cy) <= _MERGE_PX:
                    hit = m
                    break
            if hit is None:
                c = dict(c)
                c["h_idx"] = h_idx
                c["hscale"] = float(hyp["best_scale"])
                c["htheta"] = float(hyp["best_theta"])
                c["htemplate"] = hyp["best_template"]
                merged.append(c)
            elif c["corr_score"] > hit["corr_score"]:
                hit.update(corr_score=c["corr_score"], peak_x=c["peak_x"], peak_y=c["peak_y"],
                           cx=cx, cy=cy, h_idx=h_idx, hscale=float(hyp["best_scale"]),
                           htheta=float(hyp["best_theta"]), htemplate=hyp["best_template"])
    merged.sort(key=lambda c: c["corr_score"], reverse=True)
    return merged[:_POOL_CAP]


def localize_grayscale(ref_img, search_img):
    hyps = multi_hypothesis_search(ref_img, search_img, k_scale=_K_SCALE, m_rot=_M_ROT)
    top = hyps[0]
    corr_plane = top["corr_plane"]
    best_template = top["best_template"]
    est_scale = float(top["best_scale"])
    est_theta = float(top["best_theta"])

    # extract candidates per hypothesis (deep NMS, tight radius), then merge
    sh, sw = search_img.shape[:2]
    scx, scy = sw / 2.0, sh / 2.0
    hyp_pools = []
    for hi, h in enumerate(hyps):
        tw_h, th_h = h["best_template"].shape[::-1]
        ch = extract_nms_fast(h["corr_plane"], tw_h, th_h, max_k=_PER_HYP_K, r=_NMS_R)
        for c in ch:
            c["dist_to_center"] = float(np.hypot(c["cx"] - scx, c["cy"] - scy))
        hyp_pools.append((hi, h, ch))
    cands = _merge_pools(hyp_pools)
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
        hs = c.get("hscale", est_scale)
        ht = c.get("htheta", est_theta)
        tmpl = c.get("htemplate", best_template)
        ctx = verify_candidate_context(ref_img, search_img, cx, cy, hs, ht)
        phase_pen = verify_phase_consistency(search_img, tmpl, px, py)
        neigh = compute_neighborhood_consistency(search_img, tmpl, px, py, pitch_x, pitch_y)
        gncc = compute_gradient_ncc(search_img, tmpl, px, py)
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
    # frozen V25 presence indexing: df.iloc[0] is the first-extracted candidate
    # (strongest corr in the top hypothesis), NOT the ML-top row. presence.pkl was
    # trained against exactly this indexing.
    order = sorted(range(len(cands)), key=lambda i: cands[i]["ml_score"], reverse=True)
    best_i = order[0]
    second_i = order[1] if len(order) > 1 else order[0]
    best_cand = cands[best_i]

    pres_row = pd.DataFrame([{
        "top1_score": best_cand["ml_score"],
        "margin": best_cand["ml_score"] - cands[second_i]["ml_score"],
        "top1_corr": df.iloc[0]["corr_score"],
        "top1_ctx": df.iloc[0]["context_combined"],
        "top1_neigh": df.iloc[0]["neigh_cons"],
        "top1_grad": df.iloc[0]["grad_ncc"],
        "mode_strong": mode_strong,
    }])
    pres_score = float(_PRESENCE["model"].predict_proba(pres_row[_PRESENCE["features"]])[0, 1])
    found = 1 if pres_score > _PRESENCE_THRESHOLD else 0

    bhs = float(best_cand.get("hscale", est_scale))
    bht = float(best_cand.get("htheta", est_theta))
    raw_x, raw_y, _, _ = refine_pose(ref_img, search_img, bhs, bht,
                                     best_cand["peak_x"], best_cand["peak_y"],
                                     hyps[best_cand.get("h_idx", 0)]["corr_plane"])

    if found == 1:
        rx, ry = raw_x, raw_y
        theta_out, scale_out = bht, bhs
    else:
        rx = ry = 0.0
        theta_out = scale_out = 0.0

    ev = pres_row.iloc[0].to_dict()
    return {"x": float(rx), "y": float(ry), "theta": float(theta_out), "scale": float(scale_out),
            "found": int(found), "score": float(pres_score),
            "raw_x": float(raw_x), "raw_y": float(raw_y),
            "evidence": {k: float(v) for k, v in ev.items()},
            "corr_plane": corr_plane, "template": best_template,
            "est_scale": bhs if found == 1 else est_scale,
            "est_theta": bht if found == 1 else est_theta,
            "n_hypotheses": len(hyps), "pool_size": len(cands),
            "best_peak": (int(best_cand["peak_x"]), int(best_cand["peak_y"]))}


def _null_result():
    return {"x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0, "found": 0, "score": 0.0,
            "raw_x": 0.0, "raw_y": 0.0,
            "evidence": {k: 0.0 for k in ["top1_score", "margin", "top1_corr", "top1_ctx",
                                          "top1_neigh", "top1_grad", "mode_strong"]},
            "corr_plane": None, "template": None, "est_scale": 10.0, "est_theta": 0.0,
            "n_hypotheses": 0, "pool_size": 0, "best_peak": (0, 0)}
