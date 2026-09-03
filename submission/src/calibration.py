"""Confidence calibration.

Stage 1 (V41 residual mix, per-pair, exact):
    cal = 0.90*pose_score + 0.05*top1_score + 0.05*top1_corr
    (pose_score is 0 for rejected pairs -> cal = 0.05*(top1_score+top1_corr))

Stage 2 (V48 lean graded, batched over the full prediction set):
    a shallow full-fit HGB (models/calib_lean.pkl) predicts P(correct) from the
    8 V25-native evidence features + the Stage-1 score; a monotone bucketed
    regrade turns that into a graded confidence (crisp hits highest, weak
    rejections mid, missed detections low). found / x / y / theta / scale are
    never touched; rejected pairs keep zero pose.
"""
import os
import pickle
import numpy as np
import pandas as pd

_MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
_LEAN_PATH = os.path.join(_MODELS, "calib_lean.pkl")
_LEAN = None
if os.path.exists(_LEAN_PATH):
    with open(_LEAN_PATH, "rb") as f:
        _LEAN = pickle.load(f)


def residual_mix(pose_score, top1_score, top1_corr):
    return float(0.90 * pose_score + 0.05 * top1_score + 0.05 * top1_corr)


def apply_v48_lean(df):
    """df columns required: found, score (=Stage-1 cal), top1_score, margin,
    top1_corr, top1_ctx, top1_neigh, top1_grad, mode_strong.
    Returns df with `score` replaced by the graded V48 confidence.
    Falls back to the Stage-1 score if the model is unavailable or errors."""
    if _LEAN is None:
        return df
    try:
        FE = _LEAN["features"]
        rg = _LEAN["regrade"]
        X = df[FE].values.astype(float)
        pA = _LEAN["stageA"].predict_proba(X)[:, 1]
        found = df["found"].values.astype(int)
        f1 = found == 1
        f0 = ~f1
        s = np.zeros(len(df), dtype=float)
        pk = rg["pk_corr_w"] * df["top1_corr"].values + rg["pk_score_w"] * df["top1_score"].values
        if f1.sum():
            lo, hi = float(pk[f1].min()), float(pk[f1].max())
            pkn = (pk[f1] - lo) / (hi - lo + 1e-9)
            s[f1] = rg["hi_lo"] + rg["hi_span"] * (rg["pA_w"] * pA[f1] + rg["pk_w"] * pkn)
            susp = f1 & (pA < rg["susp_thr"])
            s[susp] = 0.15 + 0.20 * pA[susp]
        if f0.sum():
            p0 = pA[f0]
            lo, hi = float(p0.min()), float(p0.max())
            p0n = (p0 - lo) / (hi - lo + 1e-9)
            s[f0] = np.where(p0 >= 0.5, rg["f0_hi_lo"] + rg["f0_hi_span"] * p0n,
                             rg["f0_lo_lo"] + rg["f0_lo_span"] * p0n)
        out = df.copy()
        out["score"] = np.clip(s, 0.0, 1.0)
        return out
    except Exception:
        return df
