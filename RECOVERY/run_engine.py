"""Clean-room runner for the recovered original V25 engine.

    python RECOVERY/run_engine.py --engine B --input <pairs.csv> --output <preds.csv>
    python RECOVERY/run_engine.py --engine C --input <pairs.csv> --output <preds.csv>

Engine B = original V25 exactly as recovered (RECOVERY/V25_ORIGINAL), no cache,
           no pair_id logic, from image pixels only.
Engine C = Engine B  ->  V39 surgical pose refinement (from the current runtime)
           ->  V41 residual-mix + monotonic confidence for calibration.

No network. Deterministic. pair_id is only copied to the output.
"""
import argparse
import hashlib
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_V25 = os.path.join(_HERE, "V25_ORIGINAL")

# resolve CLI paths against the real cwd BEFORE we chdir into the recovered tree
_ap = argparse.ArgumentParser()
_ap.add_argument("--engine", choices=["B", "C"], required=True)
_ap.add_argument("--input", required=True)
_ap.add_argument("--output", required=True)
_A = _ap.parse_args()
_INPUT = os.path.abspath(_A.input)
_OUTPUT = os.path.abspath(_A.output)

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
cv2.setNumThreads(1)

# the recovered v25_pipeline.py uses relative sys.path.append('phase2') etc.
os.chdir(_V25)
for p in (_V25, os.path.join(_V25, "phase2"), os.path.join(_V25, "fallbacks"),
          os.path.join(_V25, "team", "akhilesh-localization"),
          os.path.join(_V25, "phase2", "V25_CHAMPIONSHIP"),
          os.path.join(_V25, "production_engine")):
    if p not in sys.path:
        sys.path.insert(0, p)

# v25_pipeline does `from inference_phase2 import cluster_replica_families,
# verify_candidate_context, verify_phase_consistency` -- those 3 live in
# family_clustering / context_matcher / phase_verifier. Importing the real
# inference_phase2 drags in a dead PACE dependency chain, so stub it with the
# genuine implementations from their own modules (no behaviour change).
import types                                     # noqa: E402
from family_clustering import cluster_replica_families   # noqa: E402
from context_matcher import verify_candidate_context     # noqa: E402
from phase_verifier import verify_phase_consistency      # noqa: E402
_ip = types.ModuleType("inference_phase2")
_ip.cluster_replica_families = cluster_replica_families
_ip.verify_candidate_context = verify_candidate_context
_ip.verify_phase_consistency = verify_phase_consistency
sys.modules["inference_phase2"] = _ip

from v25_pipeline import run_v25_localization   # noqa: E402  (RECOVERY/V25_ORIGINAL/phase2/V25_CHAMPIONSHIP)

# --- Engine C extras: V39 pose + V41 calibration, pulled from the current runtime ---
_RUNTIME_SRC = os.path.join(_HERE, "..", "FINAL_SUBMISSION", "runtime", "src")
sys.path.insert(0, os.path.abspath(_RUNTIME_SRC))


def _v39_refine(ref, srch, x, y, th, sc):
    from pose_estimator import refine_pose_v39
    rx, ry, rt, rs, _ = refine_pose_v39(ref, srch, x, y, th, sc, max_displacement_px=1.0)
    return float(rx), float(ry), float(rt), float(rs)


def _v41_score(pres, top1_corr):
    # residual mix; monotonic in the presence score, which is what AUC rewards
    return float(np.clip(0.90 * pres + 0.10 * top1_corr, 0.0, 1.0))


def run(engine, input_csv, output_csv):
    df = pd.read_csv(input_csv)
    data_dir = os.path.dirname(os.path.abspath(input_csv))
    rows, per_pair_t = [], []
    t0 = time.time()
    for i, r in df.iterrows():
        pid = r["pair_id"]
        rp = str(r["reference_path"]).replace("\\", "/")
        sp = str(r["search_path"]).replace("\\", "/")
        rp = rp if os.path.isabs(rp) else os.path.join(data_dir, rp)
        sp = sp if os.path.isabs(sp) else os.path.join(data_dir, sp)
        ref = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
        ts = time.time()
        if ref is None or srch is None:
            pred = {"x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0, "found": 0, "score": 0.0}
        else:
            pred = run_v25_localization(ref, srch, verbose=False)
            if engine == "C" and int(pred["found"]) == 1:
                try:
                    x, y, th, sc = _v39_refine(ref, srch, pred["x"], pred["y"], pred["theta"], pred["scale"])
                    pred["x"], pred["y"], pred["theta"], pred["scale"] = x, y, th, sc
                except Exception:
                    pass
                pred["score"] = _v41_score(pred["score"], pred["score"])
        dt = time.time() - ts
        per_pair_t.append(dt)
        if int(pred["found"]) == 0:
            pred["x"] = pred["y"] = pred["theta"] = pred["scale"] = 0.0
        rows.append({"pair_id": pid, "x": pred["x"], "y": pred["y"], "theta": pred["theta"],
                     "scale": pred["scale"], "found": int(pred["found"]), "score": float(pred["score"])})
        if (i + 1) % 20 == 0:
            sys.stderr.write(f"[{i+1}/{len(df)}] {(time.time()-t0)/(i+1):.2f}s/pair\n")
    out = pd.DataFrame(rows)[["pair_id", "x", "y", "theta", "scale", "found", "score"]]
    out.to_csv(output_csv, index=False)
    pt = np.array(per_pair_t)
    sys.stderr.write(f"engine {engine}: {len(out)} pairs, median {np.median(pt):.2f}s, "
                     f"mean {pt.mean():.2f}s, p90 {np.percentile(pt,90):.2f}s, max {pt.max():.2f}s "
                     f"-> {output_csv}\n")
    # timing sidecar
    pd.DataFrame({"pair_id": df["pair_id"], "runtime_s": pt}).to_csv(
        os.path.splitext(output_csv)[0] + "_timing.csv", index=False)


if __name__ == "__main__":
    run(_A.engine, _INPUT, _OUTPUT)
