"""Score a predictions.csv on a verified-generator dataset with the pptx rubric.

Localization = mean tiered credit over ALL present pairs (a rejected present pair
scores 0). Pose is awarded only where localization credit > 0. Rejection = F1 on
the positive class found==0. Calibration = ROC AUC of `score` against "this
prediction is within 5 px". Same tier tables as phase2/V48_MAX/score_phase2_official.py.
"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def lc(e): return np.select([e <= 1, e <= 2, e <= 3, e <= 5], [1., .8, .6, .4], 0.)
def sc(p): return np.select([p <= .01, p <= .02, p <= .05], [1., .6, .3], 0.)
def rc(x): return np.select([x <= .25, x <= .5, x <= 1.], [1., .6, .3], 0.)


def score(pred_csv, data_dir, label=""):
    gt = pd.read_csv(f"{data_dir}/ground_truth.csv")[["pair_id", "present", "x", "y"]] \
        .rename(columns={"x": "gx", "y": "gy"})
    man = pd.read_csv(f"{data_dir}/manifest.csv")[["pair_id", "z", "theta"]] \
        .rename(columns={"theta": "gtheta"})
    P = pd.read_csv(pred_csv).merge(gt, on="pair_id").merge(man, on="pair_id")
    A = P[P.present == 1]

    err = np.where(A.found == 1, np.hypot(A.x - A.gx, A.y - A.gy), 1e9)
    L = lc(err)
    ds = (A.scale - A.z).abs() / A.z
    dt = (A.theta - A.gtheta).abs()
    pose = np.where(L > 0, .5 * sc(ds.values) + .5 * rc(dt.values), 0.)

    yt = (P.present == 0).astype(int)
    yp = (P.found == 0).astype(int)
    f1 = f1_score(yt, yp, zero_division=0)

    corr = np.zeros(len(P), int)
    corr[(P.present == 1).values] = (np.hypot(A.x - A.gx, A.y - A.gy) <= 5).astype(int)
    auc = roc_auc_score(corr, P.score) if len(set(corr)) > 1 else 0.5

    return dict(label=label, loc=L.mean() * 40, pose=pose.mean() * 20, rej=f1 * 15,
                cal=auc * 10, le1=int((err <= 1).sum()), le5=int((err <= 5).sum()),
                n_present=len(A), rej_f1=f1, auc=auc,
                found_present=int(A.found.sum()),
                fp_absent=int(P[P.present == 0].found.sum()),
                n_absent=int((P.present == 0).sum()))


if __name__ == "__main__":
    data = sys.argv[1]
    rows = [score(p, data, l) for l, p in (a.split("=", 1) for a in sys.argv[2:])]
    R = pd.DataFrame(rows)
    R["sum60"] = R.loc_ if False else R["loc"] + R["pose"]
    R["sum85"] = R["loc"] + R["pose"] + R["rej"] + R["cal"]
    print(R.round(3).to_string(index=False))
