"""EXP006 -- fit the presence/confidence stage on TRAIN data only.

EXP004 left ~13 localization points on the floor: the V25 presence model rejects
31 of 78 present pairs because it was fitted in V25-ranker feature space and its
0.843 threshold is meaningless against context scores. Calibration AUC was already
1.000, so the ordering is right and only the decision boundary is wrong.

What is fitted here, and on what:
  * a logistic regression over engine_v3's evidence, trained on the generator
    TRAIN corpus (seed 100000+, disjoint from the pilot's 20260904+);
  * its decision threshold, chosen on a held-out split OF THE TRAIN CORPUS.

The pilot set is never touched during fitting -- it is the evaluation set.
`data/phase2_dev` is never touched at all: not for training, not for threshold
selection, not for feature selection, not for model selection.

Target: `score` must rank "this prediction is correct" (present AND within 5 px),
because that is exactly what the rubric's calibration AUC measures. Predicting
bare presence would be the wrong target -- a present pair we localize to the
wrong replica should get a LOW score, not a high one.

    python fit_presence.py --train-evidence E.csv --train-data DIR --out DIR
"""
import argparse, json, os, pickle
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score

FEATURES = ["ctx", "ctx_margin", "corr", "corr_margin"]


def label(ev, data_dir):
    gt = pd.read_csv(os.path.join(data_dir, "ground_truth.csv"))[["pair_id", "present", "x", "y"]]
    D = ev.merge(gt, on="pair_id", suffixes=("", "_gt"))
    err = np.hypot(D.x - D.x_gt, D.y - D.y_gt)
    D["correct"] = ((D.present == 1) & (err <= 5.0)).astype(int)
    D["err"] = np.where(D.present == 1, err, np.nan)
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-evidence", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    D = label(pd.read_csv(a.train_evidence), a.train_data).dropna(subset=FEATURES)
    X, y = D[FEATURES].values, D.correct.values
    print(f"train: {len(D)} pairs, {y.sum()} correct, {len(y)-y.sum()} not "
          f"({int((D.present==0).sum())} genuinely absent)")

    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))

    # honest in-corpus estimate of AUC and of the threshold, by grouped CV so no
    # pair contributes to both the fit and the threshold that judges it
    oof = np.zeros(len(D))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups=np.arange(len(D)) % 5):
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof)

    # threshold maximising rejection F1 (positive class = found==0 = "not correct")
    grid = np.unique(np.round(oof, 4))
    best = max(grid, key=lambda t: f1_score((y == 0).astype(int), (oof < t).astype(int),
                                            zero_division=0))
    f1 = f1_score((y == 0).astype(int), (oof < best).astype(int), zero_division=0)
    print(f"out-of-fold AUC {auc:.4f}   threshold {best:.4f}   rejection F1 {f1:.4f}")

    model.fit(X, y)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "presence_v2.pkl"), "wb") as f:
        pickle.dump({"model": model, "features": FEATURES, "threshold": float(best)}, f)
    json.dump({"n_train": int(len(D)), "features": FEATURES, "oof_auc": float(auc),
               "threshold": float(best), "oof_rejection_f1": float(f1),
               "train_evidence": os.path.abspath(a.train_evidence),
               "note": "fitted on generator TRAIN corpus only; pilot and phase2_dev untouched"},
              open(os.path.join(a.out, "fit_report.json"), "w"), indent=2)
    print(f"wrote {a.out}/presence_v2.pkl")


if __name__ == "__main__":
    main()
