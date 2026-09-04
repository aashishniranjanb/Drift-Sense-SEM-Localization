"""Turn engine_v3 evidence into competition-schema predictions using presence_v2.

    python predict.py --evidence E.csv --model presence_v2.pkl --out predictions.csv

`score` is the fitted probability that the prediction is correct (present AND
within 5 px) -- the quantity the rubric's calibration AUC actually measures.
`found` is that probability against the threshold chosen on TRAIN data. Rejected
pairs get zeroed pose columns, per the output contract.
"""
import argparse, pickle
import numpy as np, pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    M = pickle.load(open(a.model, "rb"))
    E = pd.read_csv(a.evidence)
    X = E.reindex(columns=M["features"])
    ok = X.notna().all(axis=1)

    score = np.zeros(len(E))
    score[ok.values] = M["model"].predict_proba(X[ok].values)[:, 1]
    found = (score >= M["threshold"]).astype(int)

    P = pd.DataFrame({"pair_id": E.pair_id,
                      "x": np.where(found == 1, E.x.fillna(0.0), 0.0),
                      "y": np.where(found == 1, E.y.fillna(0.0), 0.0),
                      "theta": np.where(found == 1, E.theta.fillna(0.0), 0.0),
                      "scale": np.where(found == 1, E.scale.fillna(0.0), 0.0),
                      "found": found, "score": score})
    P.to_csv(a.out, index=False)
    print(f"{len(P)} rows -> {a.out}   found={int(found.sum())}  "
          f"threshold={M['threshold']:.4f}")


if __name__ == "__main__":
    main()
