"""EXP002b -- shortlist-then-rerank.

Naked argmax(context) over all 200 candidates recovers 34 but breaks 2: context
occasionally crowns a far-field site that raw correlation never supported.
Classic fix: let correlation propose a shortlist, let context dispose.

Sweeps shortlist source x depth x rerank key. Fits nothing; K is chosen by
reading the table, and the choice is then re-verified on held-out data before
any promotion.
"""
import sys
import numpy as np, pandas as pd

D = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "pool_pilot100.csv")
P = D[D.present == 1].copy()
P["err"] = np.where(P.refined_gterr >= 0, P.refined_gterr, P.gterr)


def credit(e):
    return np.select([e <= 1, e <= 2, e <= 3, e <= 5], [1.0, .8, .6, .4], 0.0)


base = P.loc[P.groupby("pair_id")["v25"].idxmax()].set_index("pair_id")["err"]
base_ok = base <= 5

rows = []
for src in ("corr_score", "v25"):
    for K in (2, 3, 5, 8, 10, 15, 20, 30, 50, 100, 200):
        for key in ("context_combined", "grad_ncc"):
            sel = (P.sort_values(src, ascending=False).groupby("pair_id", sort=False).head(K)
                    .sort_values(key, ascending=False).groupby("pair_id", sort=False).head(1)
                    .set_index("pair_id")["err"])
            ok = sel <= 5
            c = credit(sel.values)
            rows.append(dict(shortlist=src, K=K, rerank=key,
                             le1=int((sel <= 1).sum()), le5=int(ok.sum()),
                             recovered=int((~base_ok & ok).sum()),
                             broken=int((base_ok & ~ok).sum()),
                             loc40=round(float(c.mean()) * 40, 2)))
R = pd.DataFrame(rows)
clean = R[R.broken == 0].sort_values("loc40", ascending=False)
print("=== zero-breakage configurations ===")
print(clean.head(15).to_string(index=False))
print("\n=== best overall regardless of breakage ===")
print(R.sort_values("loc40", ascending=False).head(10).to_string(index=False))
R.to_csv("shortlist_sweep.csv", index=False)

# forensics on the two pairs the naked swap broke
for pid in ("v00015", "v00041"):
    g = P[P.pair_id == pid]
    if not len(g):
        continue
    w = g.loc[g.context_combined.idxmax()]
    b = g.loc[g.v25.idxmax()]
    r = g.loc[g.corr_score.idxmax()]
    print(f"\n--- {pid} ---")
    for tag, s in (("ctx-max", w), ("v25-max", b), ("corr-max", r)):
        print(f"  {tag:9s} err={s.err:8.2f}  corr={s.corr_score:.4f} ctx={s.context_combined:.4f} "
              f"grad={s.grad_ncc:.4f} corr_rank={int((g.corr_score > s.corr_score).sum())+1}")
