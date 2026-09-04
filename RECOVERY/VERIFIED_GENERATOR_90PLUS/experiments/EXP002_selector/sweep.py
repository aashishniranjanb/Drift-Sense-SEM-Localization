"""EXP002 -- selector sweep over the FULL 200-candidate pool.

Hypothesis (from EXP001): the V25 learned ranker is anti-correlated with truth on
verified labels; raw structural evidence (NCC / context) selects the correct site
far more often. Test every single-signal and a few fused selectors over the whole
pool, scored by the official localization tiers.

Reads only the dumped pool. Fits nothing. GT is used to score, never to select.
"""
import sys
import numpy as np, pandas as pd

POOL = sys.argv[1] if len(sys.argv) > 1 else "pool_pilot100.csv"
D = pd.read_csv(POOL)
P = D[D.present == 1].copy()
err = np.where(P.refined_gterr >= 0, P.refined_gterr, P.gterr)
P["err"] = err
n_pairs = P.pair_id.nunique()


def credit(e):
    return np.select([e <= 1, e <= 2, e <= 3, e <= 5], [1.0, .8, .6, .4], 0.0)


def z(g, col):
    s = g[col].std()
    return (g[col] - g[col].mean()) / (s if s > 1e-12 else 1.0)


# fused selectors, computed per pair
def add_fused(df):
    out = []
    for _, g in df.groupby("pair_id", sort=False):
        g = g.copy()
        g["f_corr_ctx"] = z(g, "corr_score") + z(g, "context_combined")
        g["f_corr_ctx_grad"] = z(g, "corr_score") + z(g, "context_combined") + z(g, "grad_ncc")
        g["f_ctx_grad"] = z(g, "context_combined") + z(g, "grad_ncc")
        g["f_all4"] = (z(g, "corr_score") + z(g, "context_combined")
                       + z(g, "grad_ncc") - z(g, "phase_penalty"))
        g["f_prod"] = g["corr_score"].clip(0) * g["context_combined"].clip(0)
        out.append(g)
    return pd.concat(out)


P = add_fused(P)

SELECTORS = [
    ("v25 (baseline ranker)", "v25", False),
    ("corr_score", "corr_score", False),
    ("context_combined", "context_combined", False),
    ("context_128", "context_128", False),
    ("grad_ncc", "grad_ncc", False),
    ("neigh_cons", "neigh_cons", False),
    ("phase_penalty (low)", "phase_penalty", True),
    ("dist_to_center (low)", "dist_to_center", True),
    ("fuse corr+ctx", "f_corr_ctx", False),
    ("fuse corr+ctx+grad", "f_corr_ctx_grad", False),
    ("fuse ctx+grad", "f_ctx_grad", False),
    ("fuse all4", "f_all4", False),
    ("prod corr*ctx", "f_prod", False),
]

rows = []
for name, col, asc in SELECTORS:
    idx = P.groupby("pair_id")[col].idxmin() if asc else P.groupby("pair_id")[col].idxmax()
    sel = P.loc[idx]
    c = credit(sel["err"].values)
    rows.append(dict(selector=name,
                     le1=int((sel.err <= 1).sum()), le2=int((sel.err <= 2).sum()),
                     le3=int((sel.err <= 3).sum()), le5=int((sel.err <= 5).sum()),
                     mean_credit=round(float(c.mean()), 4),
                     loc_pts_of_40=round(float(c.mean()) * 40, 2)))
R = pd.DataFrame(rows).sort_values("mean_credit", ascending=False)
print(f"present pairs: {n_pairs}   pool/pair: {int(P.pool_size.iloc[0])}\n")
print(R.to_string(index=False))
R.to_csv("selector_sweep.csv", index=False)

# oracle ceiling
orc = P.groupby("pair_id")["err"].min()
print(f"\nORACLE (best candidate in pool): <=1px {int((orc<=1).sum())}  <=5px {int((orc<=5).sum())}"
      f"  mean_credit {credit(orc.values).mean():.4f}  -> {credit(orc.values).mean()*40:.2f}/40")
